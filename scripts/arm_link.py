#!/usr/bin/env python3
"""One way to open the arm's serial port, safely, from anywhere.

Why this exists
---------------
Thirteen entry points in this repo opened /dev/ttyTHS1 independently and none
of them coordinated. Two of them running at once is not a small problem: the
tty interleaves their bytes, so a 5-byte GET_ANGLES spliced into another
process's write produces a frame whose length field does not match its
contents. That is EXACTLY what the Atom reports as

    cmd_len error

and repeated malformed frames are what precede its LoadProhibited panic. So
concurrent access does not merely lose replies -- it manufactures the
firmware crash that then gets blamed on wiring, torque or a bad flash.

Diagnosed 2026-08-12 after a long session of chasing it the other way round:
a "dead link" turned out to be `calibrate_zero.py` sitting at its
`Aligned and ready? [y/N]` prompt, holding the port, while a diagnostic
polled the same tty. Every reading taken during that window was worthless,
and several confident conclusions were drawn from them anyway.

What this gives you
-------------------
* An exclusive lock, so a second opener WAITS instead of corrupting.
* A named holder when it cannot get in -- "held by PID 3766
  calibrate_zero.py" instead of silence you have to guess at.
* Correct framing: the reply header is SEARCHED for, never assumed to be at
  offset 0. The host UART carries the arm's internal Feetech servo bus and
  the ESP32's console output alongside real replies, so a valid answer
  routinely arrives behind 100+ bytes of other traffic.

Use it
------
    from arm_link import arm_serial, arm_mycobot, read_angles

    with arm_serial() as sp:
        angles = read_angles(sp)

    with arm_mycobot() as mc:
        mc.send_angles([0, 90, -149, 55, 0, 0], 40)

Both release the lock on the way out, including on exception, so a crashed
script does not strand the port the way a SIGKILL does.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import time

PORT = os.environ.get('MYCOBOT_PORT', '/dev/ttyTHS1')
BAUD = int(os.environ.get('MYCOBOT_BAUD', '1000000'))

# Lock file rather than the device node: flock on a tty is not reliably
# honoured across all drivers, and a plain file works everywhere. /tmp is
# fine -- the lock only needs to outlive the processes, not reboots.
LOCK_PATH = '/tmp/mycobot-ttyTHS1.lock'

GET_ANGLES = bytes([0xfe, 0xfe, 0x02, 0x20, 0xfa])
REPLY_HEADER = b'\xfe\xfe\x0e\x20'


class PortBusy(RuntimeError):
    """Another process holds the arm port. Names it, so you can go look."""


def _holders() -> list[str]:
    """Who has the port open, as 'PID cmd' strings. Best effort."""
    out = []
    try:
        r = subprocess.run(['fuser', PORT], capture_output=True, text=True,
                           timeout=5)
        pids = r.stdout.split() + r.stderr.replace(PORT + ':', '').split()
    except Exception:
        return out
    for pid in {p.strip() for p in pids if p.strip().isdigit()}:
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                cmd = f.read().replace(b'\0', b' ').decode(errors='replace')
            out.append(f'PID {pid} {cmd.strip()[:70]}')
        except OSError:
            out.append(f'PID {pid}')
    return out


@contextlib.contextmanager
def arm_lock(timeout: float = 20.0):
    """Exclusive access to the arm port, or a clear explanation of who has it."""
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    deadline = time.monotonic() + timeout
    warned = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    who = _holders()
                    raise PortBusy(
                        f'{PORT} is in use and did not free up in '
                        f'{timeout:.0f}s.\n  ' +
                        ('\n  '.join(who) if who else
                         'holder not identifiable (it may be mid-exit)') +
                        '\nTwo writers on this tty interleave bytes and the '
                        'arm reports that as `cmd_len error`, so this waits '
                        'rather than corrupting the link.')
                if not warned:
                    who = _holders()
                    print('waiting for the arm port'
                          + (f' (held by {who[0]})' if who else '') + '...',
                          flush=True)
                    warned = True
                time.sleep(0.25)
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextlib.contextmanager
def arm_serial(timeout: float = 1.0, settle: float = 2.0, lock_timeout=20.0):
    """A raw pyserial handle to the arm, held under the lock."""
    import serial
    with arm_lock(lock_timeout):
        sp = serial.Serial(port=PORT, baudrate=BAUD, bytesize=8, parity='N',
                           stopbits=1, timeout=timeout, xonxoff=False,
                           rtscts=False, dsrdtr=False)
        time.sleep(settle)
        sp.reset_input_buffer()
        try:
            yield sp
        finally:
            with contextlib.suppress(Exception):
                sp.close()


@contextlib.contextmanager
def arm_mycobot(settle: float = 2.0, lock_timeout: float = 20.0):
    """A pymycobot handle to the arm, held under the lock."""
    from pymycobot import MyCobot
    with arm_lock(lock_timeout):
        mc = MyCobot(PORT, BAUD)
        time.sleep(settle)
        try:
            yield mc
        finally:
            with contextlib.suppress(Exception):
                del mc


def read_angles(sp, tries: int = 6, wait: float = 0.35):
    """Six joint angles in degrees, or None.

    Searches for the reply header. Do NOT assume offset 0: a real 17-byte
    answer was observed sitting at the end of a 163-byte frame, behind the
    internal servo bus. Requiring data[0]==0xfe scored that as a failure and
    sent a working arm off for a power cycle.
    """
    for _ in range(tries):
        sp.reset_input_buffer()
        sp.write(GET_ANGLES)
        sp.flush()
        time.sleep(wait)
        d = sp.read(16384)
        if not d:
            continue
        i = d.find(REPLY_HEADER)
        if i >= 0 and len(d) >= i + 17:
            body = d[i + 4:i + 16]
            out = []
            for k in range(6):
                v = int.from_bytes(body[k * 2:k * 2 + 2], 'big', signed=True)
                out.append(v / 100.0)
            return out
    return None


def link_health(sp, n: int = 20):
    """(valid, crash, silent) -- an honest measure of the link."""
    valid = crash = silent = 0
    for _ in range(n):
        sp.reset_input_buffer()
        sp.write(GET_ANGLES)
        sp.flush()
        time.sleep(0.3)
        d = sp.read(16384)
        if not d:
            silent += 1
        elif b'cmd_len' in d or b'Guru' in d:
            crash += 1
        elif d.find(REPLY_HEADER) >= 0:
            valid += 1
    return valid, crash, silent


SERIAL_DEV = '3100000.serial'
DRIVER_DIR = '/sys/bus/platform/drivers/serial-tegra'


def rebind_uart(verbose: bool = True) -> bool:
    """Unbind and rebind the Tegra UART controller. No power cycle needed.

    The controller wedges: the arm goes completely silent -- zero bytes, no
    crash text, nothing unsolicited -- and no amount of reopening the port,
    resetting termios or cycling modem lines brings it back. A rebind does,
    immediately.

    Needs root, but only for two specific sysfs writes. Grant exactly those
    and nothing else, so this can self-heal without a password prompt:

        sudo visudo -f /etc/sudoers.d/mycobot-uart

        jonathan ALL=(root) NOPASSWD: /usr/bin/tee /sys/bus/platform/drivers/serial-tegra/unbind, /usr/bin/tee /sys/bus/platform/drivers/serial-tegra/bind

    Returns True if both writes were accepted. False means sudo refused (the
    rule is missing) -- run `sudo ./scripts/reset_uart.sh` by hand instead.
    """
    ok = True
    for action in ('unbind', 'bind'):
        try:
            r = subprocess.run(
                ['sudo', '-n', 'tee', f'{DRIVER_DIR}/{action}'],
                input=SERIAL_DEV, capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                ok = False
                if verbose:
                    why = (r.stderr or '').strip()
                    print(f'  {action} refused: {why[:120]}')
            time.sleep(1.0)
        except Exception as e:  # noqa: BLE001
            ok = False
            if verbose:
                print(f'  {action} failed: {e}')
    if ok and verbose:
        print('  UART controller rebound; waiting for the ESP32...')
        time.sleep(4.0)
    return ok


def ensure_link(attempts: int = 12, verbose: bool = True,
                settle_after_rebind: float = 4.0):
    """Get a working link, recovering a wedged controller on the way.

    Returns the angles it read, or None. Call this instead of telling a human
    to power cycle.

    Persistent by design. Recovery here is not reliably first-time: the same
    rebind-then-read sequence has produced a perfect frame on one attempt and
    nothing on the next three, with no code change between. Rather than
    declaring failure and handing the problem to a person, this keeps
    alternating read/rebind and lengthens the post-rebind settle each round,
    since the ESP32 needs an unpredictable moment to start answering.

    Twelve attempts is roughly a minute of trying. That is far cheaper than
    an interrupted session.
    """
    import serial  # noqa: F401  (imported for the clear error if missing)
    settle = settle_after_rebind
    for attempt in range(1, attempts + 1):
        try:
            with arm_serial() as sp:
                a = read_angles(sp, tries=8)
                if a is not None:
                    if verbose and attempt > 1:
                        print(f'  link recovered on attempt {attempt}')
                    return a
                if verbose:
                    v, c, s = link_health(sp, n=5)
                    note = ''
                    if c:
                        note = ('  <- crash frames: check for a second '
                                'writer on this port')
                    print(f'  attempt {attempt}/{attempts}: valid {v}/5 '
                          f'crash {c} silent {s}{note}')
        except PortBusy as e:
            if verbose:
                print(e)
            return None
        if attempt < attempts:
            if not rebind_uart(verbose=False):
                if verbose:
                    print('  cannot rebind: the sudoers rule is missing -- '
                          'see rebind_uart().')
                return None
            time.sleep(settle)
            # Back off gently; a controller that needs several goes tends to
            # need a longer pause each time rather than a faster retry.
            settle = min(settle * 1.3, 12.0)
    return None


@contextlib.contextmanager
def arm_session(verbose: bool = True):
    """One recovered, locked, long-lived handle. Do ALL your work inside it.

    Open the port once and keep it. Repeatedly closing and reopening is
    itself unreliable here -- measured: a session immediately after a
    controller rebind reads perfectly, and a fresh open moments later reads
    1/20 with no other change. Every script in this repo used to open, do one
    thing, and close, which turned one flaky moment into a flaky everything.

        with arm_session() as sp:
            angles = read_angles(sp)
            ...                       # keep using sp, do not reopen

    Raises RuntimeError if the link cannot be recovered at all, so callers
    fail loudly instead of proceeding against a dead port and reporting
    nonsense (or worse, commanding motion into it).
    """
    import serial
    with arm_lock(30.0):
        sp = None
        settle = 4.0
        for attempt in range(1, 13):
            if sp is None:
                sp = serial.Serial(port=PORT, baudrate=BAUD, bytesize=8,
                                   parity='N', stopbits=1, timeout=1.0,
                                   xonxoff=False, rtscts=False, dsrdtr=False)
                time.sleep(2.0)
                sp.reset_input_buffer()
            if read_angles(sp, tries=8) is not None:
                if verbose and attempt > 1:
                    print(f'  link up on attempt {attempt}')
                break
            if verbose:
                print(f'  attempt {attempt}/12: silent; rebinding')
            with contextlib.suppress(Exception):
                sp.close()
            sp = None
            if not rebind_uart(verbose=False):
                raise RuntimeError(
                    'The UART is not answering and the rebind could not run '
                    '(sudoers rule missing -- see rebind_uart()).')
            time.sleep(settle)
            settle = min(settle * 1.3, 12.0)
        else:
            with contextlib.suppress(Exception):
                if sp:
                    sp.close()
            raise RuntimeError(
                'The arm did not answer after 12 attempts with rebinds '
                'between each. Nothing else holds the port.')
        try:
            yield sp
        finally:
            with contextlib.suppress(Exception):
                sp.close()


def main() -> int:
    """`./scripts/arm_link.py` -- who holds the port, and is the arm alive?"""
    who = _holders()
    print(f'port {PORT} @ {BAUD}')
    print('holders: ' + ('\n         '.join(who) if who else 'none'))
    try:
        # One session for everything -- see arm_session().
        with arm_session() as sp:
            a = read_angles(sp, tries=8)
            print('angles:', [round(x, 2) for x in a] if a else 'unreadable')
            v, c, s = link_health(sp)
            print(f'link: valid {v}/20  crash {c}  silent {s}')
            if c:
                print('crash frames present. Check for a second writer on '
                      'this port before suspecting firmware -- interleaved '
                      'bytes are reported as cmd_len error.')
    except PortBusy as e:
        print(e)
        return 1
    except RuntimeError as e:
        print(e)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
