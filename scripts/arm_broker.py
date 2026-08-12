#!/usr/bin/env python3
"""One process owns the arm's serial port. Everyone else talks to it.

The problem this removes
------------------------
Thirteen entry points opened /dev/ttyTHS1 independently. Two at once
interleave their bytes on the tty, producing frames whose length field does
not match their payload -- which the Atom reports as `cmd_len error`, and
repeated malformed frames precede its panic. So concurrent access does not
just lose replies, it manufactures the firmware crash that then gets blamed
on the flash.

Locking (see arm_link.arm_lock) makes that impossible but not pleasant: every
tool still opens and closes the port, and repeated open/close is itself
unreliable on this controller -- a session straight after a rebind reads
perfectly while a fresh open moments later reads 1/20.

So: open it ONCE, here, and keep it for the life of the machine. Clients
connect over a Unix socket, which needs no TCP port and cannot collide with
anything.

    ./scripts/arm_broker.py                 # run it (or install the unit below)
    ./scripts/arm_broker.py --status        # ask a running one how it is doing

Deliberately many-client. pi/server.py did this over TCP 9000 with listen(1),
so one connection locked out everybody -- a documented, recurring waste of
time in CLAUDE.md. Any number of readers are fine here; only commands are
serialised, and only against each other.

Protocol: newline-delimited JSON, one request per line, one reply per line.

    {"cmd": "state"}
      -> {"ok": true, "angles": [...], "age_ms": 412, "link": {...}}
         The LAST KNOWN pose, with its age. Reads never wait on the wire, so
         a flaky link degrades to slightly stale rather than to failure. Check
         age_ms before trusting it for anything that matters.

    {"cmd": "send_angles", "angles": [0,90,-149,55,0,0], "speed": 40}
      -> {"ok": true}
         Refused unless the link is healthy: the myCobot protocol carries no
         checksum, so a corrupted SEND_ANGLES is simply a joint angle the arm
         obeys. Pass "force": true to override that, knowingly.

    {"cmd": "call", "method": "set_servo_calibration", "args": [6]}
      -> {"ok": true, "value": -1}
         Any whitelisted pymycobot method, run under the broker's lock. "ok"
         means the call was made, NOT that the arm acted: pymycobot returns
         -1 for anything it cannot parse, and power_on / focus_all_servos
         both return -1 while working perfectly. Verify by reading back.

    {"cmd": "health"}   -> link statistics
    {"cmd": "rebind"}   -> force a UART controller rebind
    {"cmd": "stop"}     -> mc.stop()

Install it so it survives a logout:

    sudo cp /home/jonathan/mycobot_project/scripts/mycobot-broker.service \
            /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now mycobot-broker

The unit sets PrivateTmp=no deliberately. With a private /tmp the socket
exists only inside the service, every client falls back to opening the serial
port directly, and the concurrent-access corruption this exists to prevent
comes straight back -- while the broker looks perfectly healthy from inside
its own namespace.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_link import (  # noqa: E402
    BAUD, GET_ANGLES, PORT, REPLY_HEADER, arm_lock, rebind_uart,
)

SOCK_PATH = os.environ.get('MYCOBOT_SOCK', '/tmp/mycobot-arm.sock')

# Health is judged over a rolling window of recent polls, so one bad reply
# does not look like a dead arm and one good one does not look like a fixed
# one.
WINDOW = 40
# How long to wait for a GET_ANGLES reply before giving up on that poll. The
# arm has been measured replying in 727ms; anything shorter than that throws
# away good answers and reports a healthy link as dead.
POLL_WINDOW_S = 1.5
# Below this fraction of the window answering, commanding motion is refused.
HEALTHY_FRACTION = 0.35


class ArmState:
    """The single owner of the serial port, plus the latest known truth."""

    # pymycobot methods clients may invoke through the broker.
    #
    # A whitelist, not a passthrough. set_servo_data writes raw Feetech
    # registers, and addresses 5 and 6 in that map are servo ID and baud --
    # a wrong write there takes a joint off the bus entirely. Reading those
    # registers is fine and is how temperature is obtained.
    POLL_WINDOW = POLL_WINDOW_S

    ALLOWED_CALLS = {
        # reads
        'get_angles', 'get_coords', 'get_encoder', 'get_encoders',
        'get_servo_data', 'get_servo_error', 'get_servo_max_temperature',
        'get_servo_max_voltage', 'get_servo_firmware_version',
        'is_servo_enable', 'is_all_servo_enable', 'is_power_on',
        'get_fresh_mode',
        # motion and power
        'send_angles', 'send_angle', 'send_coords', 'stop',
        'power_on', 'power_off', 'focus_servo', 'focus_all_servos',
        'release_servo', 'release_all_servos', 'set_color', 'set_fresh_mode',
        # calibration -- the reason this exists
        'set_servo_calibration',
    }

    def __init__(self, poll_hz: float = 4.0, verbose: bool = True):
        self._lock = threading.Lock()
        self._sp = None
        self._mc = None
        self._mc280 = None
        self._verbose = verbose
        self._poll_interval = 1.0 / poll_hz

        self.angles = None
        self.angles_time = 0.0
        self.recent = []           # bools, most recent last
        self.rebinds = 0
        self.started = time.time()
        self.last_error = None

    # ---- serial ----

    def _open(self):
        """Open the port once, as both a raw handle and a pymycobot handle.

        Raw for polling, because searching for the fe fe 0e 20 header is more
        reliable than pymycobot's parser on a link that also carries the
        internal Feetech bus. pymycobot for everything else, because
        reimplementing its frames by guessing opcodes is how you write a bad
        value into a servo's ID or baud register.

        pymycobot opens its own file descriptor on the same tty. That is only
        safe because THIS process is the single owner and serialises every
        use behind self._lock -- which is the entire point of the broker.
        """
        import serial
        self._sp = serial.Serial(port=PORT, baudrate=BAUD, bytesize=8,
                                 parity='N', stopbits=1, timeout=1.0,
                                 xonxoff=False, rtscts=False, dsrdtr=False)
        time.sleep(2.0)
        self._sp.reset_input_buffer()
        # BOTH classes, because neither covers the whole surface this repo
        # uses and they differ in both directions:
        #   MyCobot280 has set_fresh_mode/get_fresh_mode (the driver needs
        #     them) but no get_servo_data/get_servo_error;
        #   MyCobot has the raw servo-register reads -- which is how servo
        #     temperature is obtained, the only torque proxy on this arm --
        #     but no fresh-mode calls.
        # Holding one and not the other means some caller fails at runtime
        # with 'pymycobot has no ...' after everything looked fine.
        # NOT opened here. pymycobot opens its own file descriptor on the
        # same tty, so holding both plus the raw handle means three readers on
        # one port -- and a read() on any of them can consume bytes meant for
        # another. Polling only needs the raw handle; the pymycobot ones are
        # opened on first use and only matter while a call is in flight.
        self._mc = None
        self._mc280 = None

    def _ensure_pymycobot(self):
        """Open the pymycobot handles on demand, under the lock."""
        if self._mc is not None:
            return
        from pymycobot import MyCobot, MyCobot280
        self._mc = MyCobot(PORT, BAUD)
        self._mc280 = MyCobot280(PORT, str(BAUD))
        time.sleep(0.5)

    def _reopen_after_rebind(self):
        """Recover a wedged controller without anyone having to notice."""
        try:
            if self._sp:
                self._sp.close()
        except Exception:
            pass
        self._sp = None
        self._mc = None
        self._mc280 = None
        self.rebinds += 1
        if self._verbose:
            print(f'  link silent; rebinding UART (#{self.rebinds})',
                  flush=True)
        rebind_uart(verbose=False)
        time.sleep(4.0)
        try:
            self._open()
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)

    def _poll_once(self) -> bool:
        """One GET_ANGLES on the wire. Caller holds self._lock."""
        if self._sp is None:
            self._open()
        self._sp.reset_input_buffer()
        self._sp.write(GET_ANGLES)
        self._sp.flush()

        # Read until the reply appears, up to POLL_WINDOW -- do NOT sleep a
        # fixed guess and read once.
        #
        # The fixed 250ms wait was silently discarding good replies. Measured
        # 2026-08-12: a successful get_angles took 727ms, so the read found an
        # empty buffer and the NEXT poll's reset_input_buffer() threw away the
        # reply that had since arrived. Reads sat at 2% valid while writes
        # worked perfectly -- which reads as a dead link and is not one.
        #
        # Exits as soon as a complete frame is seen, so a healthy arm still
        # polls fast; only a slow one costs the extra wait.
        d = b''
        deadline = time.monotonic() + self.POLL_WINDOW
        i = -1
        while time.monotonic() < deadline:
            chunk = self._sp.read(4096)
            if chunk:
                d += chunk
                i = d.find(REPLY_HEADER)
                if i >= 0 and len(d) >= i + 17:
                    break
            else:
                time.sleep(0.02)
        if not d:
            return False
        i = d.find(REPLY_HEADER)
        if i < 0 or len(d) < i + 17:
            return False
        body = d[i + 4:i + 16]
        self.angles = [
            int.from_bytes(body[k * 2:k * 2 + 2], 'big', signed=True) / 100.0
            for k in range(6)]
        self.angles_time = time.time()
        return True

    def poll_forever(self):
        consecutive_bad = 0
        while True:
            try:
                with self._lock:
                    ok = self._poll_once()
                self.recent.append(ok)
                del self.recent[:-WINDOW]
                consecutive_bad = 0 if ok else consecutive_bad + 1
                # Long enough to be sure it is not one unlucky frame, short
                # enough that a wedged controller does not cost a minute.
                if consecutive_bad >= 12:
                    with self._lock:
                        self._reopen_after_rebind()
                    consecutive_bad = 0
            except Exception as e:  # noqa: BLE001
                self.last_error = str(e)
                with self._lock:
                    self._reopen_after_rebind()
                consecutive_bad = 0
            time.sleep(self._poll_interval)

    # ---- state ----

    def health(self) -> dict:
        n = len(self.recent) or 1
        good = sum(self.recent)
        return {
            'valid': good,
            'window': len(self.recent),
            'fraction': round(good / n, 3),
            'healthy': (good / n) >= HEALTHY_FRACTION,
            'rebinds': self.rebinds,
            'uptime_s': round(time.time() - self.started, 1),
            'last_error': self.last_error,
        }

    def send_angles(self, angles, speed: int, force: bool = False) -> dict:
        h = self.health()
        if not h['healthy'] and not force:
            return {
                'ok': False,
                'error': (f'link is at {h["fraction"]:.0%} and motion is '
                          'refused below '
                          f'{HEALTHY_FRACTION:.0%}. The protocol has no '
                          'checksum, so a corrupted SEND_ANGLES is just a '
                          'joint angle the arm obeys. Pass "force": true if '
                          'you accept that.'),
                'link': h,
            }
        if len(angles) != 6:
            return {'ok': False, 'error': 'need exactly 6 angles'}
        payload = bytearray([0xfe, 0xfe, 0x0f, 0x22])
        for a in angles:
            payload += int(round(a * 100)).to_bytes(2, 'big', signed=True)
        payload += bytes([int(speed) & 0xff, 0xfa])
        with self._lock:
            if self._sp is None:
                self._open()
            self._sp.write(bytes(payload))
            self._sp.flush()
        return {'ok': True, 'link': h}

    def call(self, method: str, args, kwargs=None) -> dict:
        """Invoke a whitelisted pymycobot method under the broker's lock."""
        if method not in self.ALLOWED_CALLS:
            return {'ok': False,
                    'error': f'{method!r} is not allowed through the broker. '
                             f'Allowed: {sorted(self.ALLOWED_CALLS)}'}
        with self._lock:
            if self._sp is None:
                self._open()
            self._ensure_pymycobot()
            # Model-specific class first, then the generic one. See _open().
            fn = None
            for handle in (self._mc280, self._mc):
                if handle is not None and hasattr(handle, method):
                    fn = getattr(handle, method)
                    break
            if fn is None:
                return {'ok': False,
                        'error': f'neither MyCobot280 nor MyCobot has '
                                 f'{method!r} (pymycobot version mismatch?)'}
            try:
                value = fn(*args, **(kwargs or {}))
            except Exception as e:  # noqa: BLE001
                return {'ok': False, 'error': f'{type(e).__name__}: {e}'}
        # -1 is pymycobot's "could not parse", which is NOT the same as
        # failure -- power_on and focus_all_servos both return -1 while
        # working. Report it and let the caller judge.
        return {'ok': True, 'value': value}

    def raw(self, data: bytes) -> dict:
        with self._lock:
            if self._sp is None:
                self._open()
            self._sp.write(data)
            self._sp.flush()
        return {'ok': True}


STATE: ArmState | None = None


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except Exception:
                self._reply({'ok': False, 'error': 'malformed JSON'})
                continue
            self._reply(self._dispatch(req))

    def _reply(self, obj):
        self.wfile.write((json.dumps(obj) + '\n').encode())
        self.wfile.flush()

    def _dispatch(self, req) -> dict:
        cmd = req.get('cmd')
        assert STATE is not None
        if cmd == 'state':
            age = (time.time() - STATE.angles_time) * 1000 if STATE.angles_time else None
            return {'ok': STATE.angles is not None,
                    'angles': STATE.angles,
                    'age_ms': round(age) if age is not None else None,
                    'link': STATE.health()}
        if cmd == 'health':
            return {'ok': True, 'link': STATE.health()}
        if cmd == 'send_angles':
            return STATE.send_angles(req.get('angles', []),
                                     int(req.get('speed', 40)),
                                     bool(req.get('force', False)))
        if cmd == 'call':
            return STATE.call(req.get('method', ''), req.get('args', []),
                              req.get('kwargs', {}))
        if cmd == 'rebind':
            with STATE._lock:
                STATE._reopen_after_rebind()
            return {'ok': True, 'link': STATE.health()}
        if cmd == 'stop':
            return STATE.raw(bytes([0xfe, 0xfe, 0x02, 0x29, 0xfa]))
        return {'ok': False, 'error': f'unknown cmd {cmd!r}'}


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def request(obj, timeout: float = 5.0, sock_path: str = SOCK_PATH):
    """Ask a running broker something. Returns None if none is running."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(sock_path)
    except OSError:
        return None
    try:
        s.sendall((json.dumps(obj) + '\n').encode())
        buf = b''
        while not buf.endswith(b'\n'):
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode() or '{}')
    finally:
        s.close()


class BrokerMyCobot:
    """Quacks like a pymycobot MyCobot, but goes through the broker.

    Drop-in: existing code calling mc.get_encoder(3) or
    mc.set_servo_calibration(6) needs no changes, it just stops opening the
    serial port itself. That matters because a script holding the tty while
    it waits at a prompt is what stranded the link for a whole session.

    Returns pymycobot's own values, including -1, so callers that already
    know -1 is not necessarily failure keep working.
    """

    def __init__(self, sock_path: str = SOCK_PATH):
        self.sock_path = sock_path
        if request({'cmd': 'health'}, sock_path=sock_path) is None:
            raise ConnectionError(f'no broker listening on {sock_path}')

    def __getattr__(self, method: str):
        def call(*args, **kwargs):
            # kwargs matter: the driver calls send_angles(..., _async=True),
            # and an *args-only proxy raises TypeError deep inside homing --
            # which surfaced as 'Homing failed: got an unexpected keyword
            # argument' and left the arm at an unknown pose.
            r = request({'cmd': 'call', 'method': method,
                         'args': list(args), 'kwargs': kwargs},
                        timeout=20.0, sock_path=self.sock_path)
            if r is None:
                raise ConnectionError('broker went away mid-call')
            if not r.get('ok'):
                raise RuntimeError(r.get('error', 'broker refused the call'))
            return r.get('value')
        return call


def connect_arm(port: str, direct_factory, sock_path: str = SOCK_PATH,
                verbose: bool = True):
    """Broker if it owns `port`, otherwise open directly. Use this everywhere.

    Port-aware on purpose. probe_usb_arm.py and probe_uart_bridge.py exist to
    talk to a DIFFERENT interface -- the Atom's USB-C console, or a USB-TTL
    adapter on /dev/ttyUSB0 -- and silently routing those through the broker
    would answer questions about the wrong wire, which is worse than not
    running them.
    """
    if os.path.realpath(port) != os.path.realpath(PORT):
        return direct_factory()
    try:
        mc = BrokerMyCobot(sock_path)
        if verbose:
            print(f'using the arm broker (nothing else opens {port})')
        return mc
    except ConnectionError:
        if verbose:
            print(f'no broker running; opening {port} directly. '
                  'Start ./scripts/arm_broker.py to avoid contention.')
        return direct_factory()


def connect_via_broker_or_direct(direct_factory, sock_path: str = SOCK_PATH,
                                 verbose: bool = True):
    """Prefer the broker; fall back to opening the port directly.

    Scripts should use this rather than constructing MyCobot themselves. If
    the broker is up, nothing else touches the tty; if it is not, behaviour
    is exactly as before.
    """
    try:
        mc = BrokerMyCobot(sock_path)
        if verbose:
            print(f'using the arm broker on {sock_path} '
                  '(nothing else touches the port)')
        return mc
    except ConnectionError:
        if verbose:
            print('no broker running; opening the port directly. '
                  'Start ./scripts/arm_broker.py to avoid contention.')
        return direct_factory()


def main() -> int:
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument('--sock', default=SOCK_PATH)
    ap.add_argument('--poll-hz', type=float, default=4.0)
    ap.add_argument('--status', action='store_true',
                    help='query a running broker and exit')
    args = ap.parse_args()

    if args.status:
        r = request({'cmd': 'state'}, sock_path=args.sock)
        if r is None:
            print(f'no broker listening on {args.sock}')
            return 1
        print(json.dumps(r, indent=2))
        return 0

    if os.path.exists(args.sock):
        if request({'cmd': 'health'}, sock_path=args.sock) is not None:
            print(f'a broker is already running on {args.sock}')
            return 1
        os.unlink(args.sock)      # stale socket from an unclean exit

    STATE = ArmState(poll_hz=args.poll_hz)

    # Hold the same lock every other tool respects, so a broker and a legacy
    # direct-opening script can never both be on the wire.
    with arm_lock(30.0):
        try:
            STATE._open()
        except Exception as e:  # noqa: BLE001
            print(f'could not open {PORT}: {e}')
            return 1
        threading.Thread(target=STATE.poll_forever, daemon=True).start()
        srv = Server(args.sock, Handler)
        os.chmod(args.sock, 0o666)
        print(f'broker on {args.sock}, owning {PORT} @ {BAUD}', flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print('\nshutting down')
        finally:
            srv.server_close()
            with contextlib_suppress():
                os.unlink(args.sock)
    return 0


class contextlib_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


if __name__ == '__main__':
    raise SystemExit(main())
