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
    # How long one poke is given before re-poking, and how many pokes a poll
    # may spend. Measured, not guessed -- see _poll_once.
    POLL_ATTEMPT_S = 0.05
    POLL_ATTEMPTS = 5
    # Re-apply termios before each poll. See _poll_once.
    RECONFIGURE_EACH_POLL = True
    BREAK_BEFORE_POLL = True

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
        # Rebinds since the last time a poll actually succeeded. Used to back
        # off: hammering the controller every few seconds forever is not
        # recovery, it is churn -- 194 rebinds at 0% valid, each one closing
        # the port and reopening it 4s later, possibly on top of whatever
        # recovery was in progress.
        self.rebinds_since_good = 0
        self.started = time.time()
        self.last_error = None
        # Pokes the last successful poll needed. 1 means the link answered
        # first time; a figure creeping toward POLL_ATTEMPTS is the link
        # degrading while the valid fraction still reads as healthy.
        self.poll_attempts_used = 0

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
        self.rebinds_since_good += 1
        if self._verbose:
            print(f'  link silent; rebinding UART (#{self.rebinds})',
                  flush=True)
        rebind_uart(verbose=False)
        # Back off once rebinding is clearly not working. Doubling from 4s to
        # a 60s ceiling means a genuinely wedged controller is still retried,
        # without the port being torn down every few seconds indefinitely.
        settle = min(4.0 * (2 ** min(self.rebinds_since_good, 4)), 60.0)
        if self._verbose and self.rebinds_since_good:
            print(f'  rebind #{self.rebinds_since_good} since last good read; '
                  f'waiting {settle:.0f}s', flush=True)
        time.sleep(settle)
        try:
            self._open()
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)

    def _poll_once(self) -> bool:
        """One GET_ANGLES on the wire. Caller holds self._lock."""
        if self._sp is None:
            self._open()
        # These two together took the read path from ~10% valid to 60-69%
        # sustained. They are NOT independent, and the interaction is the
        # whole point -- measured 2026-08-13, 12 trials per cell on a live
        # arm, via probe(reconf=, brk=):
        #
        #     reconf  break   hits    bytes read
        #     no      no      3/12    6021
        #     no      yes     7/12    5733
        #     yes     no      1/12     152     <- WORSE than doing nothing
        #     yes     yes     9/12    1415
        #
        # So the break is the mechanism that actually elicits a reply, and
        # reasserting termios only pays off alongside it. Do not "simplify"
        # this by keeping the reconfigure and dropping the break: that is the
        # 1/12 cell, the worst of the four.
        #
        # The byte counts say why. Without the reconfigure the port takes
        # ~6KB of junk per cell and the reply is buried in it; with it the
        # channel is quiet (152 bytes) but nothing answers unless the break
        # prompts it. Quiet-and-prompted is the combination that parses.
        #
        # Caveat on the numbers: 12 trials per cell. The ORDERING is solid --
        # it reproduced the four-way ranking -- but treat 75% as indicative.
        # A 3/3 result earlier in the same investigation evaporated when
        # rerun at 10 trials, which is why the cell size is written down.
        # ASK AGAIN rather than wait longer. Two measurements, 2026-08-13:
        #
        #   how long the read waits    30ms 60ms 120ms 250ms 500ms 1000ms
        #   replies                     67%  80%   57%   73%   80%    70%
        #
        # Flat. A reply that is coming has arrived inside 30ms; past that
        # there is nothing to wait for, and the 1.5s window was spending a
        # second and a half per failure to learn that. Repeating the poke
        # instead moves the number that matters:
        #
        #   pokes per attempt   1     2     3
        #   replies            76%   86%   92%     (80 trials per cell)
        #
        # Consistent with each poke failing on its own merits about a quarter
        # of the time -- the outcome sequence is close to independent, runs
        # test z = -1.35 over 160 -- which is why a second ask recovers most
        # of what the first lost, and why waiting cannot.
        #
        # So a failed poll is now CHEAPER as well as rarer: several attempts
        # of 50ms fit in the window that one attempt used to occupy alone.
        #
        # The 727ms reply this window was originally sized for was very
        # probably a straggler answering an EARLIER poke, not one reply
        # taking that long. Nothing in the sweep above reproduces it.
        deadline = time.monotonic() + self.POLL_WINDOW
        for attempt in range(self.POLL_ATTEMPTS):
            # The full recipe every attempt: quiet the channel, then prompt
            # it. See the table above -- the break is what elicits a reply and
            # the reconfigure is what stops it arriving buried in junk.
            if self.RECONFIGURE_EACH_POLL:
                try:
                    self._sp.baudrate = BAUD   # tcsetattr even if unchanged
                except Exception:
                    pass
            if self.BREAK_BEFORE_POLL:
                try:
                    self._sp.send_break(0.01)
                except Exception:
                    pass
            self._sp.reset_input_buffer()
            self._sp.write(GET_ANGLES)
            self._sp.flush()

            d = b''
            until = min(time.monotonic() + self.POLL_ATTEMPT_S, deadline)
            while time.monotonic() < until:
                chunk = self._sp.read(4096)
                if chunk:
                    d += chunk
                    i = d.find(REPLY_HEADER)
                    if i >= 0 and len(d) >= i + 17:
                        body = d[i + 4:i + 16]
                        self.angles = [
                            int.from_bytes(body[k * 2:k * 2 + 2],
                                           'big', signed=True) / 100.0
                            for k in range(6)]
                        self.angles_time = time.time()
                        self.poll_attempts_used = attempt + 1
                        return True
                else:
                    time.sleep(0.005)
            if time.monotonic() >= deadline:
                break
        self.poll_attempts_used = self.POLL_ATTEMPTS
        return False

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
            'rebinds_since_good': self.rebinds_since_good,
            'poll_attempts_used': self.poll_attempts_used,
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

    def sniff(self, seconds: float = 3.0, poke: bool = True) -> dict:
        """What is actually on the wire, as bytes.

        The whole point of the broker is that nothing else may open this port,
        which also means nothing else can watch it. When reads fail but writes
        land, the question is whether replies are ABSENT or arriving MALFORMED
        -- and those need opposite fixes. Only raw bytes distinguish them.
        """
        with self._lock:
            if self._sp is None:
                self._open()
            self._sp.reset_input_buffer()
            if poke:
                self._sp.write(GET_ANGLES)
                self._sp.flush()
            buf = b''
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                c = self._sp.read(4096)
                if c:
                    buf += c
                else:
                    time.sleep(0.02)
        printable = sum(1 for b in buf if 32 <= b < 127 or b in (10, 13))
        return {
            'ok': True,
            'bytes': len(buf),
            'hex': buf[:400].hex(' '),
            'printable_ratio': round(printable / len(buf), 2) if buf else None,
            'has_reply_header': buf.find(REPLY_HEADER) >= 0,
            'has_servo_bus': buf.find(b'\xff\xff') >= 0,
            'has_crash_text': (b'cmd_len' in buf) or (b'Guru' in buf),
        }

    # Commands worth poking with. Different opcodes take different paths
    # through the Atom's firmware, so one may answer when another does not.
    PROBE_CMDS = {
        'get_angles':   bytes([0xfe, 0xfe, 0x02, 0x20, 0xfa]),
        'get_coords':   bytes([0xfe, 0xfe, 0x02, 0x23, 0xfa]),
        'get_encoders': bytes([0xfe, 0xfe, 0x02, 0x35, 0xfa]),
        'is_power_on':  bytes([0xfe, 0xfe, 0x02, 0x12, 0xfa]),
        'get_fresh':    bytes([0xfe, 0xfe, 0x02, 0x3d, 0xfa]),
    }

    def probe(self, cmd='get_angles', wait=0.5, flush=True, repeat=1,
              trials=6, gap=0.15, dtr=None, rts=None,
              reconf=None, brk=None) -> dict:
        """One parameterised read attempt, repeated, reporting the hit rate.

        Exists so the read strategy can be SWEPT rather than guessed. Every
        knob that could plausibly matter -- how long to wait, whether to flush
        first, how many times to poke, which opcode, the modem lines -- is a
        separate argument, and the caller can walk the space and measure.

        reconf/brk default to whatever _poll_once() is doing, so a bare probe
        measures the SAME strategy the poll uses. They were once absent here
        and present there, which made probe report 0/12 while the poll ran at
        60% -- the probe was measuring a different read path and looked like a
        dead link.
        """
        if reconf is None:
            reconf = self.RECONFIGURE_EACH_POLL
        if brk is None:
            brk = self.BREAK_BEFORE_POLL
        hits = 0
        total_bytes = 0
        with self._lock:
            if self._sp is None:
                self._open()
            sp = self._sp
            if dtr is not None:
                try: sp.setDTR(bool(dtr))
                except Exception: pass
            if rts is not None:
                try: sp.setRTS(bool(rts))
                except Exception: pass
            payload = self.PROBE_CMDS.get(cmd, self.PROBE_CMDS['get_angles'])
            for _ in range(trials):
                if reconf:
                    try: sp.baudrate = BAUD
                    except Exception: pass
                if brk:
                    try: sp.send_break(0.01)
                    except Exception: pass
                if flush:
                    sp.reset_input_buffer()
                for _ in range(repeat):
                    sp.write(payload)
                    sp.flush()
                    if repeat > 1:
                        time.sleep(0.02)
                d = b''
                end = time.monotonic() + wait
                while time.monotonic() < end:
                    c = sp.read(4096)
                    if c:
                        d += c
                        if d.find(b'\xfe\xfe') >= 0 and len(d) >= 5:
                            break
                    else:
                        time.sleep(0.01)
                total_bytes += len(d)
                if d.find(b'\xfe\xfe') >= 0:
                    hits += 1
                time.sleep(gap)
        return {'ok': True, 'hits': hits, 'trials': trials,
                'rate': round(hits / trials, 3), 'bytes': total_bytes}

    def probe_rx(self, rx_baud=None, wait=0.8, trials=4, mode='normal',
                 parity=None, stopbits=None, brk=False, reopen=False) -> dict:
        """Poke at 1000000, then read under DIFFERENT line settings.

        Motivated by bytes that arrive but do not parse: 738 and 2693 byte
        bursts with no `fe fe` anywhere. That is what a receive-side rate
        mismatch looks like, and nothing so far has proven the Atom transmits
        at the rate it receives -- writes landing only proves the RX side of
        the ARM is right.

        Writes always go out at 1000000, because that demonstrably works. Only
        the read settings vary.
        """
        import serial
        hits = 0
        total = 0
        best_hex = ''
        with self._lock:
            if self._sp is None:
                self._open()
            sp = self._sp
            orig_baud = sp.baudrate
            orig_par, orig_stop = sp.parity, sp.stopbits
            try:
                for _ in range(trials):
                    sp.baudrate = orig_baud
                    sp.reset_input_buffer()
                    if brk:
                        try:
                            sp.send_break(0.01)
                        except Exception:
                            pass
                    sp.write(GET_ANGLES)
                    sp.flush()
                    if rx_baud and rx_baud != orig_baud:
                        sp.baudrate = rx_baud
                    if parity is not None:
                        sp.parity = parity
                    if stopbits is not None:
                        sp.stopbits = stopbits
                    d = b''
                    end = time.monotonic() + wait
                    while time.monotonic() < end:
                        if mode == 'osread':
                            try:
                                import os as _os
                                c = _os.read(sp.fileno(), 4096)
                            except Exception:
                                c = b''
                        else:
                            c = sp.read(4096)
                        if c:
                            d += c
                        else:
                            time.sleep(0.01)
                    total += len(d)
                    if d.find(b'\xfe\xfe') >= 0:
                        hits += 1
                        if not best_hex:
                            best_hex = d[:60].hex(' ')
                    elif d and not best_hex:
                        best_hex = d[:60].hex(' ')
            finally:
                sp.baudrate = orig_baud
                try:
                    sp.parity, sp.stopbits = orig_par, orig_stop
                except Exception:
                    pass
        return {'ok': True, 'hits': hits, 'trials': trials, 'bytes': total,
                'sample': best_hex}

    # struct serial_icounter_struct opens with 24 ints, the first ten being
    # cts, dsr, rng, dcd, rx, tx, frame, overrun, parity, brk.
    _ICOUNT_FIELDS = ('cts', 'dsr', 'rng', 'dcd', 'rx', 'tx',
                      'frame', 'overrun', 'parity', 'brk')
    TIOCGICOUNT = 0x545D

    def _icount(self) -> dict:
        """What the KERNEL says crossed this port. Caller holds self._lock."""
        import fcntl
        import struct
        buf = fcntl.ioctl(self._sp.fileno(), self.TIOCGICOUNT,
                          struct.pack('24i', *([0] * 24)))
        return dict(zip(self._ICOUNT_FIELDS, struct.unpack('24i', buf)[:10]))

    def counters(self, poke: bool = True, wait: float = 0.4,
                 trials: int = 20, rx_baud: int | None = None,
                 gap: float = 0.05,
                 reconf: bool | None = None, brk: bool | None = None) -> dict:
        """Per-transaction kernel counters, which userspace cannot infer.

        Every probe so far measures the same thing: whether a frame turned up
        in a buffer. That cannot separate the three mechanisms behind a
        missing reply, and they need different fixes:

            tx advanced, rx did not           nothing came back at all
            rx advanced, frame/overrun too    bytes came back corrupted
            rx advanced, counters clean       bytes came back and we lost them

        TIOCGICOUNT is the driver's own tally of characters and line errors,
        sampled either side of one transaction, so it answers that directly
        rather than by inference. It also confirms the write left: a poll that
        does not advance tx by 5 never reached the wire, and no amount of
        read-side tuning would have helped it.

        Uses the poll's own line discipline (reconfigure + break), so it
        measures the read path actually in service -- the mistake that
        invalidated the earlier strategy sweep.

        rx_baud reads back at a different rate from the one written at, which
        is what makes the framing-error count a BAUD measurement. Hit rate
        cannot do that job: 40 trials carry about +/-8% of noise, so a 2%
        clock error is invisible in it, while the same run yields thousands
        of characters whose stop bits either land or do not.

        The framing-error count doubles as a CRASH counter. The Atom's boot
        output comes out at the ESP32 ROM's own rate, not this port's, so a
        reboot lands as a burst of a few hundred framing errors -- measured
        at ~556 each, arriving in exact multiples. That gives a count of
        reboots per run, which hit rate cannot separate from ordinary silence.
        """
        out = {'absent': 0, 'framed': 0, 'unparsed': 0, 'line_errors': 0}
        deltas = {k: 0 for k in self._ICOUNT_FIELDS}
        rx_when_absent = []
        # Per-trial outcome, in order. rx=0 cannot by itself say whether the
        # arm stayed silent or this end went deaf -- but the two differ in
        # SHAPE. A receiver that wedges fails in runs; a command lost on its
        # own merits fails independently. Only the sequence shows that.
        seq = []
        if reconf is None:
            reconf = self.RECONFIGURE_EACH_POLL
        if brk is None:
            brk = self.BREAK_BEFORE_POLL
        t0 = time.monotonic()
        with self._lock:
            if self._sp is None:
                self._open()
            sp = self._sp
            for _ in range(trials):
                before = self._icount()
                if reconf:
                    try:
                        sp.baudrate = BAUD
                    except Exception:
                        pass
                if brk:
                    try:
                        sp.send_break(0.01)
                    except Exception:
                        pass
                sp.reset_input_buffer()
                if poke:
                    sp.write(GET_ANGLES)
                    sp.flush()
                # Written at BAUD, which demonstrably works. Only the read
                # rate varies, so the errors counted below are the read's.
                if rx_baud and rx_baud != sp.baudrate:
                    try:
                        sp.baudrate = rx_baud
                    except Exception:
                        pass
                d = b''
                end = time.monotonic() + wait
                while time.monotonic() < end:
                    c = sp.read(4096)
                    if c:
                        d += c
                        i = d.find(REPLY_HEADER)
                        if i >= 0 and len(d) >= i + 17:
                            break
                    else:
                        time.sleep(0.01)
                after = self._icount()
                if rx_baud:
                    try:
                        sp.baudrate = BAUD
                    except Exception:
                        pass
                for k in self._ICOUNT_FIELDS:
                    deltas[k] += after[k] - before[k]
                if any(after[k] - before[k]
                       for k in ('frame', 'overrun', 'parity')):
                    out['line_errors'] += 1
                i = d.find(REPLY_HEADER)
                if i >= 0 and len(d) >= i + 17:
                    out['framed'] += 1
                    seq.append('.')
                elif d:
                    out['unparsed'] += 1
                    seq.append('?')
                else:
                    out['absent'] += 1
                    rx_when_absent.append(after['rx'] - before['rx'])
                    seq.append('X')
                time.sleep(gap)
        rx = deltas['rx'] or 1
        elapsed = time.monotonic() - t0
        return {'ok': True, 'trials': trials, 'rx_baud': rx_baud or BAUD,
                'reconf': reconf, 'brk': brk,
                # ~556 framing errors per reboot; see the docstring.
                'reboots': round(deltas['frame'] / 556.0, 1),
                'elapsed_s': round(elapsed, 1),
                'reboots_per_min': round(deltas['frame'] / 556.0
                                         * 60.0 / max(elapsed, 1e-6), 1),
                **out,
                'sequence': ''.join(seq),
                'frame_err_per_char': round(deltas['frame'] / rx, 4),
                'rx_chars_when_no_bytes_read': rx_when_absent,
                'icount_delta': deltas}

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
        if cmd == 'probe_rx':
            return STATE.probe_rx(
                rx_baud=req.get('rx_baud'), wait=float(req.get('wait', 0.8)),
                trials=int(req.get('trials', 4)),
                mode=req.get('mode', 'normal'),
                parity=req.get('parity'), stopbits=req.get('stopbits'),
                brk=bool(req.get('brk', False)),
                reopen=bool(req.get('reopen', False)))
        if cmd == 'probe':
            return STATE.probe(
                cmd=req.get('probe_cmd', 'get_angles'),
                wait=float(req.get('wait', 0.5)),
                flush=bool(req.get('flush', True)),
                repeat=int(req.get('repeat', 1)),
                trials=int(req.get('trials', 6)),
                gap=float(req.get('gap', 0.15)),
                dtr=req.get('dtr'), rts=req.get('rts'),
                reconf=req.get('reconf'), brk=req.get('brk'))
        if cmd == 'sniff':
            return STATE.sniff(float(req.get('seconds', 3.0)),
                               bool(req.get('poke', True)))
        if cmd == 'counters':
            return STATE.counters(
                poke=bool(req.get('poke', True)),
                wait=float(req.get('wait', 0.4)),
                trials=int(req.get('trials', 20)),
                rx_baud=req.get('rx_baud'),
                gap=float(req.get('gap', 0.05)),
                reconf=req.get('reconf'), brk=req.get('brk'))
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
