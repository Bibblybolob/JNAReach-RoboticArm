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

    {"cmd": "health"}   -> link statistics
    {"cmd": "rebind"}   -> force a UART controller rebind
    {"cmd": "stop"}     -> mc.stop()

Install it so it survives a logout:

    sudo tee /etc/systemd/system/mycobot-broker.service >/dev/null <<'UNIT'
    [Unit]
    Description=myCobot arm serial broker
    After=multi-user.target
    [Service]
    ExecStart=/usr/bin/python3 /home/jonathan/mycobot_project/scripts/arm_broker.py
    User=jonathan
    Restart=always
    RestartSec=3
    [Install]
    WantedBy=multi-user.target
    UNIT
    sudo systemctl enable --now mycobot-broker
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
# Below this fraction of the window answering, commanding motion is refused.
HEALTHY_FRACTION = 0.35


class ArmState:
    """The single owner of the serial port, plus the latest known truth."""

    def __init__(self, poll_hz: float = 4.0, verbose: bool = True):
        self._lock = threading.Lock()
        self._sp = None
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
        import serial
        self._sp = serial.Serial(port=PORT, baudrate=BAUD, bytesize=8,
                                 parity='N', stopbits=1, timeout=1.0,
                                 xonxoff=False, rtscts=False, dsrdtr=False)
        time.sleep(2.0)
        self._sp.reset_input_buffer()

    def _reopen_after_rebind(self):
        """Recover a wedged controller without anyone having to notice."""
        try:
            if self._sp:
                self._sp.close()
        except Exception:
            pass
        self._sp = None
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
        time.sleep(0.25)
        d = self._sp.read(16384)
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
