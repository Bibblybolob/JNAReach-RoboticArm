"""Checks on the Pi camera server, runnable without a Pi or a camera.

    python3 pi/test_camera_stream.py

This file is deployed to the robot and debugged over the network, which is
slow and unpleasant, so the three faults that produced "the camera goes
unresponsive and then comes back" are pinned here against a stub camera:

  - two clients must not halve each other's frame rate. They used to: every
    client ran its own capture loop, so N clients meant N camera.read() calls
    per frame period contending on a lock.
  - a client that vanishes without closing must not leak a thread. Writes to
    a dead peer disappear into the socket buffer and raise nothing for
    minutes, so those threads accumulated across runs of the host stack and
    kept reading and encoding the whole time.
  - a failing camera must back off, not spin. `if not ret: continue` had no
    sleep in it, so a camera that started failing pegged a core on a 2GB Pi
    that is also running the arm server -- which is why one overloaded Pi
    showed up as a slow arm connect AND a homing timeout AND a dead camera.

Remember that editing pi/ does nothing until ./scripts/redeploy_pi.sh runs.
"""
import importlib.util
import socket
import sys
import threading
import time

import numpy as np

import os
spec = importlib.util.spec_from_file_location(
    'cs', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'camera_stream.py'))
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


class StubCap:
    """Fake VideoCapture. `failing` makes read() return False like a dead cam."""

    def __init__(self, fps=30):
        self.reads = 0
        self.failing = False
        self._period = 1.0 / fps
        self._next = time.monotonic()

    def read(self):
        self.reads += 1
        if self.failing:
            return False, None
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + self._period, time.monotonic())
        return True, np.zeros((120, 160, 3), dtype=np.uint8)

    def release(self):
        pass


def read_stream(host, port, seconds, hard_kill=False):
    """Count MJPEG boundaries received over `seconds`."""
    s = socket.create_connection((host, port), timeout=5)
    s.sendall(b'GET /?action=stream HTTP/1.1\r\nHost: x\r\n\r\n')
    end = time.monotonic() + seconds
    count = 0
    s.settimeout(1.0)
    try:
        while time.monotonic() < end:
            chunk = s.recv(65536)
            if not chunk:
                break
            count += chunk.count(b'--frame')
    except socket.timeout:
        pass
    if hard_kill:
        # Abrupt close with SO_LINGER 0 sends RST -- the closest thing to a
        # machine disappearing that can be done locally.
        import struct
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                     struct.pack('ii', 1, 0))
    s.close()
    return count


def main():
    from http.server import ThreadingHTTPServer

    cap = StubCap(fps=30)
    cam = cs.Camera(cap, fps=30, quality=75)
    cam.start()
    cs.StreamHandler.camera = cam
    srv = ThreadingHTTPServer(('127.0.0.1', 0), cs.StreamHandler)
    srv.daemon_threads = True
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.5)

    print('1. ONE CLIENT')
    n1 = read_stream('127.0.0.1', port, 2.0)
    print(f'   {n1/2.0:.1f} frames/s')

    print('2. TWO CLIENTS AT ONCE (old code halved each; should not now)')
    res = {}
    ts = [threading.Thread(target=lambda i=i: res.__setitem__(
        i, read_stream('127.0.0.1', port, 2.0))) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f'   client A {res[0]/2.0:.1f} f/s, client B {res[1]/2.0:.1f} f/s')
    reads_before = cap.reads
    time.sleep(1.0)
    print(f'   camera reads while 0 clients connected: '
          f'{cap.reads - reads_before} in 1s (capture runs continuously)')

    print('3. CLIENT KILLED ABRUPTLY (RST)')
    before = threading.active_count()
    read_stream('127.0.0.1', port, 1.0, hard_kill=True)
    time.sleep(cs.StreamHandler.write_timeout + 2.0)
    after = threading.active_count()
    print(f'   threads {before} -> {after}, clients counter = '
          f'{cs.StreamHandler.clients}')
    assert cs.StreamHandler.clients == 0, 'leaked a client'

    print('4. CAMERA STARTS FAILING (old code spun a core here)')
    cap.failing = True
    r0 = cap.reads
    t0 = time.monotonic()
    time.sleep(2.0)
    rate = (cap.reads - r0) / (time.monotonic() - t0)
    print(f'   {rate:.0f} failed reads/s while down '
          f'(unbounded spin would be 10k+)')
    assert rate < 200, f'still spinning: {rate:.0f}/s'
    cap.failing = False
    time.sleep(1.0)
    print('5. RECOVERS')
    n = read_stream('127.0.0.1', port, 1.5)
    print(f'   {n/1.5:.1f} frames/s after recovery')
    assert n > 10, 'did not recover'

    cam.stop()
    srv.shutdown()
    print('\nall checks passed')


if __name__ == '__main__':
    sys.exit(main())
