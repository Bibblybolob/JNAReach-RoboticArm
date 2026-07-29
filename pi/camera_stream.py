"""
Lightweight MJPEG HTTP server for the myCobot 280 Pi camera.
Streams /dev/video0 as MJPEG over HTTP on port 8080.

Usage:
    python3 camera_stream.py
    python3 camera_stream.py --fps 30 --width 640 --height 480 --quality 75

ONE CAPTURE THREAD, MANY CLIENTS

Every connected client used to run its own copy of the capture loop: read the
camera, encode a JPEG, write it out. Three things follow from that, and all
three were visible from the host as "the camera goes unresponsive and then
comes back".

  1. Two clients meant two camera.read() calls per frame period, contending
     on a lock, each getting roughly every other frame. Effective rate halves
     for both while CPU doubles.
  2. A client that dies without closing the socket -- a killed node, a
     Ctrl-C'd launch, a dropped WiFi link -- leaves its thread reading and
     encoding forever, because writes vanish into the socket buffer and raise
     nothing for a long time. Those threads accumulate across runs.
  3. `if not ret: continue` had no sleep in it. A camera that starts failing
     reads spins that loop as fast as the CPU allows, holding the lock, on a
     2GB Pi that is also running the arm server.

The Pi does not have the headroom to absorb that. When it saturates, the
symptom is not just a slow stream: `server.py` stops being serviced promptly
and the arm's TCP accept and joint reads go with it, so the driver takes ten
seconds to connect and homing times out. One overloaded Pi looks like three
unrelated faults.

So the camera is read by exactly ONE background thread, which encodes each
frame once and hands the bytes to whoever is connected. Clients never touch
the camera. A dead client's socket write times out and that thread exits. A
failing camera backs off and says so instead of spinning.

FRAME RATE NOTES

  1. Capture format. Most USB webcams default to raw YUYV, which at 640x480
     is about 18 MB/s and saturates USB 2.0 -- cameras cope by dropping to
     10-15 FPS. Asking for MJPG instead makes the camera do the compression
     in hardware and typically unlocks 30 FPS at the same resolution. This is
     usually the single biggest win, and it costs nothing.

  2. Buffering. OpenCV queues frames by default, so under load you read
     progressively staler images. For visual servoing that is worse than a
     low frame rate: the control loop reacts to where the hand WAS. Buffer
     size is set to 1 so reads always return the newest frame.

  3. Re-encoding, which used to be the binding constraint and is now avoided.
     OpenCV decodes the camera's MJPG into BGR on read, and we then encoded a
     fresh JPEG from it -- two full codec passes per frame, for bytes the
     camera had already produced. Measured on this Pi with NO clients
     connected: 18.7 FPS against the 30 requested. Not the network, not client
     contention, just codec work.

     CAP_PROP_CONVERT_RGB=0 gets the compressed buffer instead, and it is
     forwarded untouched. --quality then does nothing, because the camera's
     own encoder decides. Verified at startup rather than assumed -- backend
     support varies -- and the mode actually in use is printed. Fall back with
     --no-passthrough.

If WiFi is the bottleneck rather than the camera, drop resolution before
dropping frame rate -- tracking degrades more gracefully with smaller frames
than with stale ones.

Achieved FPS and the number of connected clients are logged every 5s, so a
run that is quietly serving two stale clients is visible rather than merely
slow.
"""

import argparse
import socket
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2


class Camera:
    """Reads the camera in one thread and publishes the latest encoded frame.

    Clients wait on `latest()` rather than touching the device, so the cost of
    running the camera is paid once regardless of how many are connected, and
    a slow or dead client cannot slow the capture loop down.
    """

    def __init__(self, cap, fps, quality, passthrough=False):
        self._cap = cap
        self._interval = 1.0 / max(1, fps)
        self._quality = quality
        # True when the camera hands us JPEG directly and we forward it
        # untouched. See try_passthrough().
        self._passthrough = passthrough
        self._cond = threading.Condition()
        self._jpeg = None
        self._captured_us = 0
        self._seq = 0
        self._stop = threading.Event()
        self._frames = 0
        self._fail_streak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def _run(self):
        next_frame = time.monotonic()
        window_start = time.monotonic()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                # Back off rather than spin. A camera that has gone away
                # returns False as fast as it can be asked, and the old code
                # asked in a tight loop while holding the device lock, which
                # is enough on its own to starve everything else on the Pi.
                self._fail_streak += 1
                if self._fail_streak in (1, 10, 100) or \
                        self._fail_streak % 500 == 0:
                    print(f'camera read failed ({self._fail_streak} in a row)',
                          flush=True)
                time.sleep(min(0.5, 0.01 * self._fail_streak))
                continue
            if self._fail_streak:
                print(f'camera recovered after {self._fail_streak} failed '
                      'reads', flush=True)
                self._fail_streak = 0

            # Stamped as close to the read as possible: this is the number the
            # host subtracts to find out how old a frame is by the time
            # anything looks at it, and every line of code between the capture
            # and the stamp is latency that hides from that measurement.
            captured_us = int(time.time() * 1e6)

            if self._passthrough:
                # Already a JPEG straight off the camera. Nothing to do.
                data = frame.tobytes()
            else:
                ok, jpeg = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
                if not ok:
                    continue
                data = jpeg.tobytes()

            # Encoded once here, not once per client.
            with self._cond:
                self._jpeg = data
                self._captured_us = captured_us
                self._seq += 1
                self._cond.notify_all()

            self._frames += 1
            elapsed = time.monotonic() - window_start
            if elapsed >= 5.0:
                print(f'streaming {self._frames / elapsed:.1f} FPS '
                      f'({len(data) // 1024} KB/frame), '
                      f'{StreamHandler.clients} client(s)', flush=True)
                self._frames = 0
                window_start = time.monotonic()

            next_frame += self._interval
            sleep_for = next_frame - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Behind schedule: resync instead of accumulating debt and
                # then bursting.
                next_frame = time.monotonic()

    def latest(self, after_seq, timeout=5.0):
        """Block until a frame newer than `after_seq`.

        Returns (seq, bytes, captured_us). A client slower than the capture
        rate is handed the CURRENT frame rather than the next one in order,
        so it drops intermediate frames instead of falling progressively
        further behind. Skipping is the right failure here: a servo loop wants
        the newest frame, not every frame.
        """
        with self._cond:
            if not self._cond.wait_for(
                    lambda: self._stop.is_set() or self._seq > after_seq,
                    timeout=timeout):
                return after_seq, None, 0
            if self._stop.is_set():
                return after_seq, None, 0
            return self._seq, self._jpeg, self._captured_us

    def snapshot(self):
        with self._cond:
            return self._jpeg


def try_passthrough(cap):
    """Ask the camera for its JPEG frames raw, skipping decode and re-encode.

    The capture loop used to decode the camera's MJPG into BGR and then encode
    a fresh JPEG from it, per frame. That is two full codec passes for a byte
    sequence the camera had already produced, and on a 2GB Pi also running the
    arm server it is the difference between the requested 30 FPS and the ~18.7
    actually achieved -- measured with NO clients connected, so it was never
    the network or client contention.

    CAP_PROP_CONVERT_RGB=0 makes read() return the compressed buffer instead.
    Support varies by backend and camera, so this verifies rather than assumes:
    a real JPEG starts FF D8 and ends FF D9. Anything else and we hand the
    capture back for the normal decode-and-encode path.

    Returns True if passthrough is usable. --quality has no effect when it is;
    the camera's own encoder decides, which is the point.
    """
    try:
        if not cap.set(cv2.CAP_PROP_CONVERT_RGB, 0):
            return False
    except Exception:
        return False

    for _ in range(10):          # first frames after a format change are junk
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        buf = frame.tobytes()
        if len(buf) > 4 and buf[:2] == b'\xff\xd8' and buf[-2:] == b'\xff\xd9':
            return True
    # Not JPEG, so put the camera back the way it was.
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception:
        pass
    return False


class StreamHandler(BaseHTTPRequestHandler):
    camera = None
    # Seconds a single write may block before the client is treated as gone.
    # Without this a client that vanished without closing its socket keeps a
    # thread alive for as long as the kernel keeps retransmitting, which is
    # minutes -- and those threads survive across runs of the host stack.
    write_timeout = 5.0
    clients = 0
    _lock = threading.Lock()

    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        if self.path in ('/', '/?action=stream'):
            self._stream()
        elif self.path == '/?action=snapshot':
            self._snapshot()
        else:
            self.send_error(404)

    def _stream(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()

        with StreamHandler._lock:
            StreamHandler.clients += 1
        # Bound how much the kernel will hold for a client that is not
        # draining. Frames already handed to the socket cannot be replaced
        # with fresher ones, so a deep send buffer is latency the skipping
        # above cannot undo -- a couple of frames' worth is plenty.
        try:
            self.connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, 128 * 1024)
        except OSError:
            pass
        # A blocking write to a dead peer is the thing that used to leak
        # threads; a timeout turns it into an exception this thread can exit on.
        self.connection.settimeout(self.write_timeout)
        seq = 0
        try:
            while True:
                seq, data, captured_us = self.camera.latest(seq)
                if data is None:
                    break            # shutting down, or no frames for 5s
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(data)}\r\n'.encode())
                # When this frame came off the sensor, by the Pi's clock. The
                # host cannot trust the absolute value -- the two clocks are
                # not synchronised -- but the OFFSET between them is constant,
                # so variation in (arrival - capture) is real variation in
                # latency. That is what "the camera sometimes lags" needs:
                # not an absolute figure, a reliable way to see a spike.
                self.wfile.write(
                    f'X-Capture-Us: {captured_us}\r\n'.encode())
                self.wfile.write(b'\r\n')
                self.wfile.write(data)
                self.wfile.write(b'\r\n')
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # OSError covers the socket timeout above. All of these mean the
            # same thing here: this client is gone, stop working for it.
            pass
        finally:
            with StreamHandler._lock:
                StreamHandler.clients -= 1

    def _snapshot(self):
        data = self.camera.snapshot()
        if data is None:
            self.send_error(503, 'Camera not available')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        pass  # suppress per-request logging


def main():
    parser = argparse.ArgumentParser(description='MJPEG Camera Stream Server')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30,
                        help='target frame rate (default 30)')
    parser.add_argument('--quality', type=int, default=75,
                        help='JPEG quality 1-100 (default 75)')
    parser.add_argument('--no-mjpg', action='store_true',
                        help='do not request MJPG capture format')
    parser.add_argument('--no-auto-exposure', action='store_true',
                        help='disable auto exposure. A UVC camera in dim '
                             'light lengthens exposure to brighten the image, '
                             'and frame time cannot be shorter than exposure '
                             'time -- so it silently caps the rate while '
                             'still reporting 30fps when asked. Measured on '
                             'this webcam: 10.2 fps auto, 30.2 fps manual. '
                             'Costs image brightness; detection needs a rate '
                             'more than it needs a pretty picture, but too '
                             'dark breaks detection outright, so measure')
    parser.add_argument('--exposure', type=float, default=0,
                        help='manual exposure value, with --no-auto-exposure. '
                             '0 leaves the driver default. Units are '
                             'driver-specific; smaller is shorter')
    parser.add_argument('--no-passthrough', action='store_true',
                        help='decode and re-encode every frame instead of '
                             'forwarding the camera JPEG untouched. Costs '
                             'roughly half the achievable frame rate on a Pi; '
                             'only useful if the camera JPEG is unusable')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device)

    # Request MJPG before setting size: some drivers only expose the higher
    # frame rates for a given resolution once the format is MJPG, and the
    # order of these calls matters on V4L2.
    if not args.no_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    # Always read the freshest frame. Stale frames hurt a servo loop more than
    # a lower frame rate does.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass  # not supported by every backend

    # See --no-auto-exposure. This is a frame rate control as much as a
    # brightness one, and it is the least obvious limit in the whole pipeline.
    if args.no_auto_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = manual in V4L2/UVC
        if args.exposure > 0:
            cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)

    if not cap.isOpened():
        print(f'ERROR: Cannot open camera /dev/video{args.device}')
        return

    # Report what the camera actually accepted, which is often not what was
    # requested -- a silent fallback here is a common reason "I set 30 FPS"
    # does not produce 30 FPS.
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = ''.join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(f'Requested: {args.width}x{args.height} @ {args.fps} FPS'
          f'{"" if args.no_mjpg else " MJPG"}')
    print(f'Camera gave: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS '
          f'[{fourcc_str}] auto-exposure '
          f'{"OFF" if args.no_auto_exposure else "on"}')
    print('NOTE: the FPS above is what the camera CLAIMS. Watch the '
          '"streaming N FPS" lines for what it delivers -- on auto exposure '
          'in dim light those differ by 3x.')
    if fourcc_str.strip('\x00') not in ('MJPG', '') and not args.no_mjpg:
        print('NOTE: camera did not switch to MJPG; frame rate may be capped '
              'by USB bandwidth at this resolution.')

    passthrough = False
    if not args.no_passthrough:
        passthrough = try_passthrough(cap)
    if passthrough:
        print('JPEG passthrough ON -- forwarding camera frames untouched, '
              'no decode/encode (--quality is ignored)')
    else:
        print(f'JPEG passthrough unavailable; decoding and re-encoding at '
              f'quality {args.quality}. Expect roughly half the frame rate '
              f'this Pi could otherwise manage.')

    camera = Camera(cap, args.fps, args.quality, passthrough=passthrough)
    camera.start()
    StreamHandler.camera = camera

    # Threaded so a snapshot cannot stall a stream and a dropped client is
    # cleaned up without blocking the next connection. Capture no longer
    # happens in these threads, so a stalled client costs a thread and
    # nothing else.
    server = ThreadingHTTPServer(('0.0.0.0', args.port), StreamHandler)
    server.daemon_threads = True
    print(f'Camera stream: http://0.0.0.0:{args.port}/?action=stream')
    print(f'Snapshot:      http://0.0.0.0:{args.port}/?action=snapshot')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cap.release()
        server.server_close()


if __name__ == '__main__':
    main()
