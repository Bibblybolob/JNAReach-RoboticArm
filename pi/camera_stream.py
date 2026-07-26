"""
Lightweight MJPEG HTTP server for the myCobot 280 Pi camera.
Streams /dev/video0 as MJPEG over HTTP on port 8080.

Usage:
    python3 camera_stream.py
    python3 camera_stream.py --fps 30 --width 640 --height 480 --quality 75

FRAME RATE NOTES

The old version was pinned to ~10 FPS by a hardcoded sleep. Three things
actually limit throughput, and the sleep was only the most visible:

  1. Capture format. Most USB webcams default to raw YUYV, which at 640x480
     is about 18 MB/s and saturates USB 2.0 -- cameras cope by dropping to
     10-15 FPS. Asking for MJPG instead makes the camera do the compression
     in hardware and typically unlocks 30 FPS at the same resolution. This is
     usually the single biggest win, and it costs nothing.

  2. Buffering. OpenCV queues frames by default, so under load you read
     progressively staler images. For visual servoing that is worse than a
     low frame rate: the control loop reacts to where the hand WAS. Buffer
     size is set to 1 so reads always return the newest frame.

  3. Re-encoding. We decode the camera's JPEG and re-encode it, which costs
     CPU on a 2GB Pi. Lower --quality reduces both CPU and bandwidth; 75 is
     visually fine for detection and noticeably cheaper than 80+.

If WiFi is the bottleneck rather than the camera, drop resolution before
dropping frame rate -- tracking degrades more gracefully with smaller frames
than with stale ones.

Actual achieved FPS is logged every 5s so you can see what you are really
getting rather than what you asked for.
"""

import argparse
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import cv2


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    camera = None
    lock = threading.Lock()
    frame_interval = 1.0 / 30.0
    quality = 75

    def do_GET(self):
        if self.path == '/' or self.path == '/?action=stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()

            frames = 0
            window_start = time.monotonic()
            next_frame = time.monotonic()
            try:
                while True:
                    with self.lock:
                        ret, frame = self.camera.read()
                    if not ret:
                        continue

                    _, jpeg = cv2.imencode(
                        '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    data = jpeg.tobytes()
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(data)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(data)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()

                    frames += 1
                    elapsed = time.monotonic() - window_start
                    if elapsed >= 5.0:
                        print(f'streaming {frames / elapsed:.1f} FPS '
                              f'({len(data) // 1024} KB/frame)', flush=True)
                        frames = 0
                        window_start = time.monotonic()

                    # Pace to the target rate. Sleep only the remaining time
                    # rather than a fixed amount, so encode and network time
                    # counts toward the interval instead of adding to it.
                    next_frame += self.frame_interval
                    sleep_for = next_frame - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        # Behind schedule: resync instead of accumulating debt
                        # and then bursting.
                        next_frame = time.monotonic()
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == '/?action=snapshot':
            with self.lock:
                ret, frame = self.camera.read()
            if ret:
                _, jpeg = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                data = jpeg.tobytes()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(503, 'Camera not available')
        else:
            self.send_error(404)

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
          f'[{fourcc_str}]')
    if fourcc_str.strip('\x00') not in ('MJPG', '') and not args.no_mjpg:
        print('NOTE: camera did not switch to MJPG; frame rate may be capped '
              'by USB bandwidth at this resolution.')

    MJPEGStreamHandler.camera = cap
    MJPEGStreamHandler.frame_interval = 1.0 / max(1, args.fps)
    MJPEGStreamHandler.quality = args.quality

    # Threaded so a snapshot request cannot stall the stream, and so a dropped
    # client is cleaned up without blocking the next connection.
    server = ThreadingHTTPServer(('0.0.0.0', args.port), MJPEGStreamHandler)
    server.daemon_threads = True
    print(f'Camera stream: http://0.0.0.0:{args.port}/?action=stream')
    print(f'Snapshot:      http://0.0.0.0:{args.port}/?action=snapshot')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        server.server_close()


if __name__ == '__main__':
    main()
