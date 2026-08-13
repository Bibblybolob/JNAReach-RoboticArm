#!/usr/bin/env python3
"""MJPEG stream of the RealSense colour feed, with live button detections.

Same wire format as pi/camera_stream.py (multipart/x-mixed-replace on 8080),
so anything that pointed at the Pi's stream works unchanged -- but this reads
a RealSense locally rather than a UVC device, and annotates with the button
detector so you can see what the model sees while positioning a panel.

    ./scripts/stream_camera.py                 # detections on, port 8080
    ./scripts/stream_camera.py --no-detect     # raw feed, no GPU use
    ./scripts/stream_camera.py --port 8081

Then open http://<jetson>:8080/ in a browser.

NOTE: the RealSense is exclusive. This holds the camera, so
capture_panel_dataset.py and the detection scripts cannot run at the same
time -- stop the stream first.
"""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

ENGINE = '/home/jonathan/mycobot_project/elevator_buttons.engine'

_lock = threading.Lock()
_latest: bytes | None = None
_stats = {'fps': 0.0, 'det': 0, 'ms': 0.0}


def grabber(conf: float, detect: bool, every: int) -> None:
    """Own the camera, publish the most recent encoded JPEG."""
    global _latest
    import pyrealsense2 as rs

    model = None
    if detect:
        import torch
        torch.backends.cudnn.enabled = False
        from ultralytics import YOLO
        model = YOLO(ENGINE, task='detect')

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)

    n = 0
    t0 = time.time()
    boxes: list = []
    try:
        while True:
            frames = pipe.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())

            # Detection is the expensive part and the scene rarely changes
            # between adjacent frames, so run it every Nth and redraw the last
            # result in between. Keeps the stream at camera rate.
            if model is not None and n % every == 0:
                t = time.time()
                r = model.predict(img, imgsz=640, conf=conf, verbose=False)[0]
                _stats['ms'] = (time.time() - t) * 1000
                boxes = [(r.names[int(b.cls[0])], float(b.conf[0]),
                          *b.xyxy[0].tolist()) for b in r.boxes]
                _stats['det'] = len(boxes)

            for cls, cf, x1, y1, x2, y2 in boxes:
                col = ((0, 255, 0) if cf >= 0.5
                       else (0, 200, 255) if cf >= 0.25 else (0, 140, 255))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                cv2.putText(img, f'{cls} {cf:.2f}', (int(x1), max(12, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            cv2.drawMarker(img, (320, 240), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            n += 1
            if n % 15 == 0:
                _stats['fps'] = 15.0 / (time.time() - t0)
                t0 = time.time()
            cv2.putText(img,
                        f'{_stats["fps"]:.1f} fps | {_stats["det"]} det | '
                        f'{_stats["ms"]:.0f} ms',
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _lock:
                    _latest = enc.tobytes()
    finally:
        pipe.stop()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # quiet
        pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            body = (b'<html><body style="margin:0;background:#111">'
                    b'<img src="/stream" style="width:100%">'
                    b'</body></html>')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path != '/stream':
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with _lock:
                    buf = _latest
                if buf is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(buf)}\r\n'
                                 f'X-Capture-Us: {int(time.time()*1e6)}\r\n\r\n'
                                 .encode())
                self.wfile.write(buf)
                self.wfile.write(b'\r\n')
                time.sleep(1 / 30)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--conf', type=float, default=0.10)
    ap.add_argument('--no-detect', action='store_true')
    ap.add_argument('--every', type=int, default=3,
                    help='run detection every Nth frame')
    args = ap.parse_args()

    threading.Thread(target=grabber,
                     args=(args.conf, not args.no_detect, args.every),
                     daemon=True).start()

    srv = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    print(f'streaming on http://0.0.0.0:{args.port}/  (Ctrl-C to stop)')
    srv.serve_forever()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
