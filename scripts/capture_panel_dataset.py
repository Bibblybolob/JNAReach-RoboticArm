#!/usr/bin/env python3
"""Capture a labelling set of the REAL elevator panel, from the arm's camera.

The shipped weights are trained on the public ENTC Roboflow set -- other
people's panels -- which is why detections on ours score 0.1-0.4 while the
model reports mAP50 0.93 on its own validation data. That gap closes with a
few hundred images of the actual target, not with a bigger backbone.

Viewpoint diversity is what stops the fine-tune memorising one pose, so this
walks joint1 (pan) and joint5 (tilt) over a small grid and shoots at each
stop rather than burst-capturing one view. Move the panel between runs to
vary distance -- 15-40cm is the useful band (the D405 gives valid depth from
7cm, and buttons want to fill the frame).

    ./scripts/capture_panel_dataset.py --out ~/panel_dataset --shots 2

Arm is returned home afterwards. Images only -- labelling happens elsewhere
(Roboflow, labelImg); export YOLO format with the SAME 17 class names in the
SAME order as entc-elevator-button-detection-1/data.yaml or the class ids
will not line up.
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

HOME = [0, 90, -149, 55, 0, 0]

# Small offsets around home. Enough to change the view meaningfully without
# swinging the panel out of frame -- joint5 is tilt and the panel leaves the
# top of the image fast, hence the tighter range there.
PAN_OFFSETS = [-12, -6, 0, 6, 12]        # joint1, degrees
TILT_OFFSETS = [-8, 0, 8]                # joint5, degrees


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/panel_dataset'))
    ap.add_argument('--shots', type=int, default=2,
                    help='frames per arm pose')
    ap.add_argument('--serial-port', default='/dev/ttyTHS1')
    ap.add_argument('--baud', type=int, default=1000000)
    ap.add_argument('--speed', type=int, default=40)
    ap.add_argument('--settle', type=float, default=1.2,
                    help='seconds to wait after a move before shooting')
    ap.add_argument('--no-arm', action='store_true',
                    help='just shoot from wherever it is pointing')
    args = ap.parse_args()

    import pyrealsense2 as rs

    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)

    mc = None
    if not args.no_arm:
        from pymycobot import MyCobot
        mc = MyCobot(args.serial_port, args.baud)
        time.sleep(2)
        if mc.get_angles() is None:
            print('arm not answering -- check the UART wiring, or use --no-arm')
            return 1
        mc.send_angles(HOME, args.speed)
        time.sleep(3)

    poses = ([(p, t) for t in TILT_OFFSETS for p in PAN_OFFSETS]
             if mc else [(0, 0)])

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)

    n = len(os.listdir(img_dir))
    start = n
    try:
        for pan, tilt in poses:
            if mc:
                a = list(HOME)
                a[0] += pan
                a[4] += tilt
                mc.send_angles(a, args.speed)
                time.sleep(args.settle)

            for _ in range(args.shots):
                for _ in range(6):          # drain stale frames
                    frames = pipe.wait_for_frames()
                img = np.asanyarray(frames.get_color_frame().get_data())
                bright = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
                path = os.path.join(img_dir, f'panel_{n:04d}.jpg')
                cv2.imwrite(path, img)
                print(f'  {os.path.basename(path)}  pan{pan:+3d} tilt{tilt:+3d}'
                      f'  brightness {bright:.0f}')
                n += 1
                time.sleep(0.3)
    finally:
        pipe.stop()
        if mc:
            mc.send_angles(HOME, args.speed)
            time.sleep(2)

    print(f'\n{n - start} new images -> {img_dir}  ({n} total)')
    print('Vary distance (15-40cm), lighting and panel angle between runs.')
    print('Target ~150-200 images before fine-tuning.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
