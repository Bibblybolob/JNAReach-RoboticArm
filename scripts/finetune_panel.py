#!/usr/bin/env python3
"""Fine-tune the button detector on OUR panel.

Starts from the ENTC-trained checkpoint rather than COCO. The class count
changes (17 -> 14), so Ultralytics discards the detection head and keeps the
backbone -- which is the part worth having, since it already knows what
elevator buttons look like. That is why this takes 1-2 hours instead of the
6.75h the original run took.

    ./scripts/finetune_panel.py                     # from the 960 run
    ./scripts/finetune_panel.py --weights elevator_buttons.pt --imgsz 640

Match --imgsz to what you will deploy at: the TensorRT engine is built for one
resolution, and inference cost scales with it (640 -> 19.5ms, 960 -> ~26ms
against a 33ms frame budget). 640 is the default because 960 was measured to
buy nothing on a set this small -- see DEFAULT_WEIGHTS.
"""
from __future__ import annotations

import argparse
import os

import torch

# cuDNN 9 wheels fail at runtime on this Jetson's CUDA 12.2 -- see the header
# of button_detector_node.py. Native CUDA convolutions instead.
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

# The original yolo11n@640 weights, deliberately.
#
# An overnight yolo11s@960 run on the same 393-image set was measured WORSE:
# mAP50 0.9252 / mAP50-95 0.5366 at its best epoch, against 0.9292 / 0.5446
# for these. With that little data the model is data-limited, not capacity-
# limited, and the larger backbone just overfits sooner. Starting a fine-tune
# from the weaker checkpoint would inherit that for nothing.
DEFAULT_WEIGHTS = '/home/jonathan/mycobot_project/elevator_buttons.pt'
FALLBACK_WEIGHTS = '/home/jonathan/mycobot_project/runs_960/yolo11s_960/weights/best.pt'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/panel_dataset/data.yaml'))
    ap.add_argument('--weights', default=None)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--hours', type=float, default=3.0)
    ap.add_argument('--name', default='panel_finetune')
    args = ap.parse_args()

    w = args.weights
    if w is None:
        w = DEFAULT_WEIGHTS if os.path.exists(DEFAULT_WEIGHTS) else FALLBACK_WEIGHTS
    if not os.path.exists(w):
        print(f'no weights at {w}')
        return 1
    print(f'starting from {w}')

    model = YOLO(w)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        time=args.hours,
        batch=args.batch,
        workers=4,
        cache=False,
        patience=30,
        # A few hundred images of one panel overfits fast. Keep the geometric
        # augmentation that mimics the arm moving, drop mosaic near the end so
        # the model finishes on realistic whole-panel views.
        degrees=8.0,
        translate=0.15,
        scale=0.4,
        fliplr=0.0,          # button layout is not mirror-symmetric
        mosaic=1.0,
        close_mosaic=15,
        project='/home/jonathan/mycobot_project/runs_panel',
        name=args.name,
        exist_ok=True,
        device=0,
        plots=True,
    )
    print('FINETUNE COMPLETE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
