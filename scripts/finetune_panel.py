#!/usr/bin/env python3
"""Fine-tune the button detector on OUR panel.

Starts from the ENTC-trained checkpoint rather than COCO. The class count
changes (17 -> 14), so Ultralytics discards the detection head and keeps the
backbone -- which is the part worth having, since it already knows what
elevator buttons look like. That is why this takes 1-2 hours instead of the
6.75h the original run took.

    ./scripts/finetune_panel.py                     # from elevator_buttons.pt
    ./scripts/finetune_panel.py --weights runs_960/yolo11s_960/weights/best.pt

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

    # Check the dataset BEFORE handing it to Ultralytics, which reports a
    # missing split as a FileNotFoundError buried in two chained tracebacks --
    # and does it AFTER printing eighty lines of hyperparameters, so the real
    # message is off the top of the screen.
    import yaml
    if not os.path.isfile(args.data):
        print(f'\nNo dataset at {args.data}.')
        return 1
    cfg = yaml.safe_load(open(args.data)) or {}
    root = cfg.get('path') or os.path.dirname(os.path.abspath(args.data))
    missing = [k for k in ('train', 'val')
               if not os.path.isdir(os.path.join(root, str(cfg.get(k, ''))))]
    if missing:
        imgs = os.path.join(root, 'images')
        n_img = len(os.listdir(imgs)) if os.path.isdir(imgs) else 0
        n_lbl = len([f for f in os.listdir(os.path.join(root, 'labels'))
                     if f.endswith('.txt')]) \
            if os.path.isdir(os.path.join(root, 'labels')) else 0
        print(f'\n{args.data} has no {"/".join(missing)} split yet.')
        print(f'  {root}/images : {n_img} image(s)')
        print(f'  {root}/labels : {n_lbl} label file(s)')
        if n_img and not n_lbl:
            print('\nThe images are not labelled, so there is nothing to '
                  'train on. Label them with the 14 classes in that '
                  'data.yaml, put the .txt files in labels/, then:')
            print(f'    ./scripts/split_panel_dataset.py --root {root}')
            print('\nFaster than labelling from scratch: '
                  './scripts/autolabel_panel.py writes a first pass with a '
                  'trained model, and you correct it.')
        elif n_lbl:
            print(f'\nLabels exist but the split has not been made:')
            print(f'    ./scripts/split_panel_dataset.py --root {root}')
        return 1

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
