#!/usr/bin/env python3
"""Stage A: train the 9-class button detector.

    ./scripts/train_button_detector.py                      # yolo11m @ 640
    ./scripts/train_button_detector.py --model yolo11s.pt --epochs 300

Run scripts/build_button_dataset.py first; this trains on what that writes.


Why 640 and not 960
-------------------
The Sun Moon images are **416x416 on disk** -- checked 2026-08-16, every one
of them. Training at 960 therefore feeds the network a 2.3x upscale of an
image with no detail above 416 to recover, which costs 2.2x the time per
epoch and adds no information. docs/button_datasets.md already recorded a
yolo11s@960 run coming in WORSE than yolo11n@640 on the ENTC set and put it
down to overfitting; the resolution ceiling is at least as likely a cause.

640 is a mild upsample, which YOLO does benefit from for small objects, and it
matches the D405's 640x480 delivery so the deployed engine is built at the
resolution it will actually see.

Why yolo11s and not yolo11m
---------------------------
Inference budget is not the binding constraint. Jonathan's constraint
(2026-08-16) is that the arm reads a panel while STATIONARY and the whole
press takes under 5 seconds, which leaves hundreds of milliseconds for
inference against the 18.7ms the old yolo11n engine used. On budget alone a
yolo11m or l would be free.

Two other things bind first.

**The data does not support it.** docs/button_datasets.md records a yolo11s@960
run coming in WORSE than yolo11n@640 on 393 images -- mAP50 0.9252 against
0.9292 -- and concludes the model was data-limited rather than
capacity-limited. This set is 5x larger at 1957 train images, which is what
justifies stepping up from n to s, but it is not the tens of thousands that
would justify an m.

**Training time is not actually free, measured.** yolo11n@640 batch 8 on this
Orin runs 1.8 it/s, 245 iterations to the epoch -- 140s, so 300 epochs is
11.6h. yolo11m is roughly 5x that compute: 58 hours, to overfit a 2000-image
set harder. yolo11s at ~2.2x lands near 12 hours for 150 epochs, which is an
overnight run.

So the compute goes into a long schedule on the smaller backbone. If a bigger
model is wanted later the thing to spend on first is MORE IMAGES -- the same
document's own conclusion, and the reason `--src` is repeatable in the
builder.


The augmentation, and the two flips that must stay off
------------------------------------------------------
**fliplr must be 0.0, and the reason is not the digits.** The obvious argument
-- that a mirrored numeral is not that numeral -- does not apply at this
stage, because stage A only says `floor` and never reads the number. The real
reason is that the door buttons are mirror images OF EACH OTHER: `open` is
|<->| and `close` is ->||<-. A horizontal flip does not corrupt those two
classes, it SWAPS them, which is worse than noise -- it is a consistent wrong
label on 2074 instances. The reader has its own, separate reason to keep the
flip off.

**flipud must be 0.0** for the matching reason one level up: a vertical flip
turns `up` into `down`, which are the two highest-priority classes here.

Rotation is capped at 12 degrees for the same reason. It is a real
augmentation for this task -- the arm approaches the panel off-axis, which
CLAUDE.md notes is part of why the shipped model saw nothing on the printed
panel -- but an arrow rotated far enough stops being the arrow it was
labelled. 12 degrees is well short of ambiguous and covers the approach angles
the arm actually reaches.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# cuDNN OFF before any CUDA work, exactly as button_detector_node.py does it
# and for the same reason: NVIDIA's torch 2.5.0a0 for JetPack needs
# libcudnn.so.9 present to import, but the cuDNN 9 wheels fail at runtime on
# this board's CUDA 12.2 the moment a convolution runs. Native CUDA
# convolutions instead. Deleting this line does not speed training up, it
# crashes it on the first batch.
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'mycobot_perception'))
from mycobot_perception.button_classes import DETECT_CLASSES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser(
        '~/datasets/buttons/detect/data.yaml'))
    ap.add_argument('--model', default='yolo11s.pt',
                    help='yolo11n/s/m/l.pt, or a checkpoint to resume from')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=8,
                    help='8 fits yolo11m@640 in the Orin Nano 7GB shared '
                         'pool with AMP. Raise it on a real GPU.')
    ap.add_argument('--hours', type=float, default=0.0,
                    help='wall-clock cap; 0 = no cap, run the epochs')
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--device', default='0')
    ap.add_argument('--name', default='detect_v1')
    args = ap.parse_args()

    if not os.path.isfile(args.data):
        print(f'No dataset at {args.data}.\n'
              'Build it first:\n'
              '  ./scripts/build_button_dataset.py --src '
              '~/datasets/sunmoon-buttons --out ~/datasets/buttons')
        return 1

    # Catch a desynchronised class list here rather than after an eight-hour
    # run. This is failure #11 in CLAUDE.md: ids are integers, so a data.yaml
    # written against a different ordering trains happily and mislabels
    # everything, with no error at any point.
    import yaml
    cfg = yaml.safe_load(open(args.data)) or {}
    names = cfg.get('names')
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if list(names or []) != DETECT_CLASSES:
        print('!! data.yaml disagrees with button_classes.DETECT_CLASSES')
        print(f'   data.yaml : {names}')
        print(f'   expected  : {DETECT_CLASSES}')
        print('   Rebuild the dataset; do not edit the yaml.')
        return 1

    print(f'training {args.model} @ {args.imgsz} on {args.data}')
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        time=args.hours or None,
        batch=args.batch,
        workers=4,
        device=args.device,
        amp=True,
        patience=args.patience,
        cache=False,

        # --- geometry: mimic the arm's approach, never the mirror ---------
        degrees=12.0,      # off-axis approach; see the module docstring
        translate=0.15,
        scale=0.5,         # the arm closes from ~40cm to ~10cm
        shear=4.0,
        perspective=0.0005,
        fliplr=0.0,        # would SWAP open and close. Not negotiable.
        flipud=0.0,        # would swap up and down. Not negotiable.

        # --- appearance: room light, glare, and the D405's colour ---------
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,         # generous: button panels are backlit and specular
        erasing=0.3,       # occlusion by the arm's own tool

        mosaic=1.0,
        close_mosaic=20,   # finish on realistic whole-panel views
        mixup=0.1,

        project=os.path.join(REPO, 'runs_buttons'),
        name=args.name,
        exist_ok=True,
        plots=True,
        seed=0,
    )
    print('DETECTOR TRAINING COMPLETE')
    print(f'weights: {REPO}/runs_buttons/{args.name}/weights/best.pt')
    print('Next: ./scripts/export_button_engine.py')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
