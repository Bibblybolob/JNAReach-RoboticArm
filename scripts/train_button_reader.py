#!/usr/bin/env python3
"""Stage B: train the classifier that reads the legend off a button crop.

    ./scripts/train_button_reader.py
    ./scripts/train_button_reader.py --model yolo11m-cls.pt --imgsz 160

Run scripts/build_button_dataset.py first.

Stage A finds a button and calls it `floor`. This says WHICH floor: 47 legends
(0-36, B, B1, B2, B3, G, L, LG, M, CH, -1) plus `unreadable`.


Why a classifier instead of more detector classes
-------------------------------------------------
This is the whole reason the recogniser is two-stage. Asking the detector to
name the floor means one class per legend, and the measured support collapses
down the range: floor 2 has 1200 instances and floor 33 has 55. A detector
spends its capacity on localisation AND classification jointly, over a 640px
image in which the button is maybe 30px across.

The classifier starts from a crop that is already found, centred and scaled,
and resamples it to 128px -- so the numeral that occupied 20 pixels in the
frame now occupies most of the input. Same photons, an order of magnitude more
of the network looking at them.

It is also much cheaper to run than it looks: the crops are small, they batch,
and a panel has a dozen buttons. Budget on this board is a few ms per crop
against the 5-second whole-press allowance.


fliplr must be 0.0, and here it IS about the digits
---------------------------------------------------
Stage A keeps the horizontal flip off because `open` and `close` are mirror
images of each other. At this stage the reason is the ordinary one: a mirrored
2 is not a 2, and there is no legend in the vocabulary that a mirror maps onto
another valid legend. Turning it on would teach the model that backwards
digits are digits, which is exactly the confusion that produces a wrong floor.

Rotation is allowed a little more room than in stage A (15 degrees) because a
crop is rotated about the button's own centre rather than the panel's, so the
legend stays legible where a whole-panel rotation would start clipping.


The confusions that matter are not symmetric
--------------------------------------------
6/9 is the dangerous pair: they are a 180-degree rotation apart, so any
rotation augmentation pushes them toward each other, and pressing 9 instead of
6 is a real error with no cue that it happened. That is the reason `degrees`
is capped at 15 rather than the 30-45 a generic classifier would use, and the
reason the per-class confusion matrix is worth reading after the run rather
than the headline top-1.
"""
from __future__ import annotations

import argparse
import os

import torch

# See train_button_detector.py -- cuDNN off before any CUDA work, or the first
# convolution crashes on this board.
torch.backends.cudnn.enabled = False

from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser(
        '~/datasets/buttons/read'))
    ap.add_argument('--model', default='yolo11s-cls.pt')
    ap.add_argument('--imgsz', type=int, default=128,
                    help='crops are small; 128 is already an upsample for '
                         'most of them')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--device', default='0')
    ap.add_argument('--name', default='read_v1')
    args = ap.parse_args()

    data = os.path.expanduser(args.data)
    train_dir = os.path.join(data, 'train')
    if not os.path.isdir(train_dir):
        print(f'No crop dataset at {data}.\n'
              'Build it first:\n'
              '  ./scripts/build_button_dataset.py --src '
              '~/datasets/sunmoon-buttons --out ~/datasets/buttons')
        return 1

    labels = sorted(d for d in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, d)))
    counts = {l: len(os.listdir(os.path.join(train_dir, l))) for l in labels}
    print(f'{len(labels)} legends, {sum(counts.values())} train crops')
    thin = {l: n for l, n in counts.items() if n < 20}
    if thin:
        # Not fatal -- the builder has already applied its own threshold, and
        # the val split can legitimately be thinner. Worth saying out loud,
        # because a legend the model half-learns is the one that produces a
        # confident wrong floor.
        print('  thin after the split: '
              + ', '.join(f'{l}({n})' for l, n in sorted(thin.items())))

    model = YOLO(args.model)
    model.train(
        data=data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=4,
        device=args.device,
        amp=True,
        patience=args.patience,

        degrees=15.0,      # capped: 6 and 9 are 180 degrees apart
        translate=0.10,
        scale=0.3,
        fliplr=0.0,        # a mirrored 2 is not a 2
        flipud=0.0,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        erasing=0.2,

        project=os.path.join(REPO, 'runs_buttons'),
        name=args.name,
        exist_ok=True,
        plots=True,
        seed=0,
    )
    print('READER TRAINING COMPLETE')
    print(f'weights: {REPO}/runs_buttons/{args.name}/weights/best.pt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
