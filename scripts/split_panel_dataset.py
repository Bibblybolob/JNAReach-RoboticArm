#!/usr/bin/env python3
"""Split labelled panel images into train/ and val/ for YOLO.

Expects everything flat first -- images in <root>/images and their YOLO .txt
labels in <root>/labels (which is what a Roboflow or labelImg export gives
you) -- and moves matched pairs into train/{images,labels} and
val/{images,labels}.

    ./scripts/split_panel_dataset.py --root ~/panel_dataset --val-frac 0.15

Images with no label file are reported and skipped rather than silently
becoming background examples: an unlabelled frame that DOES contain buttons
teaches the model those buttons are background, which is worse than omitting
it. Genuinely empty frames are fine as backgrounds -- pass --keep-unlabelled
if that is what they are.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/panel_dataset'))
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep-unlabelled', action='store_true')
    args = ap.parse_args()

    src_img = os.path.join(args.root, 'images')
    src_lbl = os.path.join(args.root, 'labels')
    if not os.path.isdir(src_img):
        print(f'no {src_img}')
        return 1

    imgs = sorted(f for f in os.listdir(src_img)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    paired, orphan = [], []
    for f in imgs:
        stem = os.path.splitext(f)[0]
        lbl = os.path.join(src_lbl, stem + '.txt')
        (paired if os.path.exists(lbl) else orphan).append((f, lbl))

    if orphan:
        print(f'{len(orphan)} image(s) with no label file')
        for f, _ in orphan[:5]:
            print(f'    {f}')
        if len(orphan) > 5:
            print(f'    ... and {len(orphan) - 5} more')
        if not args.keep_unlabelled:
            print('  skipping them (--keep-unlabelled to treat as background)')
        else:
            paired += [(f, None) for f, _ in orphan]

    if not paired:
        print('nothing to split -- label the images first')
        return 1

    random.Random(args.seed).shuffle(paired)
    n_val = max(1, round(len(paired) * args.val_frac))
    splits = {'val': paired[:n_val], 'train': paired[n_val:]}

    for split, items in splits.items():
        di = os.path.join(args.root, split, 'images')
        dl = os.path.join(args.root, split, 'labels')
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        for f, lbl in items:
            shutil.copy2(os.path.join(src_img, f), os.path.join(di, f))
            stem = os.path.splitext(f)[0]
            if lbl:
                shutil.copy2(lbl, os.path.join(dl, stem + '.txt'))
            else:
                open(os.path.join(dl, stem + '.txt'), 'w').close()
        print(f'{split}: {len(items)} images -> {di}')

    print(f'\ndata.yaml should point at {args.root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
