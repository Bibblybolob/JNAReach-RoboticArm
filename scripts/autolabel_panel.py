#!/usr/bin/env python3
"""Write a first-pass set of labels for our panel images, to be CORRECTED.

Labelling 150-200 images by hand is the slowest step in this project and the
only one a person has to do. But we are about to have a model that knows what
buttons 1-10 look like -- trained on the remapped public set -- and this panel
is the same fourteen classes. So let it draw the boxes, and correct them.

Correcting is much faster than drawing: the boxes are already in roughly the
right places with roughly the right labels, and the work becomes deleting
false positives, dragging a few corners, and fixing the digits it misread.

    ./scripts/autolabel_panel.py --weights runs_panel/public_pretrain/weights/best.pt
    ./scripts/autolabel_panel.py --conf 0.15 --review    # also write previews

This writes YOLO .txt files into <root>/labels, which is exactly where
split_panel_dataset.py expects them.

**These labels are a draft, not ground truth.** A model that produced perfect
labels for our panel would mean we did not need to fine-tune at all. Expect it
to be confident and wrong about which DIGIT a button is -- that is precisely
the domain gap being closed here -- so check every class id even where the box
looks right. `--review` writes annotated previews so that pass is quick.

Conventionally called pseudo-labelling. Its failure mode is that errors are
self-confirming: label a 6 as an 8, train on it, and the model learns to call
6s 8s with more confidence. Which is why this is a draft for a human, and
why the previews exist.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

torch.backends.cudnn.enabled = False  # see button_detector_node's header

import cv2  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_WEIGHTS = ('/home/jonathan/mycobot_project/runs_panel/'
                   'public_pretrain/weights/best.pt')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/panel_dataset'))
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    ap.add_argument('--conf', type=float, default=0.25,
                    help='lower catches more buttons but writes more rubbish '
                         'to delete; 0.15-0.30 is the useful band')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--review', action='store_true',
                    help='also write annotated previews to <root>/review')
    ap.add_argument('--overwrite', action='store_true',
                    help='replace existing label files. Off by default so a '
                         'second run cannot destroy corrections already made')
    args = ap.parse_args()

    img_dir = os.path.join(args.root, 'images')
    lbl_dir = os.path.join(args.root, 'labels')
    if not os.path.isdir(img_dir):
        print(f'no images at {img_dir}')
        return 1
    if not os.path.exists(args.weights):
        print(f'no weights at {args.weights}')
        print('Train on the public set first:')
        print('  ./scripts/finetune_panel.py --data '
              '~/datasets/public_panel/data.yaml --name public_pretrain')
        return 1

    data_yaml = os.path.join(args.root, 'data.yaml')
    target = []
    if os.path.isfile(data_yaml):
        cfg = yaml.safe_load(open(data_yaml)) or {}
        names = cfg.get('names')
        target = ([names[k] for k in sorted(names)]
                  if isinstance(names, dict) else list(names or []))

    os.makedirs(lbl_dir, exist_ok=True)
    if args.review:
        os.makedirs(os.path.join(args.root, 'review'), exist_ok=True)

    model = YOLO(args.weights)
    model_names = [model.names[k] for k in sorted(model.names)]

    # A mismatch here silently mislabels everything: the .txt holds class
    # INDICES, so if the model's order differs from data.yaml's, every id
    # points at the wrong name and nothing complains.
    if target and model_names != target:
        print('CLASS ORDER MISMATCH -- refusing to write labels.')
        print(f'  model    : {model_names}')
        print(f'  data.yaml: {target}')
        print('Labels are written as indices, so a different order silently '
              'relabels every box. Retrain with matching classes, or fix '
              'data.yaml.')
        return 1

    imgs = sorted(f for f in os.listdir(img_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    written = skipped = 0
    per_class = {}
    empty = []

    for fn in imgs:
        stem = os.path.splitext(fn)[0]
        out = os.path.join(lbl_dir, stem + '.txt')
        if os.path.exists(out) and not args.overwrite:
            skipped += 1
            continue
        path = os.path.join(img_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf,
                            verbose=False)[0]

        lines = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cid = int(b.cls[0])
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f'{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
            nm = model.names[cid]
            per_class[nm] = per_class.get(nm, 0) + 1

        with open(out, 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        written += 1
        if not lines:
            empty.append(fn)

        if args.review:
            ann = img.copy()
            for b in res.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(ann, f'{model.names[int(b.cls[0])]} '
                                 f'{float(b.conf[0]):.2f}',
                            (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 200, 255), 1)
            cv2.imwrite(os.path.join(args.root, 'review', fn), ann)

    print(f'{written} label file(s) written, {skipped} left alone')
    if per_class:
        print('\nwhat it thinks is there:')
        for k in sorted(per_class, key=lambda k: -per_class[k]):
            print(f'  {k:<12}{per_class[k]:>5}')
    if empty:
        print(f'\n{len(empty)} image(s) got NO boxes -- the panel may be out '
              f'of frame, or too small. e.g. {", ".join(empty[:4])}')
    print('\nNOW CHECK THEM. These are a draft: the model is expected to be '
          'confident and wrong about which digit a button is, which is the '
          'whole reason for fine-tuning on our own panel. Training on '
          'uncorrected labels teaches it its own mistakes.')
    if args.review:
        print(f'Previews: {os.path.join(args.root, "review")}')
    print(f'\nThen: ./scripts/split_panel_dataset.py --root {args.root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
