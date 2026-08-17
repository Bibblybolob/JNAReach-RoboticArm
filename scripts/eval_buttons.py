#!/usr/bin/env python3
"""Score the two-stage recogniser END TO END, on the priorities that matter.

    ./scripts/eval_buttons.py --src ~/datasets/sunmoon-buttons --split test

Neither trainer measures this. `train_button_detector.py` reports mAP over
boxes and knows nothing about legends; `train_button_reader.py` reports
top-1 over GROUND-TRUTH crops, which is not the number that matters -- in
service the reader sees crops the DETECTOR produced, shifted and scaled
slightly differently, and its accuracy on those is what the arm actually gets.

So this runs the real pipeline against held-out images and reports:

  1. Per-class detection precision/recall, with up and down first.
  2. End-to-end legend accuracy: of the floor buttons that were found, how
     many were read correctly.
  3. **The wrong-floor rate**, which is the number this project is built
     around: how often the pipeline confidently reports a legend that is not
     the one on the button. A miss costs a retry. A confident misread sends
     the arm to the wrong storey with nothing in the logs to say so, and the
     two must never be averaged into one accuracy figure.

Read against the source dataset rather than the built one, because the built
detection labels have already collapsed every floor to `floor` -- the legend
ground truth only exists upstream.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import torch

torch.backends.cudnn.enabled = False

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'mycobot_perception'))
from mycobot_perception.button_classes import (  # noqa: E402
    DETECT_CLASSES, UNREADABLE, to_detect_class, to_reader_label)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.expanduser(
        '~/datasets/sunmoon-buttons'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--detect-weights', default=os.path.join(
        REPO, 'runs_buttons', 'detect_v1', 'weights', 'best.pt'))
    ap.add_argument('--read-weights', default=os.path.join(
        REPO, 'runs_buttons', 'read_v1', 'weights', 'best.pt'))
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--read-min', type=float, default=0.95,
                    help='must match the node\'s reader_min_confidence')
    ap.add_argument('--crop-pad', type=float, default=0.12,
                    help="must match the builder's --crop-pad")
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--device', default='0')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--json', default='',
                    help='also write the numbers here, for compare_buttons.py')
    ap.add_argument('--label', default='',
                    help='name for this run in the comparison table')
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    img_dir = os.path.join(src, args.split, 'images')
    lbl_dir = os.path.join(src, args.split, 'labels')
    if not os.path.isdir(img_dir):
        print(f'no {args.split} split under {src}')
        return 1
    if not os.path.isfile(args.detect_weights):
        print(f'no detector at {args.detect_weights} -- train it first')
        return 1

    names = yaml.safe_load(open(os.path.join(src, 'data.yaml')))['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    det = YOLO(args.detect_weights)

    # Build the model's own id -> our taxonomy table, once. A model whose
    # names already match is the identity; the old ENTC-trained
    # elevator_buttons.pt is folded onto the same 9 classes here.
    det_name_map: dict[int, str] = {}
    raw = det.names
    raw = raw if isinstance(raw, dict) else dict(enumerate(raw))
    for i, nm in raw.items():
        det_name_map[int(i)] = nm if nm in DETECT_CLASSES else to_detect_class(nm)
    unmapped = sorted({nm for i, nm in raw.items()
                       if det_name_map.get(int(i)) is None})
    print(f'detector: {args.detect_weights}')
    print(f'  {len(raw)} model classes -> '
          f'{len({v for v in det_name_map.values() if v})} of ours'
          + (f'; ignoring {unmapped}' if unmapped else ''))

    reader = None
    if os.path.isfile(args.read_weights):
        reader = YOLO(args.read_weights)
    else:
        print(f'no reader at {args.read_weights} -- detection only')

    tp = Counter(); fp = Counter(); fn = Counter()
    legend_right = legend_wrong = legend_declined = 0
    confusions: Counter = Counter()

    files = sorted(f for f in os.listdir(lbl_dir) if f.endswith('.txt'))
    if args.limit:
        files = files[:args.limit]

    for n, fn_txt in enumerate(files):
        stem = os.path.splitext(fn_txt)[0]
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png'):
            p = os.path.join(img_dir, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        im = cv2.imread(img_path)
        if im is None:
            continue
        H, W = im.shape[:2]

        # --- ground truth, mapped through the same taxonomy ---------------
        gt = []   # (kind, legend|None, x1,y1,x2,y2, matched)
        for line in open(os.path.join(lbl_dir, fn_txt)):
            p = line.split()
            if len(p) < 5:
                continue
            sid = int(p[0])
            if sid >= len(names):
                continue
            kind = to_detect_class(names[sid])
            if kind is None:
                continue
            x, y, bw, bh = (float(v) for v in p[1:5])
            gt.append([kind, to_reader_label(names[sid]),
                       (x - bw / 2) * W, (y - bh / 2) * H,
                       (x + bw / 2) * W, (y + bh / 2) * H, False])

        res = det.predict(im, conf=args.conf, device=args.device,
                          verbose=False)[0]
        boxes = res.boxes
        preds = []
        if boxes is not None and len(boxes):
            xy = boxes.xyxy.cpu().numpy()
            cl = boxes.cls.cpu().numpy().astype(int)
            cf = boxes.conf.cpu().numpy()
            order = np.argsort(-cf)
            for i in order:
                # Map by NAME, never by index. For the new detector the model's
                # names ARE DETECT_CLASSES and this is the identity; for the
                # OLD 17-class elevator_buttons.pt it puts that model's
                # predictions through exactly the same taxonomy as the ground
                # truth, which is what makes a before/after comparison honest
                # rather than a comparison of two different questions.
                #
                # Mapping by index instead would silently score the old model
                # against a class list it was never trained on -- failure #11,
                # and it would read as the old model being catastrophically
                # bad rather than as a bug here.
                kind = det_name_map.get(int(cl[i]))
                if kind is not None:
                    preds.append((kind, *xy[i]))

        # --- match, greedily, highest confidence first --------------------
        for kind, x1, y1, x2, y2 in preds:
            best, best_i = 0.0, -1
            for i, g in enumerate(gt):
                if g[6] or g[0] != kind:
                    continue
                v = iou((x1, y1, x2, y2), g[2:6])
                if v > best:
                    best, best_i = v, i
            if best >= args.iou:
                gt[best_i][6] = True
                tp[kind] += 1
                # --- stage B, on the DETECTED crop, not the GT box --------
                if kind == 'floor' and reader is not None:
                    truth = gt[best_i][1]
                    pw, ph = (x2 - x1) * args.crop_pad, (y2 - y1) * args.crop_pad
                    cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
                    cx2, cy2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
                    crop = im[cy1:cy2, cx1:cx2]
                    if crop.size == 0 or truth is None:
                        continue
                    r = reader.predict(crop, device=args.device,
                                       verbose=False)[0]
                    conf = float(r.probs.top1conf)
                    got = reader.names[int(r.probs.top1)]
                    if conf < args.read_min or got == UNREADABLE:
                        legend_declined += 1
                    elif got == truth:
                        legend_right += 1
                    else:
                        legend_wrong += 1
                        confusions[(truth, got)] += 1
            else:
                fp[kind] += 1

        for g in gt:
            if not g[6]:
                fn[g[0]] += 1

        if n and n % 50 == 0:
            print(f'  {n}/{len(files)}...', flush=True)

    # ---------------- report ----------------
    print('\n' + '=' * 58)
    print(f'DETECTION  ({len(files)} images, conf>={args.conf}, '
          f'IoU>={args.iou})')
    print('=' * 58)
    print(f'{"class":<10}{"n":>6}{"precision":>11}{"recall":>9}')
    print('-' * 36)
    for c in DETECT_CLASSES:
        n_gt = tp[c] + fn[c]
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else float('nan')
        rec = tp[c] / n_gt if n_gt else float('nan')
        star = '  <--' if c in ('up', 'down') else ''
        print(f'{c:<10}{n_gt:>6}{prec:>11.3f}{rec:>9.3f}{star}')

    if reader is not None:
        total = legend_right + legend_wrong + legend_declined
        print('\n' + '=' * 58)
        print('END TO END -- reading the floor legend off DETECTED crops')
        print('=' * 58)
        if total:
            print(f'  correct   {legend_right:>6}  '
                  f'{100 * legend_right / total:5.1f}%')
            print(f'  declined  {legend_declined:>6}  '
                  f'{100 * legend_declined / total:5.1f}%   '
                  '(publishes as `floor`; costs a retry)')
            print(f'  WRONG     {legend_wrong:>6}  '
                  f'{100 * legend_wrong / total:5.1f}%   '
                  '(confident and incorrect -- the failure that matters)')
            if legend_right + legend_wrong:
                print(f'\n  of the ones it committed to: '
                      f'{100 * legend_right / (legend_right + legend_wrong):.1f}% right')
        if confusions:
            print('\n  worst confusions (truth -> reported):')
            for (t, g), c in confusions.most_common(12):
                print(f'    {t:>4} -> {g:<4}  x{c}')
            print('\n  Raise reader_min_confidence to trade these for '
                  'declines. A decline is a retry; one of these is a trip to '
                  'the wrong floor.')

    if args.json:
        import json
        out = {
            'label': args.label or os.path.basename(args.detect_weights),
            'weights': args.detect_weights,
            'reader': args.read_weights if reader is not None else None,
            'src': src, 'split': args.split, 'images': len(files),
            'conf': args.conf, 'iou': args.iou,
            'classes': {}, 'legend': None,
        }
        for c in DETECT_CLASSES:
            n_gt = tp[c] + fn[c]
            out['classes'][c] = {
                'n': n_gt, 'tp': tp[c], 'fp': fp[c], 'fn': fn[c],
                'precision': (tp[c] / (tp[c] + fp[c])) if (tp[c] + fp[c]) else None,
                'recall': (tp[c] / n_gt) if n_gt else None,
            }
        if reader is not None and (legend_right + legend_wrong + legend_declined):
            out['legend'] = {
                'right': legend_right, 'wrong': legend_wrong,
                'declined': legend_declined,
                'confusions': [[t, g, c] for (t, g), c in
                               confusions.most_common(20)],
            }
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nwrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
