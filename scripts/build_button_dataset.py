#!/usr/bin/env python3
"""Build BOTH training sets for the two-stage button recogniser, from one or
more public YOLO datasets.

    ./scripts/build_button_dataset.py --src ~/datasets/sunmoon-buttons --dry-run
    ./scripts/build_button_dataset.py --src ~/datasets/sunmoon-buttons \
        --src ~/datasets/entc --out ~/datasets/buttons

Writes two things:

  <out>/detect/     a 9-class YOLO detection set (see button_classes.py)
  <out>/read/       a classification set of floor-button CROPS, one directory
                    per legend, in Ultralytics' folder format

Replaces scripts/remap_dataset.py, which collapsed everything onto 14 flat
classes that stopped at button-10 and dropped `up` and `down` entirely -- the
two highest-priority classes in the project. That script is kept for
reproducing the old numbers in docs/button_datasets.md and nothing else.


The oversampling, and why it is here rather than in the trainer
--------------------------------------------------------------
Measured over the Sun Moon set: `floor` has 13,871 instances and `up` has 417,
a 33:1 imbalance on the class that matters most. YOLO's loss does not
reweight, and Ultralytics has no per-class sampler, so the only lever that
works end to end is showing the rare images more often.

`--oversample` (default 3) writes that many copies of every TRAIN image
containing an up or a down. The copies are hardlinks where the filesystem
allows it, so 3x costs no disk. It is applied to train only -- oversampling
the validation split would make the headline mAP a report on the duplication
rather than on the model.

A duplicate is not the same as an augmentation: the copies differ only through
whatever the trainer applies on top. That is still worth it here because the
augmentation is aggressive and the alternative is the class being outvoted 33
to 1 in every batch, but it is the reason `--oversample 8` is not better than
3 -- past a point it only overfits the same 349 images harder.


Reject classes are built, not dropped
-------------------------------------
`empty`, `blur` and `unknown` become `other` for the detector and
`unreadable` for the reader. This is deliberate and is the safety-relevant
choice in the whole pipeline: the failure this project cares about is pressing
the WRONG floor, so the recogniser needs a way to say "there is a button here
and I cannot read it". A dropped class cannot say that -- it produces no
detection, which reads identically to no button being there at all.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src', 'mycobot_perception'))

try:
    import yaml
except ImportError:
    sys.exit('pyyaml is required: pip install pyyaml')

try:
    import cv2
except ImportError:
    sys.exit('opencv is required for the crop stage: pip install opencv-python')

from mycobot_perception.button_classes import (
    DETECT_CLASSES, UNREADABLE, to_detect_class, to_reader_label)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp')


def split_dirs(src: str):
    """Yield (canonical_split, images_dir, labels_dir) for splits that exist.

    `valid` and `val` are the same split under two conventions; both are
    reported as `val` so that two sources using different names do not produce
    two validation sets, one of which is then silently never used.
    """
    for split in ('train', 'valid', 'val', 'test'):
        img = os.path.join(src, split, 'images')
        lbl = os.path.join(src, split, 'labels')
        if os.path.isdir(img) and os.path.isdir(lbl):
            yield ('val' if split == 'valid' else split), img, lbl


def source_names(src: str) -> list[str] | None:
    p = os.path.join(src, 'data.yaml')
    if not os.path.isfile(p):
        return None
    cfg = yaml.safe_load(open(p)) or {}
    n = cfg.get('names')
    if isinstance(n, dict):
        n = [n[k] for k in sorted(n)]
    return n or None


def find_image(img_dir: str, stem: str) -> str | None:
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def link_or_copy(src: str, dst: str) -> None:
    """Hardlink, so oversampling and a second dataset copy cost no disk."""
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', action='append', required=True,
                    help='YOLO dataset root with a data.yaml (repeatable)')
    ap.add_argument('--out', default=os.path.expanduser('~/datasets/buttons'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--oversample', type=int, default=3,
                    help='copies of each train image containing up/down')
    ap.add_argument('--min-reader-support', type=int, default=25,
                    help='drop reader classes with fewer crops than this.')
    ap.add_argument('--reader-balance', type=int, default=700,
                    help='hardlink copies of under-represented reader '
                         'classes up to about this many crops (0 = off)')
    ap.add_argument('--max-reader-copies', type=int, default=4,
                    help='cap on those copies; past a point it only overfits '
                         'the same few images harder')
    ap.add_argument('--min-crop-px', type=int, default=12,
                    help='skip crops smaller than this on either side')
    ap.add_argument('--crop-pad', type=float, default=0.12,
                    help='fraction of box size added around each crop, so the '
                         'reader sees the button rim and not just the digit')
    ap.add_argument('--keep-background', action='store_true',
                    help='keep images that end up with no boxes')
    args = ap.parse_args()

    det_root = os.path.join(args.out, 'detect')
    read_root = os.path.join(args.out, 'read')

    det_counts: Counter = Counter()
    read_counts: Counter = Counter()
    dropped: Counter = Counter()
    kept_imgs: Counter = Counter()
    reader_by_split: dict[str, list] = defaultdict(list)
    n_oversampled = 0

    for src in args.src:
        src = os.path.expanduser(src)
        names = source_names(src)
        if names is None:
            print(f'!! no usable data.yaml under {src}, skipping')
            continue

        # Build the id tables once per source and show what is thrown away,
        # because "which of the 368 names did you actually use" is the
        # question that decides whether this was worth running.
        det_map: dict[int, int] = {}
        read_map: dict[int, str] = {}
        unmapped = []
        for i, nm in enumerate(names):
            d = to_detect_class(nm)
            if d is None:
                unmapped.append(nm)
                continue
            det_map[i] = DETECT_CLASSES.index(d)
            if d == 'floor':
                lab = to_reader_label(nm)
                if lab:
                    read_map[i] = lab
            elif d == 'other':
                read_map[i] = UNREADABLE

        print(f'\n=== {src}')
        print(f'  {len(names)} source classes -> {len(det_map)} mapped, '
              f'{len(unmapped)} dropped')
        if unmapped:
            head = ', '.join(str(u) for u in unmapped[:12])
            print(f'  dropped: {head}'
                  + (f' ... (+{len(unmapped) - 12})' if len(unmapped) > 12 else ''))

        for split, img_dir, lbl_dir in split_dirs(src):
            out_img = os.path.join(det_root, split, 'images')
            out_lbl = os.path.join(det_root, split, 'labels')
            if not args.dry_run:
                os.makedirs(out_img, exist_ok=True)
                os.makedirs(out_lbl, exist_ok=True)

            for fn in sorted(os.listdir(lbl_dir)):
                if not fn.endswith('.txt'):
                    continue
                stem = os.path.splitext(fn)[0]

                rows = []            # (det_id, x, y, w, h) normalised
                crops = []           # (reader_label, x, y, w, h)
                for line in open(os.path.join(lbl_dir, fn)):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    sid = int(parts[0])
                    if sid >= len(names):
                        continue
                    if sid not in det_map:
                        dropped[names[sid]] += 1
                        continue
                    try:
                        box = [float(v) for v in parts[1:5]]
                    except ValueError:
                        continue
                    rows.append((det_map[sid], *box))
                    det_counts[DETECT_CLASSES[det_map[sid]]] += 1
                    if sid in read_map:
                        crops.append((read_map[sid], *box))

                if not rows and not args.keep_background:
                    continue

                src_img = find_image(img_dir, stem)
                if src_img is None:
                    continue

                # Oversample only the rare, top-priority classes, and only in
                # train. `up`/`down` are ids 0 and 1.
                reps = 1
                if split == 'train' and args.oversample > 1 \
                        and any(r[0] in (0, 1) for r in rows):
                    reps = args.oversample

                ext = os.path.splitext(src_img)[1]
                # Namespace by source so two datasets cannot collide on a
                # stem -- Roboflow exports use short numeric names and a
                # silent overwrite would look like a smaller dataset.
                tag = os.path.basename(src.rstrip('/'))
                for r in range(reps):
                    suffix = '' if r == 0 else f'__x{r}'
                    base = f'{tag}__{stem}{suffix}'
                    kept_imgs[split] += 1
                    if r:
                        n_oversampled += 1
                    if args.dry_run:
                        continue
                    link_or_copy(src_img, os.path.join(out_img, base + ext))
                    with open(os.path.join(out_lbl, base + '.txt'), 'w') as f:
                        for cid, x, y, w, h in rows:
                            f.write(f'{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

                # Crops come from the ORIGINAL image once, never from the
                # oversampled copies -- the reader has its own balance and
                # duplicating floor crops here would undo it.
                if crops:
                    reader_by_split[split].append((src_img, crops))
                    for lab, *_ in crops:
                        read_counts[lab] += 1

    # ---------------- detection report ----------------
    print('\n' + '=' * 46)
    print('STAGE A -- detection')
    print('=' * 46)
    print(f'{"class":<10}{"instances":>12}')
    print('-' * 22)
    for c in DETECT_CLASSES:
        n = det_counts.get(c, 0)
        flag = '  <-- ABSENT' if n == 0 else ('  <-- thin' if n < 100 else '')
        print(f'{c:<10}{n:>12}{flag}')
    print(f'\nimages: ' + ', '.join(f'{k} {v}' for k, v in sorted(kept_imgs.items())))
    if n_oversampled:
        print(f'  of which {n_oversampled} are up/down oversample copies '
              f'(x{args.oversample}, train only)')

    # ---------------- reader: decide the vocabulary ----------------
    keep = {k for k, v in read_counts.items()
            if v >= args.min_reader_support or k == UNREADABLE}
    # UNREADABLE always survives: it is how the arm refuses to guess.
    cut = sorted((k for k in read_counts if k not in keep),
                 key=lambda k: -read_counts[k])

    print('\n' + '=' * 46)
    print('STAGE B -- reading the legend')
    print('=' * 46)
    ordered = sorted(keep, key=lambda k: (-read_counts[k], k))
    for k in ordered:
        print(f'  {k:<12}{read_counts[k]:>7}')
    if cut:
        print(f'\n  below --min-reader-support={args.min_reader_support}, '
              f'dropped {len(cut)} legends: '
              + ', '.join(f'{k}({read_counts[k]})' for k in cut[:20]))
        print('  A button with one of those legends still DETECTS as a floor '
              'button; the reader just returns low confidence and the arm '
              'declines to press it. That is the intended behaviour.')

    if args.dry_run:
        print('\n--dry-run: nothing written.')
        return 0

    # ---------------- write the crops ----------------
    print('\nwriting crops...')
    written = Counter()
    for split, items in reader_by_split.items():
        for img_path, crops in items:
            im = cv2.imread(img_path)
            if im is None:
                continue
            H, W = im.shape[:2]
            for i, (lab, x, y, w, h) in enumerate(crops):
                if lab not in keep:
                    continue
                pw, ph = w * args.crop_pad, h * args.crop_pad
                x0 = int(max(0, (x - w / 2 - pw) * W))
                y0 = int(max(0, (y - h / 2 - ph) * H))
                x1 = int(min(W, (x + w / 2 + pw) * W))
                y1 = int(min(H, (y + h / 2 + ph) * H))
                if x1 - x0 < args.min_crop_px or y1 - y0 < args.min_crop_px:
                    continue
                crop = im[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                d = os.path.join(read_root, split, lab)
                os.makedirs(d, exist_ok=True)
                stem = os.path.splitext(os.path.basename(img_path))[0]
                base = os.path.join(d, f'{stem}_{i}.jpg')
                cv2.imwrite(base, crop)
                written[lab] += 1

                # BALANCE THE PRIOR, train split only.
                #
                # Measured 2026-08-17: every legend confusion in the
                # end-to-end eval ran from a RARER class to a COMMONER one --
                # 17(149 crops) -> 7(408), 19(142) -> 9(344), 8(385) -> B(126)
                # being the exception that proves it. A classifier with no
                # reweighting learns the prior, and the prior says a numeral
                # is more likely to be single-digit.
                #
                # Copies are hardlinks, so this costs no disk. Capped, because
                # past a point it stops adding balance and only overfits the
                # same few crops harder -- the same reason the detector's
                # up/down oversample is 3x and not 8x.
                if split == 'train' and args.reader_balance > 0:
                    have = read_counts.get(lab, 1)
                    reps = min(args.max_reader_copies,
                               max(1, args.reader_balance // max(1, have)))
                    for r in range(1, reps):
                        link_or_copy(base, os.path.join(
                            d, f'{stem}_{i}__b{r}.jpg'))
                        written[lab] += 1

    # Ultralytics classification wants train/ and val/ to hold the same class
    # directories. A legend that happens to appear only in train produces a
    # KeyError at validation time that reads as a corrupt dataset.
    all_labs = set()
    for s in ('train', 'val', 'test'):
        d = os.path.join(read_root, s)
        if os.path.isdir(d):
            all_labs |= {x for x in os.listdir(d)
                         if os.path.isdir(os.path.join(d, x))}
    for split in ('train', 'val', 'test'):
        base = os.path.join(read_root, split)
        if os.path.isdir(base):
            for lab in all_labs:
                os.makedirs(os.path.join(base, lab), exist_ok=True)

    # ---------------- data.yaml, generated ----------------
    with open(os.path.join(det_root, 'data.yaml'), 'w') as f:
        f.write('# GENERATED by scripts/build_button_dataset.py -- do not '
                'hand-edit.\n')
        f.write('# The class list is owned by '
                'src/mycobot_perception/mycobot_perception/button_classes.py;\n'
                '# editing it here only desynchronises it from the node.\n')
        f.write(f'path: {det_root}\n')
        f.write('train: train/images\n')
        f.write('val: val/images\n')
        if os.path.isdir(os.path.join(det_root, 'test', 'images')):
            f.write('test: test/images\n')
        f.write(f'\nnc: {len(DETECT_CLASSES)}\n')
        f.write('names:\n')
        for i, c in enumerate(DETECT_CLASSES):
            f.write(f'  {i}: {c}\n')

    with open(os.path.join(read_root, 'classes.txt'), 'w') as f:
        f.write('\n'.join(ordered) + '\n')

    print(f'\nwrote {det_root}/data.yaml  ({len(DETECT_CLASSES)} classes)')
    print(f'wrote {read_root}/  ({sum(written.values())} crops, '
          f'{len(written)} legends)')
    print('\nNext:')
    print(f'  ./scripts/train_button_detector.py --data {det_root}/data.yaml')
    print(f'  ./scripts/train_button_reader.py   --data {read_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
