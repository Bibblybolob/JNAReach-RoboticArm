#!/usr/bin/env python3
"""Remap a public elevator-button dataset onto OUR panel's 14 classes.

Why
---
The shipped model is trained on the ENTC set, whose 17 classes have no
button-4..button-10 and no P. Our panel has exactly those, so most of it
cannot be labelled correctly no matter how confident the model gets -- watched
live, it called button 10 'floor-ground' and button 6 'close'.

Public sets exist that DO cover numbered floors (Sun Moon University's has
0-50+), but with hundreds of classes over a couple of thousand images, which
averages a handful of examples each. Collapsing them onto the 14 classes this
panel actually has concentrates that support where it is needed and drops the
rest.

This fixes CLASS COVERAGE. It does not fix domain gap -- these are still other
people's panels. The intended use is two-stage: train on the remapped public
set for coverage and volume, then fine-tune on our own images for domain.

    ./scripts/remap_dataset.py --src ~/downloads/elevator-1 --out ~/public_panel
    ./scripts/remap_dataset.py --src ... --out ... --dry-run     # just report

Reports per-class counts before and after, because the number that decides
whether this was worth doing is how many examples the RARE classes end up
with. A button-7 with four examples will not be learned, and you want to know
that before spending an hour training.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import Counter

try:
    import yaml
except ImportError:
    sys.exit('pyyaml is required: pip install pyyaml')

# Our panel. Order is the contract -- it must match ~/panel_dataset/data.yaml
# exactly, or a model trained here and fine-tuned there learns shifted ids.
TARGET = ['button-1', 'button-2', 'button-3', 'button-4', 'button-5',
          'button-6', 'button-7', 'button-8', 'button-9', 'button-10',
          'button-P', 'open', 'close', 'alarm']

# Names that mean the same button under different dataset conventions.
# Deliberately conservative: anything not listed here is DROPPED rather than
# guessed at, because a wrong mapping is worse than a missing one -- it
# teaches the model that a thing is a button it is not.
ALIASES = {
    'open': 'open', 'door-open': 'open', 'openbutton': 'open',
    'close': 'close', 'door-close': 'close', 'closebutton': 'close',
    'alarm': 'alarm', 'bell': 'alarm', 'emergency': 'alarm',
    'call': 'alarm',
    'p': 'button-P', 'parking': 'button-P', 'park': 'button-P',
}

# Strip the prefixes different sets put in front of the same thing:
# ENTC uses button-1 and floor-1, Sun Moon uses bare '1'.
PREFIX = re.compile(r'^(button[-_ ]?|floor[-_ ]?|btn[-_ ]?)', re.I)


def map_name(name: str) -> str | None:
    """Source class name -> one of TARGET, or None to drop it."""
    n = name.strip().lower()
    n = PREFIX.sub('', n)
    if n in ALIASES:
        return ALIASES[n]
    # A bare number is a floor button. Only 1-10 exist on our panel; 12, 27a
    # and the like are real buttons on other panels and are dropped, not
    # squeezed into a class they are not.
    if n.isdigit() and 1 <= int(n) <= 10:
        return f'button-{int(n)}'
    return None


def split_dirs(src: str):
    """Yield (split, images_dir, labels_dir) for whatever splits exist."""
    for split in ('train', 'valid', 'val', 'test'):
        img = os.path.join(src, split, 'images')
        lbl = os.path.join(src, split, 'labels')
        if os.path.isdir(img) and os.path.isdir(lbl):
            yield split, img, lbl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='YOLO dataset root (has data.yaml)')
    ap.add_argument('--out', default=os.path.expanduser('~/public_panel'))
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would happen; write nothing')
    ap.add_argument('--keep-empty', action='store_true',
                    help='keep images that end up with no boxes, as '
                         'background examples. Off by default: a set that is '
                         'mostly background teaches the model to predict '
                         'nothing.')
    ap.add_argument('--min-per-class', type=int, default=15,
                    help='warn about classes with fewer examples than this')
    args = ap.parse_args()

    data_yaml = os.path.join(args.src, 'data.yaml')
    if not os.path.isfile(data_yaml):
        return int(bool(print(f'no data.yaml under {args.src}')) or 1)
    cfg = yaml.safe_load(open(data_yaml))
    names = cfg.get('names')
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        return int(bool(print('data.yaml has no names')) or 1)

    print(f'source: {len(names)} classes, {args.src}')

    # Build the id->id table once, and show what is being thrown away.
    id_map: dict[int, int] = {}
    dropped_names = []
    for i, nm in enumerate(names):
        t = map_name(nm)
        if t is None:
            dropped_names.append(nm)
        else:
            id_map[i] = TARGET.index(t)
    print(f'  mapped {len(id_map)} source classes onto {len(TARGET)} targets, '
          f'dropping {len(dropped_names)}')
    if dropped_names:
        head = ', '.join(dropped_names[:14])
        print(f'  dropped: {head}' + (' ...' if len(dropped_names) > 14 else ''))

    before, after = Counter(), Counter()
    kept_imgs = dropped_imgs = 0

    for split, img_dir, lbl_dir in split_dirs(args.src):
        out_img = os.path.join(args.out, split, 'images')
        out_lbl = os.path.join(args.out, split, 'labels')
        if not args.dry_run:
            os.makedirs(out_img, exist_ok=True)
            os.makedirs(out_lbl, exist_ok=True)

        n_split = 0
        for fn in sorted(os.listdir(lbl_dir)):
            if not fn.endswith('.txt'):
                continue
            lines_out = []
            for line in open(os.path.join(lbl_dir, fn)):
                parts = line.split()
                if len(parts) < 5:
                    continue
                src_id = int(parts[0])
                if src_id < len(names):
                    before[names[src_id]] += 1
                if src_id in id_map:
                    tgt = id_map[src_id]
                    after[TARGET[tgt]] += 1
                    lines_out.append(' '.join([str(tgt)] + parts[1:]))

            if not lines_out and not args.keep_empty:
                dropped_imgs += 1
                continue

            stem = os.path.splitext(fn)[0]
            src_img = None
            for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG'):
                cand = os.path.join(img_dir, stem + ext)
                if os.path.exists(cand):
                    src_img = cand
                    break
            if src_img is None:
                continue
            kept_imgs += 1
            n_split += 1
            if not args.dry_run:
                shutil.copy2(src_img, os.path.join(out_img,
                                                   os.path.basename(src_img)))
                with open(os.path.join(out_lbl, fn), 'w') as f:
                    f.write('\n'.join(lines_out) + ('\n' if lines_out else ''))
        print(f'  {split}: kept {n_split} images')

    print()
    print(f'{kept_imgs} images kept, {dropped_imgs} dropped for having no '
          'class we use')
    print()
    print(f'{"class":<12}{"examples":>10}')
    print('-' * 22)
    thin = []
    for t in TARGET:
        n = after.get(t, 0)
        flag = ''
        if n == 0:
            flag = '  <-- ABSENT'
            thin.append(t)
        elif n < args.min_per_class:
            flag = '  <-- thin'
            thin.append(t)
        print(f'{t:<12}{n:>10}{flag}')

    if thin:
        print()
        print('These will not be learned reliably: ' + ', '.join(thin))
        print('A class with a handful of examples is worse than absent -- the '
              'model fires on it rarely and wrongly. Either find another '
              'source for those buttons, or drop them from data.yaml and '
              'accept the model cannot name them.')

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'data.yaml'), 'w') as f:
            f.write(f'# Remapped from {args.src} by scripts/remap_dataset.py\n')
            f.write('# Class order MUST match ~/panel_dataset/data.yaml.\n')
            f.write(f'path: {args.out}\n')
            f.write('train: train/images\n')
            val = 'valid' if os.path.isdir(os.path.join(args.out, 'valid')) \
                else 'val'
            f.write(f'val: {val}/images\n\n')
            f.write(f'nc: {len(TARGET)}\n')
            f.write('names:\n')
            for i, t in enumerate(TARGET):
                f.write(f'  {i}: {t}\n')
        print(f'\nwrote {args.out}/data.yaml')
        print('Next: ./scripts/finetune_panel.py --data '
              f'{args.out}/data.yaml --name public_pretrain')
        print('then fine-tune THAT on ~/panel_dataset for our own panel.')
    else:
        print('\n--dry-run: nothing written.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
