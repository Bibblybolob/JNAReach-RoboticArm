#!/usr/bin/env python3
"""Before/after: put two eval_buttons.py runs side by side.

    ./scripts/compare_buttons.py eval_results/before.json eval_results/after.json

The whole point is to answer "did the retrain actually help, and where", on
the priorities in docs/button_datasets.md -- the hall call first, then floor
numbers, then help.

Both runs must have scored the SAME split of the SAME source, or the
comparison is between two different questions; this refuses if they did not.
That check is not pedantry: an "after" run scored on an easier split is the
most flattering mistake available here, and it looks exactly like success.


Reading it
----------
`recall` is the one to watch for the hall call. A missed `up` means the arm
never sees the button it most needs; a false positive costs it a look at
something that turns out not to be a button. So a recall gain is worth more
than an equal precision gain on `up`/`down`, and the table orders those two
first for that reason.

For the legend reader, `WRONG` is not a milder version of `declined` -- a
decline costs a retry and a wrong read sends the arm to the wrong floor with
nothing in the logs. They are reported and compared separately, never
averaged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'mycobot_perception'))
from mycobot_perception.button_classes import DETECT_CLASSES  # noqa: E402

# Priority order from Jonathan, 2026-08-16 -- not the DETECT_CLASSES order,
# because this table is read by a human deciding whether the arm got better at
# the job rather than by anything that indexes into it.
ROWS = ['up', 'down', 'floor', 'help', 'open', 'close', 'stop', 'keyhole',
        'other']


def fmt(v, w=7):
    return ' ' * (w - 1) + '-' if v is None else f'{v:>{w}.3f}'


def delta(a, b, w=8):
    if a is None or b is None:
        return ' ' * w
    d = b - a
    s = f'{d:+.3f}'
    return f'{s:>{w}}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('before')
    ap.add_argument('after')
    ap.add_argument('--force', action='store_true',
                    help='compare even if the two runs used different data')
    args = ap.parse_args()

    A = json.load(open(args.before))
    B = json.load(open(args.after))

    same = (A['src'] == B['src'] and A['split'] == B['split']
            and A['images'] == B['images'])
    if not same and not args.force:
        print('!! these two runs did not score the same data:')
        for k in ('src', 'split', 'images', 'conf', 'iou'):
            print(f'   {k:<8} before={A.get(k)!r}  after={B.get(k)!r}')
        print('\nComparing them would compare two different questions. '
              'Re-run the eval on one split, or pass --force if you know why '
              'they differ.')
        return 1

    print('=' * 72)
    print(f'BEFORE  {A["label"]}')
    print(f'AFTER   {B["label"]}')
    print(f'on {A["images"]} images of {os.path.basename(A["src"])}/'
          f'{A["split"]}, conf>={A["conf"]}, IoU>={A["iou"]}')
    print('=' * 72)

    print(f'\n{"":<9}{"n":>5}{"precision":>19}{"recall":>19}')
    print(f'{"class":<9}{"":>5}{"before":>7}{"after":>7}{"delta":>8}'
          f'{"before":>7}{"after":>7}{"delta":>8}')
    print('-' * 72)
    for c in ROWS:
        a = A['classes'].get(c, {})
        b = B['classes'].get(c, {})
        n = b.get('n', a.get('n', 0))
        star = '  <-- priority 1' if c in ('up', 'down') else ''
        print(f'{c:<9}{n:>5}'
              f'{fmt(a.get("precision"))}{fmt(b.get("precision"))}'
              f'{delta(a.get("precision"), b.get("precision"))}'
              f'{fmt(a.get("recall"))}{fmt(b.get("recall"))}'
              f'{delta(a.get("recall"), b.get("recall"))}{star}')

    # --- the hall call, called out on its own ---------------------------
    print('\n' + '-' * 72)
    for c in ('up', 'down'):
        a, b = A['classes'].get(c, {}), B['classes'].get(c, {})
        n = b.get('n') or a.get('n') or 0
        ar, br = a.get('recall'), b.get('recall')
        if n and ar is not None and br is not None:
            print(f'{c:>5}: found {int(round(br * n))} of {n} '
                  f'(was {int(round(ar * n))})')
        elif n:
            print(f'{c:>5}: {n} in the split; '
                  f'before={"n/a" if ar is None else f"{ar:.3f}"}, '
                  f'after={"n/a" if br is None else f"{br:.3f}"}')

    # --- the reader ------------------------------------------------------
    print('\n' + '=' * 72)
    print('READING THE FLOOR LEGEND')
    print('=' * 72)
    la, lb = A.get('legend'), B.get('legend')
    if not lb:
        print('  the AFTER run had no reader -- nothing to compare')
    else:
        if not la:
            print('  the BEFORE model could not read legends at all: its '
                  'class list stopped at button-3/floor-3, so every floor '
                  'above that was unnameable by construction. Treat the '
                  'before column as absent rather than as zero.')
        tot_b = lb['right'] + lb['wrong'] + lb['declined']
        rows = [('correct', 'right'), ('declined', 'declined'),
                ('WRONG', 'wrong')]
        print(f'\n{"":<10}{"before":>10}{"after":>10}{"after %":>10}')
        print('-' * 40)
        for name, key in rows:
            bv = lb[key]
            av = f'{la[key]:>10}' if la else f'{"-":>10}'
            print(f'{name:<10}{av}{bv:>10}{100 * bv / tot_b:>9.1f}%')
        if lb['right'] + lb['wrong']:
            print(f'\n  of the ones it committed to: '
                  f'{100 * lb["right"] / (lb["right"] + lb["wrong"]):.1f}% right')
        if lb.get('confusions'):
            print('\n  worst confusions after (truth -> reported):')
            for t, g, c in lb['confusions'][:8]:
                print(f'    {t:>4} -> {g:<4} x{c}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
