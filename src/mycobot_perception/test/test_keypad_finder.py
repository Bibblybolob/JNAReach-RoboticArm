#!/usr/bin/env python3
"""Pins the keypad finder's ORDERING and its refusals.

    python3 src/mycobot_perception/test/test_keypad_finder.py

Runs without ROS and without a camera: the panels are drawn here.

Everything checked below is a bug that really happened on 2026-08-17/18, and
all three had the same shape -- the numbers looked perfect and the wrong
button was chosen. A flipped row order does not crash; it presses floor 7 when
asked for floor 1. So these tests care about WHICH button gets WHICH label,
not merely that something was found.
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'mycobot_perception'))

import keypad_finder as kf  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'PASS  {name}')
    else:
        FAIL += 1
        print(f'FAIL  {name}  {detail}')


def panel(n_rows=8, rot_deg=0.0, size=(720, 900), pitch=70, col_gap=80,
          radius=22, drop_top=False):
    """A synthetic keypad: 2 columns x n_rows, rotated by rot_deg.

    Returns (image, [(x, y) centres in row-major order, top row first]).
    """
    img = np.full((size[1], size[0], 3), 235, np.uint8)
    cx, cy = size[0] / 2.0, size[1] / 2.0
    th = math.radians(rot_deg)
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th), math.cos(th)]])
    centres = []
    rows = range(1, n_rows) if drop_top else range(n_rows)
    for r in rows:
        for c in (0, 1):
            local = np.array([(c - 0.5) * col_gap,
                              (r - (n_rows - 1) / 2.0) * pitch])
            x, y = R @ local + np.array([cx, cy])
            centres.append((float(x), float(y)))
            # Filled disc, darker rim, and a mark in the middle -- the last of
            # these matters. Template matching crops a box of the DETECTED
            # radius, so a plain disc fills the crop edge to edge and every
            # patch is featureless grey: normalising it amplifies noise and
            # the correlation means nothing. Real buttons carry a printed
            # digit, which is the structure the match actually keys on.
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(img, (xi, yi), radius, (196, 196, 196), -1)
            cv2.circle(img, (xi, yi), radius, (110, 110, 110), 2)
            cv2.circle(img, (xi, yi), max(radius // 3, 3), (90, 90, 90), -1)
    return img, centres


def rows_of(img):
    rows, rad = kf.find_grid(img, n_cols=2)
    return rows, rad


# --- the grid is found at all, square-on and rotated ------------------------
for rot in (0, 8, 17, -17, 25):
    img, centres = panel(rot_deg=rot)
    try:
        rows, _ = rows_of(img)
        ok = len(rows) == 8
    except kf.KeypadError as e:
        rows, ok = [], False
    check(f'finds all 8 rows at {rot:+d}deg', ok,
          f'got {len(rows)} rows')

# --- ROW ORDER IS TOP TO BOTTOM, whatever the rotation ---------------------
# The flip bug: rows were ordered by a projection whose sign came from an
# eigenvector, so the whole layout inverted and the bottom symbol row was
# labelled 11/12.
for rot in (0, 17, -17, 25):
    img, _ = panel(rot_deg=rot)
    rows, _ = rows_of(img)
    ys = [float(np.mean([c[1] for c in r])) for r in rows]
    check(f'rows run top to bottom at {rot:+d}deg',
          all(a < b for a, b in zip(ys, ys[1:])), f'ys={[round(y) for y in ys]}')
    xs = [(r[0][0], r[1][0]) for r in rows]
    check(f'each row runs left to right at {rot:+d}deg',
          all(a < b for a, b in xs), f'xs={xs[:2]}')

# --- labelling puts the layout's first row on the panel's TOP row ----------
img, _ = panel(rot_deg=17)
lab = kf.find_labelled(img, layout=kf.PANEL_12_PRINTED)
top = min(lab.values(), key=lambda c: c[1])
bottom = max(lab.values(), key=lambda c: c[1])
check('top-most button is labelled 11 (not fan)',
      lab[11] == top, f'top={top}')
check('bottom-most button is labelled alarm (not 12)',
      lab['alarm'] == bottom, f'bottom={bottom}')

# --- a short layout must not silently take rows 2..7 ----------------------
# The one-row-out bug: the faint 11/12 row was PRESENT but undetected, so the
# six rows found were rows 2-7 and every button was labelled a row out. The
# row therefore has to be in the image and missing from `rows` -- dropping it
# from the image instead would test nothing, since then there really is
# nothing above and labelling is correct.
img, _ = panel(n_rows=8, rot_deg=10)
rows, rad = rows_of(img)
check('_row_above finds the row above when the top row is missing',
      kf._row_above(img, rows[1:], rad) is not None)
check('_row_above finds nothing above a complete panel',
      kf._row_above(img, rows, rad) is None)

# and the same panel WITH its top row present must be accepted
img, _ = panel(n_rows=8, rot_deg=10)
try:
    lab = kf.find_labelled(img, layout=kf.PANEL_12_PRINTED_NUMBERS,
                           top_anchored=True)
    top = min(lab.values(), key=lambda c: c[1])
    check('accepts a top-anchored short layout, extra rows below ignored',
          lab[11] == top and len(lab) == 12, f'{len(lab)} labelled')
except kf.KeypadError as e:
    check('accepts a top-anchored short layout, extra rows below ignored',
          False, str(e)[:60])

# --- completion must not extrapolate across a gap -------------------------
# The invented-row bug: rows at 337,388,435,687 gave a pitch of 117 against a
# true 49, and two rows were placed above the panel.
img, _ = panel(n_rows=6, rot_deg=0)
rows, rad = rows_of(img)
gappy = [rows[0], rows[1], rows[2], rows[5]]          # a hole four pitches wide
out = kf.complete_lattice(img, gappy, rad, 6)
check('refuses to complete from unevenly spaced rows',
      len(out) == len(gappy), f'completed to {len(out)}')

# --- label() still refuses a plain row-count mismatch ----------------------
img, _ = panel(n_rows=6)
rows, _ = rows_of(img)
try:
    kf.label(rows, kf.PANEL_12_PRINTED)               # 6 found, 8 expected
    check('refuses a row-count mismatch', False, 'it labelled')
except kf.KeypadError:
    check('refuses a row-count mismatch', True)

print()
print(f'{PASS} passed, {FAIL} failed')
sys.exit(1 if FAIL else 0)
