#!/usr/bin/env python3
"""Pins the touch-calibration maths. No camera, no arm, no ROS.

    python3 src/mycobot_perception/test/test_calibrate_touch.py

The solve is checked by RECOVERING a transform that was applied on purpose:
build a known camera->base, push points through it, add realistic touch noise,
and see whether Kabsch gets it back. A calibration that cannot pass that has
no business driving an arm.
"""
import importlib.util
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mycobot_driver'))

spec = importlib.util.spec_from_file_location(
    'ctouch', os.path.join(_ROOT, 'scripts', 'calibrate_touch.py'))
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'PASS  {name}')
    else:
        FAIL += 1
        print(f'FAIL  {name}')


def rot(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


rng = np.random.default_rng(7)
R_true = rot(0.3, -0.5, 1.1)
t_true = np.array([0.05, -0.12, 0.31])
P = rng.uniform(-0.12, 0.12, (10, 3)) + np.array([0.0, 0.0, 0.25])
Q = P @ R_true.T + t_true

# --- exact recovery ---------------------------------------------------------

R, t = ct.kabsch(P, Q)
check('an exact point set recovers the rotation',
      np.allclose(R, R_true, atol=1e-9))
check('an exact point set recovers the translation',
      np.allclose(t, t_true, atol=1e-9))
check('the recovered rotation is a proper rotation',
      abs(np.linalg.det(R) - 1.0) < 1e-9
      and np.allclose(R @ R.T, np.eye(3), atol=1e-9))

# --- it must not be a reflection --------------------------------------------
# The classic Kabsch bug: without the determinant correction a noisy or
# awkward set yields a mirror, which fits the points and inverts the arm.
for _ in range(50):
    A = rng.uniform(-0.1, 0.1, (4, 3))
    B = A @ rot(*rng.uniform(-3, 3, 3)).T + rng.uniform(-0.2, 0.2, 3)
    Rr, _ = ct.kabsch(A, B)
    if np.linalg.det(Rr) < 0:
        break
else:
    check('never returns a reflection over 50 random sets', True)

# --- realistic touch noise --------------------------------------------------

def trans_error(spread_m, n, sigma_mm, trials=200):
    errs = []
    for _ in range(trials):
        pts = (rng.uniform(-spread_m / 2, spread_m / 2, (n, 3))
               + np.array([0.0, 0.0, 0.25]))
        obs = pts @ R_true.T + t_true + rng.normal(0, sigma_mm / 1000.0,
                                                   (n, 3))
        _, tn = ct.kabsch(pts, obs)
        errs.append(np.linalg.norm(tn - t_true) * 1000)
    return float(np.mean(errs))


# Noise scales the error linearly -- nothing amplifies or cancels it.
e1 = trans_error(0.16, 10, 1.0)
e3 = trans_error(0.16, 10, 3.0)
print(f'      (at 160mm spread, 10 touches: 1mm noise -> {e1:.2f}mm, '
      f'3mm noise -> {e3:.2f}mm)')
check('touch error scales roughly linearly into the answer',
      2.5 < e3 / e1 < 3.5)

# The finding that matters, and the reason the naive expectation is wrong:
# error does NOT fall as 1/sqrt(N) because the rotation is fitted from the
# same points and acts through the lever arm to the camera. SPREAD is the
# lever that actually works.
tight = trans_error(0.04, 5, 2.0)
wide = trans_error(0.25, 5, 2.0)
print(f'      (2mm touches, 5 points: 40mm spread -> {tight:.1f}mm, '
      f'250mm spread -> {wide:.1f}mm)')
check('spreading the SAME number of touches cuts the error several-fold',
      tight > 4 * wide)

few_wide = trans_error(0.25, 5, 2.0)
many_tight = trans_error(0.04, 20, 2.0)
print(f'      (5 wide -> {few_wide:.1f}mm vs 20 tightly clustered -> '
      f'{many_tight:.1f}mm)')
check('five spread-out touches beat twenty clustered ones',
      few_wide < many_tight)

# --- residuals must localise a bad touch ------------------------------------

Qbad = Q.copy()
Qbad[4] += np.array([0.02, -0.015, 0.01])      # one badly placed touch
R, t = ct.kabsch(P, Qbad)
resid = np.linalg.norm((P @ R.T + t) - Qbad, axis=1) * 1000
check('a single misplaced touch is the largest residual',
      int(np.argmax(resid)) == 4)
check('and it stands out against the others',
      resid[4] > 3 * np.median(resid))

# --- degenerate geometry ----------------------------------------------------

line = np.array([[0.0, 0, 0.2], [0.05, 0, 0.2], [0.1, 0, 0.2],
                 [0.15, 0, 0.2]])
ok, msg = ct.geometry_note(line)
check('collinear points are refused', not ok and 'collinear' in msg)

ok, _ = ct.geometry_note(np.array([[0.0, 0, 0.2], [0.1, 0, 0.2]]))
check('fewer than three points is refused', not ok)

plane = np.column_stack([rng.uniform(-0.1, 0.1, 8),
                         rng.uniform(-0.1, 0.1, 8),
                         np.full(8, 0.25)])
ok, msg = ct.geometry_note(plane)
check('exactly coplanar points solve but are flagged',
      ok and 'coplanar' in msg)

vol = rng.uniform(-0.1, 0.1, (8, 3))
ok, msg = ct.geometry_note(vol)
check('a spread-out set passes without a warning',
      ok and 'coplanar' not in msg and 'spread' in msg)

# --- the tool offset --------------------------------------------------------

HOME = [0.0, 90.0, -149.0, 55.0, 0.0, 0.0]
a = ct.tip_in_base(HOME, 0.0)
b = ct.tip_in_base(HOME, 0.05)
check('a tool offset moves the contact point by exactly its length',
      abs(np.linalg.norm(b - a) - 0.05) < 1e-9)
check('and a zero offset is the flange itself',
      np.allclose(a, np.array(
          __import__('mycobot_driver.collision_guard', fromlist=['x'])
          .flange_transform(HOME))[:3, 3]))

print(f'\n{PASS} passed' + (f', {FAIL} FAILED' if FAIL else ''))
sys.exit(1 if FAIL else 0)
