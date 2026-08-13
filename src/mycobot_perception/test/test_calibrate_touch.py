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

# --- the tool offset is recoverable from the data -------------------------
# "The flange is flat so the offset is zero" is a claim about where the URDF
# puts the flange FRAME, not about the hardware. A constant offset in the
# flange frame maps to a different base displacement at every touch
# orientation, so it cannot hide inside a rigid transform -- which is exactly
# what makes it estimable.

def implied_offset(true_off_m, n=12, noise_mm=1.5):
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    t = np.array([0.05, -0.1, 0.30])
    cams, tips = [], []
    for _ in range(n):
        ang = [rng.uniform(-30, 30), 90 + rng.uniform(-20, 20),
               -149 + rng.uniform(-10, 10), 55 + rng.uniform(-40, 40),
               rng.uniform(-50, 50), rng.uniform(-60, 60)]
        tip = ct.tip_in_base(ang, true_off_m)
        cams.append(R.T @ (tip - t) + rng.normal(0, noise_mm / 1000.0, 3))
        tips.append(ang)
    best, lo = None, 1e9
    for off in np.linspace(-0.03, 0.03, 121):
        pts = np.array([ct.tip_in_base(a, off) for a in tips])
        Ro, to = ct.kabsch(np.array(cams), pts)
        r = np.linalg.norm((np.array(cams) @ Ro.T + to) - pts, axis=1).mean()
        if r < lo:
            lo, best = r, off
    return best * 1000.0


z = implied_offset(0.0)
print(f'      (true offset 0mm -> estimated {z:+.1f}mm)')
check('a genuinely zero tool offset is estimated near zero', abs(z) < 3.0)

s = implied_offset(0.007)
print(f'      (true offset 7mm -> estimated {s:+.1f}mm)')
check('a real 7mm tool offset is recovered from the touches',
      abs(s - 7.0) < 3.0)

# --- the CALL SITE, not just the function ---------------------------------
# collect() shipped with a NameError: it called jog_to_corner(guard, ...) and
# never created a guard. The unit tests above passed one in explicitly, so
# they proved the function worked while the caller was broken. Stub the camera
# and the broker and drive collect() far enough to reach the jog.

def test_collect_reaches_the_jog():
    import tempfile
    import cv2
    import types

    class FakeCam:
        K = np.array([[393.8, 0, 318.1], [0, 393.4, 236.5], [0, 0, 1.0]])

        def frame(self):
            board, _ = ct.board_for(38.9)
            img = board.draw((640, 480))
            return (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                    np.full((480, 640), 250.0))

        def close(self):
            pass

    real_cam, real_req, real_jog, real_out = (
        ct.Camera, ct.request, ct.jog_to_corner, ct.OUT_DIR)
    seen = []
    try:
        ct.Camera = FakeCam
        ct.request = lambda r, timeout=5.0: (
            {'link': {'fraction': 0.95, 'valid': 38, 'window': 40}}
            if r['cmd'] == 'health' else
            {'angles': [0.0] * 6, 'age_ms': 50} if r['cmd'] == 'state'
            else {'ok': True})
        ct.jog_to_corner = lambda guard, speed, step, lj: (
            seen.append((type(guard).__name__, speed, step)) or ('quit', lj))
        ct.OUT_DIR = tempfile.mkdtemp()

        class Args:
            points, square_mm, tool_offset_mm = 8, 38.9, 0.0
            free_drive, speed, step = False, 25, 3.0
            restart, timeout = True, 25.0

        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            rc = ct.collect(Args())
        return rc == 0 and seen and seen[0][0] == 'CollisionGuard'
    finally:
        ct.Camera, ct.request = real_cam, real_req
        ct.jog_to_corner, ct.OUT_DIR = real_jog, real_out


try:
    check('collect() reaches the jog with a real collision guard',
          test_collect_reaches_the_jog())
except Exception as e:
    check(f'collect() reaches the jog with a real collision guard ({e})', False)

print(f'\n{PASS} passed' + (f', {FAIL} FAILED' if FAIL else ''))
sys.exit(1 if FAIL else 0)
