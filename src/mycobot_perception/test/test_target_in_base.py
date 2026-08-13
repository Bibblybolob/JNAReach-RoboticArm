#!/usr/bin/env python3
"""Pins the pixel -> base-frame chain. Runs without ROS, a camera or an arm.

    python3 src/mycobot_perception/test/test_target_in_base.py

The failures this is guarding against are all silent ones. A depth in the
wrong units, a joint1 composition left out, a camera_to_flange transform used
where camera_to_base was meant -- none of them raise, and all of them put the
arm somewhere confidently wrong.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'mycobot_perception'))

from target_in_base import (  # noqa: E402
    MAX_DEPTH_MM, MIN_DEPTH_MM, TargetError, camera_to_base, pixel_to_camera,
    target_in_base,
)

# The real D405 numbers, read off the factory calibration 2026-08-13.
K = {'fx': 393.8, 'fy': 393.4, 'ppx': 318.1, 'ppy': 236.5}
PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'PASS  {name}')
    else:
        FAIL += 1
        print(f'FAIL  {name}')


def near(a, b, tol=1e-9):
    return abs(a - b) <= tol


def identity():
    return [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


# --- deprojection -----------------------------------------------------------

x, y, z = pixel_to_camera(K['ppx'], K['ppy'], 200.0,
                          K['fx'], K['fy'], K['ppx'], K['ppy'])
check('a pixel at the principal point is straight ahead',
      near(x, 0) and near(y, 0) and near(z, 0.2))

# One focal length off centre is 45 degrees, i.e. x == z.
x, y, z = pixel_to_camera(K['ppx'] + K['fx'], K['ppy'], 200.0,
                          K['fx'], K['fy'], K['ppx'], K['ppy'])
check('one focal length off centre puts x at the depth', near(x, z))

# Round trip against the projection it inverts.
for (px, py, pz) in ((0.03, -0.02, 0.25), (-0.05, 0.04, 0.12)):
    u = K['fx'] * px / pz + K['ppx']
    v = K['fy'] * py / pz + K['ppy']
    bx, by, bz = pixel_to_camera(u, v, pz * 1000.0,
                                 K['fx'], K['fy'], K['ppx'], K['ppy'])
    check(f'round trip through the pinhole model at z={pz}',
          near(bx, px, 1e-9) and near(by, py, 1e-9) and near(bz, pz, 1e-9))

# --- the units trap ---------------------------------------------------------

try:
    pixel_to_camera(320, 240, 0.2, K['fx'], K['fy'], K['ppx'], K['ppy'])
    check('metres passed where mm were wanted is refused', False)
except TargetError:
    check('metres passed where mm were wanted is refused', True)

for bad in (MIN_DEPTH_MM - 1, MAX_DEPTH_MM + 1, 0.0):
    try:
        pixel_to_camera(320, 240, bad, K['fx'], K['fy'], K['ppx'], K['ppy'])
        check(f'depth {bad}mm refused', False)
    except TargetError:
        check(f'depth {bad}mm refused', True)

# --- the joint1 composition -------------------------------------------------

p = camera_to_base((1.0, 0.0, 0.0), identity(), 0.0, 0.0)
check('no pan and an identity calibration is a no-op', p == (1.0, 0.0, 0.0))

p = camera_to_base((1.0, 0.0, 0.0), identity(), 90.0, 0.0)
check('90deg of pan rotates x onto y',
      near(p[0], 0, 1e-12) and near(p[1], 1.0, 1e-12))

# Solved at 30, queried at 30: no rotation, however non-zero the angle.
p = camera_to_base((1.0, 0.0, 0.0), identity(), 30.0, 30.0)
check('querying at the angle it was solved at applies no rotation',
      near(p[0], 1.0, 1e-12) and near(p[1], 0.0, 1e-12))

# It is the DIFFERENCE that matters, not the absolute angle. This is the bug
# worth pinning: forgetting solved_joint1_deg is correct at one pan and wrong
# everywhere else.
a = camera_to_base((0.4, 0.1, 0.2), identity(), 70.0, 40.0)
b = camera_to_base((0.4, 0.1, 0.2), identity(), 30.0, 0.0)
check('only the difference in pan matters',
      all(near(i, j, 1e-12) for i, j in zip(a, b)))

# Rotation about base z must not touch z.
p = camera_to_base((0.3, -0.2, 0.15), identity(), 57.0, 0.0)
check('pan leaves the z coordinate alone', near(p[2], 0.15, 1e-12))
check('pan preserves distance from the base axis',
      near(math.hypot(p[0], p[1]), math.hypot(0.3, -0.2), 1e-12))

# --- the whole chain --------------------------------------------------------

calib = {'camera_to_base': identity(), 'joint1_deg': 0.0}
p = target_in_base(K['ppx'], K['ppy'], 250.0, K, calib, 0.0)
check('end to end: dead centre at 250mm is 0.25m along the optical axis',
      near(p[0], 0) and near(p[1], 0) and near(p[2], 0.25))

try:
    target_in_base(320, 240, 250.0, K, {'camera_to_flange': identity()}, 0.0)
    check('a camera_to_flange calibration is refused', False)
except TargetError:
    check('a camera_to_flange calibration is refused', True)

try:
    target_in_base(320, 240, 250.0, {'fx': 0, 'fy': 0, 'ppx': 0, 'ppy': 0},
                   calib, 0.0)
    check('empty intrinsics are refused', False)
except TargetError:
    check('empty intrinsics are refused', True)

# A translation-only calibration must move the point by exactly that offset.
T = identity()
T[0][3], T[1][3], T[2][3] = 0.1, -0.05, 0.2
p = target_in_base(K['ppx'], K['ppy'], 100.0, K,
                   {'camera_to_base': T, 'joint1_deg': 0.0}, 0.0)
check('a translation-only calibration offsets the point exactly',
      near(p[0], 0.1) and near(p[1], -0.05) and near(p[2], 0.3))

print(f'\n{PASS} passed' + (f', {FAIL} FAILED' if FAIL else ''))
sys.exit(1 if FAIL else 0)
