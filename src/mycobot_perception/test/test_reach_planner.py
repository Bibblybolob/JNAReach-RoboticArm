#!/usr/bin/env python3
"""Pins base-frame point -> joint angles. Runs without ROS or an arm.

    python3 src/mycobot_perception/test/test_reach_planner.py

Checks against the arm's own FK rather than against stored numbers: a plan is
correct if running its joint angles forward puts the flange where the plan
said it would, and puts the TOOL TIP on the target.
"""
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mycobot_perception',
                                'mycobot_perception'))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mycobot_driver'))

from reach_planner import (  # noqa: E402
    ReachError, approach_direction, plan_press,
)
from mycobot_driver.collision_guard import flange_transform  # noqa: E402

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'PASS  {name}')
    else:
        FAIL += 1
        print(f'FAIL  {name}')


def flange_xyz(deg):
    return np.array(flange_transform(list(deg)))[:3, 3]


# A point out in front of the arm, inside its reach and above the desk.
TARGET = (0.18, 0.05, 0.14)

# --- the approach direction -------------------------------------------------

d = approach_direction((0.2, 0.0, 0.1))
check('approach is horizontal and outward', d == (1.0, 0.0, 0.0))

d = approach_direction((0.0, 0.2, 0.1))
check('approach follows the target in y',
      abs(d[1] - 1.0) < 1e-12 and abs(d[0]) < 1e-12)

check('approach is always horizontal', abs(approach_direction(TARGET)[2]) < 1e-12)

try:
    approach_direction((0.0, 0.0, 0.3))
    check('a target on the arm axis is refused', False)
except ReachError:
    check('a target on the arm axis is refused', True)

# --- the plan lands where it says -------------------------------------------

plan = plan_press(TARGET, tool_length_m=0.0, standoff_m=0.04)
touch = flange_xyz(plan['touch_deg'])
check('with no tool, the flange reaches the target itself',
      np.linalg.norm(touch - np.array(TARGET)) < 1.5e-3)

stand = flange_xyz(plan['standoff_deg'])
check('the standoff sits 40mm back from the touch pose',
      abs(np.linalg.norm(touch - stand) - 0.04) < 2e-3)

# The standoff must be further from the target, not merely 40mm away from the
# touch pose -- a sign error here still passes a distance check.
check('the standoff is further from the target than the touch pose',
      np.linalg.norm(stand - np.array(TARGET))
      > np.linalg.norm(touch - np.array(TARGET)))

# --- the tool offset --------------------------------------------------------

TOOL = 0.06
plan = plan_press(TARGET, tool_length_m=TOOL, standoff_m=0.03)
touch = flange_xyz(plan['touch_deg'])
d = np.array(approach_direction(TARGET))
tip = touch + d * TOOL
check('with a tool, the TIP lands on the target and the flange stops short',
      np.linalg.norm(tip - np.array(TARGET)) < 1.5e-3)
check('the flange is a tool-length short of the target',
      abs(np.linalg.norm(touch - np.array(TARGET)) - TOOL) < 1.5e-3)

# The failure this guards: forgetting the tool length puts the flange on the
# button and drives the tool a full tool-length through it.
plan0 = plan_press(TARGET, tool_length_m=0.0, standoff_m=0.03)
check('tool length actually changes the solution',
      np.linalg.norm(flange_xyz(plan0['touch_deg']) - touch) > 0.5 * TOOL)

# --- refusals ---------------------------------------------------------------

try:
    plan_press((0.9, 0.0, 0.2))
    check('an out-of-reach target is refused', False)
except ReachError:
    check('an out-of-reach target is refused', True)

try:
    plan_press((0.18, 0.05, -0.20))
    check('a target below the desk is refused', False)
except ReachError:
    check('a target below the desk is refused', True)

try:
    plan_press(TARGET, standoff_m=-0.02)
    check('a negative standoff is refused', False)
except ReachError:
    check('a negative standoff is refused', True)

try:
    plan_press(TARGET, approach_dir=(0.0, 0.0, 0.0))
    check('a zero approach direction is refused', False)
except ReachError:
    check('a zero approach direction is refused', True)

# --- the two poses stay in the same arm configuration -----------------------

plan = plan_press(TARGET, tool_length_m=0.03, standoff_m=0.05)
check('standoff and touch are adjacent in joint space',
      plan['max_joint_step_deg'] < 30.0)

# Every plan that comes back must be safe by the guard, at both ends.
from mycobot_driver.collision_guard import CollisionGuard  # noqa: E402
g = CollisionGuard(tool_offset_m=0.03)
check('both returned poses pass the collision guard',
      g.check(plan['standoff_deg'])[0] and g.check(plan['touch_deg'])[0])

# --- a spread of reachable targets ------------------------------------------

ok = 0
tried = 0
for x in (0.14, 0.18, 0.22):
    for y in (-0.08, 0.0, 0.08):
        for z in (0.10, 0.16):
            tried += 1
            try:
                p = plan_press((x, y, z), tool_length_m=0.03, standoff_m=0.04)
            except ReachError:
                continue
            tip = flange_xyz(p['touch_deg']) + np.array(
                approach_direction((x, y, z))) * 0.03
            if np.linalg.norm(tip - np.array((x, y, z))) < 2e-3:
                ok += 1
print(f'      ({ok} of {tried} sampled targets planned and verified)')
check('most of a reachable workspace plans correctly', ok >= tried * 0.6)

# --- the approach is a straight line, not an arc ----------------------------
#
# The endpoints were always right; it was the path BETWEEN them that bowed.
# The servos interpolate in JOINT space, so the only way to see this is to
# interpolate that way and run FK along it -- checking the waypoints alone
# proves nothing, because they sit on the line by construction.


def path_bow_mm(plan):
    """Worst deviation of the joint-interpolated tip path from the straight line."""
    path = [np.array(q) for q in plan['path_deg']]
    a = flange_xyz(path[0])
    b = flange_xyz(path[-1])
    d = (b - a) / np.linalg.norm(b - a)
    worst = 0.0
    for q0, q1 in zip(path[:-1], path[1:]):
        for f in np.linspace(0.0, 1.0, 21):
            p = flange_xyz(q0 + (q1 - q0) * f)
            worst = max(worst, float(np.linalg.norm((p - a) - d * ((p - a) @ d))))
    return worst * 1000.0


# Off to one side is the case that bows worst -- head-on is nearly straight
# already, so a test that only covered it would pass on a broken planner.
TGT = (0.16, 0.10, 0.24)
NRM = [0.85, 0.53, 0.0]
one = plan_press(TGT, tool_length_m=0.07, standoff_m=0.04,
                 approach_dir=NRM, approach_steps=1)
four = plan_press(TGT, tool_length_m=0.07, standoff_m=0.04,
                  approach_dir=NRM, approach_steps=4)

check('a subdivided approach emits one waypoint per segment',
      len(four['path_deg']) == 5 and len(one['path_deg']) == 2)
check('the path starts at the standoff and ends at the touch pose',
      four['path_deg'][0] == four['standoff_deg']
      and four['path_deg'][-1] == four['touch_deg'])

b1, b4 = path_bow_mm(one), path_bow_mm(four)
print(f'      (tip bows {b1:.2f}mm undivided, {b4:.2f}mm over 4 segments)')
# Measured 2026-08-16: 2.88mm against 0.31mm. Asserted loosely because the
# absolute figure depends on the target, but the RATIO is the property --
# subdividing must actually straighten the path, not merely add waypoints.
check('subdividing straightens the approach', b4 < b1 / 3.0)
check('a subdivided approach stays well inside the backlash floor', b4 < 1.0)

# Every waypoint must be safe, not just the endpoints: discovering the fourth
# is refused with the tool already at the third is the worst place to find out.
g2 = CollisionGuard(tool_offset_m=0.07)
check('every waypoint on the path passes the collision guard',
      all(g2.check(q)[0] for q in four['path_deg']))

print(f'\n{PASS} passed' + (f', {FAIL} FAILED' if FAIL else ''))
sys.exit(1 if FAIL else 0)
