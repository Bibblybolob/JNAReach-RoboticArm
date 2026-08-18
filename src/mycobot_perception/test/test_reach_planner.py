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

# Position-only first, which is the path that always exists. TARGET stopped
# admitting an AIMED plan when TOOL_AXIS's sign was corrected on 2026-08-18 --
# pointing the tool at a target is a different, harder request than pointing
# the LED face at it, and it changes which targets are feasible. That is the
# geometry changing, not the adjacency rule, so the rule is checked on a
# target that still plans as well as on the position-only path.
plan = plan_press(TARGET, tool_length_m=0.03, standoff_m=0.05,
                  orientation=False)
check('standoff and touch are adjacent, position-only',
      plan['max_joint_step_deg'] < 30.0)

AIM_TARGET = (0.20, 0.05, 0.15)
aimed = plan_press(AIM_TARGET, tool_length_m=0.03, standoff_m=0.05)
check('standoff and touch are adjacent, aimed',
      aimed['max_joint_step_deg'] < 30.0)

# Every plan that comes back must be safe by the guard, at both ends.
from mycobot_driver.collision_guard import CollisionGuard  # noqa: E402
g = CollisionGuard(tool_offset_m=0.03)
check('both returned poses pass the collision guard',
      g.check(plan['standoff_deg'])[0] and g.check(plan['touch_deg'])[0])
check('both aimed poses pass the collision guard',
      g.check(aimed['standoff_deg'])[0] and g.check(aimed['touch_deg'])[0])

# --- a spread of reachable targets ------------------------------------------

# Measured separately for the two paths, because they cover different amounts
# of the workspace and conflating them hides which one regressed. Correcting
# TOOL_AXIS's sign on 2026-08-18 cut the AIMED coverage -- pointing the tool
# at a target is a stricter request than pointing the flange's other face at
# it -- while position-only was unaffected. Measured that day, tool 30mm,
# standoff 40mm, over the 18 sampled targets:
#
#     aimed          10/18 verified, 8 refused, 0 off by >2mm
#     position-only  12/18 verified, 6 refused, 0 off by >2mm
#
# Note that NOTHING plans inaccurately -- every returned plan puts the tip on
# the target to under 2mm. The shortfall is refusals, which is the planner
# declining rather than guessing, so these bars are set just under the
# measured values to catch a real regression without re-tuning on noise.
def coverage(orientation):
    ok = tried = 0
    for x in (0.14, 0.18, 0.22):
        for y in (-0.08, 0.0, 0.08):
            for z in (0.10, 0.16):
                tried += 1
                try:
                    p = plan_press((x, y, z), tool_length_m=0.03,
                                   standoff_m=0.04, orientation=orientation)
                except ReachError:
                    continue
                tip = flange_xyz(p['touch_deg']) + np.array(
                    approach_direction((x, y, z))) * 0.03
                if np.linalg.norm(tip - np.array((x, y, z))) < 2e-3:
                    ok += 1
    return ok, tried

ok_pos, tried = coverage(False)
ok_aim, _ = coverage('auto')
print(f'      (position-only {ok_pos}/{tried}, aimed {ok_aim}/{tried} '
      'planned and verified)')
check('position-only plans most of a reachable workspace',
      ok_pos >= tried * 0.6)
check('aiming the tool still plans half of it', ok_aim >= tried * 0.5)

print(f'\n{PASS} passed' + (f', {FAIL} FAILED' if FAIL else ''))
sys.exit(1 if FAIL else 0)
