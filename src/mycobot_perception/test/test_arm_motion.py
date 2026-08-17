#!/usr/bin/env python3
"""Pin the streamed-motion maths. Runs without ROS and without an arm.

    python3 src/mycobot_perception/test/test_arm_motion.py

None of this commands motion -- it tests the arithmetic that decides what
WOULD be commanded, which is the part that can be got wrong silently. The
lookahead in particular is the one piece here that can drive the tool INTO the
panel if its clamp is wrong, and a clamp that fails only on some geometries is
exactly the bug that survives a hardware smoke test.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mycobot_perception'))

from arm_motion import (  # noqa: E402
    ARRIVE_TOL_DEG, BLEND_TOL_DEG, FINAL_SPEED_FRAC, LOOKAHEAD_S,
    lookahead_target, travel_budget_s)
from mycobot_perception.reach_planner import (  # noqa: E402
    REVERSAL_OK_DEG, path_reversal_deg)

PASS = FAIL = 0


def check(msg, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'PASS  {msg}')
    else:
        FAIL += 1
        print(f'FAIL  {msg}')


# --- the lookahead clamp: the safety-critical one --------------------------
#
# The final waypoint is ON the button. Commanding past it is pressing through
# the panel, so no geometry may produce a target beyond it on any joint.

path = [[0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0], [0, 0, 6, 0, 0, 0],
        [0, 0, 9, 0, 0, 0], [0, 0, 12, 0, 0, 0]]
t0 = lookahead_target(path[0], path[1], path[-1], 3.0)
check('lookahead commands PAST an intermediate waypoint', t0[2] > 3.0)

t_last = lookahead_target(path[2], path[3], path[-1], 99.0)
check('a huge lookahead is clamped to the final goal, never past',
      abs(t_last[2] - 12.0) < 1e-9)

rng = np.random.default_rng(0)
violations = 0
for _ in range(4000):
    n = int(rng.integers(3, 7))
    direction = rng.normal(size=6)
    steps = np.cumsum(np.abs(rng.random((n, 1))) * direction, axis=0)
    p = [list(np.zeros(6))] + [list(s) for s in steps]
    for i in range(len(p) - 1):
        t = lookahead_target(p[i], p[i + 1], p[-1], float(rng.random() * 10))
        for j in range(6):
            lo, hi = sorted((p[i][j], p[-1][j]))
            if not (lo - 1e-9 <= t[j] <= hi + 1e-9):
                violations += 1
check('the clamp holds over 4000 random monotonic paths (none past the goal)',
      violations == 0)

# Degenerate inputs must not explode -- a zero-length segment happens when two
# waypoints coincide, which a coarse subdivision can produce.
check('a zero-length segment returns the waypoint unchanged',
      lookahead_target([1] * 6, [1] * 6, [2] * 6, 5.0) == [1.0] * 6)
check('zero lookahead is the waypoint itself',
      lookahead_target([0] * 6, [1] * 6, [2] * 6, 0.0) == [1.0] * 6)

# --- reversal detection -----------------------------------------------------

mono = [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0]]
check('a monotonic path reports no reversal', path_reversal_deg(mono) == 0.0)

# Out to 3, back to 2: one degree of backward travel.
rev = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 3, 0],
       [0, 0, 0, 0, 2, 0]]
check('a joint that goes out and back reports its backward travel',
      abs(path_reversal_deg(rev) - 1.0) < 1e-6)

# Solver noise is not a direction change.
noise = [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0.001, 0], [2, 0, 0, 0, -0.001, 0],
         [3, 0, 0, 0, 0.001, 0]]
check('sub-noise wobble is not counted as a reversal',
      path_reversal_deg(noise) == 0.0)

check('the reversal threshold stays under the 0.79deg backlash floor',
      0.0 < REVERSAL_OK_DEG < 0.79)

# --- budgets and tolerances -------------------------------------------------

check('a long move gets a longer budget than a short one',
      travel_budget_s(90, 50) > travel_budget_s(5, 50))
check('a slower speed gets a longer budget',
      travel_budget_s(45, 25) > travel_budget_s(45, 100))
check('the blend tolerance is looser than the arrival tolerance',
      BLEND_TOL_DEG > ARRIVE_TOL_DEG)
# Below the backlash floor the wait can never be satisfied and every move
# would report a failure to arrive.
check('the arrival tolerance stays above the 0.79deg backlash floor',
      ARRIVE_TOL_DEG > 0.79)
check('the final segment is slower than the rest', 0.0 < FINAL_SPEED_FRAC < 1.0)
check('the lookahead is a fraction of a second, not a joint angle',
      0.0 < LOOKAHEAD_S < 1.0)

print(f'\n{PASS} passed' + (f', {FAIL} FAILED' if FAIL else ''))
sys.exit(1 if FAIL else 0)
