"""A point in the base frame -> joint angles that put the tool on it.

The last link in the chain this project has been building toward:

    detection -> pixel + depth -> base-frame point -> THESE JOINT ANGLES

Kept free of ROS so it tests without a workspace. It imports the arm's real
FK/IK from scripts/ik_demo.py and the real keep-out volumes from
mycobot_driver.collision_guard, rather than carrying copies -- there are
already two FK chains in this repo and they were verified identical to machine
precision on 2026-08-13, which is a property worth not breaking by adding a
third.

Two poses, not one
------------------
Pressing something is an approach followed by a push, so this returns a
STANDOFF pose and a TOUCH pose. Driving straight to the touch pose from
wherever the arm happens to be sweeps an arbitrary path through the panel --
the arm does not travel in straight lines in Cartesian space, and the button
is on a wall it can hit on the way.

The tool is not the flange
--------------------------
`ik()` solves for the FLANGE. Anything mounted past it -- a presser, a finger --
means the flange must stop short by the tool's length or the tool goes through
the button. That is `tool_length_m`, and it defaults to 0 deliberately: a
wrong non-zero default silently misses by exactly its own value, while zero is
obviously wrong the first time it is used.

**Orientation is not constrained**, and that limit is real. `ik()` solves
position only, because demanding a full pose on this arm can make a reachable
position unreachable. So this puts the flange in the right PLACE without
promising which way the tool points. For a flat panel approached roughly
head-on that is usually fine; it is not a substitute for a wrist that is aimed
properly, and `approach_dir` exists so a caller who knows the panel normal can
supply it.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))


def _load_ik():
    """The arm's real IK, loaded from the script that owns it."""
    path = os.path.join(_ROOT, 'scripts', 'ik_demo.py')
    spec = importlib.util.spec_from_file_location('_ik_demo', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReachError(ValueError):
    """A target that must not be turned into a motion command."""


def approach_direction(target_xyz):
    """Unit vector the tool travels ALONG to reach the target, default case.

    Horizontal, pointing outward from the arm's own axis toward the target --
    which is how a wall panel gets pressed. Vertical is deliberately excluded:
    a target directly over the base has no defined outward direction, and
    guessing one there produces a standoff pose in an arbitrary place.
    """
    x, y = target_xyz[0], target_xyz[1]
    r = math.hypot(x, y)
    if r < 1e-6:
        raise ReachError(
            'the target is on the arm\'s own vertical axis, so there is no '
            'outward direction to approach along. Pass approach_dir.')
    return (x / r, y / r, 0.0)


def plan_press(target_xyz, tool_length_m=0.0, standoff_m=0.04,
               approach_dir=None, seed_deg=None, guard=None):
    """Standoff and touch joint angles for putting the tool on target_xyz.

    Returns a dict with 'standoff_deg', 'touch_deg' (both six-element lists in
    DEGREES, which is what the driver and the broker speak) plus the flange
    positions solved for. Raises ReachError with a reason rather than
    returning something unreachable or unsafe.
    """
    ik_mod = _load_ik()
    if guard is None:
        from mycobot_driver.collision_guard import CollisionGuard
        # The tool offset the guard models is the camera bracket, which is no
        # longer on the flange. Model the PRESSER's length instead, so the
        # guard protects the part that now sticks out furthest.
        guard = CollisionGuard(tool_offset_m=tool_length_m)

    import numpy as np

    tgt = np.array(target_xyz, dtype=float)
    if approach_dir is None:
        approach_dir = approach_direction(tgt)
    d = np.array(approach_dir, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        raise ReachError('approach_dir is a zero vector')
    d = d / n

    if standoff_m < 0:
        raise ReachError('standoff_m must not be negative -- a negative '
                         'standoff starts the approach INSIDE the panel')

    # Back off along the approach direction: by the tool's length so the TIP
    # lands on the target rather than the flange, then by the standoff.
    flange_touch = tgt - d * tool_length_m
    flange_pre = flange_touch - d * standoff_m

    seed = None
    if seed_deg is not None:
        seed = np.array([math.radians(a) for a in seed_deg], dtype=float)

    out = {}
    for name, xyz in (('standoff', flange_pre), ('touch', flange_touch)):
        q = ik_mod.ik(xyz, seed=seed)
        if q is None:
            raise ReachError(
                f'no IK solution for the {name} pose at '
                f'({xyz[0]*1000:.0f}, {xyz[1]*1000:.0f}, {xyz[2]*1000:.0f})mm. '
                'Out of reach, or blocked by a joint limit.')
        deg = [math.degrees(a) for a in q]
        ok, why = guard.check(deg)
        if not ok:
            raise ReachError(f'the {name} pose is unsafe: {why}')
        out[f'{name}_deg'] = deg
        out[f'{name}_xyz'] = [float(v) for v in xyz]
        # Solve the touch pose from the standoff solution, so the two are
        # adjacent in joint space. Without this the solver can return a
        # different elbow configuration for two points 40mm apart, and the arm
        # flips between them -- through the panel.
        seed = q

    # Verify that adjacency rather than trusting the seed to have produced it.
    jump = max(abs(a - b) for a, b in
               zip(out['standoff_deg'], out['touch_deg']))
    if jump > 30.0:
        raise ReachError(
            f'the standoff and touch poses are {jump:.0f}deg apart in joint '
            'space for a move of a few centimetres, which means the solver '
            'changed arm configuration between them. Executing that would '
            'swing the arm through the target.')
    out['max_joint_step_deg'] = jump
    out['approach_dir'] = [float(v) for v in d]
    return out
