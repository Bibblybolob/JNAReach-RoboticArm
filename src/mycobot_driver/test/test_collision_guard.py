#!/usr/bin/env python3
"""Pins the geometric collision guard.

Runs without ROS -- collision_guard.py imports only `math`, deliberately, so
this can be checked on any machine.

The guard decides whether a commanded pose is sent to the arm at all, so its
failure modes are asymmetric: refusing a safe pose is an annoyance, and
permitting an unsafe one puts the camera through the desk. The tests below
therefore check BOTH directions -- that home and the whole search sweep stay
legal, and that poses which genuinely reach the desk or fold into the base are
stopped.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'mycobot_driver'))

from collision_guard import CollisionGuard, joint_points  # noqa: E402

HOME = [0.0, 90.0, -149.0, 55.0, 0.0, 0.0]


def check(name, got, want):
    ok = got == want
    print(f'{"PASS" if ok else "FAIL"}  {name}')
    if not ok:
        print(f'        expected {want}, got {got}')
    return ok


def main() -> int:
    g = CollisionGuard()
    r = []

    # --- must not block normal operation ---
    r.append(check('home is allowed', g.check(HOME)[0], True))
    r.append(check('all-zeros is allowed', g.check([0] * 6)[0], True))

    # The search sweep tilts joint5 across its whole range. If any of that is
    # refused the arm cannot hunt at all, which is a worse failure than the
    # collision being guarded against.
    sweep_ok = all(g.check([0, 90, -149, 55, t, 0])[0]
                   for t in range(-90, 91, 10))
    r.append(check('every search-sweep tilt is allowed', sweep_ok, True))

    # And panning, which the servo does constantly.
    pan_ok = all(g.check([p, 90, -149, 55, 0, 0])[0]
                 for p in range(-160, 161, 20))
    r.append(check('every pan angle at home is allowed', pan_ok, True))

    # --- must block what actually collides ---
    ok, why = g.check([0, -90, -135, 0, 0, 0])
    r.append(check('a pose reaching into the desk is blocked', ok, False))
    r.append(check('  and says it is the desk', 'desk' in why, True))

    ok, why = g.check([0, 120, -150, 0, 0, 0])
    r.append(check('a pose folded into the base is blocked', ok, False))
    r.append(check('  and says it is the base', 'base' in why, True))

    # --- the base column must not be treated as a collision ---
    # The mount sits at the origin and the first link runs up through the
    # base, so both are inside every keep-out by construction. The first
    # version of this checked them and refused every pose including home.
    pts = joint_points(HOME)
    r.append(check('FK puts the base at the origin', pts[0], (0.0, 0.0, 0.0)))
    r.append(check('  which does not make home illegal', g.check(HOME)[0], True))

    # --- link sampling, not just joints ---
    # A link can pass through the floor with both ENDS above it. Sampling
    # along links is what catches that; joint-only checking does not.
    from collision_guard import _sampled
    sampled = _sampled(joint_points(HOME))
    r.append(check('sampling adds points between joints',
                   len(sampled) > len(joint_points(HOME)) - 1, True))

    # --- disabling is honest ---
    off = CollisionGuard(enabled=False)
    r.append(check('disabled guard permits a colliding pose',
                   off.check([0, -90, -135, 0, 0, 0])[0], True))

    # --- margins are signed and meaningful ---
    r.append(check('home has positive margin', g.worst_margin(HOME) > 0, True))
    r.append(check('a colliding pose has negative margin',
                   g.worst_margin([0, 120, -150, 0, 0, 0]) < 0, True))

    # --- a broken pose must fail closed ---
    ok, why = g.check([0, 0, 0])          # too few joints
    r.append(check('a malformed pose is refused, not permitted', ok, False))
    r.append(check('  and says kinematics failed', 'kinematics' in why, True))

    # --- the tool is guarded, not just the flange ---
    # The camera sticks out past the flange and is the part that hits things
    # first.
    with_tool = joint_points(HOME, tool_offset_m=0.055)
    without = joint_points(HOME, tool_offset_m=0.0)
    r.append(check('tool offset adds a point beyond the flange',
                   len(with_tool) == len(without) + 1, True))

    print()
    print(f'{sum(r)}/{len(r)} passed')
    return 0 if all(r) else 1


if __name__ == '__main__':
    sys.exit(main())
