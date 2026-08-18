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


# The tool axis in FLANGE coordinates, MEASURED 2026-08-15 and not assumed.
#
# The 63.5mm marker is bolted flat to the flange face, so its normal is the
# face normal. Detected over 8 poses and pushed into the flange frame through
# FK, it came out CONSTANT at (-0.157, -0.011, -0.988) -- 3.0deg mean spread,
# 5.7deg worst. Two things follow: the marker cannot move on the flange, so
# its constancy confirms FK's ROTATION (which hand-eye verification never
# tested -- that only ever checked position); and this vector is the direction
# the tool actually faces.
#
# It is close to -z but 9deg off it, and that 9deg is worth carrying rather
# than idealising away: it is the difference between the face meeting a button
# flat and meeting it on one edge.
#
# SIGN CORRECTED 2026-08-18, and the correction is the whole point. What was
# measured above is the normal of the MARKER, and the marker is bolted to the
# face carrying the Atom and its LED. The presser is mounted on the OPPOSITE
# side, so the tool faces the other way and this vector must be negated. The
# magnitude and the 9deg tilt are unaffected -- only which end is the tip.
#
# Observed, twice, before it was believed: the arm reached the button with the
# tool pointing back at the camera and the LED at the panel, i.e. 180deg out,
# while every printed figure read well -- "aim 19.3deg off normal", "tool tip
# 9.2mm from the button".
#
# Those figures could not have caught it, and neither could projecting the
# computed tip into the photo, which was tried: `plan_press` POSITIONS the
# flange so that flange + TOOL_AXIS * tool_length lands on the button, so the
# computed tip is on the button by construction whichever way the real tool
# points. That projection is circular and proves nothing. What does test it is
# a photograph of the wrist, or projecting the FLANGE ORIGIN, which is not
# derived from this vector.
TOOL_AXIS = (0.157, 0.011, 0.988)


def approach_rotation(approach_dir, seed_R=None, tool_axis=TOOL_AXIS):
    """Flange orientation that points the tool along `approach_dir`.

    Solves R @ tool_axis = d, so the measured face direction ends up on the
    direction of travel.

    That fixes two of the three rotational degrees of freedom. The third --
    spin about the approach axis -- does not matter for putting a flat face on
    a flat button, so it is spent staying close to `seed_R` rather than being
    pinned arbitrarily. Constraining it for no reason is how a reachable
    target becomes unreachable.
    """
    import numpy as np

    d = np.array(approach_dir, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        raise ReachError('approach_dir is a zero vector')
    d = d / n
    u = np.array(tool_axis, dtype=float)
    u = u / np.linalg.norm(u)

    # Rotation taking u onto d, about their common perpendicular. Any further
    # spin about d is free and is deliberately left unpinned -- constraining
    # it for no reason is how a reachable target becomes unreachable.
    v = np.cross(u, d)
    c = float(u @ d)
    sv = np.linalg.norm(v)
    if sv < 1e-9:
        R = np.eye(3) if c > 0 else -np.eye(3)
        if c < 0:                    # antiparallel: half turn about any perp
            a = np.array([1.0, 0.0, 0.0])
            if abs(float(a @ u)) > 0.9:
                a = np.array([0.0, 1.0, 0.0])
            a = a - u * float(a @ u)
            a /= np.linalg.norm(a)
            Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]],
                           [-a[1], a[0], 0]])
            R = np.eye(3) + 2 * Kx @ Kx
        return R
    Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + Kx + Kx @ Kx * ((1 - c) / (sv ** 2))


def panel_clearance(angles_deg, panel_point, panel_normal, ignore_m=0.09):
    """How far the ARM BODY stays off the panel, in metres. Negative = through.

    The missing constraint, and the one that produced the actual failure: the
    planner put the flange 0.09mm from the button while the FOREARM lay across
    the panel. Measured 2026-08-15 at a standoff that was 40mm clear by the
    flange -- the nearest arm point was 152mm from the flange origin and only
    **5mm** off the panel. Move in the remaining 40mm and the forearm goes
    through the wall, which is why contact happened with the wrong part of the
    arm on the wrong button.

    `obstacles.yaml` is empty, so CollisionGuard knows about the base column
    and the arm itself and nothing else. The panel is not in any model, so it
    has to be passed in.

    Points within `ignore_m` of the flange are skipped: that is the part MEANT
    to touch the panel, and including it would reject every press.

    `ignore_m` is 90mm because that is how far the flange head's own material
    reaches -- measured 2026-08-15, the arm's surface around the origin sits
    41-75mm from it. At 60mm the contact surface itself was being counted as a
    collision, so the planner held the whole arm 57mm off the panel and
    NOTHING touched: the flange origin landed on the button while the metal
    stopped short. That presented as "way too far from the button".

    It still catches what it is for. The forearm intrusion this was written
    against was 152mm from the flange origin, well outside 90mm.
    """
    from mycobot_driver.collision_guard import joint_points, _sampled
    import numpy as np

    pts = joint_points(list(angles_deg))
    flange = np.array(pts[6], dtype=float)
    n = np.array(panel_normal, dtype=float)
    n = n / np.linalg.norm(n)
    c = np.array(panel_point, dtype=float)

    worst = float('inf')
    for p in _sampled(pts):
        p = np.array(p, dtype=float)
        if np.linalg.norm(p - flange) < ignore_m:
            continue
        worst = min(worst, float(-(p - c) @ n))
    return worst


def _aim(ik_mod, xyz, R, seed, pos_tol_m=0.002, continuity=0.3,
         panel=None, min_clear_m=0.02):
    """Best orientation the arm can give WITHOUT missing the button.

    Exact full pose is not generally available on this arm. Measured
    2026-08-15 against the panel's inward normal: holding the tool exactly on
    the normal costs 25.7mm of position at the current placement, with no
    joint at a limit -- it is dexterity, not limits. Full pose only solves
    with the panel at 120-150mm and the button 150-200mm up.

    So the priority is fixed rather than fudged: HIT THE BUTTON, and aim as
    squarely as the arm allows subject to that. Sweeping the rotation weight
    down trades orientation for position, and the best solution still inside
    `pos_tol_m` wins. Returns (q, alignment, position_error_m).
    """
    import numpy as np

    q0 = ik_mod.ik(xyz, seed=seed)
    if q0 is None:
        return None, None, None
    lo = np.array([c[2] for c in ik_mod.CHAIN])
    hi = np.array([c[3] for c in ik_mod.CHAIN])
    # The tool direction this R puts in the world, i.e. R @ TOOL_AXIS.
    tgt_axis = R @ (np.array(TOOL_AXIS, dtype=float)
                    / np.linalg.norm(TOOL_AXIS))

    tool = np.array(TOOL_AXIS, dtype=float)
    tool = tool / np.linalg.norm(tool)
    T0 = ik_mod.fk(np.array(q0, dtype=float))
    a0 = float((T0[:3, :3] @ tool) @ tgt_axis)
    j0 = 0.0 if seed is None else float(
        np.max(np.abs(np.array(q0, dtype=float) - np.asarray(seed, float))))
    best = (np.array(q0, dtype=float), a0,
            float(np.linalg.norm(T0[:3, 3] - xyz)), a0 - continuity * j0)

    for w in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
        W = np.diag([1.0, 1.0, 1.0, w, w, w])
        q = np.array(q0, dtype=float)
        for _ in range(300):
            T = ik_mod.fk(q)
            e = W @ ik_mod.pose_error(T, xyz, R)
            J = W @ ik_mod.jacobian(q)
            JT = J.T
            q = q + JT @ np.linalg.solve(J @ JT + 0.0025 * np.eye(6), e)
            q = np.clip(q, lo, hi)
        T = ik_mod.fk(q)
        perr = float(np.linalg.norm(T[:3, 3] - xyz))
        align = float((T[:3, :3] @ tool) @ tgt_axis)
        if perr > pos_tol_m:
            continue
        # Prefer the squarest aim, but not at the cost of jumping arm
        # configuration between the standoff and the touch. Those two are
        # 40mm apart; if the solver answers them from different branches the
        # arm swings through the panel getting from one to the other, and the
        # adjacency check downstream rightly refuses the whole plan.
        jump = 0.0 if seed is None else float(
            np.max(np.abs(q - np.asarray(seed, dtype=float))))
        # Reject any configuration that lays the arm across the panel, and
        # among the rest prefer the ones that keep well clear. Without this
        # the solver is free to drape the forearm over the buttons, which is
        # exactly what it did.
        if panel is not None:
            clear = panel_clearance([math.degrees(v) for v in q], *panel)
            if clear < min_clear_m:
                continue
        score = align - continuity * jump
        if best[1] is None or score > best[3]:
            best = (q, align, perr, score)
    return best[0], best[1], best[2]


def plan_press(target_xyz, tool_length_m=0.0, standoff_m=0.04,
               approach_dir=None, seed_deg=None, guard=None,
               orientation='auto', panel=None):
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

    # Full-pose or position-only.
    #
    #   'auto'  try full pose, fall back to position-only and SAY so
    #   True    full pose or fail
    #   False   position only  <- THE DEFAULT, deliberately, see below
    #
    # 'auto' again as of 2026-08-15, now that TOOL_AXIS is measured rather
    # than assumed. It was briefly defaulted OFF, because aiming an axis that
    # might be wrong is worse than not aiming at all.
    #
    # 'auto' is the default because a full pose is what pressing wants and a
    # position is what the arm can always give. Silently doing either would be
    # worse than both: the whole reason the flange kept contacting edge-on was
    # that nothing said orientation was unconstrained.
    want_R = None
    if orientation is not False:
        want_R = approach_rotation(d)

    out = {}
    used_full = True
    for name, xyz in (('standoff', flange_pre), ('touch', flange_touch)):
        # Aim as squarely as the arm allows without missing the button.
        if want_R is not None:
            qa, align, perr = _aim(ik_mod, xyz, want_R, seed, panel=panel)
            if qa is not None:
                out[f'{name}_alignment'] = align
                out[f'{name}_pos_err_mm'] = perr * 1000.0
                deg = [math.degrees(a) for a in qa]
                ok, why = guard.check(deg)
                if ok:
                    out[f'{name}_deg'] = deg
                    out[f'{name}_xyz'] = [float(v) for v in xyz]
                    seed = qa
                    continue
                raise ReachError(f'the {name} pose is unsafe: {why}')

        # TWO STAGE, and this is the whole trick. The full-pose solver is
        # sound -- it recovers a known pose 20/20 from a nearby seed -- but
        # only 13/20 from the default HOME seed, and 0/12 for real button
        # targets, because HOME is nowhere near a pose that presses a wall
        # panel. Damped least squares is local; it needs to start in the right
        # basin. So solve POSITION first, which succeeds from almost anywhere,
        # and hand that to the full-pose solve as its seed.
        q = None
        if want_R is not None:
            q_pos = ik_mod.ik(xyz, seed=seed)
            if q_pos is not None:
                q = ik_mod.ik(xyz, target_R=want_R, seed=q_pos)
            if q is None and q_pos is not None:
                # Still stuck: the free spin about the approach axis is ours
                # to choose, so try other spins before giving up on aiming
                # the tool at all.
                import numpy as _np
                for spin in _np.linspace(0, 2 * math.pi, 8, endpoint=False)[1:]:
                    c, sn = math.cos(spin), math.sin(spin)
                    Rz = _np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
                    q = ik_mod.ik(xyz, target_R=want_R @ Rz, seed=q_pos)
                    if q is not None:
                        break
        if q is None and want_R is not None:
            if orientation is True:
                raise ReachError(
                    f'no full-pose IK solution for the {name} pose at '
                    f'({xyz[0]*1000:.0f}, {xyz[1]*1000:.0f}, '
                    f'{xyz[2]*1000:.0f})mm with the tool along '
                    f'({d[0]:.2f},{d[1]:.2f},{d[2]:.2f}). Demanding an '
                    'orientation can make a reachable position unreachable; '
                    "pass orientation='auto' to fall back.")
            used_full = False
            q = ik_mod.ik(xyz, seed=seed)
        elif q is None:
            q = ik_mod.ik(xyz, seed=seed)
        if q is None:
            raise ReachError(
                f'no IK solution for the {name} pose at '
                f'({xyz[0]*1000:.0f}, {xyz[1]*1000:.0f}, {xyz[2]*1000:.0f})mm. '
                'Out of reach, or blocked by a joint limit.')
        if panel is not None:
            clear = panel_clearance([math.degrees(v) for v in q], *panel)
            if clear < 0.02:
                raise ReachError(
                    f'the {name} pose puts the arm body {clear*1000:.0f}mm '
                    'from the panel -- it would press with the forearm, not '
                    'the flange. Move the panel closer to the base, or give '
                    'the arm a longer tool.')
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
    out['orientation_constrained'] = bool(want_R is not None)
    if 'touch_alignment' in out:
        a = out['touch_alignment']
        out['touch_off_normal_deg'] = math.degrees(
            math.acos(max(-1.0, min(1.0, a))))
    if want_R is not None and not used_full:
        out['orientation_note'] = (
            'full-pose IK failed and this fell back to POSITION ONLY, so the '
            'tool arrives pointing an arbitrary way -- expect edge-on contact')
    return out
