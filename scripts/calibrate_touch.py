#!/usr/bin/env python3
"""Solve camera -> base by TOUCHING points the camera can see.

Why not hand-eye
----------------
Every hand-eye method needs the camera to see a target that moves with the
arm. This camera is on the first arm piece and cannot see the flange at all,
so there is nowhere to put such a target. That rules out the whole family --
not because the maths is hard, but because the observation does not exist.

What works instead is to let the camera and the arm each measure the SAME
points in the world, independently:

    camera:  a board corner, located in CAMERA coordinates (pixel + depth)
    arm:     the tip touching that same corner, in BASE coordinates (FK)

Two sets of the same points in two frames determine the rigid transform
between the frames. That is camera -> base, which is the whole objective, and
nothing has to be mounted on the arm.

    ./scripts/calibrate_touch.py --collect --square-mm 38.9
    ./scripts/calibrate_touch.py --solve

Depth, not the board's printed size
-----------------------------------
Each corner's 3D position comes from the depth map, not from the board's
geometry. That is deliberate: on 2026-08-13 the board here measured 39mm
against the 30mm the tooling assumed -- a printer "fit to page" -- and the
colour-geometry path CANNOT notice, because it solves the pose from the size
it was told. Depth measures the distance itself. --square-mm is still needed
to find the corners, but no distance depends on it.

Geometry that will not solve
----------------------------
Three non-collinear points determine a rigid transform, but points in a LINE
determine nothing and points on a PLANE are weakly conditioned out of that
plane -- and a flat board lying still is exactly a plane. So: move the board
between batches, to two or three different heights and tilts, and touch a few
corners at each. The script measures the spread it actually got and refuses a
degenerate set rather than returning a confident wrong transform.

The error budget, and why SPREAD matters more than care
-------------------------------------------------------
The intuition that averaging many touches beats one careful touch is wrong
here, and the reason is worth understanding before collecting: the rotation
is fitted from the same points, and a rotation error acts through the ~250mm
lever arm from the camera to the points, landing back in the translation. So
the transform is only as good as the ANGLE the points subtend.

Simulated through this module's own solver, 2mm of touch error throughout,
300 trials per cell -- only the spread of the touched points and their number
change:

    point spread    5 touches   10 touches   20 touches
        40mm          23.2mm      12.6mm       9.2mm
        80mm          11.3mm       7.2mm       4.3mm
       160mm           5.8mm       3.6mm       2.4mm
       250mm           3.9mm       2.5mm       1.6mm

Read that as: touching five points carefully in one corner of the board is
worse -- by six times -- than touching five sloppy points spread across the
whole workspace. Spread first, count second, precision third.

So: use corners at opposite ends of the board, and move the board between
batches to two or three different heights and tilts. `--solve` reports the
spread it actually got and jackknifes the fit, so the answer comes with its
own error bar rather than an assurance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src', 'mycobot_driver'))

import cv2  # noqa: E402

from arm_broker import request  # noqa: E402
from mycobot_driver.collision_guard import (  # noqa: E402
    CollisionGuard, flange_transform,
)

OUT_DIR = os.path.expanduser('~/hand_eye')
TOUCHES = 'touches.json'
SQUARES_X, SQUARES_Y = 5, 7
MARKER_RATIO = 0.72

# The D405 measures nothing useful outside this band.
MIN_DEPTH_MM, MAX_DEPTH_MM = 70.0, 500.0


class ReachError(ValueError):
    """Input that must not be turned into a transform."""


def board_for(square_mm):
    d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard_create(
        SQUARES_X, SQUARES_Y, square_mm / 1000.0,
        square_mm / 1000.0 * MARKER_RATIO, d), d


class Camera:
    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.scale = (self.profile.get_device().first_depth_sensor()
                      .get_depth_scale() * 1000.0)
        i = (self.profile.get_stream(rs.stream.color)
             .as_video_stream_profile().get_intrinsics())
        self.K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])
        for _ in range(40):
            self.pipe.wait_for_frames(10000)

    def frame(self):
        f = self.align.process(self.pipe.wait_for_frames(10000))
        colour = np.asanyarray(f.get_color_frame().get_data()).copy()
        depth = (np.asanyarray(f.get_depth_frame().get_data())
                 .astype(float) * self.scale)
        return colour, depth

    def close(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


def corners_3d(colour, depth, board, adict, K):
    """{corner index: (pixel, xyz in camera metres)} straight from depth."""
    gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    marks, ids, _ = cv2.aruco.detectMarkers(gray, adict)
    if ids is None or len(ids) < 4:
        return {}
    n, cc, ci = cv2.aruco.interpolateCornersCharuco(marks, ids, gray, board)
    if n is None or n < 4:
        return {}
    out = {}
    for k, idx in enumerate(ci.ravel()):
        u, v = cc[k].ravel()
        iu, iv = int(round(u)), int(round(v))
        patch = depth[max(0, iv - 2):iv + 3, max(0, iu - 2):iu + 3]
        valid = patch[(patch > MIN_DEPTH_MM) & (patch < MAX_DEPTH_MM)]
        if valid.size < 5:
            continue
        z = float(np.median(valid))
        out[int(idx)] = ((float(u), float(v)),
                         np.array([(u - K[0, 2]) * z / K[0, 0],
                                   (v - K[1, 2]) * z / K[1, 1], z]) / 1000.0)
    return out


def annotate(colour, seen, target_idx, path):
    """Save a picture with the corner to touch ringed.

    Not decoration. "Touch corner 14" is unusable on a board of 24 identical
    intersections -- the operator has to be able to see WHICH one, or they
    will touch a neighbour and the residual will blame the maths.
    """
    img = colour.copy()
    for idx, (px, _) in seen.items():
        cv2.circle(img, (int(px[0]), int(px[1])), 3, (0, 160, 0), -1)
    px = seen[target_idx][0]
    c = (int(px[0]), int(px[1]))
    cv2.circle(img, c, 22, (0, 0, 255), 3)
    cv2.line(img, (c[0] - 34, c[1]), (c[0] - 24, c[1]), (0, 0, 255), 3)
    cv2.line(img, (c[0] + 24, c[1]), (c[0] + 34, c[1]), (0, 0, 255), 3)
    cv2.line(img, (c[0], c[1] - 34), (c[0], c[1] - 24), (0, 0, 255), 3)
    cv2.line(img, (c[0], c[1] + 24), (c[0], c[1] + 34), (0, 0, 255), 3)
    cv2.putText(img, f'TOUCH THIS ({target_idx})', (max(5, c[0] - 90),
                max(24, c[1] - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    cv2.imwrite(path, img)


JOG_HELP = """    CARTESIAN -- move the TIP, in millimetres, which is what you want here:
    x +10     tip 10mm along base +x       z -5      tip 5mm down
    y -20     tip 20mm along base -y       +10/-10   repeat the last axis

    JOINT -- when a wrist needs turning:
    2 +5      move joint2 by +5 deg        2 =90     drive joint2 TO 90 deg

    p  print the pose      ok  record this touch
    s  skip this corner    q   stop collecting
    step 3  default step   speed 20  move speed"""

_IK = None


def _ik():
    """The arm's real IK, from the script that owns it.

    Loaded rather than reimplemented: ik_demo.fk was verified identical to
    collision_guard.flange_transform to 4e-16 over 200 random poses on
    2026-08-13, and a third copy of this chain is the way that stops being
    true.
    """
    global _IK
    if _IK is None:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'ik_demo.py')
        spec = importlib.util.spec_from_file_location('_ik_demo', path)
        _IK = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_IK)
    return _IK


def cartesian_jog(target_deg, axis, delta_mm, tool_offset_m=0.0):
    """Move the TIP by delta_mm along a base axis. Returns joint degrees.

    Cartesian because the operator is looking at a corner and a camera image,
    not at a joint-space manifold. "Put the tip 10mm left" is answerable by
    eye; "which of six joints, and how far" is inverse kinematics done in the
    head, and doing it by hand is what made the first collection session
    unusable.

    Solved from the COMMANDED pose, not the measured one, for the same reason
    the joint jog is: seeding from the measurement re-commands untouched
    joints to wherever gravity left them.

    Returns None if the target is unreachable, or if IK answers with a
    different arm configuration -- a 200-degree elbow flip technically reaches
    the point, and would swing the arm through whatever it was about to touch.
    """
    import numpy as np

    ik = _ik()
    T = np.array(flange_transform(list(target_deg)))
    tool = T[:3, 2] * tool_offset_m
    want = T[:3, 3] + tool
    want = want.copy()
    want[axis] += delta_mm / 1000.0

    # Solve, then RE-solve against the orientation actually reached. The tool
    # points along the flange's own z, so it rotates as the wrist does, and
    # holding that vector fixed over the jog is only right for a tool of zero
    # length. Measured at a 60mm offset: one shot put the tip 11.9mm out for a
    # 10mm request, 19% wrong. Two refinements close it.
    seed = np.array([math.radians(a) for a in target_deg])
    q = None
    for _ in range(3):
        guess_tool = tool if q is None else (
            np.array(flange_transform([math.degrees(a) for a in q]))[:3, 2]
            * tool_offset_m)
        q_new = ik.ik(want - guess_tool, seed=seed)
        if q_new is None:
            return None
        q = q_new
        if tool_offset_m == 0.0:
            break

    nxt = [math.degrees(a) for a in q]
    if max(abs(a - b) for a, b in zip(nxt, target_deg)) > 45.0:
        return None

    # Verify where the tip ACTUALLY lands rather than trusting the solve.
    # The refinement above converges cleanly at zero tool length and less well
    # as the tool grows, because position-only IK leaves the orientation free
    # and the tool swings with it -- measured 14mm out on a 25mm request at a
    # 100mm offset. A jog that quietly goes somewhere else is worse than one
    # that refuses, so this is checked, not assumed.
    T2 = np.array(flange_transform(nxt))
    got = (T2[:3, 3] + T2[:3, 2] * tool_offset_m)[axis] - (
        T[:3, 3] + tool)[axis]
    if abs(got - delta_mm / 1000.0) > max(0.001, abs(delta_mm) * 0.05 / 1000):
        return None
    return nxt


def wait_for_arrival(target, travel_deg, speed, tol=1.0):
    """Block until the arm reaches `target`, or until it clearly will not.

    Replaces a slept guess, which was wrong in the expensive direction. An
    ABSOLUTE jog can be 200 degrees of travel; sleeping a fixed amount sized
    for a small step returns while the arm is still moving, reports an
    enormous divergence, and invites the operator to re-issue a command that
    was already executing -- which is exactly the mess this produced in use.

    Polls the measured pose instead, so a short move costs a short wait and a
    long one is actually waited out. Returns (arrived, worst_error_deg).

    The tolerance is 1.0deg, just above this arm's floor. Measured
    2026-08-13: commanded a two-joint move and watched it settle -- 0.79deg
    and 1.60mm of tip error at 4.5s, then completely flat out to 21s. That
    0.79 is the backlash CLAUDE.md records, so it is the hardware limit and
    waiting longer buys nothing. The tolerance was 2.0deg, which accepted two
    and a half times the achievable error, i.e. about 9mm of tip error at
    250mm of reach -- and that, not the IK, was what made jogging feel
    inaccurate. The IK itself round-trips to under 0.1mm.
    """
    # Measured 52 deg/s at speed 100, so scale from that and leave headroom
    # for acceleration and the link being slow to answer.
    budget = travel_deg / 52.0 * (100.0 / max(speed, 1)) * 2.5 + 3.0
    deadline = time.monotonic() + min(budget, 30.0)
    worst = float('inf')
    while time.monotonic() < deadline:
        st = request({'cmd': 'state'})
        if st and st.get('angles') and st.get('age_ms', 1e9) < 1500:
            worst = max(abs(a - b) for a, b in zip(st['angles'], target))
            if worst <= tol:
                return True, worst
        time.sleep(0.25)
    return False, worst


def jog_to_corner(guard, speed, step, last_joint, tool_offset_m=0.0):
    """Position the arm by COMMANDED moves, never by hand.

    Measured 2026-08-13, and it is the reason this mode exists: twelve
    commanded moves out to near the guard's maximum load produced ZERO Atom
    reboots, while moving the arm by hand reliably makes it panic and reset.
    Commanded motion drives the load; a hand back-drives the motors, and the
    arm being limp is a precondition for neither.

    So the servos stay engaged for the whole session -- no release, no
    re-engage, and none of the free-drive cycle that preceded every link
    collapse today.

    Returns (action, last_joint) where action is 'ok', 'skip' or 'quit'.
    """
    # Seeded ONCE from the measurement, then tracked as a commanded target.
    # Re-seeding from the measured pose every jog is the sag ratchet: an
    # untouched joint gets re-commanded to wherever gravity left it.
    st = request({'cmd': 'state'})
    if not st or not st.get('angles'):
        print('    no pose reading; cannot jog safely')
        return 'skip', last_joint
    target = list(st['angles'])
    last_axis = None
    print(JOG_HELP)

    while True:
        st = request({'cmd': 'state'}) or {}
        meas = st.get('angles')
        if meas:
            drift = max(abs(a - b) for a, b in zip(meas, target))
            # ~0.8deg is this arm's backlash and is normal; flagging that
            # would be crying wolf. Past ~2.5deg a joint genuinely did not
            # execute -- wait_for_arrival has already waited, so it is not
            # still travelling.
            note = (f'  <-- {drift:.1f}deg from commanded; a joint did not '
                    f'execute (try power_on)' if drift > 2.5
                    else f'  ({drift:.1f}deg out, backlash is ~0.8)')
            print(f'    at {[round(a, 1) for a in meas]}{note}')
        raw = input(f'    jog [step {step}, speed {speed}]> ').strip().lower()
        if raw in ('ok', ''):
            return 'ok', last_joint
        if raw == 's':
            return 'skip', last_joint
        if raw == 'q':
            return 'quit', last_joint
        if raw == 'p':
            continue
        if raw.startswith('step'):
            try:
                step = float(raw.split()[1])
            except Exception:
                print('    usage: step 3')
            continue
        if raw.startswith('speed'):
            try:
                speed = int(raw.split()[1])
            except Exception:
                print('    usage: speed 20')
            continue

        # x|y|z <delta_mm> | <joint> <delta> | <joint> =<abs> | <delta>
        parts = raw.split()
        nxt = None
        # j stays None on the Cartesian path, which moves several joints at
        # once and so has no "last joint" to repeat on. Initialised here
        # because the branches below do not all set it, and reading it
        # afterwards is how this crashed the first time.
        j = None
        try:
            if parts[0] in ('x', 'y', 'z') and len(parts) == 2:
                axis, mm = 'xyz'.index(parts[0]), float(parts[1])
                nxt = cartesian_jog(target, axis, mm, tool_offset_m)
                if nxt is None:
                    print(f'    cannot move the tip {mm:+.0f}mm in '
                          f'{parts[0]} -- out of reach, or IK answers with a '
                          'different arm configuration, which would swing it '
                          'through the target')
                    continue
                last_axis, last_joint = axis, None
            elif (len(parts) == 1 and parts[0][0] in '+-'
                  and last_axis is not None and last_joint is None):
                # Bare +N after a Cartesian move repeats on that axis, which
                # is the common case: nudge, look, nudge again.
                mm = float(parts[0])
                nxt = cartesian_jog(target, last_axis, mm, tool_offset_m)
                if nxt is None:
                    print(f'    cannot move {mm:+.0f}mm further that way')
                    continue
            elif len(parts) == 1 and parts[0][0] in '+-':
                if last_joint is None:
                    print('    no previous axis or joint -- try "x +10" or '
                          '"2 +5" first')
                    continue
                j, val, absolute = last_joint, float(parts[0]), False
            elif len(parts) == 2:
                j = int(parts[0])
                absolute = parts[1].startswith('=')
                val = float(parts[1].lstrip('='))
                if not 1 <= j <= 6:
                    raise ValueError
                last_axis = None
            else:
                raise ValueError
        except Exception:
            print('    did not understand that\n' + JOG_HELP)
            continue

        if nxt is None:
            nxt = list(target)
            nxt[j - 1] = val if absolute else nxt[j - 1] + val
        ok, why = guard.check(nxt)
        if not ok:
            # Refuse rather than clip: a silently shortened jog leaves the
            # operator believing the arm is somewhere it is not.
            print(f'    refused, that pose is unsafe: {why}')
            continue
        r = request({'cmd': 'send_angles', 'angles': nxt, 'speed': speed,
                     'force': True}, timeout=30)
        if not r or not r.get('ok'):
            print(f'    refused: {(r or {}).get("error", "no reply")}')
            continue
        travel = max(abs(a - b) for a, b in zip(nxt, target))
        target = nxt
        if j is not None:
            last_joint = j
        wait_for_arrival(target, travel, speed)


def pose_after(t_settled, timeout=25.0):
    """Joint angles genuinely SAMPLED after the arm stopped moving.

    Not "recent enough". A reading that predates the move pairs this touch's
    POSITION with the previous touch's ANGLES, and the solve then fits a
    consistent-looking problem with one input systematically wrong.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = request({'cmd': 'state'})
        if s and s.get('angles') and s.get('age_ms') is not None:
            if time.monotonic() - s['age_ms'] / 1000.0 >= t_settled:
                return s['angles']
        time.sleep(0.3)
    return None


def tip_in_base(angles_deg, tool_offset_m):
    """Where the touching point is, in the base frame.

    The flange is not the contact point. Anything past it -- even the
    thickness of the flange face itself -- shifts the touch by that much along
    the flange's own z, and an unmodelled offset goes straight into the
    answer.
    """
    T = np.array(flange_transform(list(angles_deg)))
    return T[:3, 3] + T[:3, 2] * tool_offset_m


def kabsch(P, Q):
    """Rigid transform taking points P onto points Q. Returns (R, t).

    No scaling term on purpose. Both sides are already metric -- depth in
    millimetres and the arm's own FK -- so a fitted scale would not be a
    degree of freedom, it would be an error absorber, hiding exactly the kind
    of defect that the 39mm board turned out to be.
    """
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def geometry_note(P):
    """(ok, message) for whether these points can determine a transform."""
    if len(P) < 3:
        return False, f'{len(P)} points; 3 non-collinear is the minimum'
    sv = np.linalg.svd(P - P.mean(0))[1]
    if sv[1] < 1e-3:
        return False, ('the touched points are collinear (singular values '
                       f'{sv[0]:.3f}/{sv[1]:.4f}/{sv[2]:.4f}). A line fixes '
                       'no rotation about itself -- spread them out.')
    if sv[2] < 0.01:
        return True, ('WARNING: the points are nearly coplanar (out-of-plane '
                      f'spread {sv[2]*1000:.1f}mm). The transform is '
                      'determined but weakly conditioned perpendicular to '
                      'that plane. Move the board to another height or tilt '
                      'and touch a few more.')
    return True, f'point spread {sv[0]*1000:.0f}/{sv[1]*1000:.0f}/{sv[2]*1000:.0f}mm'


def finish_with_touch(args, guard, R, t, sweep, shared, base_pose) -> int:
    """One touch, to settle the yaw and height the sweep cannot see."""
    import numpy as np

    # Pick the corner furthest from the base axis: the yaw is recovered from
    # an azimuth, and an azimuth is least sensitive to a sloppy touch when the
    # point is far out.
    est = {j: estimate_corner(sweep, R, t, j) for j in shared}
    idx = max(est, key=lambda j: math.hypot(*est[j][:2]))

    print(f'\nOne touch left. Everything else is already solved.')
    print(f'  put the tip on corner {idx}, then "ok".')
    print(f'  (it is the one the sweep places furthest from the base axis, '
          f'which makes the answer least sensitive to a sloppy touch)\n')

    request({'cmd': 'call', 'method': 'power_on'}, timeout=20)
    ans, _ = jog_to_corner(guard, args.speed, args.step, None,
                           args.tool_offset_mm / 1000.0)
    if ans != 'ok':
        print('stopped without a touch, so the yaw and height stay unknown; '
              'nothing written.')
        return 1

    settled = time.monotonic()
    angles = pose_after(settled, timeout=args.timeout)
    if angles is None:
        print('no joint reading after the touch; nothing written.')
        return 1
    touched = tip_in_base(angles, args.tool_offset_mm / 1000.0)

    phi, dz, radial = fix_gauge(est[idx], touched)
    print(f'\n  yaw {math.degrees(phi):+.1f}deg, height {dz*1000:+.0f}mm')
    print(f'  consistency: the touch and the sweep put that corner '
          f'{radial:.1f}mm apart from the base axis')
    if radial > args.max_residual_mm:
        print(f'  Refusing: that should agree to within a touch\'s accuracy. '
              f'Rotation about z cannot change distance from the axis, so a '
              f'{radial:.0f}mm disagreement means the touch was not on that '
              'corner, or the board moved during the sweep.')
        return 1

    Rz = rot_z(phi)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = Rz @ R, Rz @ t + np.array([0.0, 0.0, dz])

    out = os.path.join(OUT_DIR, 'eye_to_hand.json')
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({
        'camera_to_base': X.tolist(),
        # The sweep is expressed relative to the joint1 angle it started from,
        # so that is the angle this transform is valid at.
        'joint1_deg': float(base_pose[0]),
        'method': 'joint1 sweep + 1 touch (gauge fixed)',
        'sweep_angles': len(sweep),
        'shared_corners': len(shared),
        'touched_corner': int(idx),
        'radial_consistency_mm': radial,
        'mount': 'first arm piece (link1); only joint1 moves the camera',
        'compose_note': ('base->camera at pan q1 = Rz(q1 - joint1_deg) @ '
                         'camera_to_base. See docs/camera_mount.md.'),
    }, open(out, 'w'), indent=2)
    print(f'\nwrote {out}')
    print(f'camera at ({X[0,3]*1000:+.0f}, {X[1,3]*1000:+.0f}, '
          f'{X[2,3]*1000:+.0f})mm in the base frame')
    return 0


def rodrigues(v):
    import numpy as np
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def rot_z(theta):
    import numpy as np
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def solve_from_sweep(sweep):
    """camera->base from joint1 rotation alone, up to two gauge freedoms.

    `sweep` is {joint1_radians: {corner index: xyz in camera metres}}.

    The idea, and why it is worth the machinery: joint1 is the ONLY joint that
    moves this camera, and it moves it in a way the arm already knows exactly.
    So rotating joint1 and watching a stationary board says a great deal about
    where the camera sits -- with no touching, no marker, and nothing that
    needs the camera to see the arm.

    A board corner is at a fixed unknown point p in the base frame, so for
    every angle it must satisfy  Rz(theta) (R c + t) = p. Fitting R and t is
    therefore just asking that all the angles AGREE about where each corner
    is; the corner's actual position never has to be known.

    What it cannot see, measured from the rank of the residual Jacobian rather
    than assumed: 4 of 6 degrees of freedom are determined, and the two that
    are not are **yaw about the base z axis** and **translation along it** --
    which is exactly right, since spinning about an axis tells you nothing
    about your angle around it or your height along it. Those are left for one
    touched point to fix; see fix_gauge().
    """
    import numpy as np
    from scipy.optimize import least_squares

    angles = sorted(sweep)
    # A corner needs to be seen at SEVERAL angles, not at all of them. The
    # camera turns with joint1, so the board leaves the frame at the ends of
    # the sweep -- measured here, 14 corners at centre, 6 at -30deg, none at
    # +30. Demanding every angle threw away a whole usable sweep for want of
    # the two corners that survived the extremes, when each corner only has
    # to constrain its own agreement.
    at_least = max(3, min(3, len(angles)))
    seen_at = {}
    for a in angles:
        for j in sweep[a]:
            seen_at.setdefault(j, []).append(a)
    shared = sorted(j for j, aa in seen_at.items() if len(aa) >= at_least)
    if len(shared) < 4:
        raise ReachError(
            f'only {len(shared)} corners were seen at {at_least}+ sweep '
            f'angles; need at least 4. Narrow --sweep so the board stays in '
            'view, or move it back so more of it fits the frame.')

    def residual(x):
        R, t = rodrigues(x[:3]), x[3:6]
        out = []
        for j in shared:
            aa = seen_at[j]
            pts = [rot_z(a) @ (R @ sweep[a][j] + t) for a in aa]
            mean = np.mean(pts, axis=0)
            for p in pts:
                out.extend(p - mean)
        return np.array(out)

    sol = least_squares(residual, np.zeros(6), method='lm',
                        xtol=1e-14, ftol=1e-14)
    R, t = rodrigues(sol.x[:3]), sol.x[3:6]
    spread = float(np.abs(residual(sol.x)).max()) * 1000
    return R, t, shared, spread


def estimate_corner(sweep, R, t, idx):
    """Where a corner lands in the (gauge-free) base frame, averaged."""
    import numpy as np
    pts = [rot_z(a) @ (R @ sweep[a][idx] + t) for a in sorted(sweep)
           if idx in sweep[a]]
    return np.mean(pts, axis=0)


def fix_gauge(est, touched):
    """Resolve the two unobservable freedoms from one touched point.

    The sweep leaves yaw about base z and height along it undetermined, so a
    single point whose base position is known settles both -- three equations
    for two unknowns, and the leftover one is a free consistency check worth
    reporting rather than discarding.
    """
    import numpy as np
    phi = math.atan2(touched[1], touched[0]) - math.atan2(est[1], est[0])
    dz = float(touched[2] - est[2])
    # The check: rotation about z preserves distance from the axis, so if the
    # touch and the estimate disagree about that, they are not the same point
    # -- a mis-touched corner, or a board that moved during the sweep.
    radial = abs(math.hypot(*touched[:2]) - math.hypot(*est[:2])) * 1000
    return phi, dz, radial


def auto(args) -> int:
    """Sweep joint1, solve, then ask for ONE touch. See solve_from_sweep()."""
    import numpy as np

    h = request({'cmd': 'health'}, timeout=15)
    if h is None:
        print('No broker running. Start ./scripts/arm_broker.py first.')
        return 1
    frac = h.get('link', {}).get('fraction', 0)
    print(f'link {frac:.0%} valid')
    if frac < 0.6:
        print('Refusing below 60%: every sweep angle needs a joint reading '
              'taken after the arm stopped.')
        return 1

    st = request({'cmd': 'state'})
    if not st or not st.get('angles'):
        print('no pose reading')
        return 1
    base_pose = list(st['angles'])
    guard = CollisionGuard(tool_offset_m=args.tool_offset_mm / 1000.0)

    offsets = [float(v) for v in args.sweep.split(',')]
    print(f'sweeping joint1 through {offsets} degrees about '
          f'{base_pose[0]:.1f}, capturing the board at each\n')

    board, adict = board_for(args.square_mm)
    cam = Camera()
    sweep = {}
    try:
        # Look BEFORE moving. The sweep is centred on wherever joint1 happens
        # to be, and after an earlier sweep that is wherever the last angle
        # left it -- measured here, a run started at joint1=30deg and swept
        # 14..46deg, where the board is not visible at all, wasting five moves
        # to discover it. One frame first answers that for free.
        colour, depth = cam.frame()
        here = corners_3d(colour, depth, board, adict, cam.K)
        print(f'  before moving: {len(here)} corners visible at joint1='
              f'{base_pose[0]:.1f}deg')
        if len(here) < 6:
            print('\nRefusing to sweep: the board is barely in view from the '
                  'starting pose, so rotating away from it will only make '
                  'that worse. Point the camera at the board first -- home '
                  'the arm, or jog joint1 until it is centred -- then re-run.')
            return 1

        for off in offsets:
            pose = list(base_pose)
            pose[0] = base_pose[0] + off
            ok, why = guard.check(pose)
            if not ok:
                print(f'  joint1{off:+.0f}: skipped, unsafe -- {why}')
                continue
            request({'cmd': 'call', 'method': 'power_on'}, timeout=20)
            r = request({'cmd': 'send_angles', 'angles': pose,
                         'speed': args.speed, 'force': True}, timeout=30)
            if not r or not r.get('ok'):
                print(f'  joint1{off:+.0f}: refused')
                continue
            arrived, err = wait_for_arrival(pose, abs(off), args.speed)
            meas = request({'cmd': 'state'})
            if not arrived or not meas or not meas.get('angles'):
                print(f'  joint1{off:+.0f}: did not arrive ({err:.0f}deg out)')
                continue
            # The MEASURED angle, not the commanded one. The arm settles about
            # a degree short, and at 250mm that degree is 4mm of error fed
            # straight into the solve.
            theta = math.radians(meas['angles'][0] - base_pose[0])
            time.sleep(0.4)
            colour, depth = cam.frame()
            seen = corners_3d(colour, depth, board, adict, cam.K)
            if len(seen) < 4:
                print(f'  joint1{off:+.0f}: only {len(seen)} corners visible')
                continue
            sweep[theta] = {k: v[1] for k, v in seen.items()}
            print(f'  joint1{off:+.0f}: measured {math.degrees(theta):+.1f}deg,'
                  f' {len(seen)} corners')
    finally:
        cam.close()

    if len(sweep) < 3:
        print(f'\nonly {len(sweep)} usable sweep angles; need at least 3.')
        return 1

    try:
        R, t, shared, spread = solve_from_sweep(sweep)
    except ReachError as e:
        print(f'\n{e}')
        return 1
    print(f'\nsolved from {len(sweep)} angles and {len(shared)} shared '
          f'corners')
    print(f'  angles agree on each corner to {spread:.1f}mm')
    if spread > 8.0:
        print('  That is poor. The board moved during the sweep, or the depth '
              'is noisy at this range. Both invalidate the solve.')
        return 1
    return finish_with_touch(args, guard, R, t, sweep, shared, base_pose)


def collect(args) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    h = request({'cmd': 'health'}, timeout=15)
    if h is None:
        print('No broker running. Start ./scripts/arm_broker.py first.')
        return 1
    frac = h.get('link', {}).get('fraction', 0)
    print(f'link {frac:.0%} valid')
    if frac < 0.6:
        # Calibration is exactly the task that must not run on bad readings:
        # a stale pose pairs a touch with the wrong angles, silently.
        print('Refusing below 60%. Every touch depends on a joint reading '
              'taken after the arm stopped, and this link cannot supply them.')
        return 1

    # Every jog is checked against this before it is sent. The tool offset is
    # whatever actually contacts the board, so the guard protects the part
    # that sticks out furthest rather than the flange origin.
    guard = CollisionGuard(tool_offset_m=args.tool_offset_mm / 1000.0)

    board, adict = board_for(args.square_mm)
    cam = Camera()
    print(f'camera intrinsics fx={cam.K[0,0]:.1f} fy={cam.K[1,1]:.1f}')

    records = []
    last_joint = None
    path = os.path.join(OUT_DIR, TOUCHES)
    if os.path.isfile(path) and not args.restart:
        records = json.load(open(path))
        print(f'continuing from {len(records)} touches already recorded '
              '(--restart to discard them)')

    try:
        while len(records) < args.points:
            colour, depth = cam.frame()
            seen = corners_3d(colour, depth, board, adict, cam.K)
            if len(seen) < 3:
                # Usually the ARM, not the board: it is parked on the last
                # corner it touched, in front of the very corners this needs
                # to find. Say that first, since it is the common case and the
                # fix is different from moving the board.
                print(f'only {len(seen)} corners visible with valid depth.')
                print('  Most likely the arm is still over the board from the '
                      'last touch and is blocking the view -- it gets '
                      'retracted automatically, so if this persists the board '
                      'itself has moved or drifted outside 70-500mm.')
                input('  clear the view, then press Enter to retry: ')
                continue

            # Prefer corners far from the ones already used, so the point set
            # spreads rather than clustering in one region of the board.
            used = [np.array(r['camera_xyz']) for r in records]
            def spread(i):
                p = seen[i][1]
                return min((np.linalg.norm(p - u) for u in used), default=1e9)
            idx = max(seen, key=spread)

            img_path = os.path.join(OUT_DIR, f'touch_{len(records):02d}.png')
            annotate(colour, seen, idx, img_path)
            cam_xyz = seen[idx][1]

            st = request({'cmd': 'state'})
            # Retry rather than give up on one stale read. This is only a
            # place to RETURN to, so a couple of seconds of staleness is
            # harmless -- whereas silently having nowhere to go leaves the arm
            # parked on the board with the next capture blaming the board.
            retract_to = None
            for _ in range(4):
                st = request({'cmd': 'state'})
                if st and st.get('angles') and st.get('age_ms', 1e9) < 8000:
                    retract_to = list(st['angles'])
                    break
                time.sleep(1.0)

            print(f'\n--- touch {len(records)+1} of {args.points}')
            print(f'    target: corner {idx}, {cam_xyz[2]*1000:.0f}mm from '
                  f'the camera')
            print(f'    picture: {img_path}  (the ringed corner)')

            if args.free_drive:
                print('    releasing the servos -- SUPPORT THE ARM, it will '
                      'go limp')
                print('    NOTE: hand-moving this arm reliably reboots the '
                      'Atom. --jog drives it by command instead, which '
                      'measured zero reboots.')
                request({'cmd': 'call', 'method': 'release_all_servos'},
                        timeout=15)
                ans = input('    put the tip on that corner, then Enter '
                            '(s to skip, q to stop): ').strip().lower()
                request({'cmd': 'call', 'method': 'focus_all_servos'},
                        timeout=15)
                time.sleep(1.0)
                ans = {'q': 'quit', 's': 'skip'}.get(ans, 'ok')
            else:
                # Servos stay engaged throughout. See jog_to_corner().
                request({'cmd': 'call', 'method': 'power_on'}, timeout=20)
                ans, last_joint = jog_to_corner(
                    guard, args.speed, args.step, last_joint,
                    args.tool_offset_mm / 1000.0)
            if ans == 'quit':
                break
            if ans == 'skip':
                continue

            settled = time.monotonic()
            angles = pose_after(settled, timeout=args.timeout)
            if angles is None:
                print('    no joint reading arrived after the touch; skipping '
                      'rather than recording a stale one')
                continue
            base_xyz = tip_in_base(angles, args.tool_offset_mm / 1000.0)
            records.append({'corner': int(idx),
                            'camera_xyz': [float(v) for v in cam_xyz],
                            'base_xyz': [float(v) for v in base_xyz],
                            'angles': list(angles),
                            'image': img_path})
            json.dump(records, open(path, 'w'), indent=2)
            print(f'    recorded: camera ({cam_xyz[0]*1000:+.0f}, '
                  f'{cam_xyz[1]*1000:+.0f}, {cam_xyz[2]*1000:+.0f})mm  '
                  f'base ({base_xyz[0]*1000:+.0f}, {base_xyz[1]*1000:+.0f}, '
                  f'{base_xyz[2]*1000:+.0f})mm')

            # Get out of the camera's way before the next capture. The arm
            # ends each touch resting ON the board, directly in front of the
            # corners the next iteration has to find -- so without this the
            # detection degrades touch by touch as the arm works its way
            # across the board, and the failure looks like the board drifting.
            #
            # Commanded, never by hand: that is what makes it free. Twelve
            # commanded moves at near-maximum load produced zero Atom reboots.
            # Silence here is the worst outcome: the arm stays on the board,
            # the next capture sees nothing, and the message blames the board.
            # So every branch says what happened.
            if retract_to is None:
                print('    NOT retracting: no fresh pose was recorded before '
                      'jogging, so there is nowhere known-good to return to. '
                      'Move the arm clear by hand-free jogging if the next '
                      'capture finds no corners.')
            else:
                ok, why = guard.check(retract_to)
                if not ok:
                    print(f'    NOT retracting -- that pose is unsafe: {why}')
                else:
                    travel = max(abs(a - b)
                                 for a, b in zip(retract_to, angles))
                    print(f'    retracting {travel:.0f}deg to clear the view')
                    request({'cmd': 'call', 'method': 'power_on'}, timeout=20)
                    r = request({'cmd': 'send_angles',
                                 'angles': [float(a) for a in retract_to],
                                 'speed': args.speed, 'force': True},
                                timeout=30)
                    if not r or not r.get('ok'):
                        print(f'    retract refused: '
                              f'{(r or {}).get("error", "no reply")}')
                    else:
                        arrived, err = wait_for_arrival(
                            retract_to, travel, args.speed)
                        if not arrived:
                            print(f'    retract incomplete, {err:.0f}deg '
                                  'short -- the view may still be blocked')
    finally:
        cam.close()

    print(f'\n{len(records)} touches in {path}')
    if len(records) >= 3:
        ok, msg = geometry_note(np.array([r['base_xyz'] for r in records]))
        print(msg)
        print('\nNow: ./scripts/calibrate_touch.py --solve')
    return 0


def solve(args) -> int:
    path = os.path.join(OUT_DIR, TOUCHES)
    if not os.path.isfile(path):
        print(f'no touches at {path} -- run --collect first')
        return 1
    recs = json.load(open(path))
    P = np.array([r['camera_xyz'] for r in recs])   # camera frame
    Q = np.array([r['base_xyz'] for r in recs])     # base frame
    print(f'{len(recs)} touches')

    ok, msg = geometry_note(Q)
    print(msg)
    if not ok:
        return 1

    R, t = kabsch(P, Q)
    resid = np.linalg.norm((P @ R.T + t) - Q, axis=1) * 1000.0

    print('\n  touch   corner   residual')
    for i, r in enumerate(resid):
        flag = '   <-- worst' if i == int(np.argmax(resid)) else ''
        print(f'  {i:5d}   {recs[i]["corner"]:6d}   {r:6.1f}mm{flag}')
    print(f'\nrms {resid.std():.1f}mm, mean {resid.mean():.1f}mm, '
          f'worst {resid.max():.1f}mm')

    # One bad touch is a bad touch, not a bad fit. Say so, because the fix is
    # to redo that point rather than to distrust the whole calibration.
    if len(resid) > 3 and resid.max() > 3 * max(np.median(resid), 1.0):
        i = int(np.argmax(resid))
        print(f'\nTouch {i} is a clear outlier against the rest. That is a '
              f'misplaced touch, not a bad solve -- see {recs[i]["image"]}, '
              'delete that entry from touches.json and redo it.')

    if resid.mean() > args.max_residual_mm:
        print(f'\nRefusing: mean residual {resid.mean():.1f}mm is above the '
              f'{args.max_residual_mm:.0f}mm bar. Both measurements claim '
              'better than that, so the disagreement is real -- misplaced '
              'touches, a board that moved between capture and touch, or a '
              'wrong --tool-offset-mm.')
        return 1

    # Is the contact point really the flange ORIGIN? "The flange is flat and
    # nothing is mounted" says the offset is zero, but that is a claim about
    # where the URDF puts the flange frame, not about the hardware -- a frame
    # sitting a few mm inside the wrist makes zero wrong with nothing visibly
    # amiss.
    #
    # The data can answer it. A constant offset in the FLANGE frame maps to a
    # different base-frame displacement at every touch orientation, so it
    # cannot be absorbed into a single rigid transform: it shows up as
    # residual, and the offset that minimises residual is an estimate of the
    # real one. Diagnostic only -- it never silently changes the answer.
    if all('angles' in r for r in recs) and len(recs) >= 5:
        offsets = np.linspace(-0.03, 0.03, 121)
        curve = []
        for off in offsets:
            tips = np.array([tip_in_base(r['angles'], off) for r in recs])
            Ro, to = kabsch(P, tips)
            curve.append(np.linalg.norm((P @ Ro.T + to) - tips, axis=1).mean())
        curve = np.array(curve)
        best = float(offsets[int(np.argmin(curve))]) * 1000.0
        at_zero = float(curve[int(np.argmin(np.abs(offsets)))]) * 1000.0
        improvement = at_zero - float(curve.min()) * 1000.0
        print(f'\ntool offset implied by the data: {best:+.1f}mm '
              f'(assumed {args.tool_offset_mm:+.1f}mm)')
        if abs(best - args.tool_offset_mm) > 3.0 and improvement > 0.5:
            print(f'  Using it would cut the mean residual by '
                  f'{improvement:.1f}mm. That is the gap between the flange '
                  'FRAME and whatever actually touched the board. Re-run '
                  f'--solve with --tool-offset-mm {best:.1f} if it is real; '
                  'a systematic offset biases every point the arm is later '
                  'sent to.')
        else:
            print('  consistent with what was assumed -- no evidence of an '
                  'unmodelled offset')

    # Jackknife: refit with each point left out and see how far the answer
    # moves. Residuals say how well the fit describes the points it was given;
    # this says how much the ANSWER depends on which points those were, which
    # is the number to quote. A small residual with a large jackknife spread
    # is a set that is too clustered to determine the transform -- and that is
    # the failure this method is prone to, since rotation error acts through
    # the ~250mm lever arm to the camera and lands back in the translation.
    jack = []
    if len(recs) >= 4:
        for i in range(len(recs)):
            m = np.ones(len(recs), dtype=bool)
            m[i] = False
            Ri, ti = kabsch(P[m], Q[m])
            jack.append(ti)
        jack = np.array(jack)
        uncertainty = float(np.linalg.norm(jack.std(axis=0)) * 1000
                            * math.sqrt(len(recs) - 1))
        print(f'jackknife uncertainty on the camera position: '
              f'+/-{uncertainty:.1f}mm')
        spread = float(np.linalg.svd(Q - Q.mean(0))[1][0] * 1000)
        print(f'point spread {spread:.0f}mm across {len(recs)} touches')
        if uncertainty > 5.0:
            print('  That is loose. Spread matters more than care here -- '
                  'simulated at 2mm touches, five points over 40mm give '
                  '23mm of error while five over 250mm give 3.9mm. Touch '
                  'corners at opposite ends of the board, and move the board '
                  'to another height before touching more.')
    else:
        uncertainty = float('nan')
        print('fewer than 4 touches, so no uncertainty estimate is possible')

    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t
    j1 = float(np.mean([r['angles'][0] for r in recs]))
    j1_spread = float(np.ptp([r['angles'][0] for r in recs]))
    out = os.path.join(OUT_DIR, 'eye_to_hand.json')
    json.dump({
        'camera_to_base': X.tolist(),
        'joint1_deg': j1,
        'joint1_spread_deg': j1_spread,
        'method': 'touch (Kabsch on corresponding points)',
        'points': len(recs),
        'residual_mean_mm': float(resid.mean()),
        'residual_max_mm': float(resid.max()),
        # The honest error bar: how much the answer moves when the point set
        # changes, not how well it fits the points it was handed.
        'uncertainty_mm': uncertainty,
        'mount': 'first arm piece (link1); only joint1 moves the camera',
        'compose_note': ('base->camera at pan q1 = Rz(q1 - joint1_deg) @ '
                         'camera_to_base. See docs/camera_mount.md.'),
    }, open(out, 'w'), indent=2)
    print(f'\nwrote {out}')
    print(f'camera sits at ({t[0]*1000:+.0f}, {t[1]*1000:+.0f}, '
          f'{t[2]*1000:+.0f})mm in the base frame')
    if j1_spread > 1.0:
        # joint1 carries the camera, so it moving mid-collection means the
        # camera moved and the single transform above is an average of two.
        print(f'\nWARNING: joint1 varied by {j1_spread:.1f}deg across the '
              'touches, and joint1 is the joint that MOVES this camera. The '
              'transform is an average over that, not a measurement. Redo '
              'with joint1 held still.')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--auto', action='store_true',
                    help='sweep joint1 and solve automatically, then ask for '
                         'ONE touch to fix the yaw and height the sweep '
                         'cannot see. Far fewer manual steps than --collect.')
    ap.add_argument('--sweep', default='-16,-8,0,8,16',
                    help='joint1 offsets in degrees for --auto')
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--solve', action='store_true')
    ap.add_argument('--points', type=int, default=8,
                    help='how many touches to collect (default 8)')
    ap.add_argument('--square-mm', type=float, default=38.9,
                    help='PRINTED square size. Measure it -- '
                         'calibrate_hand_eye.py --measure-square does it with '
                         'depth (default 38.9, measured on this board)')
    ap.add_argument('--tool-offset-mm', type=float, default=0.0,
                    help='distance from the flange origin to the contact '
                         'point, along the flange z (default 0)')
    ap.add_argument('--free-drive', action='store_true',
                    help='release the servos and position the arm BY HAND. '
                         'Not the default, and not recommended: hand-moving '
                         'this arm reliably reboots the Atom, while commanded '
                         'motion measured zero reboots over 12 moves at near '
                         'maximum load. Use the default jog mode instead.')
    ap.add_argument('--speed', type=int, default=25,
                    help='speed for jog moves (default 25)')
    ap.add_argument('--step', type=float, default=3.0,
                    help='default jog step in degrees (default 3)')
    ap.add_argument('--restart', action='store_true',
                    help='discard any touches already recorded')
    ap.add_argument('--timeout', type=float, default=25.0)
    ap.add_argument('--max-residual-mm', type=float, default=8.0)
    args = ap.parse_args()

    if args.auto:
        return auto(args)
    if args.collect:
        return collect(args)
    if args.solve:
        return solve(args)
    ap.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
