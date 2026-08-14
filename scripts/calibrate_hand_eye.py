#!/usr/bin/env python3
"""Find where the camera sits relative to the flange (eye-in-hand).

What this is for
----------------
The servo currently steers in IMAGE space: it knows "the button is left of
centre" and jogs until it is not. That is enough to centre a target and
nothing more. To say "the button is 180mm in front of the base, 40mm left"
-- which is what pressing needs -- the stack has to turn a pixel plus a depth
into a point in the ARM's frame, and that requires knowing how the camera is
mounted.

Which of these actually changes when the robot moves
----------------------------------------------------
Worth being clear, because "portable" suggests recalibrating constantly and
that is the wrong picture:

  camera -> flange    FIXED by the mounting bracket. Only changes if the
                      camera is physically remounted or knocked. THIS is what
                      the script solves, and it is a one-off.
  base -> panel       Changes every time the robot is moved. NOT calibration
                      -- it is perception, estimated live from depth on every
                      run. Nothing here needs redoing for it.

So in a changing environment you calibrate once, on a desk, with a sheet of
paper -- and then never need a rig in the field.

Using it
--------
    ./scripts/calibrate_hand_eye.py --make-board board.png   # print this
    ./scripts/calibrate_hand_eye.py --collect                # move + capture
    ./scripts/calibrate_hand_eye.py --solve                  # compute + save

Print the board at 100% scale (NO "fit to page" -- it rescales and every
distance comes out wrong by that factor), measure one square with a ruler,
and pass --square-mm with what you measured rather than what was intended.

Lay it flat where the arm can see it from several angles, then --collect.
The arm visits a set of poses, every one checked by the collision guard
first, and records image + joint angles at each.

Why the pose set looks like that
--------------------------------
Hand-eye is solved from the RELATIVE motions between poses, and rotation is
what constrains it. A set of poses that only translate is degenerate: the
maths runs, returns something, and it is wrong with no warning. So the poses
deliberately vary joint5 and joint6 -- which rotate the camera without moving
the arm much -- and the script REFUSES to solve if the rotational spread is
too small, rather than handing back a confident wrong answer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'mycobot_driver',
                                'mycobot_driver'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from collision_guard import CollisionGuard, flange_transform  # noqa: E402

OUT_DIR = os.path.expanduser('~/hand_eye')
HOME = [0.0, 90.0, -149.0, 55.0, 0.0, 0.0]

# A5-ish board: big enough to detect at 20-40cm, small enough to print on one
# sheet and lie flat on a desk.
SQUARES_X, SQUARES_Y = 5, 7
DEFAULT_SQUARE_MM = 30.0
MARKER_RATIO = 0.72


def make_board(square_mm: float):
    d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard_create(
        SQUARES_X, SQUARES_Y, square_mm / 1000.0,
        square_mm / 1000.0 * MARKER_RATIO, d), d


def make_tag_grid(args):
    """A 2x2 grid of small markers -- the target that fits a flange face.

    Measured 2026-08-13 with 0.2px of corner noise, 12 trials, against the
    hand-eye solve itself:

        one 27mm tag              17.2mm mean,  56.0mm worst
        one 40mm tag               5.5mm mean,  39.0mm worst
        one 60mm tag               4.6mm mean,  39.3mm worst
        FOUR 20mm tags in 43mm     2.1mm mean,   4.1mm worst

    So four small markers in the same footprint beat one large one twice over
    on the mean and ten times on the worst case. The worst column is the point:
    a single square has a two-fold pose ambiguity that occasionally resolves
    the wrong way and throws the answer 40mm out, and neighbouring markers
    disambiguate each other. That is why this exists rather than "print it
    bigger" -- bigger does not fit on a flange, and would not fix the outliers
    anyway.

    Ids start at 20 so they can never collide with the 5x7 board's 0-16.
    """
    d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.GridBoard_create(
        2, 2, args.tag_mm / 1000.0, args.tag_gap_mm / 1000.0, d,
        args.tag_first_id), d


def cmd_make_tag(args) -> int:
    board, _ = make_tag_grid(args)
    span = 2 * args.tag_mm + args.tag_gap_mm
    px = int(span * 12)
    img = board.draw((px, px))
    cv2.imwrite(args.make_tag, img)
    print(f'wrote {args.make_tag}')
    print(f'  2x2 markers, ids {args.tag_first_id}-{args.tag_first_id + 3}, '
          f'{args.tag_mm:.0f}mm each with {args.tag_gap_mm:.0f}mm gaps')
    print(f'  print so the WHOLE PATTERN spans {span:.0f}mm, then measure one '
          f'black square and pass --tag-mm with what you measured')
    print(f'  ids start at {args.tag_first_id}, so they cannot be confused '
          'with the 5x7 board (which uses 0-16)')
    return 0


def cmd_measure_square(args) -> int:
    """Measure the PRINTED square with depth, instead of trusting the print.

    Why this exists, measured 2026-08-13: the board in use here is printed at
    **39mm squares, not the 30mm the script assumed** -- a 30% scale error,
    almost certainly a printer "fit to page". Scale in the target is scale in
    the answer, so every translation the old calibration produced was 30% out.

    Nothing in the colour-only path can catch that. The board's pose is solved
    FROM the assumed square size, so a wrong size yields a wrong range that is
    perfectly self-consistent -- and all four solvers agree, because they were
    handed the same wrong number. "Solvers agree to 1mm" measures agreement,
    not accuracy.

    Depth is independent of all of it. It measures the distance between
    adjacent corners directly, in millimetres, with no reference to what the
    board is supposed to be. On this board, 34 spacings gave 38.99mm with a
    0.57mm spread -- and depth reconstructed a known-flat board to 1.31mm rms
    at the same time, which is what says the sensor is worth believing.

    Run this once per printed board, then pass the number to --square-mm.
    """
    import numpy as np
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(40):          # let auto-exposure settle
            frames = pipe.wait_for_frames(10000)
        frames = align.process(frames)
        colour = np.asanyarray(frames.get_color_frame().get_data()).copy()
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_mm = (np.asanyarray(frames.get_depth_frame().get_data())
                    .astype(float) * scale * 1000.0)
    finally:
        pipe.stop()

    K = np.array(args.K, dtype=float).reshape(3, 3) if args.K else np.array(
        [[393.8, 0, 318.1], [0, 393.4, 236.5], [0, 0, 1.0]])
    board, adict = make_board(args.square_mm)
    gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, adict)
    if ids is None or len(ids) < 4:
        print('No board in view. Point the camera at it, filling a good part '
              'of the frame, and try again.')
        return 1
    n, ch_c, ch_i = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board)
    if n is None or n < 6:
        print(f'Only {n} board corners found; need at least 6.')
        return 1

    pts = {}
    for k, idx in enumerate(ch_i.ravel()):
        u, v = ch_c[k].ravel()
        iu, iv = int(round(u)), int(round(v))
        patch = depth_mm[max(0, iv - 2):iv + 3, max(0, iu - 2):iu + 3]
        valid = patch[patch > 0]
        if valid.size < 5:
            continue
        z = float(np.median(valid))
        pts[int(idx)] = np.array([(u - K[0, 2]) * z / K[0, 0],
                                  (v - K[1, 2]) * z / K[1, 1], z])

    # The inner corners of an X-by-Y board form an (X-1) by (Y-1) grid.
    gx = SQUARES_X - 1
    gy = SQUARES_Y - 1
    spacings = []
    for idx, P in pts.items():
        r, c = divmod(idx, gx)
        for nr, nc in ((r, c + 1), (r + 1, c)):
            if nc < gx and nr < gy and (nr * gx + nc) in pts:
                spacings.append(float(np.linalg.norm(P - pts[nr * gx + nc])))
    if len(spacings) < 5:
        print(f'Only {len(spacings)} corner spacings had valid depth. Move the '
              'board into the 70-500mm band where a D405 actually measures.')
        return 1

    s = np.array(spacings)
    measured = float(np.median(s))

    # Flatness is the sensor's own credibility check. A known-flat board that
    # depth renders as flat means the depth is worth believing; if this is
    # poor, so is the measurement above.
    P = np.array(list(pts.values()))
    centred = P - P.mean(axis=0)
    normal = np.linalg.svd(centred)[2][2]
    flat_rms = float(np.abs(centred @ normal).std())

    print(f'{len(s)} adjacent-corner spacings measured by depth')
    print(f'  median {measured:.2f}mm, mean {s.mean():.2f}mm, '
          f'spread {s.std():.2f}mm')
    print(f'  depth renders this flat board flat to {flat_rms:.2f}mm rms '
          f'-- that is what makes the number above trustworthy')
    print()
    print(f'configured square size: {args.square_mm:.2f}mm')
    err = 100.0 * (measured / args.square_mm - 1.0)
    if abs(err) < 2.0:
        print(f'measured agrees to {err:+.1f}% -- the configured size is right')
        return 0
    print(f'measured is {err:+.1f}% off the configured size.')
    print(f'\n  Use --square-mm {measured:.1f}')
    print('\nScale in the target is scale in the answer: a calibration solved '
          f'at {args.square_mm:.0f}mm from a board that is really '
          f'{measured:.0f}mm has every translation wrong by {err:+.0f}%, and '
          'nothing in the colour-only path can notice.')
    return 0


def cmd_make_board(args) -> int:
    board, _ = make_board(args.square_mm)
    # 10 px/mm gives a crisp print at any sane printer DPI.
    px = (int(SQUARES_X * args.square_mm * 10), int(SQUARES_Y * args.square_mm * 10))
    img = board.draw(px)
    cv2.imwrite(args.make_board, img)
    print(f'wrote {args.make_board}  ({SQUARES_X}x{SQUARES_Y} squares, '
          f'{args.square_mm:.1f}mm each)')
    print('Print at 100% scale -- "fit to page" silently rescales it and every')
    print('distance downstream is wrong by that factor. Then MEASURE a square')
    print('and pass --square-mm with the measured value.')
    return 0


def capture_poses(args) -> int:
    """Move through the pose set, recording image + joint angles at each."""
    import time
    import pyrealsense2 as rs
    from arm_broker import request

    guard = CollisionGuard()
    os.makedirs(OUT_DIR, exist_ok=True)

    def pose_after(t_settled, timeout=25.0):
        """A reading genuinely SAMPLED after the arm stopped moving.

        Not "recent enough": on a lossy link the freshest available sample can
        predate the move entirely, and accepting it pairs this pose's IMAGE
        with the previous pose's ANGLES. Hand-eye would then solve a
        consistent-looking problem with one input systematically wrong, and
        return a confident wrong transform with no error anywhere.

        The broker stamps each reading with its age, so the sample time is
        now - age. Require that to be after the arm settled.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = request({'cmd': 'state'})
            if s and s.get('angles') and s.get('age_ms') is not None:
                sampled_at = time.monotonic() - s['age_ms'] / 1000.0
                if sampled_at >= t_settled:
                    return s['angles']
            time.sleep(0.4)
        return None

    if args.eye_to_hand and args.stand:
        # A stand-mounted camera does not move with the arm at all, so
        # nothing has to be held still and EVERY joint can vary. That matters:
        # hand-eye is constrained by relative rotation, and freeing joint1
        # roughly doubles the rotational spread available compared with the
        # arm-mounted case that has to pin it.
        poses = []
        for pan in (-25, 0, 25):
            for lift in (-15, 10):
                for wrist in (-35, 0, 35):
                    for roll in (-40, 0, 40):
                        q = list(HOME)
                        q[0] += pan
                        q[1] += lift
                        q[3] += wrist
                        q[5] += roll
                        poses.append(q)
    elif args.eye_to_hand:
        # The camera rides on the first arm piece, so joint1 -- and ONLY
        # joint1 -- moves it. Hold joint1 still and the camera is a static
        # observer, which is the eye-to-hand case: the BOARD moves, on the
        # flange, and the arm's own FK says where it is.
        #
        # So the pose set has to do the opposite of the eye-in-hand one. There
        # the arm barely moved and the camera turned; here the camera cannot
        # turn at all, so the flange must sweep rotation in front of it while
        # staying in view. joint4/5/6 supply the rotation, joint2/3 move the
        # board around the frame and change its range.
        poses = []
        for lift in (-15, 0, 15):
            for reach in (-15, 5):
                for wrist in (-35, 0, 35):
                    for roll in (-40, 0, 40):
                        q = list(HOME)
                        q[0] = args.fixed_joint1   # never varies. See above.
                        q[1] += lift
                        q[2] += reach
                        q[3] += wrist
                        q[5] += roll
                        poses.append(q)
    else:
        # Rotation-rich, translation-poor. joint5 and joint6 turn the camera
        # while barely moving the arm, which is exactly the motion hand-eye
        # needs and the motion a naive "wave it around" pose set lacks.
        poses = []
        for pan in (-20, 0, 20):
            for tilt in (-25, -10, 5, 20):
                for roll in (-40, 0, 40):
                    q = list(HOME)
                    q[0] += pan
                    q[4] += tilt
                    q[5] += roll
                    poses.append(q)

    safe = [q for q in poses if guard.check(q)[0]]
    print(f'{len(safe)} of {len(poses)} candidate poses pass the collision '
          f'guard')
    if len(safe) < 8:
        print('Too few safe poses to calibrate from.')
        return 1

    # Refuse to start on a link that cannot answer. Calibration is precisely
    # the task that must not run on unreliable readings: a pose read that
    # arrives late or not at all pairs an image with the wrong joint angles,
    # and hand-eye then returns a confident wrong transform. Measured the hard
    # way -- started at 15% valid, ground through poses for 10 minutes waiting
    # 25s each for readings that never came, and recorded nothing.
    h = request({'cmd': 'health'}, timeout=15)
    frac = (h or {}).get('link', {}).get('fraction')
    if frac is None:
        print('No broker is running. Start ./scripts/arm_broker.py first.')
        return 1
    if args.commanded_angles:
        # Degraded mode, for when writes land but reads do not -- a real and
        # separately-failing pair on this arm, confirmed by the LED cycling on
        # command while reads sat at 2%.
        #
        # The cost is real and worth stating: the arm settles about a degree
        # from where it was told to go, and that error goes straight into the
        # transform instead of being measured out. Roughly 4mm of position
        # error at 250mm. Much better than no calibration, clearly worse than
        # a clean one -- redo it properly when reads come back.
        print(f'link {frac*100:.0f}% valid -- COMMANDED-ANGLE MODE')
        print('  Using commanded joint angles, not measured ones. The arm '
              'settles ~1deg off target and that error enters the result: '
              'expect ~4mm of position error at 250mm.')
        print('  Redo this without --commanded-angles once reads recover.')
    elif frac < args.min_link:
        print(f'Link is {frac*100:.0f}% valid; calibration needs at least '
              f'{args.min_link*100:.0f}%.')
        print('Readings that arrive late get paired with the wrong image, '
              'which yields a confident wrong transform rather than an error. '
              'Not starting.')
        print('Override with --min-link 0 if you know what you are doing.')
        return 1
    print(f'link {frac*100:.0f}% valid -- proceeding')

    # Start from home. The pose set is defined as offsets from it, so
    # beginning anywhere else means the FIRST move can be enormous -- the arm
    # was 93deg away on joint4 once, missed the settle window, and every pose
    # was then rejected for not having arrived.
    print('homing before the pose set...')
    request({'cmd': 'send_angles', 'angles': HOME, 'speed': 30,
             'force': True}, timeout=25)
    time.sleep(6.0)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    records = []
    consecutive_failures = 0
    last_cmd = list(HOME)
    try:
        for i, q in enumerate(safe):
            request({'cmd': 'send_angles', 'angles': q, 'speed': 30,
                     'force': True}, timeout=25)
            # Wait in proportion to how far it actually has to travel. A fixed
            # settle is wrong at both ends: consecutive poses here differ by up
            # to 80deg of roll, which at speed 30 (~29deg/s measured) takes
            # nearly 3s on its own, while a small move wastes the same wait.
            travel = max(abs(a - b) for a, b in zip(q, last_cmd))
            time.sleep(args.settle + travel / 25.0)
            last_cmd = list(q)
            settled = time.monotonic()
            if args.commanded_angles:
                # Give it longer to arrive, since nothing will confirm it did.
                time.sleep(args.settle)
                actual = list(q)
            else:
                actual = pose_after(settled)
            if actual is None:
                print(f'  {i:02d}: no post-move pose read within 25s '
                      '-- skipped rather than guessed')
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print('\nThree poses in a row could not be read. The link '
                          'has gone during the run -- stopping rather than '
                          'spending 25s on each of the remaining poses to '
                          'collect nothing.')
                    break
                continue
            # A pose that never arrived is worse than a skipped one: it means
            # the arm is somewhere else entirely, and the image would be
            # attributed to where we THINK it went.
            if max(abs(a - c) for a, c in zip(actual, q)) > 8.0:
                print(f'  {i:02d}: arm is {max(abs(a-c) for a,c in zip(actual,q)):.0f}deg '
                      'from the commanded pose -- skipped')
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print('\nThree poses in a row where the arm did not '
                          'arrive. It is not executing commands -- stopping.')
                    break
                continue
            consecutive_failures = 0
            for _ in range(8):
                frames = pipe.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
            path = os.path.join(OUT_DIR, f'pose_{i:02d}.png')
            cv2.imwrite(path, img)
            # Record the MEASURED angles, not the commanded ones. The arm
            # trails its target by a degree or so, and calibrating against
            # where it was told to be rather than where it is puts that error
            # straight into the result.
            records.append({'image': path, 'angles': actual})
            print(f'  {i:02d}: {[round(a, 1) for a in actual]}')
    finally:
        pipe.stop()
        request({'cmd': 'send_angles', 'angles': HOME, 'speed': 30,
                 'force': True}, timeout=25)

    with open(os.path.join(OUT_DIR, 'poses.json'), 'w') as f:
        json.dump(records, f, indent=2)
    print(f'\n{len(records)} captures -> {OUT_DIR}')
    print('Now: ./scripts/calibrate_hand_eye.py --solve')
    return 0


def solve(args) -> int:
    from arm_broker import request  # noqa: F401  (kept for symmetry)

    rec_path = os.path.join(OUT_DIR, 'poses.json')
    if not os.path.isfile(rec_path):
        print(f'no captures at {rec_path} -- run --collect first')
        return 1
    records = json.load(open(rec_path))

    K = np.array(args.K, dtype=float).reshape(3, 3) if args.K else None
    if K is None:
        # Factory intrinsics from the D405, as camera_node reports them.
        K = np.array([[393.8, 0, 318.1], [0, 393.4, 236.5], [0, 0, 1.0]])
        print('using D405 factory intrinsics fx=393.8 fy=393.4 '
              'cx=318.1 cy=236.5')
    dist = np.zeros(5)

    board, adict = make_board(args.square_mm)

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    j1 = []
    used = 0
    for r in records:
        img = cv2.imread(r['image'])
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, adict)
        if ids is None:
            continue

        if args.tag_grid:
            # Pose from ALL visible markers of the grid at once, which is what
            # makes a grid better than its biggest marker: neighbours resolve
            # each other's pose ambiguity.
            tb, _ = make_tag_grid(args)
            keep = [k for k, m in enumerate(ids.ravel())
                    if args.tag_first_id <= int(m) <= args.tag_first_id + 3]
            if len(keep) < 2:
                continue
            # OpenCV returns (count, rvec, tvec) here in 4.x, and writes in
            # place in some builds. Handle both rather than assume.
            out = cv2.aruco.estimatePoseBoard(
                [corners[k] for k in keep],
                np.array([[int(ids.ravel()[k])] for k in keep]),
                tb, K, dist, np.zeros(3), np.zeros(3))
            if isinstance(out, tuple):
                n, rvec, tvec = out
            else:
                n = out
            if n < 2:
                continue
            rvec = np.asarray(rvec).reshape(3)
            tvec = np.asarray(tvec).reshape(3)
        elif args.tag_id is not None:
            # A SINGLE marker as the target, which is what fits on a flange
            # face -- a whole ChArUco board does not. Less accurate than a
            # board (four corners rather than dozens, and its pose is more
            # sensitive to corner noise the more square-on it is), but it is
            # what can physically be mounted, and the solver disagreement
            # check will say if that accuracy is not enough.
            hit = [k for k, m in enumerate(ids.ravel())
                   if int(m) == args.tag_id]
            if not hit:
                continue
            if len(hit) > 1:
                # The same id twice means the tag shares an id with something
                # else in view, and there is no way to tell which is the
                # flange. Refusing beats picking one.
                print(f'  {os.path.basename(r["image"])}: marker '
                      f'{args.tag_id} appears {len(hit)} times -- ambiguous, '
                      'skipped')
                continue
            rv, tv, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[hit[0]]], args.tag_mm / 1000.0, K, dist)
            rvec, tvec = rv[0][0], tv[0][0]
        else:
            if len(ids) < 4:
                continue
            n, ch_c, ch_i = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if n is None or n < 6:
                continue
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                ch_c, ch_i, board, K, dist, None, None)
            if not ok:
                continue
        T = np.array(flange_transform(r['angles']))
        if args.eye_to_hand:
            # Same solver, transforms inverted. cv2.calibrateHandEye solves
            # AX=XB for the camera's pose relative to whatever frame the
            # "gripper" transforms describe. Feed it BASE->FLANGE instead of
            # flange->base and the X it returns is the camera's pose in the
            # BASE frame -- which is the static-camera case, no separate
            # algorithm needed.
            T = np.linalg.inv(T)
        R_g2b.append(T[:3, :3])
        t_g2b.append(T[:3, 3])
        R_t2c.append(cv2.Rodrigues(rvec)[0])
        t_t2c.append(tvec.reshape(3))
        j1.append(r['angles'][0])
        used += 1

    what = ('tag grid' if args.tag_grid else
            f'marker {args.tag_id}' if args.tag_id is not None else 'board')
    print(f'{used} of {len(records)} captures had a usable {what} view')
    if used < 6:
        print('Not enough. The board must be visible and reasonably large in '
              'the frame -- move it closer, light it better, or print bigger.')
        return 1

    if args.eye_to_hand and args.stand:
        print('camera is stand-mounted, so joint1 is free to vary and no '
              'fixed-joint1 check applies')
    elif args.eye_to_hand:
        # The failure mode unique to this mode, and it is silent. Eye-to-hand
        # assumes the camera did not move. Here the camera rides on the first
        # arm piece, so joint1 moving IS the camera moving -- and the solve
        # would absorb that into the transform and return something confident
        # and wrong, with every other check still passing.
        j1_spread = float(np.max(j1) - np.min(j1)) if j1 else 0.0
        print(f'joint1 varied by {j1_spread:.2f}deg across the captures')
        if j1_spread > 1.0:
            print('Refusing: joint1 moved, and joint1 is the one joint that '
                  'carries the camera. The static-camera assumption this mode '
                  'rests on is broken, and nothing downstream would notice. '
                  'Recapture with joint1 held still (--fixed-joint1).')
            return 1

    # Rotational spread. Hand-eye is constrained by RELATIVE rotation between
    # poses; a set that only translates is degenerate and the solvers will
    # still return an answer, silently wrong. Refuse instead.
    angles = []
    for i in range(len(R_g2b)):
        for j in range(i + 1, len(R_g2b)):
            Rrel = R_g2b[i].T @ R_g2b[j]
            angles.append(np.degrees(
                np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
    spread = float(np.max(angles)) if angles else 0.0
    print(f'largest relative rotation between poses: {spread:.0f}deg')
    if spread < 25.0:
        print('Too little rotation for hand-eye to be determined. The maths '
              'will happily return a confident wrong answer from a set like '
              'this, so it is refused. Vary joint5/joint6 more.')
        return 1

    results = {}
    for name, method in (('TSAI', cv2.CALIB_HAND_EYE_TSAI),
                         ('PARK', cv2.CALIB_HAND_EYE_PARK),
                         ('HORAUD', cv2.CALIB_HAND_EYE_HORAUD),
                         ('DANIILIDIS', cv2.CALIB_HAND_EYE_DANIILIDIS)):
        try:
            Rc, tc = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c,
                                          method=method)
            results[name] = (Rc, tc.reshape(3))
        except Exception as e:  # noqa: BLE001
            print(f'  {name}: failed ({e})')

    if not results:
        print('every solver failed')
        return 1

    print()
    print(f'solver      camera position in the '
          f'{"BASE" if args.eye_to_hand else "flange"} frame (mm)')
    ts = []
    for name, (Rc, tc) in results.items():
        print(f'  {name:<11} x={tc[0]*1000:+7.1f}  y={tc[1]*1000:+7.1f}  '
              f'z={tc[2]*1000:+7.1f}')
        ts.append(tc)
    spread_mm = float(np.max(np.ptp(np.array(ts), axis=0)) * 1000)
    print()
    print(f'disagreement between solvers: {spread_mm:.1f}mm')
    # Independent algorithms on the same data should land in the same place.
    # When they do not, the data is the problem -- not the choice of solver.
    if spread_mm > 15.0:
        print('That is too much to trust. The solvers are independent, so '
              'wide disagreement means the INPUT is bad: not enough rotation, '
              'a mis-measured square size, a board that moved during capture, '
              'or joint angles that did not match the images. Recapture '
              'rather than picking a favourite.')
        return 1

    Rc, tc = results.get('PARK', next(iter(results.values())))
    X = np.eye(4)
    X[:3, :3] = Rc
    X[:3, 3] = tc
    out = os.path.join(OUT_DIR,
                       'eye_to_hand.json' if args.eye_to_hand
                       else 'hand_eye.json')
    key = 'camera_to_base' if args.eye_to_hand else 'camera_to_flange'
    extra = {}
    if args.eye_to_hand:
        # Solved at ONE joint1 angle, and joint1 rotates the camera. Record it,
        # because base->camera at any other pan is this transform composed with
        # Rz(q1 - joint1_deg) -- and a consumer that does not know the angle it
        # was solved at cannot do that composition.
        if args.stand:
            extra = {'mount': 'fixed stand; the camera does not move with '
                              'the arm',
                     'compose_note': ('camera_to_base is absolute -- do NOT '
                                      'compose it with joint1. See '
                                      'docs/camera_mount.md.')}
        else:
            extra = {'joint1_deg': float(np.mean(j1)),
                     'joint1_spread_deg': j1_spread,
                     'mount': 'first arm piece (link1); only joint1 moves it',
                     'compose_note': (
                         'base->camera at pan q1 = Rz(q1 - joint1_deg) @ '
                         'camera_to_base. See docs/camera_mount.md.')}
    with open(out, 'w') as f:
        json.dump({key: X.tolist(),
                   **extra,
                   'solver': 'PARK',
                   'captures_used': used,
                   'rotation_spread_deg': spread,
                   'solver_disagreement_mm': spread_mm,
                   'square_mm': args.square_mm,
                   'commanded_angles': bool(args.commanded_angles),
                   'accuracy_note': (
                       'DEGRADED: solved from commanded joint angles because '
                       'reads were unavailable. Expect ~4mm position error. '
                       'Recalibrate without --commanded-angles.'
                       if args.commanded_angles else
                       'solved from measured joint angles')},
                  f, indent=2)
    print(f'\nwrote {out}')
    print('This is FIXED to the mounting. Redo it only if the camera is '
          'remounted or knocked -- not when the robot is moved somewhere new.')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--make-board', metavar='PNG',
                    help='write a printable ChArUco board and exit')
    ap.add_argument('--collect', action='store_true',
                    help='move the arm through the pose set and capture')
    ap.add_argument('--solve', action='store_true',
                    help='compute the transform from captured poses')
    ap.add_argument('--square-mm', type=float, default=DEFAULT_SQUARE_MM,
                    help='MEASURED square size of the printed board')
    ap.add_argument('--settle', type=float, default=2.5)
    ap.add_argument('--measure-square', action='store_true',
                    help='measure the PRINTED square with depth and stop. '
                         'Run this once per printed board -- a printer '
                         '"fit to page" put 30%% of scale error into this '
                         "project's board and nothing in the colour path "
                         'could see it.')
    ap.add_argument('--eye-to-hand', action='store_true',
                    help='camera is STATIC and the board rides on the flange, '
                         'which is the case once the camera leaves the flange. '
                         'Solves camera->base instead of camera->flange.')
    ap.add_argument('--tag-grid', action='store_true',
                    help='target is a 2x2 GRID of small markers on the flange. '
                         'Beats one large tag: 2.1mm mean error against 4.6mm '
                         'for a single 60mm tag, and 4mm worst case against '
                         '39mm, because neighbours resolve each other\'s pose '
                         'ambiguity. Fits where a big tag does not.')
    ap.add_argument('--make-tag', metavar='PNG', default=None,
                    help='write the printable 2x2 tag grid and stop')
    ap.add_argument('--tag-gap-mm', type=float, default=3.0,
                    help='gap between markers in the grid (default 3)')
    ap.add_argument('--tag-first-id', type=int, default=20,
                    help='first marker id in the grid; 20+ never collides '
                         'with the 5x7 board (default 20)')
    ap.add_argument('--tag-id', type=int, default=None,
                    help='use a SINGLE aruco marker of this id as the target '
                         'instead of the ChArUco board -- what fits on a '
                         'flange face. Give --tag-mm as well.')
    ap.add_argument('--tag-mm', type=float, default=27.0,
                    help='the printed side of that marker, measured across '
                         'the BLACK SQUARE only, not the white surround. '
                         'Scale here is scale in the answer.')
    ap.add_argument('--stand', action='store_true',
                    help='the camera is on a FIXED STAND rather than the arm. '
                         'Frees joint1 in the pose set, drops the fixed-joint1 '
                         'check, and records the transform as absolute so '
                         'nothing composes it with joint1 later.')
    ap.add_argument('--fixed-joint1', type=float, default=0.0,
                    help='joint1 angle to hold throughout an eye-to-hand '
                         'capture. joint1 is the only joint that moves the '
                         'camera, so it must not vary (default 0)')
    ap.add_argument('--commanded-angles', action='store_true',
                    help='record commanded instead of measured joint angles. '
                         'For when writes land but reads do not. Costs ~4mm '
                         'of accuracy; redo properly when reads recover')
    ap.add_argument('--min-link', type=float, default=0.6,
                    help='refuse to collect below this link reliability; '
                         'calibrating on unreliable readings produces a '
                         'confident wrong answer, not an error')
    ap.add_argument('--K', type=float, nargs=9, default=None,
                    help='camera matrix, row-major; defaults to D405 factory')
    args = ap.parse_args()

    if args.make_tag:
        return cmd_make_tag(args)
    if args.measure_square:
        return cmd_measure_square(args)
    if args.make_board:
        return cmd_make_board(args)
    if args.collect:
        return capture_poses(args)
    if args.solve:
        return solve(args)
    ap.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
