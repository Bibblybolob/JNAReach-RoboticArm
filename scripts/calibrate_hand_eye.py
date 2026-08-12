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

    def pose(tries=25, max_age=12000):
        for _ in range(tries):
            s = request({'cmd': 'state'})
            if s and s.get('angles') and (s.get('age_ms') or 9e9) < max_age:
                return s['angles']
            time.sleep(0.6)
        return None

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

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    records = []
    try:
        for i, q in enumerate(safe):
            request({'cmd': 'send_angles', 'angles': q, 'speed': 30,
                     'force': True}, timeout=25)
            time.sleep(args.settle)
            actual = pose(tries=10, max_age=9000)
            if actual is None:
                print(f'  {i:02d}: no pose read, skipping')
                continue
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
    used = 0
    for r in records:
        img = cv2.imread(r['image'])
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, adict)
        if ids is None or len(ids) < 4:
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
        R_g2b.append(T[:3, :3])
        t_g2b.append(T[:3, 3])
        R_t2c.append(cv2.Rodrigues(rvec)[0])
        t_t2c.append(tvec.reshape(3))
        used += 1

    print(f'{used} of {len(records)} captures had a usable board view')
    if used < 6:
        print('Not enough. The board must be visible and reasonably large in '
              'the frame -- move it closer, light it better, or print bigger.')
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
    print('solver      camera offset from flange (mm)')
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
    out = os.path.join(OUT_DIR, 'hand_eye.json')
    with open(out, 'w') as f:
        json.dump({'camera_to_flange': X.tolist(),
                   'solver': 'PARK',
                   'captures_used': used,
                   'rotation_spread_deg': spread,
                   'solver_disagreement_mm': spread_mm,
                   'square_mm': args.square_mm}, f, indent=2)
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
    ap.add_argument('--K', type=float, nargs=9, default=None,
                    help='camera matrix, row-major; defaults to D405 factory')
    args = ap.parse_args()

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
