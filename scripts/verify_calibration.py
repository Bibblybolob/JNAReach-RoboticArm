#!/usr/bin/env python3
"""Check a camera->base calibration against something it did not fit.

Why the solve's own numbers are not enough
------------------------------------------
`calibrate_hand_eye.py` refuses when its four solvers disagree, and that
catches a mis-measured target or too little rotation. It cannot catch the
failure this arm's small flange tag is prone to: a single square has a
two-fold pose ambiguity, and when it resolves the wrong way EVERY solver is
handed the same flipped poses. They then agree beautifully with each other and
are wrong together -- so solver agreement is not evidence.

The independent check
---------------------
The tag is bolted to the flange, so its position in the FLANGE frame is a
physical constant. Nothing in the calibration is allowed to change it.

    tag in base   = camera_to_base @ (tag in camera, measured now)
    tag in flange = inv(flange in base, from FK) @ tag in base

Compute that at several poses and the spread IS the error. It uses no
quantity the solve fitted: the tag position comes from the live camera, the
flange from the arm's own encoders. A calibration that is wrong makes the
"constant" wander as the arm moves, and by how much.

Two independent measurements of the tag are compared, because they fail
differently:

  - IMAGE: the tag's pose from its corners, which depends on the printed size
  - DEPTH: the tag centre straight from the depth map, which does not

If those two disagree, the printed size is wrong -- the same defect that made
this project's board 39mm when the tooling said 30mm.

    ./scripts/verify_calibration.py --poses 8
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
HOME = [0.0, 90.0, -149.0, 55.0, 0.0, 0.0]


def load_calibration(path):
    with open(path) as f:
        c = json.load(f)
    if 'camera_to_base' not in c:
        raise SystemExit(
            f'{path} has no camera_to_base -- that is an eye-IN-hand result, '
            'which describes a camera mounted on the flange.')
    return c


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calibration',
                    default=os.path.join(OUT_DIR, 'eye_to_hand.json'))
    ap.add_argument('--poses', type=int, default=8)
    ap.add_argument('--tag-id', type=int, default=2)
    ap.add_argument('--tag-mm', type=float, default=27.0)
    ap.add_argument('--speed', type=int, default=30)
    ap.add_argument('--contact-sheet',
                    default=os.path.join(OUT_DIR, 'verify_sheet.png'),
                    help='one tiled image of every pose the arm visited, with '
                         'the detected tag ringed. A picture per move is the '
                         'only way to see that the arm went where it was told '
                         'and that the tag was found on the flange rather '
                         'than somewhere else in the scene.')
    args = ap.parse_args()

    import pyrealsense2 as rs

    calib = load_calibration(args.calibration)
    X = np.array(calib['camera_to_base'], dtype=float)
    print(f'checking {args.calibration}')
    print(f"  method: {calib.get('method', 'unknown')}")
    print(f"  mount:  {calib.get('mount', 'unrecorded')}\n")

    guard = CollisionGuard()
    h = request({'cmd': 'health'}, timeout=15)
    if h is None or h.get('link', {}).get('fraction', 0) < 0.6:
        print('link too poor to verify -- every pose needs a joint reading '
              'taken after the arm stopped.')
        return 1

    rng = np.random.default_rng(0)
    poses = []
    while len(poses) < args.poses:
        q = [HOME[0] + rng.uniform(-25, 25), HOME[1] + rng.uniform(-12, 10),
             HOME[2], HOME[3] + rng.uniform(-30, 30),
             HOME[4] + rng.uniform(-35, 35), HOME[5] + rng.uniform(-40, 40)]
        if guard.check(q)[0]:
            poses.append(q)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale() * 1000.0
    i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])
    adict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

    from_image, from_depth = [], []
    shots = []
    try:
        for _ in range(40):
            pipe.wait_for_frames(10000)
        print(f"{'pose':>5s} {'tag in flange frame (mm), from image':>38s}"
              f" {'from depth':>22s}")
        for k, q in enumerate(poses):
            request({'cmd': 'call', 'method': 'power_on'}, timeout=20)
            r = request({'cmd': 'send_angles', 'angles': [float(v) for v in q],
                         'speed': args.speed, 'force': True}, timeout=30)
            if not r or not r.get('ok'):
                print(f'{k:5d}   move refused')
                continue
            time.sleep(6.0)
            st = request({'cmd': 'state'})
            if not st or not st.get('angles') or st.get('age_ms', 1e9) > 3000:
                print(f'{k:5d}   no fresh joint reading')
                continue
            meas = list(st['angles'])
            time.sleep(0.4)
            f = align.process(pipe.wait_for_frames(10000))
            colour = np.asanyarray(f.get_color_frame().get_data())
            depth = (np.asanyarray(f.get_depth_frame().get_data())
                     .astype(float) * scale)
            corners, ids, _ = cv2.aruco.detectMarkers(
                cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY), adict)
            # A frame per move, whether or not the tag was found -- a missing
            # tag is exactly the case worth being able to look at.
            shot = cv2.aruco.drawDetectedMarkers(colour.copy(), corners, ids)
            cv2.putText(shot, f'pose {k}  j=[{",".join(f"{a:.0f}" for a in meas)}]',
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(shot, f'pose {k}  j=[{",".join(f"{a:.0f}" for a in meas)}]',
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if ids is None or args.tag_id not in [int(x) for x in ids.ravel()]:
                cv2.putText(shot, 'TAG NOT FOUND', (6, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                shots.append(shot)
                print(f'{k:5d}   tag not visible')
                continue
            j = [int(x) for x in ids.ravel()].index(args.tag_id)

            # MEASURED pose, not commanded -- the arm settles about a degree
            # short and that degree is millimetres at this reach.
            T_bf = np.array(flange_transform(meas))
            T_fb = np.linalg.inv(T_bf)

            rv, tv, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[j]], args.tag_mm / 1000.0, K, np.zeros(5))
            p_cam_img = np.asarray(tv[0][0], dtype=float).reshape(3)

            pts = corners[j].reshape(4, 2)
            cx, cy = pts.mean(axis=0)
            patch = depth[max(0, int(cy) - 3):int(cy) + 4,
                          max(0, int(cx) - 3):int(cx) + 4]
            val = patch[patch > 0]
            p_cam_dep = None
            if val.size >= 5:
                z = float(np.median(val)) / 1000.0
                p_cam_dep = np.array([(cx - K[0, 2]) * z / K[0, 0],
                                      (cy - K[1, 2]) * z / K[1, 1], z])

            out = []
            for src, p_cam in (('img', p_cam_img), ('dep', p_cam_dep)):
                if p_cam is None:
                    out.append(None)
                    continue
                p_base = X[:3, :3] @ p_cam + X[:3, 3]
                p_fl = T_fb[:3, :3] @ p_base + T_fb[:3, 3]
                out.append(p_fl)
            cv2.putText(shot, f'{np.linalg.norm(p_cam_img)*1000:.0f}mm',
                        (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
            shots.append(shot)
            if out[0] is not None:
                from_image.append(out[0])
            if out[1] is not None:
                from_depth.append(out[1])
            s0 = ('(%+7.1f,%+7.1f,%+7.1f)' % tuple(out[0] * 1000)
                  if out[0] is not None else 'n/a')
            s1 = ('(%+6.0f,%+6.0f,%+6.0f)' % tuple(out[1] * 1000)
                  if out[1] is not None else 'n/a')
            print(f'{k:5d} {s0:>38s} {s1:>22s}')
    finally:
        pipe.stop()

    if shots:
        cols = 4
        rows = (len(shots) + cols - 1) // cols
        th, tw = 240, 320
        sheet = np.full((rows * th, cols * tw, 3), 30, np.uint8)
        for n, im in enumerate(shots):
            r, c = divmod(n, cols)
            sheet[r*th:(r+1)*th, c*tw:(c+1)*tw] = cv2.resize(im, (tw, th))
        cv2.imwrite(args.contact_sheet, sheet)
        print(f'\ncontact sheet of all {len(shots)} poses: {args.contact_sheet}')

    print()
    ok = True
    for name, pts in (('image', from_image), ('depth', from_depth)):
        if len(pts) < 3:
            print(f'{name}: only {len(pts)} usable poses, cannot judge')
            continue
        P = np.array(pts)
        spread = float(np.linalg.norm(P.std(axis=0)) * 1000)
        worst = float(np.max(np.linalg.norm(P - P.mean(axis=0), axis=1)) * 1000)
        print(f'{name:>6s}: the tag sits at '
              f'({P.mean(axis=0)[0]*1000:+.0f},{P.mean(axis=0)[1]*1000:+.0f},'
              f'{P.mean(axis=0)[2]*1000:+.0f})mm in the flange frame, '
              f'spread {spread:.1f}mm, worst {worst:.1f}mm')
        if spread > 5.0:
            ok = False
    if from_image and from_depth:
        n = min(len(from_image), len(from_depth))
        d = np.linalg.norm(np.array(from_image[:n]) - np.array(from_depth[:n]),
                           axis=1).mean() * 1000
        print(f'\nimage and depth disagree by {d:.1f}mm on average')
        if d > 8.0:
            print('  They measure the same tag, so a large gap means the '
                  'printed --tag-mm is wrong: the image estimate scales with '
                  'it and the depth one does not.')
    print()
    print('VERDICT: ' + ('the tag holds still in the flange frame, so the '
                         'calibration is consistent'
                         if ok else
                         'the tag WANDERS in the flange frame -- it is bolted '
                         'to the flange and cannot move, so the calibration '
                         'is wrong by roughly that much'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
