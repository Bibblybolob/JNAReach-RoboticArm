#!/usr/bin/env python3
"""Measure how far the tool sticks out past the flange, for --tool-mm.

    ./scripts/measure_tool.py

`reach_planner.plan_press` drives the FLANGE ORIGIN onto the target, so
anything mounted past it lands short or long by its own length. That is
`tool_length_m`, it defaults to 0, and a wrong non-zero default would silently
miss by exactly its own value -- so it has to be measured rather than guessed.

How
---
`TOOL_AXIS` is known: (-0.157, -0.011, -0.988) in flange coordinates, measured
from the flange marker over 8 poses. So put the wrist somewhere with clear air
in front of it, take the depth cloud, transform it into the FLANGE frame, and
the tool length is simply how far the arm's own material reaches along that
axis. No fixture, no touching anything.

Two things this deliberately does NOT do:

  - it does not touch a surface to find the tip. Contact measurement is more
    accurate and needs a known plane, a controlled approach and a force the
    arm cannot sense; this needs one frame.
  - it does not assume the tip is on the axis. It reports the spread of the
    furthest points too, because a presser mounted off-centre is a real thing
    and would otherwise show up later as a constant sideways miss.

Run it with the tool FITTED. Run it again after refitting -- this is a
property of the mounting, not of the arm.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mycobot_perception'))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mycobot_driver'))
sys.path.insert(0, _HERE)

import cv2  # noqa: E402

from arm_broker import request  # noqa: E402
from mycobot_driver.collision_guard import CollisionGuard, flange_transform  # noqa: E402
from mycobot_perception.reach_planner import TOOL_AXIS  # noqa: E402

CALIB = os.path.expanduser('~/hand_eye/eye_to_hand.json')
# Wrist held out in clear air, pointing across the camera's view so the tool
# is silhouetted rather than end-on. An end-on tool is a few pixels across and
# its tip is exactly the part depth loses first.
POSES = [
    [-35.0, 20.0, -95.0, 10.0, 0.0, 0.0],
    [-35.0, 10.0, -85.0, 25.0, 30.0, 0.0],
    [-20.0, 25.0, -100.0, 15.0, -30.0, 0.0],
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calibration', default=CALIB)
    ap.add_argument('--radius-mm', type=float, default=220.0,
                    help='how much of the arm around the flange to consider')
    ap.add_argument('--axis-mm', type=float, default=45.0,
                    help='how close to the tool axis material must be to '
                         'count as tool rather than arm')
    ap.add_argument('--speed', type=int, default=25)
    ap.add_argument('--save', default=None, metavar='PNG')
    args = ap.parse_args()

    import pyrealsense2 as rs

    if not os.path.isfile(os.path.expanduser(args.calibration)):
        print(f'no calibration at {args.calibration}')
        return 1
    calib = json.load(open(os.path.expanduser(args.calibration)))
    X = np.array(calib['camera_to_base'], dtype=float)
    u = np.array(TOOL_AXIS, dtype=float)
    u /= np.linalg.norm(u)
    guard = CollisionGuard()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    prof = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    sc = prof.get_device().first_depth_sensor().get_depth_scale() * 1000.0
    it = (prof.get_stream(rs.stream.color).as_video_stream_profile()
          .get_intrinsics())

    reaches = []
    try:
        for k, q in enumerate(POSES):
            ok, why = guard.check(q)
            if not ok:
                print(f'{k}: guard refused -- {why}')
                continue
            request({'cmd': 'send_angles', 'angles': q, 'speed': args.speed,
                     'force': True}, timeout=30)
            time.sleep(7.5)
            st = request({'cmd': 'state'}, timeout=20)
            if not st or not st.get('angles'):
                print(f'{k}: no joint reading')
                continue
            meas = list(st['angles'])
            for _ in range(25):
                fr = align.process(pipe.wait_for_frames(10000))
            img = np.asanyarray(fr.get_color_frame().get_data())
            dep = (np.asanyarray(fr.get_depth_frame().get_data())
                   .astype(float) * sc)
            if args.save and k == 0:
                cv2.imwrite(args.save, img)

            T = np.array(flange_transform(meas))
            origin = T[:3, 3]

            ys, xs = np.nonzero(dep > 0)
            z = dep[ys, xs] / 1000.0
            P = np.stack([(xs - it.ppx) * z / it.fx,
                          (ys - it.ppy) * z / it.fy, z], 1)
            Pb = (X[:3, :3] @ P.T).T + X[:3, 3]
            near = Pb[np.linalg.norm(Pb - origin, axis=1)
                      < args.radius_mm / 1000.0]
            if len(near) < 300:
                print(f'{k}: only {len(near)} points near the flange')
                continue

            F = (T[:3, :3].T @ (near - origin).T).T      # flange frame, metres
            along = F @ u
            lat_all = np.linalg.norm(F - np.outer(along, u), axis=1)
            # ONLY material near the axis LINE counts. Without this the answer
            # is set by the forearm, which reaches a long way along the tool
            # axis while sitting far off it -- a bare arm measured 97.5mm that
            # way, with its "tip" 166mm off-axis. A presser is a slender thing
            # close to the axis, so require that.
            on_axis = lat_all < args.axis_mm / 1000.0
            if on_axis.sum() < 60:
                print(f'{k}: only {int(on_axis.sum())} points within '
                      f'{args.axis_mm:.0f}mm of the tool axis -- nothing '
                      'slender is mounted, or it is out of view')
                continue
            along = along[on_axis]
            F = F[on_axis]
            # 99.5th rather than the max: one stray depth pixel beyond the tool
            # would otherwise set the answer.
            reach = float(np.percentile(along, 99.5)) * 1000.0
            tipish = F[along > np.percentile(along, 99.0)]
            lateral = np.linalg.norm(tipish - np.outer(tipish @ u, u), axis=1)
            reaches.append(reach)
            print(f'{k}: material reaches {reach:6.1f}mm along the tool axis, '
                  f'tip off-axis by {np.median(lateral)*1000:.1f}mm')
    finally:
        pipe.stop()

    if len(reaches) < 2:
        print('\nnot enough usable views -- is the wrist in clear view?')
        return 1
    r = np.array(reaches)
    spread = float(r.max() - r.min())
    print(f'\ntool reach: {r.mean():.1f}mm mean, spread {spread:.1f}mm '
          f'over {len(r)} poses')

    # REFUSE rather than emit a number nobody should use. A real tool is rigid
    # and reads the same from every angle; disagreement across poses means the
    # thing being measured is not a tool. Measured on a BARE arm: 146mm of
    # spread, because with nothing slender mounted the answer is set by
    # whichever bit of flange body happens to fall near the axis, and that
    # changes with the view. Printing a --tool-mm there would put a confident
    # wrong offset straight into every press.
    if spread > 8.0:
        print()
        print(f'REFUSING to report a tool length: {spread:.0f}mm of '
              'disagreement between poses.')
        print('A rigid tool measures the same from every angle. This does not,')
        print('which means no slender tool is fitted and what was measured is')
        print('the flange body seen from three directions. Fit the presser and')
        print('run this again.')
        return 1

    print(f'\n    ./scripts/press_button.py 5 --tool-mm {r.mean():.0f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
