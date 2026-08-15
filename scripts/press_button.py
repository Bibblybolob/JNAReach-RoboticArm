#!/usr/bin/env python3
"""Press a numbered button on the panel: see it, place it, plan it, touch it.

    ./scripts/press_button.py 5              # press button 5
    ./scripts/press_button.py 5 --dry-run    # plan and print, command nothing
    ./scripts/press_button.py --look         # just detect, no motion at all

The whole chain in one command:

    keypad_finder  ->  depth  ->  target_in_base  ->  plan_press  ->  send_angles

Each stage refuses rather than guessing, because every one of them can produce
a confident wrong answer that ends with the arm driving somewhere real:

  - the finder REFUSES a frame whose row count does not match the layout,
    since a missed row shifts every number and presses the wrong floor
  - target_in_base REFUSES a depth outside the D405's usable band
  - plan_press REFUSES a standoff/touch pair that changes arm configuration,
    which would swing the arm through the panel on the way

Resolution matters and is not a detail. Measured 2026-08-14: the finder gets
all twelve buttons on 10/12 frames at 1280x720 and only 2/6 at 640x480, where
the buttons are r~12px and rows drop out. Hence the default.

The two known limits, both stated in the output rather than hidden:

  - `--tool-mm` defaults to 0, so the FLANGE ORIGIN is driven onto the button
    and whatever protrudes past it lands short or long by that much. Measured
    ~13mm on the first presses. Pass the real number once measured.
  - the tool cannot generally be held square to the panel: exact aim costs
    ~25mm of position at most placements. This aims as squarely as the arm
    allows while still hitting the button, and prints how square that was.
    ~21deg is typical; under 5deg needs the panel at 120-150mm and the button
    150-200mm up.
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
from mycobot_driver.collision_guard import (  # noqa: E402
    CollisionGuard, flange_transform)
from mycobot_perception import keypad_finder as kf  # noqa: E402
from mycobot_perception import reach_planner  # noqa: E402
from mycobot_perception.target_in_base import TargetError, target_in_base  # noqa: E402

CALIB = os.path.expanduser('~/hand_eye/eye_to_hand.json')
# Clear of the camera's view of the panel: keypad 0.1% blocked against 27% at
# home, measured 2026-08-14 by counting depth pixels nearer than the panel.
PARK = [-60.0, 30.0, -110.0, 15.0, 0.0, 0.0]


def look(args):
    """Grab one frame and locate the buttons. Returns (found, colour, depth)."""
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height,
                      rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, args.width, args.height,
                      rs.format.z16, 30)
    prof = pipe.start(cfg)
    try:
        align = rs.align(rs.stream.color)
        scale = prof.get_device().first_depth_sensor().get_depth_scale() * 1000
        it = (prof.get_stream(rs.stream.color).as_video_stream_profile()
              .get_intrinsics())
        # Retry across frames: the finder REFUSES an incomplete view rather
        # than mislabelling, so a failure is a retry and not an error.
        last = 'no frames'
        for attempt in range(args.tries):
            for _ in range(15):
                fr = align.process(pipe.wait_for_frames(10000))
            img = np.asanyarray(fr.get_color_frame().get_data())
            dep = (np.asanyarray(fr.get_depth_frame().get_data())
                   .astype(float) * scale)
            try:
                found = kf.find_labelled(
                    img, depth_mm=dep, depth_range=(args.near, args.far))
                intr = {'fx': it.fx, 'fy': it.fy, 'cx': it.ppx, 'cy': it.ppy}
                return found, img, dep, intr
            except kf.KeypadError as e:
                last = str(e)
        raise SystemExit(f'could not read the keypad in {args.tries} frames: '
                         f'{last}')
    finally:
        pipe.stop()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('button', nargs='?', type=int,
                    help='which numbered button to press')
    ap.add_argument('--look', action='store_true',
                    help='detect and report only; command no motion')
    ap.add_argument('--dry-run', action='store_true',
                    help='plan and print the joint angles, command no motion')
    ap.add_argument('--tool-mm', type=float, default=0.0,
                    help='flange origin to the contact face, in mm. 0 drives '
                         'the ORIGIN onto the button, which lands short or '
                         'long by whatever sticks out past it')
    ap.add_argument('--standoff-mm', type=float, default=40.0)
    ap.add_argument('--speed', type=int, default=20)
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--near', type=float, default=280.0)
    ap.add_argument('--far', type=float, default=470.0)
    ap.add_argument('--tries', type=int, default=5)
    ap.add_argument('--calibration', default=CALIB)
    ap.add_argument('--save', default=None, metavar='PNG',
                    help='write the annotated detection here')
    args = ap.parse_args()

    if args.button is None and not args.look:
        ap.error('give a button number, or --look')

    found, img, dep, intr = look(args)
    print(f'detected {len(found)} buttons: {sorted(found)}')
    if args.save:
        vis = img.copy()
        for n, (x, y, a, b) in found.items():
            cv2.ellipse(vis, (int(x), int(y)), (int(a), int(b)), 0, 0, 360,
                        (0, 255, 0), 2)
            cv2.putText(vis, str(n), (int(x) - 12, int(y) - int(b) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(args.save, vis)
        print(f'wrote {args.save}')
    if args.look:
        return 0

    if args.button not in found:
        print(f'button {args.button} is not among the detected ones')
        return 1

    u, v = found[args.button][0], found[args.button][1]
    patch = dep[int(v) - 4:int(v) + 5, int(u) - 4:int(u) + 5]
    val = patch[patch > 0]
    if val.size < 5:
        print(f'no usable depth at button {args.button} ({u:.0f},{v:.0f})')
        return 1
    d = float(np.median(val))
    print(f'button {args.button}: pixel ({u:.0f},{v:.0f}), depth {d:.0f}mm '
          f'({val.size}/81 valid)')

    if not os.path.isfile(os.path.expanduser(args.calibration)):
        print(f'no calibration at {args.calibration} -- run '
              'calibrate_hand_eye.py, then verify_calibration.py')
        return 1
    calib = json.load(open(os.path.expanduser(args.calibration)))
    try:
        p = target_in_base(u, v, d, intr, calib, 0.0)
    except TargetError as e:
        print(f'refused: {e}')
        return 1
    print(f'  base frame ({p[0]*1000:+.1f}, {p[1]*1000:+.1f}, '
          f'{p[2]*1000:+.1f})mm, {np.linalg.norm(p)*1000:.0f}mm from origin')

    # The panel's own normal, from the buttons themselves. plan_press
    # otherwise assumes a radial-from-base approach, which on this panel is
    # ~30deg off the truth and costs exactly that much squareness at contact
    # (37.5deg against 21deg, measured 2026-08-15). The twelve buttons are
    # coplanar by construction, so they define the plane better than any
    # assumption -- and if fewer than four convert cleanly, fall back to the
    # radial default rather than fitting a plane to noise.
    approach = None
    panel = None
    pts = []
    for n, (bu, bv, _a, _b) in found.items():
        pw = dep[int(bv) - 3:int(bv) + 4, int(bu) - 3:int(bu) + 4]
        pv = pw[pw > 0]
        if pv.size < 5:
            continue
        try:
            pts.append(target_in_base(bu, bv, float(np.median(pv)), intr,
                                      calib, 0.0))
        except TargetError:
            continue
    if len(pts) >= 4:
        P = np.array(pts)
        _u, _s, vt = np.linalg.svd(P - P.mean(axis=0))
        nrm = vt[2] / np.linalg.norm(vt[2])
        # Point it INTO the panel, i.e. away from the base.
        if nrm @ P.mean(axis=0) < 0:
            nrm = -nrm
        approach = [float(v) for v in nrm]
        # Hand the plane to the planner as a keep-out. Without it nothing in
        # the chain knows the panel exists -- obstacles.yaml is empty -- and
        # the solver is free to choose a configuration that lays the forearm
        # across the buttons while the flange sits neatly on the target.
        panel = (P.mean(axis=0), nrm)
        print(f'  panel normal from {len(pts)} buttons: '
              f'({approach[0]:+.2f},{approach[1]:+.2f},{approach[2]:+.2f})')

    guard = CollisionGuard()
    try:
        plan = reach_planner.plan_press(
            p, tool_length_m=args.tool_mm / 1000.0,
            standoff_m=args.standoff_mm / 1000.0, guard=guard,
            approach_dir=approach, panel=panel)
    except reach_planner.ReachError as e:
        print(f'refused: {e}')
        return 1

    if panel is not None:
        from mycobot_perception.reach_planner import panel_clearance
        for nm_ in ('standoff', 'touch'):
            cl = panel_clearance(plan[f'{nm_}_deg'], *panel) * 1000
            print(f'  {nm_}: arm body clears the panel by {cl:.0f}mm')

    off = plan.get('touch_off_normal_deg')
    if off is not None:
        print(f'  aim: {off:.1f}deg off the panel normal, '
              f'position error {plan.get("touch_pos_err_mm", 0):.2f}mm')
    print(f'  standoff {[round(a, 1) for a in plan["standoff_deg"]]}')
    print(f'  touch    {[round(a, 1) for a in plan["touch_deg"]]}')
    if args.tool_mm == 0.0:
        print('  NOTE --tool-mm is 0, so the FLANGE ORIGIN goes on the '
              'button; anything protruding past it contacts off by that much')
    if args.dry_run:
        print('\ndry run -- nothing commanded')
        return 0

    h = request({'cmd': 'health'}, timeout=15)
    frac = h.get('link', {}).get('fraction', 0) if h else 0
    if frac < 0.6:
        print(f'link is {frac*100:.0f}% valid -- too poor to move safely')
        return 1

    def go(name, q):
        ok, why = guard.check(list(q))
        if not ok:
            print(f'{name}: guard refused -- {why}')
            return False
        request({'cmd': 'send_angles', 'angles': [float(a) for a in q],
                 'speed': args.speed, 'force': True}, timeout=30)
        time.sleep(7.5)
        st = request({'cmd': 'state'}, timeout=20)
        if st and st.get('angles'):
            T = np.array(flange_transform(list(st['angles'])))
            print(f'{name}: flange {np.linalg.norm(T[:3, 3] - np.array(p))*1000:.1f}mm '
                  f'from the button')
        return True

    if go('standoff', plan['standoff_deg']):
        if go('touch', plan['touch_deg']):
            time.sleep(0.6)
        go('retract', plan['standoff_deg'])
    print('parking clear of the camera')
    request({'cmd': 'send_angles', 'angles': PARK, 'speed': args.speed,
             'force': True}, timeout=30)
    time.sleep(8.0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
