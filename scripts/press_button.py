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
import math
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

# Which panel is in front of the arm. These are NOT interchangeable: the lab's
# printed sheet has eight rows (door-open/close and fan/alarm below the
# numbers) and runs odd-left/even-right throughout, while PANEL_12 has six and
# reverses the 9/10 row. Picking the wrong one either refuses outright on the
# row count or, worse, numbers the buttons wrongly -- so it is an explicit
# choice with no clever auto-detection.
LAYOUTS = {'real': kf.PANEL_12,
           'printed': kf.PANEL_12_PRINTED,
           'printed-numbers': kf.PANEL_12_PRINTED_NUMBERS}


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
                    img, layout=LAYOUTS[args.layout],
                    depth_mm=dep, depth_range=(args.near, args.far),
                    # This layout covers only the top six rows of an
                    # eight-row panel, so it must be told to check that the
                    # rows it found really are the top ones.
                    top_anchored=args.layout == 'printed-numbers')
                intr = {'fx': it.fx, 'fy': it.fy, 'cx': it.ppx, 'cy': it.ppy}
                return found, img, dep, intr
            except kf.KeypadError as e:
                last = str(e)
        raise SystemExit(f'could not read the keypad in {args.tries} frames: '
                         f'{last}')
    finally:
        pipe.stop()


def snap(args, path):
    """One colour frame, saved. Used at the touch pose to SEE the contact."""
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height,
                      rs.format.bgr8, 30)
    try:
        pipe.start(cfg)
    except Exception as e:                                     # noqa: BLE001
        print(f'  (no photo: {type(e).__name__}: {str(e)[:60]})')
        return None
    try:
        # Let auto-exposure settle; the arm now fills a chunk of the frame and
        # the metering shifts when it arrives.
        for _ in range(20):
            fr = pipe.wait_for_frames(10000)
        img = np.asanyarray(fr.get_color_frame().get_data())
        cv2.imwrite(path, img)
        print(f'  photo at the touch pose -> {path}')
        return path
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
    ap.add_argument('--no-aim', action='store_true',
                    help='do not constrain the tool direction. With no tool '
                         'fitted this is usually what you want: aiming points '
                         'the (empty) tool axis at the panel and leaves no '
                         'metal at the button, which looks like the arm '
                         'stopping short. Unconstrained, the wrist falls where '
                         'the body can actually reach and contact happens -- '
                         'off-centre, but it happens.')
    ap.add_argument('--base-radius-mm', type=float, default=None,
                    help='shrink the modelled base column from its 60mm '
                         'radius, letting through poses that pass closer to '
                         'the base than the guard allows. The arm can hit '
                         'itself; use only with eyes on it')
    ap.add_argument('--max-aim-deg', type=float, default=20.0,
                    help='refuse to press when the tool is further off the '
                         'panel normal than this. The tip misses by '
                         'tool_length*sin(angle), so a crooked pose contacts '
                         'with the wrist instead (default: 20)')
    ap.add_argument('--photo', default=None, metavar='PNG',
                    help='where to write the photograph taken AT the touch '
                         'pose (default ~/hand_eye/press_touch_<button>.png). '
                         'This is the only unmediated check that the tool '
                         'reached the button -- every other figure printed '
                         'here is derived from FK and shares its errors')
    ap.add_argument('--layout', choices=sorted(LAYOUTS), default='real',
                    help='which panel is in front of the arm. "printed" is '
                         'the lab sheet: eight rows, odd-left throughout. '
                         '"real" is PANEL_12. Wrong choice presses the wrong '
                         'button, so there is no auto-detect (default: real)')
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
    # key=str because a layout may carry named buttons (open/close/fan/alarm)
    # alongside the numbered ones, and int and str do not compare.
    print(f'detected {len(found)} buttons: {sorted(found, key=str)}')
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

    # The base column the guard models is a cylinder of base_radius, and
    # shrinking it is the ONLY way to let a plan through that would otherwise
    # be refused for entering the base. Deliberately explicit and deliberately
    # not a boolean "off": the message that prompts this reports how far from
    # the axis the offending point is, so the operator can shrink the model to
    # just below that and see exactly how much they have given up.
    guard = (CollisionGuard() if args.base_radius_mm is None
             else CollisionGuard(base_radius=args.base_radius_mm / 1000.0))
    if args.base_radius_mm is not None:
        print(f'  base column modelled at {args.base_radius_mm:.0f}mm radius '
              f'instead of 60mm -- self-collision is NOT being checked below '
              f'that')
    try:
        plan = reach_planner.plan_press(
            p, tool_length_m=args.tool_mm / 1000.0,
            standoff_m=args.standoff_mm / 1000.0, guard=guard,
            approach_dir=approach, panel=panel,
            orientation=False if args.no_aim else 'auto')
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
    # A press that is far off the normal does not press. The tip swings
    # sideways by tool_length*sin(angle), so at 33deg a 27.7mm tool lands 15mm
    # off a button of ~14mm radius -- it misses, and what reaches the panel
    # first is the wrist. Photographed 2026-08-18: aim 33.3deg, every printed
    # figure nominal, and the cone was in mid-air beside the buttons while the
    # forearm lay across them. So refuse rather than report and continue.
    if off is not None and not args.no_aim and off > args.max_aim_deg:
        miss = (args.tool_mm or 1.0) * math.sin(math.radians(off))
        print(f'\nrefusing: {off:.1f}deg off the panel normal exceeds '
              f'--max-aim-deg {args.max_aim_deg:.0f}. A {args.tool_mm:.0f}mm '
              f'tool at that angle puts its tip ~{miss:.0f}mm to the side of '
              'the button, so the wrist reaches the panel before the tool '
              'does. Move the panel closer to the base or raise it -- see '
              'scripts/sweep_press_placement.py -- or pass a larger '
              '--max-aim-deg if you really mean to.')
        return 1

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
            # Report the TOOL TIP, not the flange origin. The flange distance
            # flatters a crooked pose: with the tool 33deg off the normal, a
            # 27.7mm tool puts its tip 27.7*sin(33) = 15mm SIDEWAYS of the
            # button -- off it entirely, on a ~14mm radius button -- while the
            # flange figure still reads a tidy "32.3mm", i.e. tool length plus
            # a few mm. Photographed 2026-08-18 doing exactly that: the cone
            # hung in mid-air beside the panel and the printout said success.
            axis = np.array(reach_planner.TOOL_AXIS, dtype=float)
            axis /= np.linalg.norm(axis)
            tip = T[:3, 3] + T[:3, :3] @ (axis * (args.tool_mm / 1000.0))
            print(f'{name}: tool tip {np.linalg.norm(tip - np.array(p))*1000:.1f}mm '
                  f'from the button (flange origin '
                  f'{np.linalg.norm(T[:3, 3] - np.array(p))*1000:.1f}mm)')
        return True

    if go('standoff', plan['standoff_deg']):
        if go('touch', plan['touch_deg']):
            time.sleep(0.6)
            # A photograph AT THE TOUCH POSE, which is the only direct
            # evidence that the tool is on the button. Every other number in
            # this script is derived: the flange distance comes from FK on the
            # measured angles, so an FK error, a calibration error or a tool
            # length error all read as success. The picture does not.
            snap(args, args.photo or os.path.expanduser(
                f'~/hand_eye/press_touch_{args.button}.png'))
        go('retract', plan['standoff_deg'])
    print('parking clear of the camera')
    request({'cmd': 'send_angles', 'angles': PARK, 'speed': args.speed,
             'force': True}, timeout=30)
    time.sleep(8.0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
