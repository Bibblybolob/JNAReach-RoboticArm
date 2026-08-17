#!/usr/bin/env python3
"""Press a NAMED button on the panel: see it, place it, plan it, touch it.

    ./scripts/press_button.py up --tool-mm 27.7   # the hall call
    ./scripts/press_button.py 5  --tool-mm 27.7   # a floor, by its legend
    ./scripts/press_button.py 5  --dry-run        # plan and print, no motion
    ./scripts/press_button.py --look              # detect only, no motion

The target is NAMED, not numbered: up, down, help, open, close, or a floor
legend (5, B1, G, LG). `up` and `down` are this project's top priority and
were not expressible at all while this took an integer.

The whole chain in one command:

    detector  ->  depth  ->  target_in_base  ->  plan_press  ->  arm_motion

Each stage refuses rather than guessing, because every one of them can produce
a confident wrong answer that ends with the arm driving somewhere real:

  - the model REFUSES to name a legend it is not sure of, so an unread button
    shows as `floor?` and cannot be asked for; the geometric finder REFUSES a
    frame whose row count does not match the layout, since a missed row shifts
    every number and presses the wrong floor
  - target_in_base REFUSES a depth outside the D405's usable band
  - plan_press REFUSES a standoff/touch pair that changes arm configuration,
    which would swing the arm through the panel on the way

Resolution matters and is not a detail. Measured 2026-08-14: the finder gets
all twelve buttons on 10/12 frames at 1280x720 and only 2/6 at 640x480, where
the buttons are r~12px and rows drop out. Hence the default.

The two known limits, both stated in the output rather than hidden:

  - `--tool-mm` defaults to 0, so the FLANGE ORIGIN is driven onto the button
    and whatever protrudes past it lands short or long by that much. The
    fitted presser is 27.7mm; pass it every time.
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

import arm_motion  # noqa: E402
from arm_broker import request  # noqa: E402

# How far short of the touch pose still counts as "the button stopped me".
#
# Above arm_motion.ARRIVE_TOL_DEG (1.0), because inside that is ordinary
# arrival. The upper end is a judgement rather than a measurement: a button
# has a few millimetres of travel, which at this arm's reach is a couple of
# degrees at the driven joints. Beyond that the arm was stopped by something
# that is not a button, and calling that a successful press would be the worst
# possible lie -- it reports a floor as selected that was never pressed.
CONTACT_MAX_DEG = 4.0
from mycobot_driver.collision_guard import (  # noqa: E402
    CollisionGuard, flange_transform)
from mycobot_perception import keypad_finder as kf  # noqa: E402
from mycobot_perception import reach_planner  # noqa: E402
from mycobot_perception.target_in_base import TargetError, target_in_base  # noqa: E402

CALIB = os.path.expanduser('~/hand_eye/eye_to_hand.json')
# Clear of the camera's view of the panel: keypad 0.1% blocked against 27% at
# home, measured 2026-08-14 by counting depth pixels nearer than the panel.
PARK = [-60.0, 30.0, -110.0, 15.0, 0.0, 0.0]


_MODELS = {}


def find_with_model(img, args):
    """Locate buttons with the trained two-stage models.

    Returns the same {label: (x, y, half_w, half_h)} shape `keypad_finder`
    produces, so it drops into the rest of the chain unchanged. Labels are
    what a caller would ask for: a floor legend (`5`, `B1`, `G`), or `up` /
    `down` / `help` / `open` / `close`.

    This is the difference between the lab and a real elevator.
    `keypad_finder` solves the printed panel GEOMETRICALLY -- equal ellipses
    on a lattice -- because no model could see it, and it is hard-coded to a
    12-button layout that `label()` refuses to fit anything else. That is
    exactly right for the bench and useless in a lift.

    A button whose legend was not read confidently comes back as `floor?`
    (numbered if there are several). It still counts for the panel plane fit
    and still shows in --look, but asking for `5` will not match it, so the
    arm cannot press it BELIEVING it is 5.
    """
    import torch
    torch.backends.cudnn.enabled = False
    from ultralytics import YOLO
    from mycobot_perception.button_classes import (
        DETECT_CLASSES, FORBIDDEN, UNREADABLE)

    if not _MODELS:
        for key, path, fallback, task in (
                ('det', args.detect_weights, 'button_detect.pt', 'detect'),
                ('rdr', args.read_weights, 'button_read.pt', 'classify')):
            mdl = None
            for cand in (path, os.path.join(_ROOT, fallback)):
                if not cand:
                    continue
                try:
                    mdl = YOLO(cand, task=task)
                    print(f'  {key}: {os.path.basename(cand)}')
                    break
                except Exception as e:  # noqa: BLE001
                    print(f'  {key}: {cand} would not load ({e})')
            _MODELS[key] = mdl
    det, rdr = _MODELS.get('det'), _MODELS.get('rdr')
    if det is None:
        raise kf.KeypadError('no button detector available')

    res = det.predict(img, conf=args.conf, device=args.device,
                      verbose=False)[0]
    if res.boxes is None or not len(res.boxes):
        raise kf.KeypadError('the detector found no buttons')

    H, W = img.shape[:2]
    xy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)

    # Read the legend off every floor crop, in one batch. A panel has a dozen
    # of them and per-crop calls would pay the inference overhead a dozen
    # times for images that are 128px.
    legend = {}
    if rdr is not None:
        crops, idx = [], []
        for i, (x1, y1, x2, y2) in enumerate(xy):
            if DETECT_CLASSES[cls[i]] != 'floor':
                continue
            pw, ph = (x2 - x1) * args.crop_pad, (y2 - y1) * args.crop_pad
            cr = img[int(max(0, y1 - ph)):int(min(H, y2 + ph)),
                     int(max(0, x1 - pw)):int(min(W, x2 + pw))]
            if cr.size:
                crops.append(cr)
                idx.append(i)
        if crops:
            for i, r in zip(idx, rdr.predict(crops, device=args.device,
                                             verbose=False)):
                conf = float(r.probs.top1conf)
                nm = rdr.names[int(r.probs.top1)]
                if conf >= args.read_min and nm != UNREADABLE:
                    legend[i] = nm

    found, unread = {}, 0
    for i, (x1, y1, x2, y2) in enumerate(xy):
        kind = DETECT_CLASSES[cls[i]]
        if kind in FORBIDDEN:
            # keyhole, stop and other. All real, all worth knowing about, and
            # none of them may become a press target -- so none is given a
            # label anyone can ask for. Withheld, not merely deprioritised.
            continue
        if kind == 'floor':
            lab = legend.get(i)
            if lab is None:
                unread += 1
                lab = 'floor?' if unread == 1 else f'floor?{unread}'
        else:
            lab = kind
        # Two of a kind on one panel (two `help` buttons, say) must not
        # overwrite each other -- the plane fit wants both.
        if lab in found:
            n = 2
            while f'{lab}#{n}' in found:
                n += 1
            lab = f'{lab}#{n}'
        found[lab] = (float((x1 + x2) / 2), float((y1 + y2) / 2),
                      float((x2 - x1) / 2), float((y2 - y1) / 2))
    if not found:
        raise kf.KeypadError('buttons detected, but none of a pressable kind')
    return found


def detect(img, dep, args):
    """Locate the buttons with whichever detector was asked for.

    `auto` runs the MODEL first and falls back to the geometric finder,
    because the two solve different panels and neither supersedes the other:
    the model generalises to real lifts, and keypad_finder is the only thing
    that has ever worked on the lab's printed panel -- measured 2026-08-14,
    the shipped detector scored zero on it at every scale and threshold, and
    YOLO-World found nothing across three prompt sets.

    Falling back is safe in the direction that matters. keypad_finder refuses
    a frame it cannot fit rather than mislabelling one, so the worst case is
    a retry.
    """
    if args.detector in ('model', 'auto'):
        try:
            return find_with_model(img, args)
        except kf.KeypadError:
            if args.detector == 'model':
                raise
    labelled = kf.find_labelled(
        img, depth_mm=dep, depth_range=(args.near, args.far))
    # Geometric labels are ints; everything downstream compares strings, so
    # that `up` and `B1` are askable in exactly the same way as `5`.
    return {str(k): v for k, v in labelled.items()}


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
                found = detect(img, dep, args)
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
    # A STRING, not an int. The whole point of the model detector is that the
    # hall call is askable: `up` and `down` are the project's top priority and
    # were not expressible at all while this took a number. `B1`, `G` and `LG`
    # come along for free.
    ap.add_argument('button', nargs='?', type=str,
                    help="which button: a floor legend (5, B1, G), or "
                         "up / down / help / open / close")
    ap.add_argument('--detector', choices=('auto', 'model', 'geometric'),
                    default='auto',
                    help='auto (default) tries the trained model and falls '
                         'back to the geometric finder, which is the only '
                         "thing that works on the lab's printed panel")
    ap.add_argument('--detect-weights',
                    default=os.path.join(_ROOT, 'button_detect.engine'),
                    help='stage A. TensorRT engine, or a .pt')
    ap.add_argument('--read-weights',
                    default=os.path.join(_ROOT, 'button_read.engine'),
                    help='stage B, the legend reader')
    ap.add_argument('--conf', type=float, default=0.5,
                    help='stage A detection threshold')
    # Must match the node's reader_min_confidence. Measured 2026-08-17 over
    # 1302 detected crops: 0.75 gave 11.7% confidently WRONG legends, 0.95
    # gives 6.8%. A decline costs a retry; a wrong read presses another floor.
    ap.add_argument('--read-min', type=float, default=0.95,
                    help='below this the legend is not published at all')
    ap.add_argument('--crop-pad', type=float, default=0.12,
                    help="must match the builder's --crop-pad")
    ap.add_argument('--device', default='0')
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

    # Case-insensitive, so `UP` and `b1` work. Matched against the exact
    # labels only -- never a prefix or a fuzzy match, because `1` matching
    # `19` is a wrong floor arrived at by string handling.
    want = str(args.button).strip().lower()
    key = next((k for k in found if k.lower() == want), None)
    if key is None:
        askable = sorted(k for k in found if not k.startswith('floor?'))
        unread = sum(1 for k in found if k.startswith('floor?'))
        print(f'button {args.button!r} is not among the detected ones')
        print(f'  askable: {askable}')
        if unread:
            # Worth separating: these ARE buttons, found and located. The
            # reader would not commit to their legend, so they are deliberately
            # unaskable rather than missing.
            print(f'  plus {unread} button(s) whose legend was not read '
                  f'confidently (--read-min {args.read_min})')
        return 1

    u, v = found[key][0], found[key][1]
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
    if args.dry_run:
        print('\ndry run -- nothing commanded')
        return 0

    h = request({'cmd': 'health'}, timeout=15)
    frac = h.get('link', {}).get('fraction', 0) if h else 0
    if frac < 0.6:
        print(f'link is {frac*100:.0f}% valid -- too poor to move safely')
        return 1

    def report(name):
        st = request({'cmd': 'state'}, timeout=20)
        if st and st.get('angles'):
            T = np.array(flange_transform(list(st['angles'])))
            print(f'  {name}: flange '
                  f'{np.linalg.norm(T[:3, 3] - np.array(p))*1000:.1f}mm '
                  'from the button')

    t0 = time.monotonic()

    # Move to the standoff and WAIT for it, rather than sleeping a guess.
    # Every move below is closed-loop; see scripts/arm_motion.py for why the
    # fixed 7.5s sleeps this replaces were wrong in both directions.
    arrived, _err = arm_motion.move_to(plan['standoff_deg'], args.speed,
                                       guard=guard, name='standoff')
    if not arrived:
        print('did not reach the standoff pose; not approaching the panel')
        return 1
    report('standoff')

    # The approach itself: standoff -> button as ONE streamed straight line.
    path = plan.get('path_deg') or [plan['standoff_deg'], plan['touch_deg']]
    status, err = arm_motion.stream_path(
        path[1:], args.speed, guard=guard, name='approach')

    # A REFUSAL is not a press, and it is not a blockage either -- nothing was
    # commanded and the arm has not left the standoff. Say so and stop, rather
    # than falling through to contact reporting (which read `inf` degrees of
    # shortfall as "something stopped the arm") and then retracting along a
    # path whose waypoints were just rejected.
    if status == 'refused':
        print('  the approach was refused before any motion; the arm is still '
              'at the standoff pose')
        arm_motion.move_to(PARK, args.speed, name='park', tol=3.0)
        return 1

    report('touch')

    # CONTACT, inferred from the arm failing to finish the last millimetres.
    #
    # Nothing here senses force, so "did it actually press" has to come from
    # somewhere else. A button that is being pressed resists: the servos stall
    # a fraction short of the commanded pose and the wait times out just
    # outside tolerance. So a small shortfall at the TOUCH pose specifically
    # is evidence of contact, not a failure -- while arriving exactly means
    # the tool met nothing, which on a plan that put it on the button means
    # the button was not where the depth said it was.
    #
    # The window matters. Below ARRIVE_TOL_DEG is ordinary arrival. Far
    # outside it is the arm being blocked by something that is not a button,
    # or not moving at all, and that must not be read as a successful press.
    if status == 'arrived':
        print(f'  contact: NONE detected -- the tool reached the touch pose '
              f'exactly ({err:.2f}deg), so it met no resistance. Either the '
              'button is further away than the depth reading, or the tool is '
              'shorter than --tool-mm says.')
    else:
        # Short of the target is necessary but NOT sufficient. The wait also
        # returns short when the budget simply expired on a slow link, and
        # calling that a press would report a floor as selected that was never
        # pressed. A stalled arm is STATIONARY; a slow one is still closing.
        still = arm_motion.is_stationary()
        if still is False:
            print(f'  contact: NOT pressed -- {err:.2f}deg short but the arm '
                  'is STILL MOVING, so the wait expired rather than the button '
                  'stopping it. Raise the speed or the budget.')
        elif still is None:
            print(f'  contact: UNKNOWN -- {err:.2f}deg short, and the arm '
                  'could not be read to tell a stall from a slow move')
        elif err <= CONTACT_MAX_DEG:
            print(f'  contact: pressed -- stopped {err:.2f}deg short of the '
                  'touch pose and held there, which is the button resisting')
        else:
            print(f'  contact: BLOCKED -- stopped {err:.2f}deg short, far '
                  'outside the press window. Something other than the button '
                  'stopped the arm; check the panel clearance above.')

    time.sleep(0.4)

    # Retract back along the same line, so the tool leaves the way it came in
    # rather than sweeping sideways across the neighbouring buttons.
    arm_motion.stream_path(list(reversed(path[:-1])), args.speed, guard=guard,
                           name='retract')

    print(f'press cycle: {time.monotonic() - t0:.1f}s')
    print('parking clear of the camera')
    arm_motion.move_to(PARK, args.speed, name='park', tol=3.0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
