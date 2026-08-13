#!/usr/bin/env python3
"""Which joints actually move the image? Measure it, do not assume it.

The question
------------
With the camera on the flange, joint1 pans the view and joint5 tilts it, and
the servo's 2x2 image Jacobian is built on that. Move the camera to the first
arm piece and only joint1 is upstream of it -- so joints 2 through 6 move the
ARM without moving the VIEW, at any pose. That is a structural claim about the
kinematic chain, and it is worth confirming with pixels rather than reasoning,
because the failure it causes is silent.

Silent because the servo's usual guard does not fire. `Jog target pinned to
the measured pose` detects a joint that will not MOVE; joint5 moves perfectly
well here, it just has no effect on the image. The lag compensator then books
the missing image motion as the target fleeing, leads harder, and tilts
further -- the documented runaway, with no warning attached.

How
---
Global phase correlation between a frame before a jog and a frame after it.
No fiducial, no detector, no printed board: it measures how far the whole
scene shifted, which is exactly what the servo's error signal is made of.
Sub-pixel, and it reports a response value so a textureless scene shows up as
untrustworthy rather than as a confident zero.

Reported in the servo's own units -- normalised error per degree, where 1.0 is
"at the edge of frame" -- so the numbers can be compared against
`assumed_deg_per_error` directly.

    ./scripts/measure_jacobian.py                 # all six joints
    ./scripts/measure_jacobian.py --joints 1 5    # just the servo's pair
    ./scripts/measure_jacobian.py --delta 8       # bigger jog, clearer signal

Reads nothing it does not need: every jog is applied from, and returned to,
the pose the arm started at, and every intermediate pose goes through
CollisionGuard first.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src', 'mycobot_driver'))

from arm_broker import SOCK_PATH  # noqa: E402
from mycobot_driver.collision_guard import CollisionGuard  # noqa: E402

# A textureless scene phase-correlates to a confident-looking zero, which is
# indistinguishable from "this joint does not move the camera" -- the exact
# thing being measured. Below this response the reading is not reported as a
# number.
MIN_RESPONSE = 0.02


def ask(sock_path, req, timeout=90.0):
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
    except (ConnectionRefusedError, FileNotFoundError) as e:
        raise SystemExit(f'no broker on {sock_path} ({e}); '
                         'start ./scripts/arm_broker.py')
    with s:
        s.sendall((json.dumps(req) + '\n').encode())
        return json.loads(s.makefile().readline())


class Camera:
    """Colour frames from the D405, as float32 grayscale."""

    def __init__(self, width=640, height=480, fps=30, warmup=30):
        import numpy as np
        import pyrealsense2 as rs
        self.np = np
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipe.start(cfg)
        # Auto-exposure hunting between the two frames of a pair changes the
        # image brightness, which phase correlation does not care about -- but
        # a still-settling exposure also changes sharpness, which it does.
        for _ in range(warmup):
            self.pipe.wait_for_frames()
        self.w, self.h = width, height

    def gray(self, average=3):
        """Mean of a few frames, to keep sensor noise out of the correlation."""
        import cv2
        acc = None
        for _ in range(average):
            frames = self.pipe.wait_for_frames()
            c = frames.get_color_frame()
            img = self.np.asanyarray(c.get_data())
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(self.np.float32)
            acc = g if acc is None else acc + g
        return acc / average

    def close(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


def shift(a, b):
    """(dx, dy, response): how far the scene moved from frame a to frame b."""
    import cv2
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return dx, dy, resp


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--joints', nargs='*', type=int,
                    default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--delta', type=float, default=6.0,
                    help='degrees to jog each joint (default 6)')
    ap.add_argument('--trials', type=int, default=2,
                    help='+delta/-delta pairs per joint (default 2)')
    ap.add_argument('--speed', type=int, default=30)
    ap.add_argument('--settle', type=float, default=2.5)
    ap.add_argument('--sock', default=SOCK_PATH)
    args = ap.parse_args()

    guard = CollisionGuard()
    st = ask(args.sock, {'cmd': 'state'})
    home = st.get('angles')
    link = st.get('link', {})
    if home is None or st.get('age_ms', 1e9) > 5000:
        print('refusing: need a fresh pose to jog from and return to.')
        return 2
    print(f'link {link.get("fraction", 0):.0%}, '
          f'starting pose {[round(a, 1) for a in home]}')

    def goto(pose, settle):
        ok, why = guard.check(pose)
        if not ok:
            return False, why
        r = ask(args.sock, {'cmd': 'send_angles', 'angles': pose,
                            'speed': args.speed, 'force': True})
        if not r.get('ok'):
            return False, r.get('error', 'refused')
        time.sleep(settle)
        return True, ''

    cam = Camera()
    print(f'camera {cam.w}x{cam.h}\n')
    results = {}
    try:
        for j in args.joints:
            samples = []
            for sign in ([+1, -1] * args.trials)[:2 * args.trials]:
                ok, why = goto(list(home), args.settle)
                if not ok:
                    print(f'  joint{j}: cannot return to start -- {why}')
                    break
                before = cam.gray()
                pose = list(home)
                pose[j - 1] += sign * args.delta
                ok, why = goto(pose, args.settle)
                if not ok:
                    print(f'  joint{j} {sign:+d}: skipped -- {why}')
                    continue
                after = cam.gray()
                dx, dy, resp = shift(before, after)
                # Per +1 degree, regardless of which way this trial went.
                samples.append((dx / (sign * args.delta),
                                dy / (sign * args.delta), resp))
            goto(list(home), args.settle)
            if not samples:
                continue
            n = len(samples)
            mdx = sum(s[0] for s in samples) / n
            mdy = sum(s[1] for s in samples) / n
            mr = sum(s[2] for s in samples) / n
            results[j] = (mdx, mdy, mr, n)
            trust = '' if mr >= MIN_RESPONSE else '   (LOW RESPONSE)'
            print(f'  joint{j}: {mdx:+7.2f} px/deg x, {mdy:+7.2f} px/deg y, '
                  f'response {mr:.3f}, n={n}{trust}')
    finally:
        cam.close()

    print(f'\n{"joint":>6s} {"px/deg x":>10s} {"px/deg y":>10s} '
          f'{"norm/deg x":>11s} {"norm/deg y":>11s} {"resp":>6s}')
    for j, (dx, dy, r, n) in results.items():
        # The servo normalises per axis against the half-frame, so 1.0 is the
        # edge. That makes these directly comparable to assumed_deg_per_error.
        print(f'{j:6d} {dx:10.2f} {dy:10.2f} '
              f'{dx / (cam.w / 2):11.4f} {dy / (cam.h / 2):11.4f} {r:6.3f}')

    moved = {j: v for j, v in results.items()
             if max(abs(v[0]), abs(v[1])) > 1.0 and v[2] >= MIN_RESPONSE}
    print(f'\njoints that actually move the image: '
          f'{sorted(moved) if moved else "none detected"}')
    if len(moved) < 2:
        print('\nFewer than two joints steer the view, so there is no 2x2 to\n'
              'invert: the servo\'s image Jacobian is singular by construction\n'
              'and no choice of vertical_joint fixes it. skip_probe:=false\n'
              'will refuse for exactly this reason.')
    print('\nA joint downstream of the camera can still shift these numbers a\n'
          'little by moving the ARM through the field of view. Read a couple\n'
          'of px/deg as that, not as steering.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
