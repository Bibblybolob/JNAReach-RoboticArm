#!/usr/bin/env python3
"""Move the arm to its least-loaded pose and leave it there.

Why this exists
---------------
The Atom's crash-loop is load-driven -- measured 2026-08-12, valid replies
went 18% -> 96% from unloading the arm alone, with no reflash and no power
cycle. So "park it before doing anything else" is a real diagnostic step and
not tidying up, and it wants to be one command rather than a hand-typed
send_angles at a moment when the link is already unreliable.

Which pose, and why not the one called "home"
---------------------------------------------
Computed rather than chosen, using the driver's own forward kinematics: lump
each link's mass at its midpoint, and take the horizontal distance from each
joint to the centre of mass of everything distal to it. That is the moment
gravity actually applies and the servo has to hold.

    pose                          peak joint load   total
    all-zeros (upright)                0.027        0.131
    operating home                     0.051        0.165
    mechanical home ("folded")         0.056        0.218
    camera-forward                     0.089        0.357

All-zeros wins because every link stacks vertically over the base, so the
moment arms go to nearly nothing. **The pose named "mechanical home" in
docs/recorded_poses.md is described there as folding the arm over, and it is
the WORST of the four** -- joint2 at +90 puts the upper arm horizontal, which
is the largest moment arm on the arm. Folding by eye picks it; folding by
arithmetic does not. Worth knowing before "fold the arm to unload it" gets
followed literally.

All-zeros is also the calibration reference pose, so parking here doubles as a
visual check that the servo zeros are still true -- if the arm does not look
straight, the zeros are gone (see calibrate_zero.py).

How it moves
------------
In guarded steps, because this protocol has no checksum: a corrupted
SEND_ANGLES is simply a joint angle the arm accepts and moves to. Every
intermediate pose goes through CollisionGuard before it is sent, so a mangled
frame that lands somewhere illegal is at least bounded by the step size, and
the settle time scales with how far each step actually travels.

    ./scripts/park_arm.py                    # to all-zeros
    ./scripts/park_arm.py --steps 8          # gentler
    ./scripts/park_arm.py --target 0 -30 -30 0 -30 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src', 'mycobot_driver'))

from arm_broker import SOCK_PATH  # noqa: E402
from mycobot_driver.collision_guard import CollisionGuard  # noqa: E402

# Every link stacked over the base. See the module docstring for the numbers.
PARKED = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# Measured 52.0 deg/s at speed 100, 29.0 at 30 (see CLAUDE.md). Slow on
# purpose: the point of this move is to reduce load, not to demonstrate speed.
SPEED = 30
DEG_PER_S = 29.0


def ask(sock_path: str, req: dict, timeout: float = 30.0) -> dict:
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
    except (ConnectionRefusedError, FileNotFoundError) as e:
        raise SystemExit(
            f'no broker on {sock_path} ({e}). This script deliberately will '
            'not open the port itself -- start ./scripts/arm_broker.py.')
    with s:
        s.sendall((json.dumps(req) + '\n').encode())
        return json.loads(s.makefile().readline())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', nargs=6, type=float, default=PARKED,
                    metavar='DEG', help='pose to park at (default all-zeros)')
    ap.add_argument('--steps', type=int, default=5,
                    help='how many intermediate poses (default 5)')
    ap.add_argument('--speed', type=int, default=SPEED)
    ap.add_argument('--sock', default=SOCK_PATH)
    ap.add_argument('--force', action='store_true',
                    help='command even if the link is below the health bar')
    args = ap.parse_args()

    guard = CollisionGuard()
    ok, why = guard.check(args.target)
    if not ok:
        print(f'refusing: the target pose itself is unsafe -- {why}')
        return 2

    st = ask(args.sock, {'cmd': 'state'})
    start = st.get('angles')
    link = st.get('link', {})
    print(f'link {link.get("fraction", 0):.0%} valid '
          f'({link.get("valid")}/{link.get("window")}), '
          f'pose is {st.get("age_ms", 0)}ms old')
    if start is None:
        print('refusing: the broker has never read a pose, so there is no '
              'starting point to interpolate from.')
        return 2
    if st.get('age_ms', 0) > 5000:
        # Interpolating from a stale pose is how a "small" step becomes a
        # large one. 93deg of unexpected travel has already cost a session.
        print(f'refusing: the last pose is {st["age_ms"]}ms old. Steps sized '
              'against a stale pose are not small steps.')
        return 2

    print(f'  from {[round(a, 1) for a in start]}')
    print(f'    to {[round(a, 1) for a in args.target]}')
    travel = max(abs(t - s) for t, s in zip(args.target, start))
    print(f'  largest joint move {travel:.0f}deg, in {args.steps} steps\n')

    prev = list(start)
    for k in range(1, args.steps + 1):
        f = k / args.steps
        pose = [s + (t - s) * f for s, t in zip(start, args.target)]
        ok, why = guard.check(pose)
        if not ok:
            print(f'refusing at step {k}: {why}')
            print('  the straight-line path between these poses is not safe; '
                  'move it in parts, or by hand.')
            return 2
        r = ask(args.sock, {'cmd': 'send_angles', 'angles': pose,
                            'speed': args.speed,
                            **({'force': True} if args.force else {})})
        if not r.get('ok'):
            print(f'step {k} refused: {r.get("error")}')
            return 1
        # Scale the wait to the distance actually covered, plus a margin. A
        # fixed settle photographs the arm before it arrives on the long steps
        # and wastes time on the short ones.
        step_deg = max(abs(p - q) for p, q in zip(pose, prev))
        settle = step_deg / DEG_PER_S * (100.0 / args.speed) * 0.6 + 0.8
        time.sleep(settle)
        got = ask(args.sock, {'cmd': 'state'})
        a, age = got.get('angles'), got.get('age_ms', 0)
        if a and age < 3000:
            err = max(abs(x - y) for x, y in zip(a, pose))
            print(f'  step {k}/{args.steps}  waited {settle:.1f}s  '
                  f'off by {err:.1f}deg')
        else:
            # Not a failure. Reads are the direction that is broken here, and
            # the write almost certainly landed -- that asymmetry is the whole
            # finding of this session. Say so rather than aborting the park.
            print(f'  step {k}/{args.steps}  waited {settle:.1f}s  '
                  f'(no fresh reading; the write path is the healthy one)')
        prev = pose

    time.sleep(1.0)
    end = ask(args.sock, {'cmd': 'state'})
    link = end.get('link', {})
    print(f'\nparked. link now {link.get("fraction", 0):.0%} valid '
          f'({link.get("valid")}/{link.get("window")})')
    if end.get('angles') and end.get('age_ms', 1e9) < 3000:
        print(f'  measured {[round(x, 1) for x in end["angles"]]}')
    print('\nGive it a minute before judging the link -- the servos shed heat '
          'slowly, and it is temperature, not the pose itself, that tracks '
          'the crash rate.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
