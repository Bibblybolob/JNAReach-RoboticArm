#!/usr/bin/env python3
"""Where must the panel be, and how long must the tool be, to press square-on?

    ./scripts/sweep_press_placement.py
    ./scripts/sweep_press_placement.py --tools 0,60,80 --az 45,55,65

Reconstructs the placement sweep whose result is quoted in commit 4acdc82 and
in README -- "a square press needs the panel at 120-150mm" and the table of
tool length against aim. That sweep was run ad-hoc and never committed, so the
numbers could not be checked when the arm later pressed edge-on. This script
exists so the claim is reproducible rather than remembered.

Commands nothing and needs no hardware: it is pure kinematics.

The conventions that matter, because getting either backwards silently
produces a much worse answer than the truth:

  - `approach_dir` points INTO the panel, i.e. AWAY from the base, which is
    the same vector `press_button.py` derives from the detected buttons and
    passes as both the approach and the panel normal.
  - the guard is a BARE CollisionGuard(), as `press_button.py` builds it.
    `plan_press`'s own default models the tool as well, which is stricter and
    refuses placements the real caller accepts.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'mycobot_perception'))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'mycobot_driver'))

from mycobot_driver.collision_guard import CollisionGuard      # noqa: E402
from mycobot_perception.reach_planner import (                 # noqa: E402
    plan_press, ReachError, panel_clearance)


def attempt(dist_m, height_m, az_deg, tool_m, standoff_m):
    """One placement. Returns (aim_deg or None, clearance_mm, note)."""
    a = math.radians(az_deg)
    tgt = np.array([dist_m * math.cos(a), dist_m * math.sin(a), height_m])
    # Panel faces the arm: its normal runs radially outward, so the approach
    # runs the same way -- into the panel, away from the base.
    nrm = np.array([math.cos(a), math.sin(a), 0.0])
    panel = (tgt, nrm)
    try:
        p = plan_press(tgt, tool_length_m=tool_m, standoff_m=standoff_m,
                       approach_dir=nrm, panel=panel,
                       guard=CollisionGuard(), orientation='auto')
    except ReachError as e:
        return None, 0.0, str(e)[:60]
    except Exception as e:                                     # noqa: BLE001
        return None, 0.0, f'{type(e).__name__}: {str(e)[:50]}'
    aim = p.get('touch_off_normal_deg')
    clear = panel_clearance(p['touch_deg'], *panel) * 1000.0
    if aim is None:
        return None, clear, 'solved position-only (orientation unconstrained)'
    return float(aim), clear, ''


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tools', default='0,60,80,130',
                    help='tool lengths in mm')
    ap.add_argument('--dist', default='100,120,140,160,180,200,220',
                    help='panel distance from the base axis, mm')
    ap.add_argument('--height', default='120,160,200,240',
                    help='panel height, mm')
    ap.add_argument('--az', default='0,25,45,55,70,90',
                    help='azimuth about the base, deg')
    ap.add_argument('--standoff-mm', type=float, default=40.0)
    ap.add_argument('--good', type=float, default=15.0,
                    help='count placements at or below this aim as square')
    args = ap.parse_args()

    nums = lambda s: [float(v) for v in s.split(',') if v != '']   # noqa: E731
    tools, dists = nums(args.tools), nums(args.dist)
    heights, azs = nums(args.height), nums(args.az)

    print(f'{len(tools)} tools x {len(dists)} distances x {len(heights)} '
          f'heights x {len(azs)} azimuths = '
          f'{len(tools)*len(dists)*len(heights)*len(azs)} placements\n')

    for tool in tools:
        rows = []
        reasons = {}
        for d in dists:
            for h in heights:
                for az in azs:
                    aim, clear, note = attempt(d / 1000.0, h / 1000.0, az,
                                               tool / 1000.0,
                                               args.standoff_mm / 1000.0)
                    if aim is None:
                        key = note.split(' at ')[0][:46]
                        reasons[key] = reasons.get(key, 0) + 1
                    else:
                        rows.append((aim, clear, d, h, az))
        print(f'=== tool {tool:.0f}mm: {len(rows)} placements solved with an '
              f'aim angle')
        if rows:
            rows.sort()
            good = [r for r in rows if r[0] <= args.good]
            print(f'    best aim {rows[0][0]:.1f}deg at '
                  f'{rows[0][2]:.0f}mm out, {rows[0][3]:.0f}mm up, '
                  f'az {rows[0][4]:.0f}deg, body clear {rows[0][1]:.0f}mm')
            print(f'    aim range {rows[0][0]:.1f}-{rows[-1][0]:.1f}deg; '
                  f'{len(good)} placement(s) at or below {args.good:.0f}deg')
            for aim, clear, d, h, az in rows[:5]:
                print(f'      {aim:6.1f}deg  {d:4.0f}mm out  {h:4.0f}mm up  '
                      f'az {az:3.0f}  clear {clear:5.0f}mm')
        for k, v in sorted(reasons.items(), key=lambda t: -t[1])[:4]:
            print(f'    refused x{v}: {k}')
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
