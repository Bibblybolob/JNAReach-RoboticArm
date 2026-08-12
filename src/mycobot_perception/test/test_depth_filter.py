#!/usr/bin/env python3
"""Pins the depth sanity filter in detection_bridge_node.

Loads _depth_ok and _panel_depth out of the node by source, so this runs
without ROS -- same approach as test_servo_math.py.

Why it matters: this decides what the arm chases. The detector currently
produces confident nonsense on this panel -- it locked onto 'down' (a class
the panel does not have) at 0.21, and the target jumped from (+0.94,-0.52) to
(+0.40,-0.96) to (+0.96,-0.60) between consecutive sightings, saturating every
jog at the 5deg clamp. A real button has a depth, inside the sensor's range,
on the surface the other buttons are on. That is what this checks.
"""
from __future__ import annotations

import ast
import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), '..', 'mycobot_perception',
                   'detection_bridge_node.py')


def _load():
    tree = ast.parse(open(os.path.abspath(SRC)).read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and 'Bridge' in n.name)
    want = {'_depth_ok', '_panel_depth'}
    fns = [n for n in cls.body
           if isinstance(n, ast.FunctionDef) and n.name in want]
    ns: dict = {'np': np}
    exec(compile(ast.Module(body=fns, type_ignores=[]), '<f>', 'exec'), ns)
    return ns['_depth_ok'], ns['_panel_depth']


depth_ok, panel_depth = _load()


class Fake:
    """Only the attributes the two functions touch."""

    DEPTH_INVALID = 65535

    def __init__(self, recent=(), require=True, lo=70.0, hi=500.0, tol=80.0):
        self._recent_depths = list(recent)
        self.require_depth = require
        self.min_depth_mm = lo
        self.max_depth_mm = hi
        self.depth_consistency_mm = tol

    _panel_depth = panel_depth


def check(name, got, want):
    ok = got == want
    print(f'{"PASS" if ok else "FAIL"}  {name}')
    if not ok:
        print(f'        expected {want}, got {got}')
    return ok


def main() -> int:
    r = []

    # A button on the panel, nothing learned yet.
    n = Fake()
    r.append(check('a plausible depth is accepted',
                   depth_ok(n, 300.0)[0], True))

    # The common false positive: detector fires on empty space, where the
    # D405 resolves nothing and returns 0.
    n = Fake()
    ok, why = depth_ok(n, 0.0)
    r.append(check('no depth is rejected', ok, False))
    r.append(check('  and says why', why, 'no_depth'))

    # ...unless the operator has deliberately allowed it.
    n = Fake(require=False)
    r.append(check('require_depth:=false lets it through',
                   depth_ok(n, 0.0)[0], True))

    # Outside what a D405 can measure. 2m is a wall behind the panel.
    n = Fake()
    r.append(check('beyond sensor range is rejected',
                   depth_ok(n, 2000.0)[0], False))
    r.append(check('closer than sensor range is rejected',
                   depth_ok(n, 30.0)[0], False))

    # With a panel distance agreed, something at a very different depth is
    # a different object -- a hand, the wall, the desk.
    n = Fake(recent=[300, 305, 298, 302])
    r.append(check('panel distance is the median of accepted depths',
                   round(panel_depth(n)), 301))
    r.append(check('on-panel detection accepted',
                   depth_ok(n, 310.0)[0], True))
    ok, why = depth_ok(n, 450.0)
    r.append(check('off-panel detection rejected', ok, False))
    r.append(check('  and says why', why, 'inconsistent'))

    # The reference must not bite before it is trustworthy. Two samples is
    # not a panel distance, and vetoing on it would reject the very
    # detections needed to establish one.
    n = Fake(recent=[300, 305])
    r.append(check('fewer than 3 samples means no reference',
                   panel_depth(n), None))
    r.append(check('  so a far-but-valid depth is still accepted',
                   depth_ok(n, 450.0)[0], True))

    # uint16 saturation is "no measurement", not a distance. Treating it as
    # one made the diagnostics read "depths seen: 2021-65535mm".
    n = Fake()
    ok, why = depth_ok(n, 65535.0)
    r.append(check('uint16 saturation counts as no depth', why, 'no_depth'))
    r.append(check('  and is rejected', ok, False))

    # Turning consistency off must not disable the range check too.
    n = Fake(recent=[300, 305, 298], tol=0.0)
    r.append(check('consistency 0 accepts any in-range depth',
                   depth_ok(n, 480.0)[0], True))
    r.append(check('  but still rejects out-of-range',
                   depth_ok(n, 900.0)[0], False))

    print()
    print(f'{sum(r)}/{len(r)} passed')
    return 0 if all(r) else 1


if __name__ == '__main__':
    sys.exit(main())
