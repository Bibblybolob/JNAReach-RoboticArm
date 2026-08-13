#!/usr/bin/env python3
"""Pins the discontinuity guard that stops a garbled read becoming a command.

Loads _angles_continuous out of the driver by source, so this runs without
ROS -- same trick as test_jog_profile.py.

The case that motivated it: the arm homed correctly, a read of
[41, 93, -64, 55, -29, 61] was accepted while it sat at home, and the jog
profiler (which seeds its goal from the measured pose) drove the arm there.
Every value in that frame is individually legal; only the distance from the
previous reading gives it away.
"""
from __future__ import annotations

import math
import os
import sys
import time
import types

SRC = os.path.join(os.path.dirname(__file__), '..', 'mycobot_driver',
                   'mycobot_hardware_node.py')


def _load():
    """Pull the guard and its constants off the class without importing ROS."""
    import ast
    tree = ast.parse(open(os.path.abspath(SRC)).read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and 'Node' in n.name)
    wanted = {'_angles_continuous'}
    consts = {}
    funcs = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            funcs.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in (
                        'MAX_SLEW_DEG_S', 'SLEW_MARGIN_DEG'):
                    consts[t.id] = ast.literal_eval(node.value)
    mod = ast.Module(body=funcs, type_ignores=[])
    ns: dict = {'math': math, 'time': time}
    exec(compile(mod, '<guard>', 'exec'), ns)
    return ns['_angles_continuous'], consts


_continuous, CONSTS = _load()

HOME = [0.0, 90.0, -149.0, 55.0, 0.0, 0.0]
GARBLED = [41.0, 93.2, -64.2, 55.5, -29.3, 61.5]   # the real one


class FakeNode:
    """Minimal stand-in: the guard only touches these five attributes."""

    JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    MAX_SLEW_DEG_S = CONSTS['MAX_SLEW_DEG_S']
    SLEW_MARGIN_DEG = CONSTS['SLEW_MARGIN_DEG']

    def __init__(self, last=None, age_s=0.0):
        self._last_good_deg = list(last) if last else None
        self._last_good_time = time.monotonic() - age_s
        self.warnings = []

    def get_logger(self):
        node = self

        class L:
            @staticmethod
            def warn(msg, **kw):
                node.warnings.append(msg)
        return L()


def check(name, got, want):
    ok = got == want
    print(f'{"PASS" if ok else "FAIL"}  {name}')
    if not ok:
        print(f'        expected {want}, got {got}')
    return ok


def main() -> int:
    results = []

    n = FakeNode()
    results.append(check('first read is always accepted',
                         _continuous(n, GARBLED), True))

    # The motivating bug: at home, a legal-looking frame 85deg away.
    n = FakeNode(HOME, age_s=0.1)
    results.append(check('garbled-but-legal frame is rejected at home',
                         _continuous(n, GARBLED), False))
    results.append(check('  and it says which joints jumped',
                         'joint3' in (n.warnings[0] if n.warnings else ''), True))

    # A rejection must not overwrite the trusted pose, or one bad frame
    # would become the new reference and let the next one through.
    results.append(check('  and does not become the new reference',
                         n._last_good_deg, HOME))

    # Honest motion must never be rejected. joint1 measured 51.6 deg/s flat
    # out, so 5 deg in 100ms (50 deg/s) is a real move at full speed.
    n = FakeNode(HOME, age_s=0.1)
    moved = list(HOME)
    moved[0] += 5.0
    results.append(check('real motion at full speed is accepted',
                         _continuous(n, moved), True))

    # Small reads during a jog: well inside budget.
    n = FakeNode(HOME, age_s=0.05)
    nudged = list(HOME)
    nudged[4] += 0.4
    results.append(check('a 0.4deg jog step is accepted',
                         _continuous(n, nudged), True))

    # Recovery: if the arm really was hand-moved, elapsed time grows the
    # budget until the new pose is reachable, so it must not latch forever.
    n = FakeNode(HOME, age_s=3.0)
    results.append(check('hand-moved arm recovers once time has passed',
                         _continuous(n, GARBLED), True))

    # An accepted read updates the reference.
    n = FakeNode(HOME, age_s=0.1)
    ok = list(HOME)
    ok[1] += 2.0
    _continuous(n, ok)
    results.append(check('accepted read updates the reference',
                         n._last_good_deg, ok))

    print()
    print(f'{sum(results)}/{len(results)} passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
