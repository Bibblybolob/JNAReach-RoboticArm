#!/usr/bin/env python3
"""Pins the base-frame output of detection_bridge_node.

Loads the methods out of the node by source so this runs without ROS -- same
approach as test_depth_filter.py and test_servo_math.py.

What is worth pinning here is the DEGRADING, not the happy path. Three inputs
can be missing (intrinsics, calibration, joint1) and every one of them is
normal at some point in a session. Getting that wrong in either direction is
bad in a different way: publish anyway and something downstream drives the arm
to a made-up point; fail loudly and /button/point_px goes down with it, taking
the servo, which needs none of this.
"""
from __future__ import annotations

import ast
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'mycobot_perception'))
from target_in_base import TargetError, target_in_base  # noqa: E402

SRC = os.path.join(os.path.dirname(__file__), '..', 'mycobot_perception',
                   'detection_bridge_node.py')

WANT = {'_load_calibration', '_on_camera_info', '_on_joint_states',
        '_publish_base_point', '_warn_once'}


def _load():
    tree = ast.parse(open(os.path.abspath(SRC)).read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and 'Bridge' in n.name)
    ns = {'json': json, 'math': math, 'os': os,
          'target_in_base': target_in_base, 'TargetError': TargetError,
          'PointStamped': _Point}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANT:
            node.decorator_list = []
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(mod), SRC, 'exec'), ns)
    missing = WANT - set(ns)
    if missing:
        raise SystemExit(f'could not load {missing} from the node')
    return ns


class _Point:
    def __init__(self):
        self.header = type('H', (), {'frame_id': ''})()
        self.point = type('P', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()


class _Log:
    def __init__(self):
        self.msgs = []

    def _rec(self, m):
        self.msgs.append(m)
    info = warn = error = debug = _rec


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, m):
        self.sent.append(m)


class Node:
    """The smallest object the extracted methods need."""

    def __init__(self, **kw):
        self._log = _Log()
        self._base_pub = _Pub()
        self._warned = set()
        self._intrinsics = None
        self._joint1_deg = None
        self._cam_to_base = None
        self._published_base_any = False
        self.pan_joint = 'joint1'
        self.base_frame = 'base_link'
        self.base_output_topic = '/button/point_base'
        self.calibration_path = ''
        self.__dict__.update(kw)

    def get_logger(self):
        return self._log


NS = _load()
for name in WANT:
    setattr(Node, name, NS[name])

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'PASS  {name}')
    else:
        FAIL += 1
        print(f'FAIL  {name}')


def ident():
    return [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


def msg_ci(k):
    return type('CI', (), {'k': k})()


def msg_js(names, positions):
    return type('JS', (), {'name': names, 'position': positions})()


# --- calibration loading ----------------------------------------------------

n = Node(calibration_path='/nonexistent/eye_to_hand.json')
check('a missing calibration returns None rather than raising',
      n._load_calibration() is None)
check('and it names the command that creates one',
      any('--eye-to-hand' in m for m in n._log.msgs))

with tempfile.TemporaryDirectory() as d:
    # The likeliest wrong file: an eye-IN-hand result.
    p = os.path.join(d, 'wrong.json')
    json.dump({'camera_to_flange': ident()}, open(p, 'w'))
    n = Node(calibration_path=p)
    check('an eye-in-hand calibration is rejected, not used',
          n._load_calibration() is None)
    check('and the message says why it is the wrong one',
          any('flange' in m for m in n._log.msgs))

    p = os.path.join(d, 'right.json')
    json.dump({'camera_to_base': ident(), 'joint1_deg': 12.0}, open(p, 'w'))
    n = Node(calibration_path=p)
    c = n._load_calibration()
    check('a valid eye-to-hand calibration loads',
          c is not None and c['joint1_deg'] == 12.0)

    p = os.path.join(d, 'broken.json')
    open(p, 'w').write('{not json')
    n = Node(calibration_path=p)
    check('unparseable JSON returns None rather than raising',
          n._load_calibration() is None)

# --- intrinsics -------------------------------------------------------------

n = Node()
n._on_camera_info(msg_ci([0.0] * 9))
check('empty CameraInfo intrinsics are refused', n._intrinsics is None)
check('and the message names source:=realsense',
      any('realsense' in m for m in n._log.msgs))

n._on_camera_info(msg_ci([393.8, 0, 318.1, 0, 393.4, 236.5, 0, 0, 1]))
check('real intrinsics are taken from k in the right order',
      n._intrinsics == {'fx': 393.8, 'fy': 393.4, 'cx': 318.1, 'cy': 236.5})

# --- joint1 -----------------------------------------------------------------

n = Node()
n._on_joint_states(msg_js(['joint1', 'joint2'], [math.radians(30.0), 0.0]))
check('joint1 is read by NAME and converted to degrees',
      abs(n._joint1_deg - 30.0) < 1e-9)

# Order must not be assumed: the name lookup is the point.
n = Node()
n._on_joint_states(msg_js(['joint3', 'joint1'], [0.0, math.radians(-45.0)]))
check('joint1 is found even when it is not first',
      abs(n._joint1_deg + 45.0) < 1e-9)

n = Node()
n._on_joint_states(msg_js(['shoulder', 'elbow'], [0.0, 0.0]))
check('a missing pan joint leaves the angle unknown', n._joint1_deg is None)

# --- the gating -------------------------------------------------------------

hdr = type('H', (), {'frame_id': ''})()
K = {'fx': 393.8, 'fy': 393.4, 'cx': 318.1, 'cy': 236.5}

n = Node(_intrinsics=K, _joint1_deg=0.0, _cam_to_base=None)
n._publish_base_point(hdr, 320, 240, 250.0)
check('no calibration means silence, not a guess', not n._base_pub.sent)

n = Node(_intrinsics=None, _joint1_deg=0.0,
         _cam_to_base={'camera_to_base': ident()})
n._publish_base_point(hdr, 320, 240, 250.0)
check('no intrinsics means silence', not n._base_pub.sent)

n = Node(_intrinsics=K, _joint1_deg=None,
         _cam_to_base={'camera_to_base': ident()})
n._publish_base_point(hdr, 320, 240, 250.0)
check('no joint1 means silence', not n._base_pub.sent)
check('and it says the driver may not be running',
      any('driver' in m for m in n._log.msgs))

# Out-of-range depth is normal, not an error: it must not spam.
n = Node(_intrinsics=K, _joint1_deg=0.0,
         _cam_to_base={'camera_to_base': ident()})
n._publish_base_point(hdr, 320, 240, 5000.0)
check('an out-of-range depth publishes nothing', not n._base_pub.sent)

# --- the happy path ---------------------------------------------------------

n = Node(_intrinsics=K, _joint1_deg=0.0,
         _cam_to_base={'camera_to_base': ident(), 'joint1_deg': 0.0})
n._publish_base_point(hdr, K['cx'], K['cy'], 250.0)
check('a good detection publishes exactly one point',
      len(n._base_pub.sent) == 1)
p = n._base_pub.sent[0]
check('dead centre at 250mm lands 0.25m along the optical axis',
      abs(p.point.x) < 1e-9 and abs(p.point.y) < 1e-9
      and abs(p.point.z - 0.25) < 1e-9)
check('the point is stamped in the base frame', p.header.frame_id == 'base_link')

# The pan angle must actually be applied -- this is the composition that is
# correct at one angle and quietly wrong everywhere else if omitted.
n = Node(_intrinsics=K, _joint1_deg=90.0,
         _cam_to_base={'camera_to_base': ident(), 'joint1_deg': 0.0})
n._publish_base_point(hdr, K['cx'], K['cy'], 250.0)
q = n._base_pub.sent[0]
check('panning 90deg rotates the point, so joint1 is really used',
      abs(q.point.x) < 1e-9 and abs(q.point.y) < 1e-9
      and abs(q.point.z - 0.25) < 1e-9)

n = Node(_intrinsics=K, _joint1_deg=90.0,
         _cam_to_base={'camera_to_base': ident(), 'joint1_deg': 0.0})
n._publish_base_point(hdr, K['cx'] + 100, K['cy'], 250.0)
off = n._base_pub.sent[0]
check('an off-centre detection at 90deg pan moves in y, not x',
      abs(off.point.y) > 0.01 and abs(off.point.x) < 1e-9)

print(f'\n{PASS} passed' + (f', {FAIL} FAILED' if FAIL else ''))
sys.exit(1 if FAIL else 0)
