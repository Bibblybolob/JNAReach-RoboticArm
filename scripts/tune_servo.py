#!/usr/bin/env python3
"""Tune the visual servo while it is actually following your hand.

    ./scripts/tune_servo.py          (with ./run.py already running)

Relaunching to try a gain meant holding your hand somewhere slightly
different every time, so two settings were never measured against the same
thing and "that felt better" was the only available verdict. This changes
parameters on the live node and scores the result, so the comparison is a
number and both settings see the same hand.

WHAT IT SHOWS

    h/v error   how far the hand is from centre on each axis, in pixels.
                Separately, because the two axes have separate scales
                (assumed_deg_per_error / assumed_v_deg_per_error) and a
                vertical axis that tracks worse than the horizontal one is
                invisible in a combined figure.
    score       mean distance from centre over the scoring window. THE
                number: lower is better, and it is what to compare between
                two settings.
    worst       the largest miss in the window -- a low score with a high
                worst is a loop that mostly sits still and lurches.
    flips       how often the error changed sign. This is oscillation, and
                it is the one failure a mean cannot show you: a loop
                buzzing evenly either side of centre scores well and looks
                terrible.
    det/s       detections per second. Below ~6 nothing tuned here helps.

HOW TO USE IT

Start a hunt, hold your hand in view, and let the score settle. Press `m`
to mark a baseline. Change one thing. Watch whether the score beats the
baseline -- and check `flips` has not gone up, because most ways of
improving the score buy it with oscillation.

Move your hand the way you actually want it tracked. A score taken with
your hand still says nothing about following, and lead_time in particular
only does anything against a hand that is moving.
"""

import argparse
import math
import select
import sys
import termios
import time
import tty
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Trigger


SERVO_NODE = 'visual_servo_node'
DRIVER_NODE = 'mycobot_hardware_node'

# node, name, label, step, minimum, maximum
#
# The driver knobs are here because the SPEED ceiling lives there, not in the
# servo. The servo can ask for 150 deg/s of goal and the profiler will still
# only deliver max_jog_speed_deg_s, so "make it move faster" is tuned on the
# driver while the score is read off the servo.
KNOBS = [
    (SERVO_NODE, 'gain',                    'gain at centre',      0.05, 0.05,  2.0),
    (SERVO_NODE, 'progressive_gain',        'gain rise w/ dist',   0.25, 0.0,   8.0),
    (SERVO_NODE, 'lead_time',               'aim ahead (s)',       0.05, 0.0,   0.6),
    (SERVO_NODE, 'deadband',                'hold-still zone',     0.01, 0.0,   0.2),
    (SERVO_NODE, 'command_lag',             'lag window (s)',      0.02, 0.0,   0.4),
    (SERVO_NODE, 'max_step_deg',            'max jog (deg)',       1.0,  1.0,  15.0),
    (SERVO_NODE, 'assumed_deg_per_error',   'horiz deg/unit',      2.0,  5.0, 200.0),
    (SERVO_NODE, 'assumed_v_deg_per_error', 'vert deg/unit',       2.0,  0.0, 200.0),
    (SERVO_NODE, 'velocity_smoothing',      'vel filter',          0.1,  0.0,  0.95),
    (SERVO_NODE, 'ki',                      'integral',            0.01, 0.0,   1.0),
    (SERVO_NODE, 'kd',                      'derivative',          0.01, 0.0,   1.0),
    (SERVO_NODE, 'integral_limit',          'integral cap',        0.1,  0.0,   2.0),
    (DRIVER_NODE, 'max_jog_speed_deg_s',   'ARM top speed d/s',   5.0,  10.0, 200.0),
    (DRIVER_NODE, 'max_jog_accel_deg_s2',  'ARM accel d/s2',    100.0, 100.0, 3000.0),
    (DRIVER_NODE, 'speed_at_100_deg_s',    'speed@100 (GUESS)',   5.0,  20.0, 300.0),
    (DRIVER_NODE, 'max_jog_deg',           'max jog step deg',    0.5,   1.0,  20.0),
]

HELP = """
  up/down or k/j   select knob            left/right or h/l   adjust
  m  mark baseline          r  reset score window
  a  toggle approach (toward/away axis)
  s  start a hunt           q  quit
"""


class Tuner(Node):

    def __init__(self, window: float):
        super().__init__('servo_tuner')
        self._window = window
        self._w = self._h = None
        self._samples = deque()       # (t, ex_px, ey_px)
        self._dets = deque()
        self._baseline = None
        self._sel = 0
        self._values = {}
        self._status = 'connecting to the servo...'

        self.create_subscription(CameraInfo, '/camera/camera_info',
                                 self._info_cb, 10)
        self.create_subscription(PointStamped, '/hand/point_px',
                                 self._point_cb, 1)
        self._set_cli = {
            n: self.create_client(SetParameters, f'/{n}/set_parameters')
            for n in (SERVO_NODE, DRIVER_NODE)}
        self._search_cli = self.create_client(Trigger, '/servo/search')

    # ---- data in ----

    def _info_cb(self, msg):
        if self._w is None and msg.width > 0:
            self._w, self._h = msg.width, msg.height

    def _point_cb(self, msg):
        now = time.monotonic()
        self._dets.append(now)
        if self._w is None:
            return
        self._samples.append((now,
                              msg.point.x - self._w / 2.0,
                              msg.point.y - self._h / 2.0))
        self._trim(now)

    def _trim(self, now):
        while self._samples and now - self._samples[0][0] > self._window:
            self._samples.popleft()
        while self._dets and now - self._dets[0] > 5.0:
            self._dets.popleft()

    # ---- scoring ----

    def score(self):
        """Mean distance from centre, worst miss, sign flips, per-axis means.

        The mean is the headline, but on its own it rewards a loop that has
        given up: a servo parked off-centre and not moving scores the same as
        one hunting evenly around it. Sign flips separate those two.
        """
        self._trim(time.monotonic())
        if not self._samples:
            return None
        xs = [s[1] for s in self._samples]
        ys = [s[2] for s in self._samples]
        dists = [math.hypot(x, y) for x, y in zip(xs, ys)]
        flips = 0
        for seq in (xs, ys):
            for a, b in zip(seq, seq[1:]):
                if a * b < 0 and abs(a) > 3 and abs(b) > 3:
                    flips += 1
        span = self._samples[-1][0] - self._samples[0][0]
        # Is the error systematically growing? A correctly signed loop pulls
        # the target in; an inverted one pushes it out, and then every knob
        # below is scaling a push. Compare the first third of the window
        # against the last.
        third = max(2, len(dists) // 3)
        diverging = (len(dists) >= 9
                     and sum(dists[-third:]) / third
                     > sum(dists[:third]) / third + 25.0)
        return dict(
            diverging=diverging,
            n=len(dists),
            mean=sum(dists) / len(dists),
            worst=max(dists),
            h=sum(abs(x) for x in xs) / len(xs),
            v=sum(abs(y) for y in ys) / len(ys),
            flips=flips / span if span > 0.5 else 0.0,
        )

    def det_rate(self):
        if len(self._dets) < 2:
            return 0.0
        span = self._dets[-1] - self._dets[0]
        return len(self._dets) / span if span > 0 else 0.0

    # ---- parameters ----

    def fetch_values(self):
        """Read current values from every node that owns a knob."""
        from rcl_interfaces.srv import GetParameters
        for node in (SERVO_NODE, DRIVER_NODE):
            names = [k[1] for k in KNOBS if k[0] == node]
            if node == SERVO_NODE:
                names = names + ['approach_enabled']
            cli = self.create_client(GetParameters,
                                     f'/{node}/get_parameters')
            if not cli.wait_for_service(timeout_sec=5.0):
                self._status = (f'no {node} on the graph -- is ./run.py '
                                'running?')
                return False
            req = GetParameters.Request()
            req.names = names
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            if fut.result() is None:
                self._status = f'{node} did not answer a parameter read'
                return False
            for name, pv in zip(names, fut.result().values):
                # INTEGER=2, DOUBLE=3, BOOL=1 -- max_jog_deg and friends come
                # back as whichever the node declared.
                self._values[name] = (
                    pv.bool_value if pv.type == 1 else
                    float(pv.integer_value) if pv.type == 2 else
                    pv.double_value)
        self._status = 'connected'
        return True

    def apply(self, name, value, node=SERVO_NODE):
        req = SetParameters.Request()
        if isinstance(value, bool):
            req.parameters = [Parameter(name, Parameter.Type.BOOL,
                                        value).to_parameter_msg()]
        else:
            req.parameters = [Parameter(name, Parameter.Type.DOUBLE,
                                        float(value)).to_parameter_msg()]
        fut = self._set_cli[node].call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        res = fut.result()
        if res is None:
            self._status = f'{name}: no reply from {node}'
            return False
        if not res.results[0].successful:
            self._status = f'{name} refused: {res.results[0].reason}'
            return False
        self._values[name] = value
        self._status = f'{name} = {value}'
        return True

    def nudge(self, direction):
        node, name, _, step, lo, hi = KNOBS[self._sel]
        cur = self._values.get(name, 0.0)
        new = max(lo, min(hi, round(cur + direction * step, 4)))
        if new != cur:
            self.apply(name, new, node)

    def toggle_approach(self):
        cur = bool(self._values.get('approach_enabled', False))
        self.apply('approach_enabled', not cur)

    def start_search(self):
        if not self._search_cli.wait_for_service(timeout_sec=2.0):
            self._status = 'no /servo/search service'
            return
        fut = self._search_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        self._status = 'hunting -- hold your hand in view'

    # ---- display ----

    def render(self):
        sc = self.score()
        out = ['\033[H\033[J', '  SERVO TUNER      (? = keys)\n']

        if sc is None:
            out.append('  waiting for a hand on /hand/point_px ...\n')
        elif sc.get('diverging'):
            # Nothing below this line is worth touching while the loop is
            # pushing the hand away: every knob scales a correction that is
            # pointed the wrong way, so they all "feel the same" and the score
            # is measuring how fast the target leaves rather than how well it
            # is held.
            out.append(
                f'  !! THE ERROR IS GROWING ({sc["mean"]:.0f} px and rising) '
                f'-- the arm is driving your hand OUT of frame.\n')
            out.append(
                '     That is a SIGN problem, not a tuning one. Tuning cannot '
                'fix it and\n     will feel like nothing does anything. '
                'Restart with skip_probe:=false to\n     measure the mounting, '
                'or flip assumed_h_sign / assumed_v_sign.\n')
        else:
            base = ''
            if self._baseline:
                d = sc['mean'] - self._baseline['mean']
                arrow = 'BETTER' if d < -0.5 else ('worse' if d > 0.5 else 'same')
                base = f'   baseline {self._baseline["mean"]:5.1f}  ({arrow} {d:+.1f})'
            out.append(
                f'  score {sc["mean"]:5.1f} px'
                f'   worst {sc["worst"]:5.1f}'
                f'   flips {sc["flips"]:4.1f}/s{base}\n')
            out.append(
                f'  h err {sc["h"]:5.1f} px     v err {sc["v"]:5.1f} px'
                f'     {self.det_rate():4.1f} det/s   n={sc["n"]}\n')
            if sc['v'] > 2.0 * sc['h'] + 5:
                out.append('  vertical is much worse than horizontal -- try '
                           'lowering vert deg/unit\n')
            elif sc['h'] > 2.0 * sc['v'] + 5:
                out.append('  horizontal is much worse than vertical -- try '
                           'lowering horiz deg/unit\n')
        out.append('\n')

        last_node = None
        for i, (node, name, label, step, _, _) in enumerate(KNOBS):
            if node != last_node:
                out.append(f'  {"-" * 4} {node} {"-" * 4}\n')
                last_node = node
            mark = '>' if i == self._sel else ' '
            val = self._values.get(name, float('nan'))
            out.append(f'  {mark} {label:<18} {val:8.3f}   ({name})\n')

        appr = bool(self._values.get('approach_enabled', False))
        out.append(f'\n  approach (toward/away): '
                   f'{"ON" if appr else "OFF -- that axis never moves"}\n')
        out.append(f'\n  {self._status}\n')
        sys.stdout.write(''.join(out))
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window', type=float, default=6.0,
                    help='seconds of history the score averages over')
    args = ap.parse_args()

    rclpy.init()
    node = Tuner(args.window)
    if not node.fetch_values():
        print(f'\n{node._status}\n')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    show_help = False
    try:
        tty.setcbreak(fd)
        last_draw = 0.0
        while True:
            rclpy.spin_once(node, timeout_sec=0.02)

            if select.select([sys.stdin], [], [], 0)[0]:
                c = sys.stdin.read(1)
                if c == '\x1b':                       # arrow keys
                    rest = sys.stdin.read(2) if select.select(
                        [sys.stdin], [], [], 0.01)[0] else ''
                    c = {'[A': 'k', '[B': 'j',
                         '[D': 'h', '[C': 'l'}.get(rest, '')
                if c in ('q', '\x03'):
                    break
                elif c == 'k':
                    node._sel = (node._sel - 1) % len(KNOBS)
                elif c == 'j':
                    node._sel = (node._sel + 1) % len(KNOBS)
                elif c == 'h':
                    node.nudge(-1)
                elif c == 'l':
                    node.nudge(+1)
                elif c == 'm':
                    node._baseline = node.score()
                    node._status = 'baseline marked'
                elif c == 'r':
                    node._samples.clear()
                    node._status = 'score window cleared'
                elif c == 'a':
                    node.toggle_approach()
                elif c == 's':
                    node.start_search()
                elif c == '?':
                    show_help = not show_help
                last_draw = 0.0

            now = time.monotonic()
            if now - last_draw > 0.2:
                node.render()
                if show_help:
                    sys.stdout.write(HELP)
                    sys.stdout.flush()
                last_draw = now
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
