"""Checks on the driver's trapezoidal jog profile.

    python3 src/mycobot_driver/test/test_jog_profile.py
    pytest src/mycobot_driver/test/

The profile decides what the arm is physically asked to do, and two of its
properties are load-bearing for everything above it:

  * it must ARRIVE at the goal, exactly, and stop there. An overshoot here is
    indistinguishable from the servo oscillating, and would be debugged in
    entirely the wrong place.
  * the velocity must never step. That is the whole reason it exists: joint1
    could not follow a step change in velocity, fell behind, and the servo
    read the motion it never got as the hand moving away from it.

profile_step is pulled out of the driver by source so this runs without rclpy
or pymycobot, and tests the shipping code rather than a copy of it.
"""

import ast
import math
import os
import threading
import time

SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'mycobot_driver', 'mycobot_hardware_node.py')

DT = 0.06
ACCEL = 1200.0   # matches max_jog_accel_deg_s2
V_MAX = 80.0


def _load(*names):
    tree = ast.parse(open(SRC).read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef)
               and n.name == 'MyCobotHardwareNode')
    fns = [n for n in cls.body
           if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {f.name for f in fns}
    assert not missing, f'not found in the driver: {sorted(missing)}'
    for f in fns:
        # Drop @staticmethod; these are exec'd as plain functions.
        f.decorator_list = []
    ns = {'math': math, 'time': time}
    exec(compile(ast.Module(body=fns, type_ignores=[]), SRC, 'exec'), ns)
    return ns


_NS = _load('profile_step', '_jog_profile_tick', '_jog_reset_profile',
            '_publish_jog_applied', '_step_speed')
profile_step = _NS['profile_step']


def drive(distance, dt=DT, accel=ACCEL, v_max=V_MAX, max_ticks=2000):
    """Run the profile to completion. Returns the per-tick step list.

    Ticks that command no motion are not recorded -- the driver skips sending
    those too, so they are not part of the move.
    """
    remaining = distance
    vel = 0.0
    steps = []
    for _ in range(max_ticks):
        step, vel = profile_step(remaining, vel, dt, accel, v_max)
        if abs(step) < 1e-12:
            break
        remaining -= step
        steps.append(step)
    return steps


# --- Arrival ----------------------------------------------------------------

def test_it_arrives_exactly_and_stops():
    for d in (0.5, 2.0, 5.0, 15.0, 45.0, -5.0, -30.0):
        steps = drive(d)
        assert abs(sum(steps) - d) < 1e-9, f'{d}: landed on {sum(steps)}'
        assert abs(steps[-1]) < abs(d) or len(steps) == 1


def test_it_never_overshoots():
    """Travelled distance must approach the goal monotonically, never past."""
    for d in (1.0, 5.0, 20.0, 60.0):
        travelled = 0.0
        for step in drive(d):
            travelled += step
            assert travelled <= d + 1e-9, f'{d}: overshot to {travelled}'


def test_a_zero_move_does_nothing():
    assert drive(0.0)[:1] in ([], [0.0])


# --- The trapezoid ----------------------------------------------------------

def test_velocity_never_steps_while_travelling():
    """The property the whole thing exists for: no jump in commanded speed.

    Held exactly for every tick except the last two, which land on the goal.
    Discrete time, exact arrival and a strict acceleration limit cannot all
    hold at once -- the final step is truncated to whatever distance is left,
    and that truncation is a deceleration nobody chose. Bounding it is the
    honest guarantee; see the next test for how big it is allowed to get.
    """
    for d in (5.0, 20.0, 60.0, 120.0):
        vels = [s / DT for s in drive(d)]
        for a, b in list(zip(vels, vels[1:]))[:-2]:
            assert abs(b - a) <= ACCEL * DT + 1e-6, f'{d}: {a} -> {b}'


def test_the_arrival_tick_overruns_the_limit_only_slightly():
    """Regression guard. Braking on the continuous sqrt(2*a*d) starts a tick
    too late and dumps the remaining speed in one command -- 80 deg/s to 12 at
    the defaults, the exact lurch this profile removes, relocated to the end
    of the move. The discrete-safe form keeps the worst arrival step inside a
    small margin of the limit."""
    for d in (5.0, 20.0, 60.0, 120.0):
        vels = [s / DT for s in drive(d)]
        worst = max(abs(b - a) for a, b in zip(vels, vels[1:]))
        assert worst <= 1.25 * ACCEL * DT, f'{d}: dumped {worst:.1f} deg/s'


def test_a_long_move_has_all_three_phases():
    """Accelerate, cruise at the ceiling, decelerate."""
    vels = [abs(s) / DT for s in drive(90.0)]
    peak = max(vels)
    assert abs(peak - V_MAX) < 1e-6, f'never reached the cruise ceiling: {peak}'
    cruise = [v for v in vels if abs(v - V_MAX) < 1e-6]
    assert len(cruise) >= 2, 'no cruise phase'
    assert vels[0] < peak and vels[-1] < peak


def test_a_medium_move_is_triangular():
    """Too short to reach the ceiling: ramp up, ramp straight back down.

    Exercised at accel 300 rather than the shipping default, because the
    triangular band runs from accel*dt^2 (below which one tick covers the
    whole move) to v_max^2/accel (above which there is room to cruise), and
    at 1200 those are 4.3 and 5.3 degrees -- a band too narrow to sit in.
    """
    vels = [abs(s) / DT for s in drive(6.0, accel=300.0)]
    assert len(vels) >= 3, 'expected a multi-tick ramp'
    assert max(vels) < V_MAX, 'a 6deg move should not reach cruise speed'
    assert vels[0] < max(vels)
    assert vels[-1] < max(vels)


def test_the_default_barely_ramps_and_that_is_deliberate():
    """The shipping default is close to a no-op, by measurement rather than
    oversight.

    At command_interval 0.06 the profile is coarse: one tick buys accel*dt,
    which at 1200 is 72 deg/s, so the cruise ceiling is reached in a single
    command and any move under 4.3 degrees is one step regardless. What it
    still guarantees is that no command demands more than 72 deg/s of velocity
    change -- the step joint1 could not follow, and the reason this exists.

    Ramping properly means dropping accel, and that costs lock-on time in
    direct proportion. Simulated through the servo maths, from a hand at the
    frame edge: 1200 acquires in 0.36s, 600 in 2.02s, and 300 fails to
    converge in nine runs out of twelve. Steadiness once locked (3px) and
    tracking of a moving hand are unchanged throughout, so the ramp costs only
    the getting there. Dropping the deceleration planning was tried on the
    theory that stopping at each streamed goal was the expense; it is not, and
    it came out slightly worse (2.74s at accel 600).
    """
    assert V_MAX / (ACCEL * DT) < 1.5, 'default should reach cruise in one tick'
    assert ACCEL * DT == 72.0
    steps = drive(90.0)
    worst = max(abs(b - a) / DT for a, b in zip(steps, steps[1:]))
    assert worst <= ACCEL * DT * 1.25


def test_a_long_move_is_dominated_by_cruise_not_ramps():
    # 90deg at 80deg/s is 1.13s of cruise; the ramps must not add much.
    assert len(drive(90.0)) * DT < 90.0 / V_MAX + 4 * DT

def test_moves_below_one_tick_of_ramp_are_a_single_step():
    """Honest limit of the profile: it cannot ramp a move smaller than one
    tick's worth of acceleration (accel * dt^2, ~2.2deg at the defaults).
    Those arrive in one command, as they did before. It matters less than it
    sounds -- a step that small is a slow step by definition, 0.4deg at 10px
    from centre being 7deg/s -- but it is a limit, not an accident, and
    shrinking command_interval is what moves it."""
    threshold = ACCEL * DT * DT
    assert len(drive(threshold * 0.8)) == 1
    assert len(drive(threshold * 2.0)) > 1


def test_it_accelerates_from_rest_at_the_limit():
    step, vel = profile_step(90.0, 0.0, DT, ACCEL, V_MAX)
    assert abs(vel - ACCEL * DT) < 1e-9
    assert abs(step - ACCEL * DT * DT) < 1e-9


def test_the_ceiling_is_respected():
    for d in (30.0, 90.0, 200.0):
        assert max(abs(s) / DT for s in drive(d)) <= V_MAX + 1e-6


# --- Direction and degenerate settings ---------------------------------------

def test_negative_moves_mirror_positive_ones():
    assert [-s for s in drive(-17.0)] == drive(17.0)


def test_a_reversal_ramps_through_zero_rather_than_snapping():
    """Target jumps behind the arm mid-travel: velocity must come down through
    zero at the acceleration limit, not invert in one tick."""
    vel = V_MAX
    seen = [vel]
    remaining = -20.0
    for _ in range(200):
        step, vel = profile_step(remaining, vel, DT, ACCEL, V_MAX)
        remaining -= step
        seen.append(vel)
        if abs(remaining) < 1e-9 and abs(vel) < 1e-9:
            break
    for a, b in list(zip(seen, seen[1:]))[:-2]:
        assert abs(b - a) <= ACCEL * DT + 1e-6, f'{a} -> {b}'
    assert min(seen) < 0 < max(seen), 'never actually reversed'
    # The point of the test: it passed through zero rather than inverting.
    assert any(abs(v) < ACCEL * DT for v in seen)


def test_zero_acceleration_falls_back_to_commanding_the_whole_step():
    step, _ = profile_step(3.0, 0.0, DT, 0.0, V_MAX)
    assert abs(step - 3.0) < 1e-9


def test_zero_dt_is_a_no_op():
    assert profile_step(5.0, 10.0, 0.0, ACCEL, V_MAX) == (0.0, 10.0)


def test_faster_acceleration_arrives_sooner():
    slow = len(drive(20.0, accel=200.0))
    fast = len(drive(20.0, accel=1200.0))
    assert fast < slow



# --- The timer body, driven against a stub arm --------------------------------
#
# The arithmetic above is only half the story. A NameError left in
# _jog_profile_tick by a refactor killed the driver the first time a jog
# arrived on real hardware -- py_compile does not catch that, and none of the
# tests above execute the method that had the bug. These do.


class _StubArm:
    def __init__(self):
        self.sent = []

    def send_angles(self, angles, speed):
        self.sent.append((list(angles), speed))


class _StubLog:
    def __init__(self):
        self.messages = []

    def __getattr__(self, level):
        return lambda msg, **kw: self.messages.append(msg)


class _StubNode:
    """The smallest object _jog_profile_tick will run against."""

    JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

    def __init__(self, target=None, accel=ACCEL):
        self._jog_profile = True
        self._in_motion = threading.Event()
        self._jog_enabled = True
        self._mc = _StubArm()
        self._lock = threading.Lock()
        self._cmd_interval = DT
        self._jog_accel = accel
        self._jog_max_speed = V_MAX
        self._jog_vel = [0.0] * 6
        self._jog_lookahead = 0.12
        self._jog_cmd_deg = [0.0] * 6
        self._jog_target_deg = list(target) if target else [0.0] * 6
        self._joint_limits_deg = [(-168.0, 168.0)] * 6
        self._adaptive_jog_speed = True
        self._jog_speed = 40
        self._min_jog_speed = 60
        self._min_speed = 15
        self._max_speed = 100
        self._speed_at_100 = 120.0
        self._speed_headroom = 1.3
        self._traj_speed = 60
        self._adaptive_speed = True
        self._last_tx_time = 0.0
        self._last_jog_time = 0.0
        self._last_angles_rad = None
        self.applied_msgs = []
        self._log = _StubLog()

    def get_logger(self):
        return self._log

    def get_clock(self):
        class _C:
            def now(self):
                class _T:
                    def to_msg(self):
                        return None
                return _T()
        return _C()

    # Captured instead of published; JointJog needs ROS to construct.
    def _publish_jog_applied(self, applied):
        self.applied_msgs.append(list(applied))


for _n in ('profile_step', '_jog_profile_tick', '_jog_reset_profile',
           '_step_speed'):
    setattr(_StubNode, _n, _NS[_n])
_StubNode.profile_step = staticmethod(_NS['profile_step'])


def test_the_timer_body_actually_runs():
    """Regression guard for the NameError that killed the driver."""
    n = _StubNode(target=[10.0, 0, 0, 0, 0, 0])
    n._jog_profile_tick()
    assert n._mc.sent, 'nothing was commanded'
    angles, speed = n._mc.sent[-1]
    assert angles[0] > 0.0
    assert speed >= n._min_jog_speed


def test_the_timer_body_walks_all_the_way_to_the_goal():
    n = _StubNode(target=[10.0, 0, 0, 0, 0, 0])
    for _ in range(200):
        n._jog_profile_tick()
    assert abs(n._jog_cmd_deg[0] - 10.0) < 1e-9
    assert abs(n._mc.sent[-1][0][0] - 10.0) < 1e-9
    # And it reports each applied step.
    assert abs(sum(m[0] for m in n.applied_msgs) - 10.0) < 1e-9


def test_the_timer_body_is_a_no_op_once_it_has_arrived():
    n = _StubNode(target=[3.0, 0, 0, 0, 0, 0])
    for _ in range(200):
        n._jog_profile_tick()
    before = len(n._mc.sent)
    for _ in range(10):
        n._jog_profile_tick()
    assert len(n._mc.sent) == before, 'kept commanding after arrival'


def test_the_timer_body_respects_the_gates():
    for attr, value in (('_jog_profile', False), ('_jog_enabled', False),
                        ('_mc', None), ('_jog_cmd_deg', None),
                        ('_jog_target_deg', None)):
        n = _StubNode(target=[10.0, 0, 0, 0, 0, 0])
        arm = n._mc
        setattr(n, attr, value)
        n._jog_profile_tick()
        assert not arm.sent, f'commanded the arm with {attr}={value}'
    n = _StubNode(target=[10.0, 0, 0, 0, 0, 0])
    n._in_motion.set()
    n._jog_profile_tick()
    assert not n._mc.sent, 'commanded the arm during a trajectory'


def test_the_timer_body_clamps_to_joint_limits():
    n = _StubNode(target=[500.0, 0, 0, 0, 0, 0])
    for _ in range(400):
        n._jog_profile_tick()
    assert max(a[0] for a, _ in n._mc.sent) <= 168.0


def test_a_link_failure_does_not_escape_the_timer_body():
    n = _StubNode(target=[10.0, 0, 0, 0, 0, 0])

    def boom(angles, speed):
        raise RuntimeError('pymycobot said no')

    n._mc.send_angles = boom
    n._jog_profile_tick()          # must not raise
    assert any('rejected' in m for m in n._log.messages)


def test_reset_clears_the_ramp():
    n = _StubNode(target=[20.0, 0, 0, 0, 0, 0])
    n._jog_profile_tick()
    assert n._jog_vel[0] != 0.0
    n._jog_reset_profile()
    assert n._jog_target_deg is None
    assert n._jog_cmd_deg is None
    assert n._jog_vel == [0.0] * 6


def test_the_command_leads_the_profile_so_the_arm_never_arrives():
    """The pulsing fix. send_angles stops on arrival, so commanding exactly
    the profiled position makes the arm reach it and wait. While travelling,
    the commanded angle must be AHEAD of the profiled one."""
    n = _StubNode(target=[60.0, 0, 0, 0, 0, 0])
    n._jog_profile_tick()
    n._jog_profile_tick()
    commanded = n._mc.sent[-1][0][0]
    assert commanded > n._jog_cmd_deg[0], (
        f'commanded {commanded:.2f} is not ahead of the profile position '
        f'{n._jog_cmd_deg[0]:.2f} -- the arm will arrive and wait')


def test_the_lookahead_never_overshoots_the_goal():
    n = _StubNode(target=[8.0, 0, 0, 0, 0, 0])
    for _ in range(200):
        n._jog_profile_tick()
    assert max(a[0] for a, _ in n._mc.sent) <= 8.0 + 1e-9


def test_the_jog_chain_compounds_off_the_profile_not_the_lookahead():
    """_last_angles_rad feeds the next jog. Compounding off a point the arm
    was merely aimed at would walk the chain forward by a lookahead per
    tick."""
    n = _StubNode(target=[60.0, 0, 0, 0, 0, 0])
    for _ in range(3):
        n._jog_profile_tick()
    assert abs(math.degrees(n._last_angles_rad[0]) - n._jog_cmd_deg[0]) < 1e-9


if __name__ == '__main__':
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {name}')
            passed += 1
    print(f'\n{passed} passed')
