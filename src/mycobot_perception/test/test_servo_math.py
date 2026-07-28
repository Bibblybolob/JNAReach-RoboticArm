"""Checks on the visual servo's control maths.

    python3 src/mycobot_perception/test/test_servo_math.py
    pytest src/mycobot_perception/test/

These exist because the sign conventions here are genuinely easy to get
wrong, and getting them wrong does not look like a crash -- it looks like the
arm driving your hand out of frame, which is indistinguishable from a badly
mounted camera. Every direction in the loop is pinned below:

  * _jinv maps image error -> the joint motion that corrects it
  * _jfwd maps joint motion -> the image error it produces
  * a correction computed from an error must SHRINK that error
  * the lag compensator must predict the error AFTER in-flight jogs land

Plus the auto_sign detector, which has to survive a waving hand without ever
flipping a correct axis -- a false flip breaks tracking outright, so it is
tested against 200 noisy runs in each direction.

The functions under test are pulled out of visual_servo_node by source, so
this runs without rclpy or a ROS environment and tests the shipping code
rather than a transcription of it.
"""

import ast
import os

import numpy as np

SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'mycobot_perception', 'visual_servo_node.py')

WANTED = ('_record_sent', '_sent_between', '_compensate', '_set_jacobian',
          '_reset_sign_estimate', '_update_sign_estimate', '_update_velocity')

GAIN = 0.7
DEG = 25.0


def _load():
    tree = ast.parse(open(SRC).read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == 'VisualServoNode')
    funcs = [n for n in cls.body
             if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    missing = set(WANTED) - {f.name for f in funcs}
    assert not missing, f'not found in visual_servo_node: {sorted(missing)}'
    ns = {'np': np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), SRC, 'exec'), ns)
    return ns


_NS = _load()
LOGS = []


class _Log:
    def __getattr__(self, level):
        return lambda msg, **kw: LOGS.append(msg)


class _Servo:
    pass


for _name in WANTED:
    setattr(_Servo, _name, _NS[_name])


def make(h=1.0, v=1.0, lag=0.15, max_comp=0.8, samples=25,
         lead=0.15, vel_smoothing=0.0, max_speed=3.0, sat_limit=8):
    s = _Servo()
    s._command_lag = lag
    s._max_comp = max_comp
    s._lag_comp = True
    # Velocity smoothing defaults to 0 here so a single measured difference
    # arrives undiluted and the arithmetic under test is the arithmetic
    # asserted on. The smoothing itself is pinned separately.
    s._lead_time = lead
    s._vel_smoothing = vel_smoothing
    s._max_target_speed = max_speed
    s._vel_sat_limit = sat_limit
    s._vel = np.zeros(2)
    s._vel_prev = None
    s._vel_saturated = 0
    s._vel_untrusted = False
    s._sent = []
    s._corr = np.zeros(2)
    s._pmag = np.zeros(2)
    s._corr_n = 0
    s._sign_verdict = [False, False]
    s._prev_meas = None
    s._auto_sign_samples = samples
    s._assumed_deg = DEG
    s._assumed_v_deg = DEG
    s._assumed_h_sign = h
    s._assumed_v_sign = v
    s.get_logger = _Log
    s._set_jacobian(np.array([[DEG * h, 0.0], [0.0, DEG * v]]))
    return s


def feed(s, ratio, n=30, step=0.12, noise=0.0, rng=None):
    """Run n detections where the image responds `ratio` x as predicted.

    ratio 1.0 is a correct model, -1.0 an inverted axis, 0.0 a joint that
    does not steer that image direction at all.
    """
    e = np.array([0.5, 0.5])
    t = 100.0
    for _ in range(n):
        s._update_sign_estimate(t, e.copy())
        d = np.array([-2.0, -2.0])
        s._record_sent(t + 0.01, *d)
        e = e + (s._jfwd @ d) * ratio
        if noise:
            e = e + rng.normal(0, noise, 2)
        t += step
    return e


# --- Jacobian directions ----------------------------------------------------

def test_jfwd_is_the_inverse_of_jinv():
    s = make()
    assert np.allclose(s._jfwd @ s._jinv, np.eye(2))


def test_a_correction_shrinks_the_error_it_came_from():
    """The one that matters: get this backwards and the arm runs away."""
    s = make()
    err = np.array([0.6, -0.4])
    delta = s._jinv @ (GAIN * err) * -1.0     # the control law
    assert np.allclose(s._jfwd @ delta, -GAIN * err)


def test_a_flipped_sign_reverses_the_correction():
    a = make(h=1.0)._jinv @ np.array([0.5, 0.0])
    b = make(h=-1.0)._jinv @ np.array([0.5, 0.0])
    assert np.allclose(a, -b)


# --- Gain profile -----------------------------------------------------------
#
# "Faster the farther out, slower as it closes in". The far half of that is
# not available -- max_step_deg clamps every error past 0.29 -- so the shape
# that matters is the near half backing off.

def _eff(gain, k, err):
    return gain * (1.0 + k * abs(err))


def test_the_gain_rises_with_distance_from_centre():
    for near, far in ((0.05, 0.15), (0.15, 0.29), (0.29, 0.8)):
        assert _eff(0.45, 2.0, near) < _eff(0.45, 2.0, far)


def test_the_profile_meets_the_old_flat_gain_where_the_clamp_starts():
    """The far field must be unchanged from the flat 0.7 it replaced: past
    the clamp point the step is capped anyway, so any difference there is
    cosmetic, and below it the profile must be GENTLER, not hotter."""
    clamp_err = 5.0 / (0.7 * DEG)          # 0.286
    assert abs(_eff(0.45, 2.0, clamp_err) - 0.7) < 0.02
    assert _eff(0.45, 2.0, 0.05) < 0.7
    assert _eff(0.45, 2.0, 0.15) < 0.7


def test_zero_progressive_gain_is_a_flat_gain():
    for e in (0.0, 0.1, 0.5, 1.0):
        assert _eff(0.6, 0.0, e) == 0.6


def test_the_step_is_still_monotonic_in_the_error():
    """Superlinear must not mean non-monotonic: a hand farther out must never
    get a SMALLER jog than one closer in."""
    steps = [min(_eff(0.45, 2.0, e) * e * DEG, 5.0)
             for e in np.arange(0.0, 1.01, 0.01)]
    assert all(b >= a - 1e-9 for a, b in zip(steps, steps[1:]))


# --- Per-axis correction scale ----------------------------------------------
#
# assumed_deg_per_error was one number for both axes. It should not be: the
# error is normalised per axis, so a unit error means "at the edge" in both
# directions, but on a 640x480 sensor those edges are ~25 and ~19 degrees away.

def test_the_two_axes_can_be_scaled_separately():
    s = _Servo()
    s._jinv = None
    s.get_logger = _Log
    s._reset_sign_estimate = lambda: None
    s._set_jacobian(np.array([[25.0, 0.0], [0.0, 19.0]]))
    # A full correction of a unit error costs 25deg horizontally, 19 vertically.
    assert np.allclose(s._jinv @ np.array([1.0, 0.0]), [25.0, 0.0])
    assert np.allclose(s._jinv @ np.array([0.0, 1.0]), [0.0, 19.0])


def test_a_weaker_vertical_axis_gets_a_bigger_correction():
    """Lowering the vertical number makes tilting move MORE per unit error,
    which is the fix when tilting under-shoots where panning does not."""
    strong = _Servo(); strong._jinv = None; strong.get_logger = _Log
    strong._reset_sign_estimate = lambda: None
    strong._set_jacobian(np.array([[25.0, 0.0], [0.0, 25.0]]))
    weak = _Servo(); weak._jinv = None; weak.get_logger = _Log
    weak._reset_sign_estimate = lambda: None
    weak._set_jacobian(np.array([[25.0, 0.0], [0.0, 19.0]]))
    err = np.array([0.0, 0.5])
    assert abs((weak._jinv @ err)[1]) < abs((strong._jinv @ err)[1])


# --- Lag compensation -------------------------------------------------------

def test_compensator_predicts_the_error_after_in_flight_jogs_land():
    s = make()
    err = np.array([0.6, -0.4])
    # Frame captured at t=10.0; a jog sent at 10.05 cannot be visible in it.
    s._record_sent(10.05, *(s._jinv @ (GAIN * err) * -1.0))
    assert np.allclose(s._compensate(10.0, err), err * (1 - GAIN))


def test_jogs_older_than_the_lag_window_are_ignored():
    s = make()
    err = np.array([0.6, -0.4])
    s._record_sent(10.05, -10.0, 10.0)
    assert np.allclose(s._compensate(20.0, err), err)


def test_nothing_in_flight_means_no_adjustment():
    err = np.array([0.6, -0.4])
    assert np.allclose(make()._compensate(10.0, err), err)


def test_compensation_is_clamped():
    """It is an open-loop prediction; unbounded means runaway."""
    s = make(max_comp=0.8)
    for i in range(20):
        s._record_sent(10.0 + i * 0.01, 5.0, 5.0)
    assert abs(np.linalg.norm(s._compensate(10.0, np.zeros(2))) - 0.8) < 1e-9


def test_disabling_compensation_is_a_passthrough():
    s = make()
    s._lag_comp = False
    s._record_sent(10.05, -10.0, -10.0)
    err = np.array([0.6, -0.4])
    assert np.allclose(s._compensate(10.0, err), err)


# --- Target velocity feedforward --------------------------------------------
#
# The lead term is what centres a MOVING hand instead of trailing it, and it
# has one failure mode that matters: mistaking the camera's own motion for the
# hand's. Doing that would have the loop chase its own jogs, so the
# own-motion subtraction is pinned hardest here.

def test_first_sighting_has_no_velocity():
    """Nothing to difference against, so the lead term must be a no-op
    rather than a guess."""
    s = make()
    assert np.allclose(s._update_velocity(10.0, np.array([0.5, 0.5])),
                       np.zeros(2))


def test_a_still_hand_has_no_velocity():
    s = make()
    err = np.array([0.5, -0.3])
    s._update_velocity(10.0, err)
    assert np.allclose(s._update_velocity(10.125, err), np.zeros(2))


def test_velocity_is_measured_in_half_frames_per_second():
    s = make()
    s._update_velocity(10.0, np.array([0.0, 0.0]))
    # Moved 0.1 right and 0.05 down over a quarter second.
    v = s._update_velocity(10.25, np.array([0.1, -0.05]))
    assert np.allclose(v, [0.4, -0.2])


def test_our_own_jog_is_not_mistaken_for_the_hand_moving():
    """The camera moving looks exactly like the hand moving the other way.
    Crediting that to the hand would make the loop chase its own motion."""
    s = make()
    s._update_velocity(10.0, np.array([0.5, 0.5]))
    # A jog lands between two frames when it was SENT one command_lag before
    # them, so 9.9 falls between 10.0-0.15 and 10.125-0.15. Shifts the image
    # by -0.2 on both axes.
    d = s._jinv @ np.array([0.2, 0.2])
    s._record_sent(9.9, *(-d))
    moved = np.array([0.5, 0.5]) - np.array([0.2, 0.2])
    assert np.allclose(s._update_velocity(10.125, moved), np.zeros(2))


def test_a_hand_moving_while_we_jog_reports_only_the_hand():
    s = make()
    s._update_velocity(10.0, np.array([0.5, 0.5]))
    d = s._jinv @ np.array([0.2, 0.0])
    s._record_sent(9.9, *(-d))
    # Image moved -0.2 from our jog, and +0.1 from the hand, over 0.125s.
    seen = np.array([0.5 - 0.2 + 0.1, 0.5])
    assert np.allclose(s._update_velocity(10.125, seen), [0.8, 0.0])


def test_velocity_is_clamped():
    """One bad detection is one enormous difference, and lead_time
    multiplies it straight into the error."""
    s = make(max_speed=3.0)
    s._update_velocity(10.0, np.array([0.0, 0.0]))
    v = s._update_velocity(10.01, np.array([1.0, 1.0]))
    assert abs(np.linalg.norm(v) - 3.0) < 1e-9


def test_a_gap_in_sightings_is_not_a_velocity():
    """Across a dropout the hand is not the same hand, and the distance it
    appears to have moved is not a speed."""
    s = make()
    s._update_velocity(10.0, np.array([0.0, 0.0]))
    assert np.allclose(s._update_velocity(14.0, np.array([0.8, 0.8])),
                       np.zeros(2))


def test_repeated_timestamps_do_not_divide_by_zero():
    s = make()
    s._update_velocity(10.0, np.array([0.1, 0.1]))
    v = s._update_velocity(10.0, np.array([0.4, 0.4]))
    assert np.all(np.isfinite(v)) and np.allclose(v, np.zeros(2))


def test_velocity_smoothing_damps_a_single_jump():
    s = make(vel_smoothing=0.6)
    s._update_velocity(10.0, np.array([0.0, 0.0]))
    v = s._update_velocity(10.25, np.array([0.1, 0.0]))
    assert np.allclose(v, [0.4 * 0.4, 0.0])


def test_disabling_lead_time_stops_measuring_velocity():
    s = make(lead=0.0)
    s._update_velocity(10.0, np.array([0.0, 0.0]))
    assert np.allclose(s._update_velocity(10.25, np.array([0.5, 0.5])),
                       np.zeros(2))


def test_leading_a_moving_hand_aims_ahead_of_it():
    """The whole point: the commanded correction must overshoot the hand's
    CURRENT position by roughly the distance it travels during the lag."""
    s = make(lead=0.2)
    s._update_velocity(10.0, np.array([0.0, 0.0]))
    v = s._update_velocity(10.25, np.array([0.1, 0.0]))
    err = np.array([0.1, 0.0]) + v * s._lead_time
    assert err[0] > 0.1
    assert np.isclose(err[0], 0.1 + 0.4 * 0.2)


# --- A joint that is commanded but does not move -----------------------------
#
# Observed on hardware: joint1 was not turning, the driver pinned its target,
# and _update_velocity booked the predicted-but-absent image motion as the
# TARGET moving at the clamp speed -- in the direction that makes the error
# look bigger. The lead then drove harder, which sustained the saturation.
# A stuck axis became a runaway. These pin the way out of that.

def _stuck_joint(s, n, dt=0.0667, jog=5.0):
    """Feed n detections where jogs are commanded and nothing moves."""
    t = 100.0
    still = np.array([0.4, 0.4])
    s._update_velocity(t, still)
    for _ in range(n):
        s._record_sent(t - s._command_lag + 1e-6, jog, jog)
        t += dt
        s._update_velocity(t, still)
    return s._vel


def test_unexecuted_jogs_are_seen_as_target_motion():
    """The mechanism, pinned so the fix is not mistaken for the bug."""
    s = make(sat_limit=10_000)          # guard off, mechanism visible
    v = _stuck_joint(s, 3)
    # Predicted image motion that never arrived, booked as the hand moving
    # the other way -- and fast.
    assert np.linalg.norm(v) > 1.0
    assert v[0] < 0 and v[1] < 0


def test_a_stuck_joint_stops_the_lead_rather_than_amplifying_it():
    LOGS.clear()
    s = make(sat_limit=8)
    v = _stuck_joint(s, 20)
    assert s._vel_untrusted
    assert np.allclose(v, np.zeros(2))
    assert any('not reaching the arm' in m for m in LOGS)


def test_one_saturated_sample_does_not_disable_the_lead():
    """A glitch is not an indictment; only a run of them is."""
    s = make(sat_limit=8)
    s._update_velocity(10.0, np.array([0.0, 0.0]))
    s._update_velocity(10.01, np.array([1.0, 1.0]))   # saturates
    assert not s._vel_untrusted
    v = s._update_velocity(10.14, np.array([1.01, 1.01]))
    assert not s._vel_untrusted
    assert s._vel_saturated == 0
    assert np.linalg.norm(v) < 3.0


def test_the_lead_is_restored_once_the_estimate_is_sane_again():
    LOGS.clear()
    s = make(sat_limit=8)
    _stuck_joint(s, 20)
    assert s._vel_untrusted
    # Arm starts moving again: predicted motion now actually shows up.
    t, pos = 200.0, np.array([0.4, 0.4])
    s._update_velocity(t, pos)
    for _ in range(3):
        t += 0.0667
        pos = pos + np.array([0.01, 0.01])
        s._update_velocity(t, pos)
    assert not s._vel_untrusted
    assert any('plausible again' in m for m in LOGS)


def test_a_genuinely_fast_hand_still_gets_led():
    """The guard must not punish a hand that really is moving quickly."""
    s = make(sat_limit=8, vel_smoothing=0.6)
    t, pos = 100.0, np.array([0.0, 0.0])
    s._update_velocity(t, pos)
    for _ in range(20):
        t += 0.0667
        pos = pos + np.array([0.08, 0.0])   # ~1.2/s, under the 3.0 ceiling
        v = s._update_velocity(t, pos)
    assert not s._vel_untrusted
    assert v[0] > 0.8


# --- auto_sign --------------------------------------------------------------

def test_correct_model_is_left_alone():
    s = make()
    feed(s, ratio=1.0)
    assert s._sign_verdict == [False, False]
    assert s._jinv[0, 0] > 0


def test_inverted_axis_is_detected_and_flipped():
    LOGS.clear()
    s = make()
    feed(s, ratio=-1.0)
    assert s._jinv[0, 0] < 0
    assert any('inverted' in m for m in LOGS)


def test_unresponsive_axis_is_reported_not_flipped():
    """A swapped axis is a different fault from an inverted one, and
    flipping the sign would not fix it."""
    LOGS.clear()
    s = make()
    feed(s, ratio=0.0)
    assert s._jinv[0, 0] > 0
    assert any('barely responds' in m for m in LOGS)


def test_weak_but_correct_axis_suggests_a_gain():
    LOGS.clear()
    s = make()
    feed(s, ratio=0.4)
    assert s._jinv[0, 0] > 0
    assert any('as strongly as assumed' in m for m in LOGS)


def test_no_spurious_flips_with_a_waving_hand():
    """Flipping a correct axis breaks tracking outright, so this is the
    expensive mistake. Hand motion is the noise."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = make()
        feed(s, ratio=1.0, noise=0.08, rng=rng)
        assert s._sign_verdict == [False, False]


def test_inverted_axis_still_caught_through_noise():
    rng = np.random.default_rng(1)
    for _ in range(200):
        s = make()
        feed(s, ratio=-1.0, noise=0.08, rng=rng)
        assert s._sign_verdict[0]


if __name__ == '__main__':
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {name}')
            passed += 1
    print(f'\n{passed} passed')
