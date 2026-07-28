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
          '_reset_sign_estimate', '_update_sign_estimate')

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


def make(h=1.0, v=1.0, lag=0.15, max_comp=0.8, samples=25):
    s = _Servo()
    s._command_lag = lag
    s._max_comp = max_comp
    s._lag_comp = True
    s._sent = []
    s._corr = np.zeros(2)
    s._pmag = np.zeros(2)
    s._corr_n = 0
    s._sign_verdict = [False, False]
    s._prev_meas = None
    s._auto_sign_samples = samples
    s._assumed_deg = DEG
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
