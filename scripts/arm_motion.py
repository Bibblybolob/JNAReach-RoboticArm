#!/usr/bin/env python3
"""Commanded motion that waits for the arm instead of guessing at it.

`wait_for_arrival` was written in calibrate_touch.py (commit 292c8e4) and
proved there; press_button.py never adopted it and still slept a fixed 7.5s
per move. That is not a small difference:

    standoff 7.5 + touch 7.5 + settle 0.6 + retract 7.5 + park 8.0 = 31.1s

against a 5-second budget for the whole press, and a fixed sleep whether the
joint travels 3 degrees or 90. So this module is the shared home for it, and
both scripts import from here.

Sleeping a guess is wrong in both directions. Too short returns while the arm
is still moving, reports an enormous divergence, and invites the operator to
re-issue a command that was already executing -- which is exactly the mess it
produced in use. Too long is the 31 seconds above.


Two consecutive reads, not one
------------------------------
The driver learned this the hard way (mycobot_hardware_node.py): one
in-tolerance read is NOT proof of arrival. The host UART carries the ESP32's
console output and the arm's internal Feetech servo bus alongside real
replies, so a garbled frame that happens to decode near the target ends the
wait early -- reporting arrival for an arm that never left. Two agreeing reads
0.25s apart costs almost nothing and rules that out.

The version in calibrate_touch.py guarded only on `age_ms`, which catches a
STALE read but not a corrupt one that is fresh. Both checks are applied here.


The tolerance is the hardware floor. Do not lower it.
----------------------------------------------------
1.0 degree, just above this arm's measured settling. 2026-08-13: commanded a
two-joint move and watched it settle to 0.79deg / 1.60mm of tip error at 4.5s,
then completely flat out to 21s. That 0.79 is the backlash CLAUDE.md records,
so it is the limit and waiting longer buys nothing.

It was 2.0deg once, which accepted two and a half times the achievable error --
about 9mm of tip error at 250mm of reach -- and THAT, not the IK, was what
made jogging feel inaccurate. The IK round-trips to under 0.1mm.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from arm_broker import request  # noqa: E402

# Measured 2026-08-01 over /dev/ttyTHS1: 51.6 deg/s at speed=100, 40.6 at 60,
# 29.0 at 30. Re-measure after any payload change with scripts/measure_arm.py.
DEG_PER_S_AT_100 = 52.0

# Just above the 0.79deg backlash floor. See the module docstring.
ARRIVE_TOL_DEG = 1.0

# Intermediate waypoints on a streamed path are not destinations, they are
# directions. Waiting for a tight tolerance at each one makes the arm stop
# dead at every step -- `send_angles` stops on arrival, which CLAUDE.md
# records as motion "arriving in pulses with a pause between each". Moving on
# once the arm is most of the way there keeps the motion flowing, and only the
# final pose is held to ARRIVE_TOL_DEG.
BLEND_TOL_DEG = 4.0

# Seconds of travel to command BEYOND each intermediate waypoint.
#
# The same trick and the same value as the driver's `jog_lookahead`, for the
# same reason, in its own words: "send_angles is point-to-point. Handed a
# fresh target every command interval it sprints the gap, stops, and waits,
# and at servo rates that stop-start IS the visible jerk."
#
# Commanding slightly past a waypoint means the arm is still accelerating
# through it when the next command lands, so the path is walked instead of
# hopped. It self-cancels: the last waypoint gets none, and the clamp below
# means the tool can never be commanded past the button.
LOOKAHEAD_S = 0.12

# Fraction of the estimated segment time to wait before issuing the next
# waypoint. Gating on a POSITION READ instead means one slow or garbled reply
# stalls the whole path -- the link carries ESP32 console output and the
# Feetech bus, so that is not hypothetical. Time is the primary gate and the
# tolerance is an early exit, so a fast segment still moves on early.
BLEND_FRAC = 0.6

# The last segment is commanded at this fraction of the speed. Slower into
# contact is gentler on the button and on the arm, and it sharpens the stall
# signal that press_button.py reads as "the button resisted" -- a fast
# approach overshoots into the panel and blurs that distinction.
FINAL_SPEED_FRAC = 0.5


def lookahead_target(w_cur, w_next, w_final, lookahead_deg):
    """`w_next`, pushed further along the direction of travel.

    Clamped per joint so it can never pass `w_final`. That clamp is the whole
    safety argument: the final waypoint is ON the button, and overshooting it
    is pressing through the panel.
    """
    import numpy as np

    c = np.asarray(w_cur, dtype=float)
    n = np.asarray(w_next, dtype=float)
    f = np.asarray(w_final, dtype=float)
    d = n - c
    L = float(np.linalg.norm(d))
    if L < 1e-9 or lookahead_deg <= 0.0:
        return [float(v) for v in n]
    t = n + d / L * lookahead_deg
    for j in range(len(t)):
        if d[j] > 0:
            t[j] = min(t[j], max(f[j], n[j]))
        elif d[j] < 0:
            t[j] = max(t[j], min(f[j], n[j]))
    return [float(v) for v in t]


def travel_budget_s(travel_deg: float, speed: int) -> float:
    """How long a move of this size may take, with headroom.

    Scaled from the measured speed, x2.5 for acceleration and a link that is
    slow to answer, plus a floor for the round trip itself.
    """
    return travel_deg / DEG_PER_S_AT_100 * (100.0 / max(speed, 1)) * 2.5 + 3.0


def wait_for_arrival(target, travel_deg, speed, tol=ARRIVE_TOL_DEG,
                     confirmations=2, poll_s=0.25, max_wait_s=30.0):
    """Block until the arm reaches `target`, or until it clearly will not.

    Returns (arrived, worst_error_deg). A short move costs a short wait.

    `confirmations` is the number of consecutive agreeing in-tolerance reads
    required -- see the module docstring. Pass 1 only when the caller has
    already established the link is clean.
    """
    deadline = time.monotonic() + min(travel_budget_s(travel_deg, speed),
                                      max_wait_s)
    worst = float('inf')
    agreed = 0
    while time.monotonic() < deadline:
        st = request({'cmd': 'state'})
        # Fresh AND plausible. `age_ms` catches a stale read; it does not
        # catch a corrupt one that arrived just now, which is what the
        # consecutive-agreement count is for.
        if st and st.get('angles') and len(st['angles']) == 6 \
                and st.get('age_ms', 1e9) < 1500:
            worst = max(abs(a - b) for a, b in zip(st['angles'], target))
            if worst <= tol:
                agreed += 1
                if agreed >= confirmations:
                    return True, worst
            else:
                agreed = 0
        time.sleep(poll_s)
    return False, worst


def is_stationary(tol_deg=0.3, gap_s=0.35):
    """Has the arm stopped moving? True / False / None if it cannot be read.

    This is what separates "the button stopped me" from "the link was slow".
    `wait_for_arrival` returns False for both -- it only knows the pose was
    not reached inside the budget -- and treating the second as a press would
    report a floor as selected that was never pressed.

    A stalled arm is STATIONARY short of its target. A slow one is still
    closing. Two reads a third of a second apart tell them apart, and the
    tolerance is below the 0.79deg backlash floor because a held pose still
    dithers by a fraction of a degree.
    """
    a = measured_angles()
    if a is None:
        return None
    time.sleep(gap_s)
    b = measured_angles()
    if b is None:
        return None
    return max(abs(x - y) for x, y in zip(a, b)) <= tol_deg


def measured_angles(retries=3):
    """Current joint angles, or None. Same freshness rule as the wait."""
    for _ in range(retries):
        st = request({'cmd': 'state'})
        if st and st.get('angles') and len(st['angles']) == 6 \
                and st.get('age_ms', 1e9) < 1500:
            return list(st['angles'])
        time.sleep(0.15)
    return None


def move_to(angles, speed, guard=None, tol=ARRIVE_TOL_DEG, name='move',
            confirmations=2, verbose=True):
    """Command one pose and wait for it. Returns (arrived, worst_err_deg).

    The guard is checked BEFORE commanding, never after -- a pose that is
    refused must not be sent at all.
    """
    angles = [float(a) for a in angles]
    if guard is not None:
        ok, why = guard.check(angles)
        if not ok:
            if verbose:
                print(f'{name}: guard refused -- {why}')
            return False, float('inf')

    here = measured_angles()
    travel = 180.0 if here is None else max(
        abs(a - b) for a, b in zip(here, angles))

    request({'cmd': 'send_angles', 'angles': angles, 'speed': speed,
             'force': True}, timeout=30)
    arrived, err = wait_for_arrival(angles, travel, speed, tol=tol,
                                    confirmations=confirmations)
    if verbose:
        state = 'arrived' if arrived else 'DID NOT ARRIVE'
        print(f'  {name}: {state} ({err:.2f}deg worst, {travel:.0f}deg travel)')
    return arrived, err


def stream_path(waypoints, speed, guard=None, blend_tol=BLEND_TOL_DEG,
                final_tol=ARRIVE_TOL_DEG, name='approach', verbose=True,
                blend_frac=BLEND_FRAC, final_speed_frac=FINAL_SPEED_FRAC):
    """Run a sequence of joint waypoints as ONE continuous motion.

    Every waypoint but the last is blended: the arm moves on once it is
    `blend_tol` of the way there rather than stopping dead on it. Only the
    final pose is held to `final_tol`.

    Returns (status, worst_err_deg_at_final), where status is:

        'arrived'   the final pose was reached inside `final_tol`
        'short'     the arm moved but stopped short of it
        'refused'   NOTHING was commanded -- a waypoint failed the guard

    `refused` has to be distinguishable from `short`. press_button reads a
    shortfall at the touch pose as evidence the BUTTON stopped the arm, and a
    refusal that looked like a shortfall would report a press that never
    happened, from an arm that never moved.

    All waypoints are guard-checked BEFORE any of them is commanded. Checking
    as you go means discovering the fifth one is unsafe with the arm already
    at the fourth, which is the worst place to find out.
    """
    wps = [[float(a) for a in w] for w in waypoints]
    if not wps:
        return 'refused', float('inf')

    if guard is not None:
        for i, w in enumerate(wps):
            ok, why = guard.check(w)
            if not ok:
                if verbose:
                    print(f'{name}: waypoint {i + 1}/{len(wps)} refused '
                          f'before anything moved -- {why}')
                return 'refused', float('inf')

    deg_per_s = DEG_PER_S_AT_100 * max(speed, 1) / 100.0
    la_deg = deg_per_s * LOOKAHEAD_S
    here = measured_angles()
    prev = here if here is not None else wps[0]

    for i, w in enumerate(wps):
        last = (i == len(wps) - 1)
        travel = max(abs(a - b) for a, b in zip(prev, w))

        # Slow into contact (step 6), and aim past the waypoint on every
        # segment but the last (step 1).
        seg_speed = int(round(speed * final_speed_frac)) if last else speed
        seg_speed = max(1, seg_speed)
        cmd = list(w) if last else lookahead_target(prev, w, wps[-1], la_deg)

        # The LOOKAHEAD target is what actually gets commanded, so that is
        # what has to be safe. Guard-checking only the waypoints would clear a
        # pose the arm is never sent while sending one nobody checked.
        if guard is not None and not last:
            ok, why = guard.check(cmd)
            if not ok:
                if verbose:
                    print(f'{name}: lookahead past waypoint {i + 1} is '
                          f'unsafe ({why}); commanding the waypoint itself')
                cmd = list(w)

        request({'cmd': 'send_angles', 'angles': cmd, 'speed': seg_speed,
                 'force': True}, timeout=30)

        if last:
            arrived, err = wait_for_arrival(w, travel, seg_speed,
                                            tol=final_tol, confirmations=2)
            if verbose:
                state = 'arrived' if arrived else 'stopped short'
                print(f'  {name}: {state} at the final pose '
                      f'({err:.2f}deg worst)')
            return ('arrived' if arrived else 'short'), err

        # Blend on TIME, with the tolerance as an early exit (step 7).
        # Progress is measured against the waypoint, never against the
        # lookahead target -- the arm is not meant to reach that one.
        budget = travel_budget_s(travel, seg_speed) * blend_frac
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            st = request({'cmd': 'state'})
            if st and st.get('angles') and len(st['angles']) == 6 \
                    and st.get('age_ms', 1e9) < 1500:
                if max(abs(a - b) for a, b in zip(st['angles'], w)) <= blend_tol:
                    break
            time.sleep(0.05)
        prev = w
    return False, float('inf')
