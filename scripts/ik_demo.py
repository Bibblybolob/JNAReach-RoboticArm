#!/usr/bin/env python3
"""Inverse kinematics for the myCobot 280: Cartesian target -> joint angles.

    ./scripts/ik_demo.py 0.15 0.0 0.20              # solve and print, no motion
    ./scripts/ik_demo.py 0.15 0.0 0.20 --move --serial-port /dev/ttyTHS1
    ./scripts/ik_demo.py --self-test                # no arm, no ROS

Nothing moves unless you pass --move.

WHY NOT send_coords()

pymycobot has send_coords(), which asks the arm's own firmware to solve this.
That works, but you cannot see the solution before it executes, cannot check
it against joint limits yourself, and cannot ask "is this reachable" without
commanding it. For pressing a button at a measured standoff -- where the whole
point is knowing where the flange will end up before committing -- solving it
here and sending plain joint angles is the safer shape.

GEOMETRY

Taken from src/mycobot_description/urdf/mycobot_280pi.urdf.xacro, which is the
same description MoveIt and the robot_state_publisher use, so this cannot
drift away from the rest of the stack. Each joint contributes a fixed
translate+rotate followed by a rotation about its own z:

    T_i = Translate(xyz_i) * RPY(rpy_i) * Rz(q_i)

VALIDATED AGAINST THE ARM

Checked against the firmware's own get_coords() at a real pose
(angles [-33.39, 93.86, -89.12, 18.8, 27.5, -0.61]):

    this FK   [-115.5,  25.1, 310.0] mm
    firmware  [-115.9,  25.7, 304.4] mm

X and Y agree to 0.6mm. Z reads 5.6mm high, consistently -- the firmware
measures to the flange FACE while this chain ends at link6_flange in the URDF,
so it is a tool-frame difference rather than an error in the kinematics.
Subtract it if you need to match the firmware's numbers exactly; for
commanding joint angles, which is what this does, it does not enter into it.

METHOD

Damped least squares on the 6-DOF pose error, seeded from a starting guess.
A myCobot 280 has no spherical wrist, so there is no closed form to reach for;
iterating is the honest approach and it converges in a few dozen steps. The
damping is what keeps it stable near singularities, where a plain pseudo-
inverse asks for enormous joint velocities.

The answer is checked before it is returned: the solution is run back through
forward kinematics and rejected if it does not land where it was asked to.
An IK routine that quietly returns its best failed attempt is worse than one
that says it could not get there.
"""
import argparse
import math
import sys

import numpy as np

# Route through the arm broker when one is running, so this script does not
# open /dev/ttyTHS1 itself. Two openers interleave bytes on that tty and the
# Atom reports the resulting bad length field as `cmd_len error` -- which
# reads as a firmware fault and has cost whole sessions.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
try:
    from arm_broker import connect_arm as _connect_arm
except Exception:  # broker module unavailable; behave exactly as before
    def _connect_arm(port, factory, **kw):
        return factory()


# (xyz, rpy, lower, upper) per joint, straight out of the URDF.
CHAIN = [
    ((0.0, 0.0, 0.13956), (0.0, 0.0, 0.0), -2.932, 2.932),
    ((0.0, 0.0, -0.001), (0.0, 1.5708, -1.5708), -2.443, 2.443),
    ((-0.1104, 0.0, 0.0), (0.0, 0.0, 0.0), -2.618, 2.618),
    ((-0.096, 0.0, 0.06462), (0.0, 0.0, -1.5708), -2.618, 2.618),
    ((0.0, -0.07318, -0.001), (1.5708, -1.5708, 0.0), -2.705, 2.792),
    ((0.0, 0.0456, 0.0), (-1.5708, 0.0, 0.0), -3.142, 3.142),
]
HOME_DEG = [0.0, 90.0, -90.0, 0.0, 0.0, 0.0]


def rpy(r, p, y):
    """URDF rpy is fixed-axis: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def rot_z(q):
    c, s = math.cos(q), math.sin(q)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fk(q, upto=6):
    """Flange pose as a 4x4, for joint angles in radians."""
    T = np.eye(4)
    for i in range(upto):
        xyz, r, _, _ = CHAIN[i]
        step = np.eye(4)
        step[:3, :3] = rpy(*r) @ rot_z(q[i])
        step[:3, 3] = xyz
        T = T @ step
    return T


def _log_so3(R):
    """Rotation matrix -> rotation vector, the error the solver drives down."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    angle = math.acos(c)
    if angle < 1e-9:
        return np.zeros(3)
    if abs(angle - math.pi) < 1e-6:
        # Near pi the usual formula loses precision; take the axis from the
        # symmetric part instead.
        w = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(w))
        axis = np.zeros(3)
        axis[i] = w[i]
        for j in range(3):
            if j != i:
                axis[j] = R[i, j] / (2.0 * w[i]) if w[i] > 1e-9 else 0.0
        return axis / np.linalg.norm(axis) * angle
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (angle / (2.0 * math.sin(angle)))


def jacobian(q, eps=1e-6):
    """Numeric 6xN. Analytic would be faster and is not the bottleneck here --
    a solve is a few hundred FK evaluations and takes under a millisecond."""
    T0 = fk(q)
    J = np.zeros((6, 6))
    for i in range(6):
        dq = list(q)
        dq[i] += eps
        T1 = fk(dq)
        J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
        J[3:, i] = _log_so3(T1[:3, :3] @ T0[:3, :3].T) / eps
    return J


def pose_error(T, target_xyz, target_R):
    e = np.zeros(6)
    e[:3] = target_xyz - T[:3, 3]
    if target_R is not None:
        e[3:] = _log_so3(target_R @ T[:3, :3].T)
    return e


def ik(target_xyz, target_R=None, seed=None, iters=200, damping=0.05,
       pos_tol=1e-4, rot_tol=1e-3):
    """Joint angles (radians) reaching the target, or None.

    target_R=None solves POSITION ONLY, leaving orientation free. That is
    usually what you want on a 6-DOF arm being asked for a point: demanding a
    full pose can make a reachable position unreachable.
    """
    q = np.array(seed if seed is not None
                 else [math.radians(d) for d in HOME_DEG], dtype=float)
    lo = np.array([c[2] for c in CHAIN])
    hi = np.array([c[3] for c in CHAIN])
    rows = 6 if target_R is not None else 3

    for _ in range(iters):
        T = fk(q)
        e = pose_error(T, target_xyz, target_R)[:rows]
        if (np.linalg.norm(e[:3]) < pos_tol
                and (rows == 3 or np.linalg.norm(e[3:]) < rot_tol)):
            break
        J = jacobian(q)[:rows]
        # Damped least squares: J^T (J J^T + lambda^2 I)^-1 e. The damping is
        # what stops a near-singular pose demanding an enormous step.
        JT = J.T
        q = q + JT @ np.linalg.solve(
            J @ JT + (damping ** 2) * np.eye(rows), e)
        q = np.clip(q, lo, hi)

    # Verify rather than trust. Clipping to joint limits inside the loop can
    # leave the solver converged onto something it cannot actually reach.
    T = fk(q)
    err = np.linalg.norm(target_xyz - T[:3, 3])
    if err > 1e-3:
        return None
    if target_R is not None:
        if np.linalg.norm(_log_so3(target_R @ T[:3, :3].T)) > 1e-2:
            return None
    return q


def self_test():
    """Round-trip: FK a random reachable pose, solve for it, check it lands.

    Catches the errors that matter -- a wrong rpy convention, a transposed
    rotation, a mis-signed axis -- none of which show up as anything but
    slightly wrong numbers otherwise.
    """
    rng = np.random.default_rng(0)
    lo = np.array([c[2] for c in CHAIN])
    hi = np.array([c[3] for c in CHAIN])

    print('FK at home', HOME_DEG, '->')
    Th = fk([math.radians(d) for d in HOME_DEG])
    print(f'   flange xyz = ({Th[0,3]:+.4f}, {Th[1,3]:+.4f}, {Th[2,3]:+.4f}) m')
    print(f'   reach from base axis = '
          f'{math.hypot(Th[0,3], Th[1,3])*1000:.0f} mm, '
          f'height {Th[2,3]*1000:.0f} mm\n')

    fails = 0
    worst = 0.0
    trials = 60
    for _ in range(trials):
        q_true = rng.uniform(lo * 0.7, hi * 0.7)
        T = fk(q_true)
        seed = q_true + rng.normal(0.0, 0.3, 6)
        q = ik(T[:3, 3], seed=np.clip(seed, lo, hi))
        if q is None:
            fails += 1
            continue
        got = fk(q)[:3, 3]
        worst = max(worst, float(np.linalg.norm(got - T[:3, 3])))
    print(f'position-only round trip: {trials - fails}/{trials} solved, '
          f'worst error {worst*1000:.4f} mm')

    fails6 = 0
    worst6 = 0.0
    for _ in range(trials):
        q_true = rng.uniform(lo * 0.7, hi * 0.7)
        T = fk(q_true)
        seed = q_true + rng.normal(0.0, 0.15, 6)
        q = ik(T[:3, 3], T[:3, :3], seed=np.clip(seed, lo, hi))
        if q is None:
            fails6 += 1
            continue
        got = fk(q)
        worst6 = max(worst6, float(np.linalg.norm(got[:3, 3] - T[:3, 3])))
    print(f'full-pose round trip:     {trials - fails6}/{trials} solved, '
          f'worst error {worst6*1000:.4f} mm')

    unreachable = ik(np.array([1.0, 0.0, 0.0]))
    print(f'unreachable target (1m away): '
          f'{"correctly refused" if unreachable is None else "WRONGLY SOLVED"}')
    ok = fails == 0 and worst < 1e-3 and unreachable is None
    print('\n' + ('SELF-TEST PASSED' if ok else 'SELF-TEST FAILED'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xyz', nargs='*', type=float,
                    help='target x y z in METRES, base frame')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--move', action='store_true',
                    help='actually command the arm (otherwise print only)')
    ap.add_argument('--serial-port', default='/dev/ttyTHS1')
    ap.add_argument('--baud', type=int, default=1000000)
    ap.add_argument('--speed', type=int, default=30)
    args = ap.parse_args()

    if args.self_test or not args.xyz:
        return self_test()
    if len(args.xyz) != 3:
        sys.exit('give exactly three numbers: x y z in metres')

    target = np.array(args.xyz, dtype=float)
    print(f'target: ({target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}) m')

    q = ik(target)
    if q is None:
        reach = np.linalg.norm(target)
        sys.exit(
            f'\nNo solution. The target is {reach*1000:.0f} mm from the base '
            f'origin;\na 280 reaches roughly 280 mm, and joint limits remove '
            f'much of that.\nTry somewhere closer, or free the orientation by '
            f'moving the point.')

    deg = [math.degrees(v) for v in q]
    print('\njoint angles (degrees):')
    for i, d in enumerate(deg, 1):
        lo, hi = math.degrees(CHAIN[i-1][2]), math.degrees(CHAIN[i-1][3])
        margin = min(d - lo, hi - d)
        flag = '   <- near limit' if margin < 10.0 else ''
        print(f'  joint{i}: {d:+8.2f}   ({lo:+.0f}..{hi:+.0f}){flag}')

    got = fk(q)[:3, 3]
    print(f'\nforward-checks to ({got[0]:+.4f}, {got[1]:+.4f}, {got[2]:+.4f}) '
          f'-- {np.linalg.norm(got-target)*1000:.3f} mm from target')

    if not args.move:
        print('\nNot moving. Re-run with --move to command the arm.')
        return 0

    try:
        from pymycobot import MyCobot280
    except ImportError:
        sys.exit('pymycobot not installed')
    print(f'\nCLEAR THE ARM. Commanding in 3s via {args.serial_port}...')
    import time
    time.sleep(3.0)
    mc = _connect_arm(args.serial_port,
                      lambda: MyCobot280(args.serial_port, str(args.baud)))
    time.sleep(0.5)
    # _async: SEND_ANGLES has no reply on the serial path and blocks 1550ms
    # waiting for one. Same reason the driver does this.
    mc.send_angles([round(d, 2) for d in deg], args.speed, _async=True)
    print('commanded.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
