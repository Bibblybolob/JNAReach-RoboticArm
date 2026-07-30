#!/usr/bin/env python3
"""First motion over a direct UART link. One joint, small, and verified.

    ./scripts/serial_move_test.py --port /dev/ttyACM0

Moves joint 1 by 15 degrees, checks it arrived, and puts it back. No ROS, no
camera, no launch files -- if this works the UART path is proven and ./run.py
connection:=serial will work; if it does not, the problem is below all of that.

WHY IT CHECKS THE LINK FIRST

The myCobot protocol has no checksum. A frame is FE FE <len> <cmd> <payload>
FA and nothing validates the payload, so a single flipped bit in a joint angle
is not a dropped message -- it is a different angle, which the arm accepts and
drives to. On a link that is 99.9% clean that is a rare large surprise rather
than a small constant error, which is the worse failure mode of the two.

So before commanding anything this reads the arm's own angles repeatedly with
the arm stationary. Every reply should agree. Replies that do not are the
corruption rate on the real wire, measured with real protocol traffic -- which
is something probe_uart_bridge.py's loopback structurally cannot tell you,
since a loopback shares its clock and its wire with itself.

It refuses to move if that check fails. --skip-link-check overrides it, and
the reason to want that is usually not a good one.

BEFORE RUNNING

Exactly one master on the bus. The Pi drives the same two lines, and stopping
its service is not sufficient -- GPIO14 stays in ALT0 actively driving pin 8:

    ssh er@<pi> 'sudo systemctl stop mycobot_server'
    ssh er@<pi> 'sudo pkill -f uart_peripheral_serial'
    ssh er@<pi> 'sudo raspi-gpio set 14 ip'

If the Pi is unreachable you cannot do that, and you cannot assume it is idle
either -- a Pi off the network is still a Pi driving the UART. Run
`probe_uart_bridge.py listen` to find out: traffic on pin 10 means something
is polling the arm, and that something is competing with you.
"""

import argparse
import sys
import time


# myCobot 280 software limits, per joint, degrees. Hardcoded rather than read
# from the arm: get_joint_min_angle is 6 more round trips on a link whose
# reliability is the thing in question, and a wrong reply here would widen a
# limit rather than narrow it.
LIMITS = {
    1: (-168.0, 168.0), 2: (-135.0, 135.0), 3: (-150.0, 150.0),
    4: (-145.0, 145.0), 5: (-165.0, 165.0), 6: (-180.0, 180.0),
}


def connect(port, baud):
    try:
        from pymycobot import MyCobot280
    except ImportError as e:
        sys.exit(f'pymycobot missing: {e}')
    print(f'Opening {port} at {baud} baud...')
    try:
        return MyCobot280(port, str(baud))
    except Exception as e:
        name = e.__class__.__name__
        if 'Permission' in name or 'Permission' in str(e):
            sys.exit(f'{port}: permission denied. '
                     f'sudo usermod -aG dialout $USER, then log back in.')
        sys.exit(f'Cannot open {port}: {name}: {e}')


def check_link(mc, samples, tolerance):
    """Measure reply integrity with the arm stationary. Returns (ok, pose)."""
    print(f'\nChecking the link: {samples} reads with the arm still.')
    good = []
    bad = 0
    for _ in range(samples):
        try:
            a = mc.get_angles()
        except Exception as e:
            print(f'  read raised {e.__class__.__name__}: {e}')
            bad += 1
            continue
        if isinstance(a, list) and len(a) == 6 and all(
                isinstance(v, (int, float)) for v in a):
            good.append([float(v) for v in a])
        else:
            bad += 1
        time.sleep(0.02)

    total = len(good) + bad
    print(f'  {len(good)}/{total} valid replies, {bad} malformed')

    if not good:
        print('\nThe arm never answered with angles.\n')
        print('The port opens and bytes go out, so this is the bus rather')
        print('than the adapter -- probe_uart_bridge.py loopback already')
        print('cleared the adapter. In order of likelihood:\n')
        print('  1. Something else is driving these lines. mycobot_server on')
        print('     the Pi, the rc.local Bluetooth bridge, or GPIO14 still in')
        print('     ALT0 because raspi-gpio set 14 ip never ran. A Pi you')
        print('     cannot SSH into is NOT a Pi that has released the bus.')
        print('  2. D0 is not on the arm\'s RX line, or the grounds are not')
        print('     tied together.')
        print('  3. The arm is not powered.\n')
        print('`probe_uart_bridge.py listen` separates 1 from 2: if you can')
        print('see frames arriving, the wiring and baud are right and you')
        print('have a contention problem.')
        return False, None

    # Stationary, so every joint should read the same each time. Spread is
    # corruption; a single flipped bit in a 2-byte angle shows up as a jump
    # far larger than sensor noise.
    spread = [max(r[j] for r in good) - min(r[j] for r in good)
              for j in range(6)]
    worst = max(spread)
    median = [sorted(r[j] for r in good)[len(good) // 2] for j in range(6)]
    print('  pose: [' + ', '.join(f'{v:.1f}' for v in median) + ']')
    print(f'  worst spread across reads: {worst:.2f} deg '
          f'(per joint: ' + ', '.join(f'{s:.1f}' for s in spread) + ')')

    if bad or worst > tolerance:
        print(f'\nThe link is not clean enough to command motion.\n')
        if bad:
            print(f'{bad} of {total} replies were malformed.')
        if worst > tolerance:
            print(f'A stationary arm read {worst:.2f} deg apart across '
                  f'reads, over the {tolerance:.1f} deg allowed.')
        print('\nThe protocol has no checksum, so corruption that shows up')
        print('here as a bad reading would show up in a command as a joint')
        print('angle the arm accepts and moves to. Fix the link first:')
        print('shorter wires, a real ground, and nothing else on the bus.')
        return False, median

    print(f'  clean (spread under {tolerance:.1f} deg)')
    return True, median


def move(mc, joint, start, delta, speed, timeout, tolerance):
    """Command one joint and report whether it actually arrived."""
    lo, hi = LIMITS[joint]
    target = max(lo, min(hi, start[joint - 1] + delta))
    travel = abs(target - start[joint - 1])
    # A move clamped down to a degree or two would "succeed" without proving
    # anything -- the arm has to visibly go somewhere for this to be evidence.
    floor = min(5.0, abs(delta))
    if travel < floor:
        print(f'\nJoint {joint} is at {start[joint - 1]:.1f} deg and its '
              f'limit is {lo}..{hi}, so --delta {delta:+.0f} leaves only '
              f'{travel:.1f} deg of travel.')
        print(f'That is too small to be evidence either way. '
              f'Try --delta {-delta:+.0f}.')
        return False

    print(f'\nJoint {joint}: {start[joint - 1]:.1f} -> {target:.1f} deg '
          f'at speed {speed}')
    mc.send_angle(joint, target, speed)

    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        time.sleep(0.15)
        try:
            a = mc.get_angles()
        except Exception:
            continue
        if not (isinstance(a, list) and len(a) == 6):
            continue
        last = float(a[joint - 1])
        if abs(last - target) <= tolerance:
            print(f'  arrived in {time.time() - t0:.2f}s '
                  f'({last:.1f} deg, {abs(last - target):.2f} off)')
            return True

    if last is None:
        print(f'  no readable position in {timeout:.0f}s -- the arm stopped '
              f'answering as soon as it was commanded, which is what a bus '
              f'with two masters on it looks like.')
    else:
        print(f'  did NOT arrive in {timeout:.0f}s (got to {last:.1f}, '
              f'wanted {target:.1f})')
        print('  A joint that is commanded and does not move is worth '
              'chasing before anything else --')
        print('  it poisons the servo\'s lag compensator and feedforward '
              'later on. Check it is')
        print('  not at a mechanical limit and that the servo is enabled.')
    return False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--baud', type=int, default=1000000)
    ap.add_argument('--joint', type=int, default=1, choices=range(1, 7),
                    help='1 = base pan, the most visible and the safest (1)')
    ap.add_argument('--delta', type=float, default=15.0,
                    help='degrees to move, clamped to the joint limit (15)')
    ap.add_argument('--speed', type=int, default=30,
                    help='0-100, deliberately low (30)')
    ap.add_argument('--samples', type=int, default=20,
                    help='link-check reads (20)')
    ap.add_argument('--tolerance', type=float, default=1.5,
                    help='degrees of disagreement allowed between reads, and '
                         'of arrival error (1.5)')
    ap.add_argument('--timeout', type=float, default=10.0)
    ap.add_argument('--no-return', action='store_true',
                    help='leave the joint where it ended up')
    ap.add_argument('--skip-link-check', action='store_true',
                    help='command motion without measuring reply integrity '
                         'first; see the module docstring for why not')
    args = ap.parse_args()

    mc = connect(args.port, args.baud)
    time.sleep(0.5)

    try:
        if args.skip_link_check:
            print('\nSkipping the link check, as asked. The protocol has no '
                  'checksum;\ncorruption will present as motion to an angle '
                  'you did not command.')
            pose = mc.get_angles()
            if not (isinstance(pose, list) and len(pose) == 6):
                sys.exit(f'Cannot read a starting pose: {pose!r}')
            pose = [float(v) for v in pose]
        else:
            ok, pose = check_link(mc, args.samples, args.tolerance)
            if not ok:
                return 1

        powered = mc.is_power_on()
        if powered == 0:
            print('\nThe arm reports its servos are off, so it will not move.')
            print('Energising a robot is not something this script does for')
            print('you -- make sure it is clear, then:\n')
            print(f"    python3 -c \"from pymycobot import MyCobot280; "
                  f"MyCobot280('{args.port}', '{args.baud}').power_on()\"")
            return 1
        if powered != 1:
            print(f'\n(is_power_on returned {powered!r} -- carrying on, but '
                  f'that is not a clean yes)')

        moved = move(mc, args.joint, pose, args.delta, args.speed,
                     args.timeout, args.tolerance)

        if not args.no_return:
            print(f'\nReturning joint {args.joint} to {pose[args.joint-1]:.1f}'
                  f' deg')
            mc.send_angle(args.joint, pose[args.joint - 1], args.speed)
            time.sleep(min(args.timeout, 3.0))

        if moved:
            print('\n' + '=' * 68)
            print('  The arm moves over the direct UART link.')
            print('=' * 68)
            print('\nThat is the whole Jetson topology validated. Run the '
                  'stack the same way:\n')
            print(f'    ./run.py connection:=serial '
                  f'serial_port:={args.port} \\')
            print(f'        serial_baud:={args.baud} source:=device device:=0')
            return 0
        return 1

    except KeyboardInterrupt:
        print('\nInterrupted -- stopping the arm.')
        try:
            mc.stop()
        except Exception:
            pass
        return 130
    finally:
        try:
            mc._serial_port.close()
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main())
