#!/usr/bin/env python3
"""
Measure the two numbers the motion tuning is currently guessing at.

Run this with the ROS 2 driver STOPPED — Server.py on the Pi accepts one
client, so this script and the driver cannot both be connected.

    python3 scripts/measure_arm.py --ip 192.168.0.15

It reports:

  1. Command round-trip time. Determines how fast we can stream: if
     command_interval is shorter than the link can sustain, commands queue and
     motion stutters. Set command_interval to roughly 2x the p95 send time.

  2. Actual joint speed at send_angles speed=100, in deg/s. Two parameters
     depend on this:
       - speed_at_100_deg_s in the driver (used to size each streaming step)
       - max_velocity in joint_limits.yaml, which MoveIt's
         AddTimeOptimalParameterization uses to time trajectories. If that
         claims the arm is faster than it is, the arm can never track the
         plan and motion degrades no matter what the driver does.

SAFETY: this MOVES THE ARM. joint1 sweeps +/-40 degrees from wherever it
currently sits, and the arm must be clear to do that. Nothing else is touched.
Keep a hand on the e-stop. Ctrl-C stops the script but NOT the arm mid-move.

    ./scripts/measure_arm.py --serial-port /dev/ttyTHS1     # Jetson on UART
    ./scripts/measure_arm.py --ip 192.168.0.15              # Pi over TCP

WHY speed_at_100_deg_s MATTERS MORE THAN IT LOOKS

The driver sizes every streamed step from it:

    speed = (needed_deg_s / speed_at_100_deg_s) * 100

so if the real figure is LOWER than the 120 currently assumed, the driver asks
for proportionally less speed than it means to and the arm falls behind its own
commanded goal. That shows up as `Jog target pinned to the measured pose`, as a
servo whose corrections never seem to land, and as tracking that feels sluggish
while every other number in the pipeline looks healthy -- detection rate fine,
latency fine, jogs being issued at full size. Guessing it high is the failure
mode that hides itself.
"""

import argparse
import os
import statistics
import sys
import time

try:
    from pymycobot import MyCobot280, MyCobot280Socket
except ImportError:
    sys.exit('pymycobot not installed: pip install pymycobot')


def measure_round_trip(mc, n=30):
    """Time get_angles(), which is the expensive has_return path."""
    times = []
    for _ in range(n):
        t0 = time.monotonic()
        mc.get_angles()
        times.append((time.monotonic() - t0) * 1000.0)
        time.sleep(0.02)
    times.sort()
    return {
        'min': times[0],
        'median': statistics.median(times),
        'p95': times[int(len(times) * 0.95) - 1],
        'max': times[-1],
    }


def measure_send_cost(mc, angles, n=20):
    """Time send_angles(), which should be fire-and-forget (not in has_return)."""
    times = []
    for _ in range(n):
        t0 = time.monotonic()
        mc.send_angles(angles, 20)
        times.append((time.monotonic() - t0) * 1000.0)
        time.sleep(0.05)
    times.sort()
    return {
        'min': times[0],
        'median': statistics.median(times),
        'p95': times[int(len(times) * 0.95) - 1],
        'max': times[-1],
    }


def measure_joint_speed(mc, start, sweep_deg, speed, settle=3.0):
    """Sweep joint1 by sweep_deg at the given speed and time the travel."""
    target = list(start)
    target[0] = start[0] + sweep_deg

    mc.send_angles(list(start), 50)
    time.sleep(settle)

    begin = mc.get_angles()
    t0 = time.monotonic()
    mc.send_angles(target, speed)

    # Poll until motion stops rather than until the target is hit: the arm may
    # stop short, and we want the speed it actually achieved.
    last = begin
    still_since = None
    while time.monotonic() - t0 < 15.0:
        time.sleep(0.05)
        cur = mc.get_angles()
        if not isinstance(cur, list) or len(cur) != 6:
            continue
        if abs(cur[0] - last[0]) < 0.3:
            if still_since is None:
                still_since = time.monotonic()
            elif time.monotonic() - still_since > 0.4:
                break
        else:
            still_since = None
        last = cur

    elapsed = (still_since or time.monotonic()) - t0
    travelled = abs(last[0] - begin[0])
    return travelled, elapsed, (travelled / elapsed if elapsed > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--ip', default=os.environ.get('MYCOBOT_IP', '192.168.0.15'),
        help='Pi address; defaults to $MYCOBOT_IP, then 192.168.0.15')
    ap.add_argument('--port', type=int, default=9000)
    # The serial path. Without this the script could only measure the arm
    # over TCP through the Pi -- which is not the connection the numbers are
    # wanted for once the Jetson drives the UART directly, and the round-trip
    # figures differ by the whole network leg.
    ap.add_argument('--serial-port', metavar='DEV',
                    help='drive the arm over serial instead of TCP, e.g. '
                         '/dev/ttyTHS1 on a Jetson. Skips the Pi entirely.')
    ap.add_argument('--baud', type=int, default=1000000,
                    help='serial baud (default 1000000, what the arm uses)')
    ap.add_argument('--sweep', type=float, default=40.0,
                    help='degrees to sweep joint1 (default 40)')
    ap.add_argument('--skip-motion', action='store_true',
                    help='only measure link timing, never move the arm')
    args = ap.parse_args()

    if args.serial_port:
        print(f'Connecting over serial: {args.serial_port} at {args.baud} ...')
        mc = MyCobot280(args.serial_port, str(args.baud))
    else:
        print(f'Connecting to {args.ip}:{args.port} ...')
        mc = MyCobot280Socket(args.ip, args.port)
    time.sleep(0.5)

    start = mc.get_angles()
    if not isinstance(start, list) or len(start) != 6:
        sys.exit(f'Could not read joint angles (got {start!r}). Is the arm powered?')
    print(f'Current angles: {[round(a, 1) for a in start]}\n')

    print('--- 1. Link timing ---')
    rt = measure_round_trip(mc)
    print(f'  get_angles()  min {rt["min"]:6.1f}ms   median {rt["median"]:6.1f}ms   '
          f'p95 {rt["p95"]:6.1f}ms   max {rt["max"]:6.1f}ms')
    st = measure_send_cost(mc, start)
    print(f'  send_angles() min {st["min"]:6.1f}ms   median {st["median"]:6.1f}ms   '
          f'p95 {st["p95"]:6.1f}ms   max {st["max"]:6.1f}ms')

    suggested_interval = max(0.03, round(st['p95'] * 2 / 1000.0, 3))
    print(f'\n  => suggested command_interval: {suggested_interval:.3f} s '
          f'(2x send p95)')

    if args.skip_motion:
        print('\n--skip-motion given; not measuring joint speed.')
        return

    print('\n--- 2. Joint speed ---')
    print(f'  About to sweep joint1 by {args.sweep:+.0f} degrees.')
    print('  MAKE SURE THE ARM IS CLEAR. Ctrl-C now to abort.')
    for i in range(5, 0, -1):
        print(f'    {i}...', end='\r', flush=True)
        time.sleep(1)
    print('    sweeping...      ')

    results = {}
    for speed in (100, 60, 30):
        travelled, elapsed, deg_s = measure_joint_speed(
            mc, start, args.sweep, speed)
        results[speed] = deg_s
        print(f'  speed={speed:3d}: {travelled:5.1f} deg in {elapsed:4.2f}s '
              f'= {deg_s:6.1f} deg/s')

    mc.send_angles(list(start), 40)
    print('\n  returning to starting pose...')
    time.sleep(3)

    v100 = results.get(100, 0.0)
    if v100 > 0:
        print(f'\n=== Apply these ===')
        print(f'  driver param  speed_at_100_deg_s: {v100:.0f}')
        print(f'  driver param  command_interval:   {suggested_interval:.3f}')
        # joint_limits wants rad/s, and MoveIt should plan a bit under what the
        # arm can actually do or it will never keep up.
        safe_rad_s = (v100 * 0.8) * 3.14159 / 180.0
        print(f'  joint_limits.yaml max_velocity:   {safe_rad_s:.2f}   '
              f'(80% of measured, all 6 joints)')


if __name__ == '__main__':
    main()
