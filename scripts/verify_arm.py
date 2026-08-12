#!/usr/bin/env python3
"""Post-flash acceptance test: does the arm read, and does it obey?

    ./scripts/verify_arm.py

Runs the checks in order of increasing commitment, stopping at the first
failure, so a broken arm is never asked to move:

  1. connect and read the firmware version
  2. read angles repeatedly -- how reliable is the link, really
  3. set_color -- a WRITE that touches no servo. If the LED changes, the
     controller executes commands. This is the check that mattered on
     2026-08-10: it separates "will not obey" from every mechanical,
     torque and power explanation in one step, because a LED cannot be
     blocked, stalled or starved of current.
  4. one small joint6 move -- the lightest joint, 10 degrees, low speed

Expected firmware is 6.2: the known-good version for a 280 Pi. 6.4 is the
version in elephantrobotics/myCobot issue #48 (motors unresponsive,
send_angles ignored) and 6.5+ is documented as failing to write.
"""

import sys
import time

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


try:
    from pymycobot import MyCobot280
except ImportError:
    sys.exit('pymycobot is not installed: pip install pymycobot')

PORT = '/dev/ttyTHS1'
BAUD = '1000000'
GOOD_FIRMWARE = 6.2


def connect(patience=40.0):
    print(f'[1/4] connecting to {PORT}...', end=' ', flush=True)
    mc = _connect_arm(PORT, lambda: MyCobot280(PORT, BAUD))
    time.sleep(3.0)
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        if isinstance(mc.get_angles(), list):
            print('ok')
            return mc
        time.sleep(0.5)
    print('no answer')
    return None


def read(mc, tries=15):
    for _ in range(tries):
        angles = mc.get_angles()
        if isinstance(angles, list) and len(angles) == 6:
            return angles
        time.sleep(0.2)
    return None


def retry(fn, *args, tries=8):
    for _ in range(tries):
        try:
            value = fn(*args)
        except Exception:
            return -1
        if value != -1:
            return value
        time.sleep(0.25)
    return -1


def main():
    mc = connect()
    if mc is None:
        sys.exit('FAIL: no reply. Power-cycle the arm and retry; if it stays '
                 'silent the flash did not take.')

    version = retry(mc.get_system_version)
    print(f'      firmware: {version}')
    if isinstance(version, (int, float)) and abs(version - GOOD_FIRMWARE) > 0.05:
        print(f'      NOTE: expected {GOOD_FIRMWARE}. 6.4 is the version in '
              'issue #48; 6.5+ cannot be written to.')

    print('[2/4] reading angles 20 times...')
    good = 0
    for _ in range(20):
        if isinstance(mc.get_angles(), list):
            good += 1
        time.sleep(0.15)
    print(f'      {good}/20 replies')
    if good == 0:
        sys.exit('FAIL: answered the handshake then went quiet -- the '
                 'degradation seen before the flash.')
    if good < 15:
        print('      WARNING: patchy. Mixed servo-bus traffic on the host '
              'UART can look like this; pymycobot cannot parse it and '
              'returns -1.')

    pose = read(mc)
    print('      pose: ' + ('  '.join(f'j{i + 1}={a:.1f}'
                                      for i, a in enumerate(pose))
                            if pose else 'unreadable'))

    print('[3/4] LED write test -- WATCH THE ATOM')
    for name, r, g, b in (('RED', 255, 0, 0), ('BLUE', 0, 0, 255),
                          ('GREEN', 0, 255, 0)):
        print(f'      -> {name}', flush=True)
        for _ in range(3):
            try:
                mc.set_color(r, g, b)
            except Exception as e:
                print(f'         set_color: {e}')
            time.sleep(0.2)
        time.sleep(1.8)

    answer = input('      Did the LED change colour? [y/N] ').strip().lower()
    if not answer.startswith('y'):
        sys.exit(
            'FAIL: reads work, writes do not -- unchanged from before the '
            'flash. set_color touches no servo, so this is not torque, '
            'power or anything mechanical. Next step is Elephant support, '
            'citing issue #48.')

    print('      writes work.')

    print('[4/4] moving joint6 by 10 degrees')
    for name in ('power_on', 'focus_all_servos'):
        fn = getattr(mc, name, None)
        if fn:
            try:
                fn()
            except Exception:
                pass
            time.sleep(0.4)

    before = read(mc)
    if before is None:
        sys.exit('FAIL: lost the arm before the move')
    target = list(before)
    target[5] = max(-170.0, min(170.0, before[5] + 10.0))
    print(f'      joint6 {before[5]:.1f} -> {target[5]:.1f}')
    for _ in range(3):
        try:
            mc.send_angles(target, 30)
        except Exception as e:
            print(f'      send_angles: {e}')
        time.sleep(2.0)

    after = read(mc)
    if after is None:
        sys.exit('FAIL: lost the arm during the move')
    moved = after[5] - before[5]
    print(f'      joint6 now {after[5]:.1f} (moved {moved:+.1f})')

    seen = input('      Did joint6 physically move? [y/N] ').strip().lower()
    print()
    if seen.startswith('y') and abs(moved) > 2.0:
        print('PASS -- the arm reads, obeys writes, and moves. Bring the '
              'stack up:')
        print('  ./run.py connection:=serial serial_port:=/dev/ttyTHS1 '
              'serial_baud:=1000000 source:=realsense')
    elif abs(moved) > 2.0:
        print('The reading moved but you did not see motion. That is the '
              'unexplained case from 2026-08-10 -- get_angles may be '
              'reporting a commanded pose. Run ./scripts/check_encoders.py')
    else:
        print('Writes reach the controller (the LED changed) but motion did '
              'not happen. That narrows it to the servo drive path, with the '
              'servo bus itself known good -- temps and voltages read live.')


if __name__ == '__main__':
    main()
