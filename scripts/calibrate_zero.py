#!/usr/bin/env python3
"""Set the current physical pose as the arm's zero. Run with joints on their notches.

    ./scripts/calibrate_zero.py

Every joint has a notch marking its origin. Line them all up by hand with the
servos released, then run this: it calls set_servo_calibration on each servo,
which writes "here is 0" into that servo. Afterwards get_angles should read
approximately [0,0,0,0,0,0] and the raw encoders approximately 2048.

**This overwrites the arm's zero references.** If it runs with the joints in
the wrong place, every angle the arm reports and every angle it is commanded
to is wrong by that much, and the fix is to do it again properly -- so it is
worth being fussy about the alignment first.

Why this is needed here: reflashing the Atom on 2026-08-10 left the arm
reporting angles that did not match its physical pose. The encoder-to-angle
maths was exact (360/4096 per count, verified against all six joints), so the
conversion was never the problem -- the zero references were.
"""

import sys
import time

try:
    from pymycobot import MyCobot280
except ImportError:
    sys.exit('pymycobot is not installed: pip install pymycobot')

PORT = '/dev/ttyTHS1'
BAUD = '1000000'


def connect(attempts=3, patience=25.0):
    for attempt in range(attempts):
        print(f'connecting ({attempt + 1})...', end=' ', flush=True)
        try:
            mc = MyCobot280(PORT, BAUD)
        except Exception as e:
            print(f'open failed: {e}')
            time.sleep(2.0)
            continue
        time.sleep(3.0)
        deadline = time.monotonic() + patience
        while time.monotonic() < deadline:
            try:
                if isinstance(mc.get_angles(), list):
                    print('ok')
                    return mc
            except Exception:
                pass
            time.sleep(0.5)
        print('no answer')
        try:
            mc._serial_port.close()
        except Exception:
            pass
        time.sleep(2.0)
    return None


def retry(fn, *args, tries=8):
    for _ in range(tries):
        try:
            value = fn(*args)
        except Exception as e:
            return f'raised {type(e).__name__}: {e}'
        if value != -1:
            return value
        time.sleep(0.25)
    return -1


def main():
    mc = connect()
    if mc is None:
        sys.exit('could not reach the arm')

    print()
    print('before calibration:')
    print('  encoders:', [retry(mc.get_encoder, j) for j in range(1, 7)])
    print('  angles:  ', mc.get_angles())
    print()
    print('This writes the CURRENT physical pose as zero for all six joints.')
    print('Every joint must be sitting on its notch right now.')
    try:
        if not input('Aligned and ready? [y/N] ').strip().lower().startswith('y'):
            sys.exit('aborted -- nothing written')
    except (EOFError, KeyboardInterrupt):
        sys.exit('\naborted -- nothing written')

    print()
    for joint in range(1, 7):
        try:
            mc.set_servo_calibration(joint)
            print(f'  joint{joint} calibrated')
        except Exception as e:
            print(f'  joint{joint} FAILED: {e}')
        time.sleep(0.5)

    time.sleep(1.5)
    print()
    print('after calibration:')
    encoders = [retry(mc.get_encoder, j) for j in range(1, 7)]
    print('  encoders:', encoders, ' (want ~2048 each)')
    angles = None
    for _ in range(12):
        angles = mc.get_angles()
        if isinstance(angles, list) and len(angles) == 6:
            break
        time.sleep(0.2)
    print('  angles:  ', angles, ' (want ~0 each)')

    if isinstance(angles, list) and len(angles) == 6:
        worst = max(abs(a) for a in angles)
        print()
        if worst < 5.0:
            print(f'GOOD -- every joint within {worst:.1f} degrees of zero.')
            print('Next: re-engage the servos and test a small move.')
        else:
            print(f'joint furthest from zero: {worst:.1f} degrees.')
            print('If that is more than the notch alignment error, the '
                  'calibration did not take on every joint.')


if __name__ == '__main__':
    main()
