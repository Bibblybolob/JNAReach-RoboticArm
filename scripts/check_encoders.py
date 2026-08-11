#!/usr/bin/env python3
"""Is get_angles reading the encoders, or echoing what we commanded?

    ./scripts/check_encoders.py

On 2026-08-10 scripts/jog_joints.py reported joint2 walking 89.7 -> 50.4
degrees in clean ~4 degree increments, each step apparently confirmed by
get_angles, while the arm physically did not move. If the controller reports
commanded position rather than encoder position, then every angle-derived
conclusion about this arm is worthless -- stalls, torque limits, homing
failures, all of it.

The test needs a human, because it is the only way to move a joint without
commanding it: release the servos and move joint2 BY HAND. A real encoder
follows your hand. An echo cannot, because nothing commanded it anywhere.

Commands NO motion. Releasing removes drive rather than adding it, which is
the one safe thing to send into an arm whose behaviour is not understood.

THE ARM WILL SAG WHEN RELEASED. Hold it.
"""

import sys
import time

try:
    from pymycobot import MyCobot280
except ImportError:
    sys.exit('pymycobot is not installed: pip install pymycobot')

PORT = '/dev/ttyTHS1'
BAUD = '1000000'
WINDOW = 40.0
HAND_MOVE_DEG = 3.0


def connect(patience=40.0):
    """One open, then patient retries -- reopening is the wrong recovery on
    this port, which needs settling time after an open."""
    print(f'connecting to {PORT}...', end=' ', flush=True)
    mc = MyCobot280(PORT, BAUD)
    time.sleep(3.0)
    deadline = time.monotonic() + patience
    dots = 0
    while time.monotonic() < deadline:
        if isinstance(mc.get_angles(), list):
            print(' ok')
            return mc
        time.sleep(0.5)
        dots += 1
        if dots % 4 == 0:
            print('.', end='', flush=True)
    print(' no answer')
    return None


def read(mc, tries=15):
    """Patient: this link drops roughly half its replies, and an impatient
    read loop reports silence from a link that is merely lossy."""
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


def release(mc):
    """Actually take the torque off, and prove it before trusting it.

    release_all_servos() on its own did NOT work here: it returned without
    error while is_free_mode stayed 0, every servo still reported enabled,
    and the arm stayed rigid enough that it could not be moved by hand. That
    turned this whole test into a false negative -- a locked arm shows zero
    movement whatever get_angles is doing.

    So try every route the firmware offers, then verify, and say plainly
    whether it took rather than assuming.
    """
    print('releasing:')
    for _ in range(2):
        fn = getattr(mc, 'release_all_servos', None)
        if fn:
            try:
                fn()
            except Exception as e:
                print(f'  release_all_servos: {e}')
        time.sleep(0.4)

    # Per-servo, because release_all does not always reach every one --
    # the mirror of focus_all_servos missing individually-dropped servos.
    fn = getattr(mc, 'release_servo', None)
    if fn:
        for joint in range(1, 7):
            try:
                fn(joint)
            except Exception:
                pass
            time.sleep(0.15)
        print('  release_servo sent per joint')

    fn = getattr(mc, 'set_free_mode', None)
    if fn:
        for _ in range(2):
            try:
                fn(1)
            except Exception:
                pass
            time.sleep(0.4)
        print('  set_free_mode(1) sent')

    time.sleep(1.0)
    free = retry(mc.is_free_mode) if hasattr(mc, 'is_free_mode') else -1
    enabled = ([retry(mc.is_servo_enable, j) for j in range(1, 7)]
               if hasattr(mc, 'is_servo_enable') else [])
    print(f'  is_free_mode: {free}   is_servo_enable: {enabled}')

    holding = [i + 1 for i, v in enumerate(enabled) if v == 1]
    if free == 1 and not holding:
        print('  released cleanly')
        return True

    # The flags disagree with each other often enough that they cannot decide
    # this. The hand is the only reliable instrument.
    print()
    print('  The flags are ambiguous -- is_servo_enable often means "present '
          'on the bus"')
    print('  rather than "driving". Settle it by feel.')
    try:
        answer = input('  Can you move joint2 by hand right now? [y/N] ')
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().lower().startswith('y')


def main():
    mc = connect()
    if mc is None:
        sys.exit('could not reach the arm -- power-cycle it and retry')

    start = read(mc)
    if start:
        print('pose now: ' + '  '.join(
            f'j{i + 1}={a:.1f}' for i, a in enumerate(start)))

    print()
    print('=' * 68)
    print('Releasing the servos. SUPPORT THE ARM -- it will sag.')
    print('=' * 68)
    try:
        input('press Enter when you are holding it... ')
    except (EOFError, KeyboardInterrupt):
        print()

    if not release(mc):
        sys.exit(
            'The servos did not release, so this test cannot run -- an arm '
            'that cannot be moved by hand reads 0.00 degrees of movement no '
            'matter what get_angles is doing, which is exactly the false '
            'negative this hit on 2026-08-10.')

    print()
    print(f'NOW MOVE JOINT 2 BY HAND, 20+ degrees, for the next '
          f'{WINDOW:.0f} seconds.')
    print('Keep moving it -- back and forth is fine.')
    print()

    samples = []
    dropped = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < WINDOW:
        angles = read(mc)
        if angles:
            samples.append(angles)
            print(f'  t={time.monotonic() - t0:4.1f}s  ' + '  '.join(
                f'j{i + 1}={a:7.1f}' for i, a in enumerate(angles)))
        else:
            dropped += 1
            print(f'  t={time.monotonic() - t0:4.1f}s  (no reply)')
        time.sleep(0.3)

    print()
    print(f'{len(samples)} readings, {dropped} with no reply')
    if len(samples) < 2:
        sys.exit(
            'Not enough readings to judge. The arm answered on connect then '
            'went silent -- the pattern that ended every attempt on '
            '2026-08-09. Power-cycle and retry.')

    spread = [max(s[j] for s in samples) - min(s[j] for s in samples)
              for j in range(6)]
    print('\nmovement seen per joint:')
    for j, s in enumerate(spread):
        print(f'  joint{j + 1}: {s:7.2f} deg')

    print()
    if spread[1] > HAND_MOVE_DEG:
        print('READINGS ARE REAL. joint2 followed your hand, so get_angles')
        print('reports the encoder. The arm genuinely was not executing')
        print('commands -- so the servos have no torque, and the power')
        print('supply is the thing to look at.')
    elif max(spread) > HAND_MOVE_DEG:
        moved = [f'joint{j + 1}' for j, s in enumerate(spread)
                 if s > HAND_MOVE_DEG]
        print(f'Readings look real -- {", ".join(moved)} moved -- but joint2')
        print('did not. Either you moved a different joint, or joint2 is')
        print('seized, or its encoder alone is not reporting.')
    else:
        print('NOTHING CHANGED while you moved it by hand.')
        print('get_angles is NOT reading the encoders -- it is echoing the')
        print('last commanded pose. Every angle measured from this arm is the')
        print("controller's intention, not the arm's position, and the whole")
        print('diagnosis has to be redone on that basis.')

    print()
    print('Servos are still RELEASED. Re-engage before letting go:')
    print('  ./scripts/jog_joints.py   then press f')


if __name__ == '__main__':
    main()
