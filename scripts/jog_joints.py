#!/usr/bin/env python3
"""Move the arm's joints one at a time, by hand, over /dev/ttyTHS1.

For when you want to position the arm without the ROS stack running, or to
find out which joints are actually responding. Nothing here plans, tracks or
servos -- it sends one send_angles per command and reports what moved.

    ./scripts/jog_joints.py

Commands (joint numbers are 1-6):

    2 +5        move joint2 by +5 degrees
    2 -10       move joint2 by -10 degrees
    2 =90       move joint2 TO 90 degrees, in steps
    p           print the current pose
    f           re-focus the servos (see below)
    r           release the servos -- THE ARM WILL SAG, hold it
    s           servo diagnostics: temps, voltages, status
    step 3      change the default step size to 3 degrees
    speed 20    change the move speed (1-100)
    q           quit

Bare `+5` / `-5` repeat on the last joint you moved, so walking one joint
along is `2 +5` then `+5` then `+5`.

Two things this knows about that a bare pymycobot session does not:

**Servos silently stop accepting motion.** They keep answering get_angles,
keep reporting enabled with no fault, and ignore every send_angles. Calling
focus_servo on each joint revives them. This script detects a command that
achieved nothing, re-focuses, and retries once -- and tells you it did, since
needing that repeatedly is a symptom worth noticing rather than papering over.

**A move is not one command.** These servos reliably manage only a few degrees
per send_angles under load, so `=90` walks there in steps rather than asking
once and reporting failure when the arm stops short.

Joint limits are the driver's own (mycobot_hardware_node.py). pymycobot
rejects anything outside roughly +/-150 on some joints with an exception
rather than clamping, which aborts the whole call, so targets are clamped here
first.
"""

import os
import sys
import time

try:
    from pymycobot import MyCobot280
except ImportError:
    sys.exit('pymycobot is not installed: pip install pymycobot')

PORT = '/dev/ttyTHS1'
BAUD = '1000000'

# Matches joint_limits_deg in mycobot_hardware_node.py. Kept a degree inside
# where pymycobot's own validator sits, because it raises rather than clamps.
LIMITS = [
    (-167.0, 167.0),
    (-139.0, 139.0),
    (-149.0, 149.0),
    (-149.0, 149.0),
    (-154.0, 159.0),
    (-179.0, 179.0),
]

DEFAULT_STEP = 5.0
DEFAULT_SPEED = 30
SETTLE = 2.0
MOVED_THRESHOLD = 0.5


def warn_if_port_busy():
    """Say so if something else already holds the port.

    A second holder does not fail loudly -- the two readers split the incoming
    bytes and both see garbage or silence, which is indistinguishable from an
    unpowered arm. A forgotten instance of this very script, or a ROS driver
    still running, is the likeliest cause and costs a power cycle to work out
    otherwise.
    """
    import subprocess
    try:
        out = subprocess.run(['fuser', PORT], capture_output=True,
                             text=True, timeout=5).stdout.split()
    except Exception:
        return
    others = [p for p in out if p.strip().isdigit()
              and int(p) != os.getpid()]
    if others:
        print(f'WARNING: {PORT} is already open by pid(s) '
              f'{", ".join(others)}.')
        print('  Two masters on one bus behaves erratically and reads as a '
              'dead arm.')
        for pid in others:
            try:
                with open(f'/proc/{pid}/cmdline') as f:
                    print(f'    {pid}: {f.read().replace(chr(0), " ").strip()}')
            except OSError:
                pass
        print()


def connect(patience=30.0, reopens=2):
    """Open the port ONCE and keep asking on it.

    Reopening is the wrong recovery on this port and was actively causing the
    failure it looked like it was working around: the Tegra UART in PIO mode
    needs settling time after an open, so a loop that opens, gets a few -1s,
    closes and opens again never gives any single connection long enough to
    start answering. Holding one open port and retrying the READ is what has
    always worked here -- it is why pymycobot succeeds where a quick raw poke
    reports a dead bus.

    So: one open, then patient retries. Reopen only as a last resort, and even
    then with a long pause first.
    """
    for reopen in range(reopens + 1):
        if reopen:
            print(f'  still nothing after {patience:.0f}s; '
                  'closing and reopening once')
            time.sleep(3.0)
        print(f'connecting to {PORT} at {BAUD}...', end=' ', flush=True)
        mc = MyCobot280(PORT, BAUD)
        time.sleep(3.0)

        deadline = time.monotonic() + patience
        dots = 0
        while time.monotonic() < deadline:
            angles = mc.get_angles()
            if isinstance(angles, list) and len(angles) == 6:
                print(' ok')
                return mc
            time.sleep(0.5)
            dots += 1
            if dots % 4 == 0:
                print('.', end='', flush=True)
        print()
        try:
            mc._serial_port.close()
        except Exception:
            pass
    return None


def read(mc, tries=15):
    """get_angles, retried -- this link drops a good fraction of them."""
    for _ in range(tries):
        angles = mc.get_angles()
        if isinstance(angles, list) and len(angles) == 6:
            return angles
        time.sleep(0.2)
    return None


def retry(fn, *args, tries=6):
    for _ in range(tries):
        try:
            value = fn(*args)
        except Exception as e:
            return f'error: {e}'
        if value != -1:
            return value
        time.sleep(0.25)
    return -1


def focus(mc, quiet=False):
    """Re-engage torque. focus_all_servos does not always reach a servo that
    has dropped out on its own, so each one is focused individually too."""
    for name in ('power_on', 'focus_all_servos'):
        fn = getattr(mc, name, None)
        if fn:
            try:
                fn()
            except Exception:
                pass
            time.sleep(0.3)
    fn = getattr(mc, 'focus_servo', None)
    if fn:
        for joint in range(1, 7):
            try:
                fn(joint)
            except Exception:
                pass
            time.sleep(0.12)
    if not quiet:
        print('servos re-focused')


def show(angles, label='pose'):
    if angles is None:
        print(f'{label}: unreadable')
        return
    print(f'{label}: ' + '  '.join(
        f'j{i + 1}={a:.1f}' for i, a in enumerate(angles)))


def clamp(index, value):
    lo, hi = LIMITS[index]
    clamped = max(lo, min(hi, value))
    if abs(clamped - value) > 1e-6:
        print(f'  clamped to joint{index + 1} limit {lo:.0f}..{hi:.0f}')
    return clamped


def move_once(mc, index, goal, speed, sends=3):
    """Command one joint, resending if nothing happens.

    This link drops roughly half its get_angles replies, and there is no
    reason writes fare better -- a dropped SEND_ANGLES is simply a move that
    never happens. Measured here: the first command after any pause almost
    always does nothing and the next one works, on joints under no load at
    all, which is a lost write rather than a weak servo.

    So resend before concluding anything. The protocol carries no checksum and
    send_angles is idempotent -- it names an absolute pose, not a delta -- so
    repeating it is safe: arriving twice is the same as arriving once.

    Returns (moved_degrees, new_angles, attempts_used).
    """
    before = read(mc)
    if before is None:
        print('  cannot read the arm')
        return 0.0, None, 0
    target = list(before)
    target[index] = clamp(index, goal)

    for attempt in range(1, sends + 1):
        try:
            mc.send_angles(target, speed)
        except Exception as e:
            print(f'  refused: {e}')
            return 0.0, before, attempt
        time.sleep(SETTLE)
        after = read(mc)
        if after is None:
            continue
        moved = after[index] - before[index]
        if abs(moved) >= MOVED_THRESHOLD:
            return moved, after, attempt

    after = read(mc)
    if after is None:
        return 0.0, None, sends
    return after[index] - before[index], after, sends


def move_to(mc, index, goal, step, speed):
    """Walk a joint to an absolute angle, re-focusing if it stalls."""
    goal = clamp(index, goal)
    rescued = False

    for _ in range(40):
        angles = read(mc)
        if angles is None:
            print('  lost the arm')
            return
        here = angles[index]
        gap = goal - here
        if abs(gap) <= max(1.0, step * 0.3):
            print(f'  joint{index + 1} at {here:.1f} (asked {goal:.1f})')
            return

        chunk = here + max(-step, min(step, gap))
        moved, after, attempts = move_once(mc, index, chunk, speed)
        if after is None:
            return

        if abs(moved) < MOVED_THRESHOLD:
            if rescued:
                print(f'  joint{index + 1} STUCK at {here:.1f} -- resent the '
                      'command and re-focused, still nothing. Either it is '
                      'mechanically blocked, or the controller needs a power '
                      'cycle. Check whether the Atom LED is steady.')
                return
            print(f'  joint{index + 1} ignored {attempts} sends; '
                  're-focusing')
            focus(mc, quiet=True)
            rescued = True
            continue

        rescued = False
        note = f'  (took {attempts} sends)' if attempts > 1 else ''
        print(f'  joint{index + 1}: {here:7.1f} -> {after[index]:7.1f} '
              f'({moved:+.1f}){note}')


def diagnostics(mc):
    for name in ('get_servo_temps', 'get_servo_voltages', 'get_servo_status',
                 'get_servo_speeds'):
        fn = getattr(mc, name, None)
        if fn is not None:
            print(f'  {name[10:]:>9}: {retry(fn)}')
    fn = getattr(mc, 'is_servo_enable', None)
    if fn is not None:
        print(f'    enabled: {[retry(fn, j) for j in range(1, 7)]}')


def main():
    warn_if_port_busy()
    mc = connect()
    if mc is None:
        sys.exit(
            'Could not reach the arm.\n'
            '  - is it powered on? The Atom LED should be lit and steady\n'
            '  - if the LED blinks out, the controller is browning out\n'
            '  - a power cycle clears a wedged controller')

    focus(mc)
    show(read(mc), 'start')
    print('\nType a command, or "?" for help, "q" to quit.')

    step = DEFAULT_STEP
    speed = DEFAULT_SPEED
    last_joint = None

    while True:
        try:
            line = input(f'[step {step:.0f} speed {speed}] > ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        low = line.lower()
        if low in ('q', 'quit', 'exit'):
            break
        if low in ('?', 'h', 'help'):
            print(__doc__)
            continue
        if low == 'p':
            show(read(mc))
            continue
        if low == 'f':
            focus(mc)
            continue
        if low == 's':
            diagnostics(mc)
            continue
        if low == 'r':
            confirm = input('  release servos? THE ARM WILL SAG. [y/N] ')
            if confirm.strip().lower() == 'y':
                for name in ('release_all_servos',):
                    fn = getattr(mc, name, None)
                    if fn:
                        try:
                            fn()
                        except Exception as e:
                            print(f'  {name}: {e}')
                print('  released -- support the arm')
            continue

        parts = low.split()
        if parts[0] == 'step' and len(parts) == 2:
            try:
                step = max(0.5, min(20.0, float(parts[1])))
                print(f'  step is now {step:.1f} degrees')
            except ValueError:
                print('  usage: step 5')
            continue
        if parts[0] == 'speed' and len(parts) == 2:
            try:
                speed = max(1, min(100, int(parts[1])))
                print(f'  speed is now {speed}')
            except ValueError:
                print('  usage: speed 30')
            continue

        # Movement: "2 +5", "2 =90", or bare "+5" to repeat the last joint.
        if parts[0].isdigit() and len(parts) == 2:
            joint = int(parts[0])
            arg = parts[1]
        elif len(parts) == 1 and parts[0][0] in '+-=':
            if last_joint is None:
                print('  no joint chosen yet -- try "2 +5"')
                continue
            joint = last_joint + 1
            arg = parts[0]
        else:
            print('  do not understand that. "?" for help')
            continue

        if not 1 <= joint <= 6:
            print('  joints are 1 to 6')
            continue
        index = joint - 1
        last_joint = index

        angles = read(mc)
        if angles is None:
            print('  cannot read the arm')
            continue

        try:
            if arg.startswith('='):
                goal = float(arg[1:])
            else:
                goal = angles[index] + float(arg)
        except ValueError:
            print('  usage: 2 +5   or   2 =90')
            continue

        move_to(mc, index, goal, step, speed)
        show(read(mc))

    print('done -- servos left holding position')


if __name__ == '__main__':
    main()
