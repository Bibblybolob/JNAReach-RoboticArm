#!/usr/bin/env python3
"""Clear latched servo faults without reaching for the power switch.

Why this exists
---------------
The arm gets into a state where the Atom crash-loops -- `cmd_len error`,
LoadProhibited panic, reboot, repeat -- and only a physical power cycle
recovers it. Reflashing does not (tried three times on 2026-08-11), and the
ESP32's own reboot does not either.

The reason the reboot does not help is that the fault is not in the part that
reboots. Feetech servos LATCH overload/overheat faults and keep the latch
across an ESP32 restart, because they have their own power. The Atom comes
back up, polls the bus, gets a malformed or absent reply from the latched
joint, and its parser panics -- forever. Pulling the mains clears the servos
along with everything else, which is why that works.

A latch clears when torque is toggled off and back on for that servo. So this
does in software what the power switch does mechanically, without dropping
power to the controller.

    ./scripts/reset_arm.py                 # every joint, safest ordering
    ./scripts/reset_arm.py --joint 5       # just the one you suspect
    ./scripts/reset_arm.py --check         # read fault codes, change nothing

Load-bearing joints (2 and 3 hold the arm up against gravity) are released for
`--off-time` only, defaulting to a fraction of a second, so the arm sags
minimally. Do not raise that without supporting the arm.

The link is usually crash-looping when this is needed, so commands mostly do
not land. That is expected: it retries through the lucid windows between
crashes rather than giving up on the first silence.
"""
from __future__ import annotations

import argparse
import time

PORT = '/dev/ttyTHS1'
BAUD = 1000000

# Joints that hold the arm's weight. Released for the shortest possible time.
LOAD_BEARING = (2, 3, 4)


def link_health(port: str, baud: int, n: int = 20) -> tuple[int, int, int]:
    """(valid, crash, silent) over n raw GET_ANGLES pokes.

    Raw bytes deliberately: pymycobot returns -1 for anything it cannot parse,
    which reads identically to a dead link, to a crash-loop, and to a healthy
    arm answering alongside ESP32 console output. The distinction is the whole
    question here, so read the wire.
    """
    import serial
    sp = serial.Serial(port, baud, timeout=1.0)
    time.sleep(2)
    sp.reset_input_buffer()
    valid = crash = silent = 0
    try:
        for _ in range(n):
            sp.reset_input_buffer()
            sp.write(bytes([0xfe, 0xfe, 0x02, 0x20, 0xfa]))
            sp.flush()
            time.sleep(0.25)
            d = sp.read(16384)
            if not d:
                silent += 1
            elif b'cmd_len' in d or b'Guru' in d:
                crash += 1
            elif d.find(b'\xfe\xfe\x0e\x20') >= 0:
                valid += 1
    finally:
        sp.close()
    return valid, crash, silent


def read_faults(mc, joints) -> dict:
    """Latched error code per joint, or None where it could not be read."""
    out = {}
    for j in joints:
        out[j] = None
        for _ in range(5):
            try:
                v = mc.get_servo_error(j)
            except Exception:
                break
            if isinstance(v, int) and v != -1:
                out[j] = v
                break
            time.sleep(0.2)
    return out


def read_temps(mc, joints) -> dict:
    """Servo temperature in C. Feetech STS register 63, read raw.

    Reliable where load and current are not: temperature is slow-moving, so it
    survives a lossy link, and sustained overload is exactly what heats a
    servo. j5 measured 58C at rest while j1 sat at 32C.
    """
    out = {}
    for j in joints:
        out[j] = None
        for _ in range(6):
            try:
                v = mc.get_servo_data(j, 63, 0)
            except TypeError:
                v = mc.get_servo_data(j, 63)
            except Exception:
                break
            if isinstance(v, int) and 0 < v < 100:
                out[j] = v
                break
            time.sleep(0.2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default=PORT)
    ap.add_argument('--baud', type=int, default=BAUD)
    ap.add_argument('--joint', type=int, default=None,
                    help='reset only this joint (1-6)')
    ap.add_argument('--off-time', type=float, default=0.4,
                    help='seconds a load-bearing joint stays released')
    ap.add_argument('--off-time-free', type=float, default=1.5,
                    help='seconds a non-load-bearing joint stays released')
    ap.add_argument('--rounds', type=int, default=3,
                    help='attempts, since most commands are lost mid-crash-loop')
    ap.add_argument('--check', action='store_true',
                    help='report fault codes and temperatures, change nothing')
    args = ap.parse_args()

    from pymycobot import MyCobot

    joints = [args.joint] if args.joint else [1, 2, 3, 4, 5, 6]

    before = link_health(args.port, args.baud)
    print(f'link before:  valid {before[0]}/20  crash {before[1]}  '
          f'silent {before[2]}')

    mc = MyCobot(args.port, args.baud)
    time.sleep(2)

    faults = read_faults(mc, joints)
    temps = read_temps(mc, joints)
    print('\njoint  fault  temp')
    for j in joints:
        f = faults[j]
        t = temps[j]
        flag = ''
        if f not in (0, None):
            flag = '  <-- LATCHED FAULT'
        elif t is not None and t >= 55:
            flag = '  <-- hot'
        print(f'  j{j}   {str(f):>5}  {str(t) + "C" if t else "  ?":>5}{flag}')

    if args.check:
        print('\n--check: nothing changed.')
        return 0

    print(f'\nToggling torque to clear latches ({args.rounds} round(s))...')
    for r in range(args.rounds):
        for j in joints:
            off = (args.off_time if j in LOAD_BEARING else args.off_time_free)
            try:
                mc.release_servo(j)
                time.sleep(off)
                mc.focus_servo(j)
                time.sleep(0.2)
            except Exception as e:  # noqa: BLE001
                print(f'  j{j}: {e}')
        try:
            mc.power_on()
        except Exception:
            pass
        time.sleep(1.0)
        print(f'  round {r + 1} done')

    after = link_health(args.port, args.baud)
    print(f'\nlink after:   valid {after[0]}/20  crash {after[1]}  '
          f'silent {after[2]}')

    if after[0] > before[0]:
        print('Improved. If it is still short of ~18/20 with 0 crashes, run '
              'again -- a joint can re-latch while another is being cleared.')
    elif after[1] or after[0] == 0:
        print('No better. The latch did not clear, or the fault is not a '
              'latch. Power cycle, then run with --check FIRST to catch the '
              'fault code before it is wiped.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
