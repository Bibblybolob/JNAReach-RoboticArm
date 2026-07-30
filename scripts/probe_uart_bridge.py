#!/usr/bin/env python3
"""Test a USB-to-TTL bridge before trusting it with the arm's UART.

    ./scripts/probe_uart_bridge.py loopback          # jumper TX to RX
    ./scripts/probe_uart_bridge.py listen            # passive tap, decodes frames

This is the cheap de-risking step for putting a Jetson (or anything else) on
the arm's UART in place of the Raspberry Pi. Two questions, in order:

  1. Does my adapter actually run at 1000000 baud?     -> loopback
  2. Is the wire I tapped really the arm's TX line?    -> listen

Answer both before commanding motion. probe_usb_arm.py is the next step after
these pass; it opens the same port and asks the arm for joint angles.

USING AN ARDUINO UNO R3 AS THE ADAPTER

An Uno works, but not the way people first try it. It has ONE hardware UART
and it is already wired to the USB chip, so a bridging sketch is not
available -- and SoftwareSerial tops out around 115200, nowhere near the
1000000 this arm runs at. The route that works is to remove the sketch from
the picture entirely:

    Connect the Uno's RESET pin to GND.

That holds the ATmega328P in reset with its pins high-impedance, leaving the
ATmega16U2 USB chip wired straight through to D0/D1. The board is now a plain
USB-to-TTL adapter. (16 MHz in double-speed mode divides to exactly 1000000
baud with zero error, which is better than some dedicated adapters manage.)

The direction of D0/D1 INVERTS when you do this, and it is the single most
common way to waste an evening here. Those labels describe the 328P, which is
now switched off. From the outside:

    D0, labelled RX, is an OUTPUT  (it is the USB chip's TX)  -> target's RX
    D1, labelled TX, is an INPUT   (it is the USB chip's RX)  <- target's TX

So you wire label-to-same-label, NOT crossed. Plus a shared ground.

VOLTAGE -- READ THIS BEFORE CONNECTING D0

The Uno is a 5V board. The ESP32 in the arm, and the Pi driving it, are 3.3V
and are not 5V tolerant. Driving D0 into the arm's RX line at 5V risks
damaging it permanently.

    listening (D1) is safe    -- it is an input, nothing is driven
    transmitting (D0) is not  -- put a divider in it

Add ONE resistor, 2k from D0 to ground -- not two. The Uno R3 already has a 1k
in series between the 16U2's TX and the D0 header pin (it is there so an
external device can override the USB chip), so 2k to ground completes a 1k/2k
divider at 3.33V. Adding your own series resistor on top makes it 1k+1k/2k =
2.5V, and the ESP32's input threshold is 0.75 x VDD = 2.475V -- a 25mV margin,
i.e. a line that reads as neither high nor low depending on temperature.

Clones vary, so measure rather than assume: UART idles high, so D0 with the
resistor fitted and nothing transmitting should sit at ~3.3V. If it reads 5V
the on-board resistor is absent and you need a 1k in series as well.

This is why `listen` exists as a separate mode: it uses only D1, an input, so
it answers the pin-mapping question with no divider and no risk. Do it first.

THE OTHER MASTER

The Pi drives the same two lines. Two push-pull outputs on one net fight each
other, so before transmitting, take the Pi off the bus:

    ssh er@192.168.0.15 'sudo systemctl stop mycobot_server'
    ssh er@192.168.0.15 'sudo pkill -f uart_peripheral_serial'
    ssh er@192.168.0.15 'sudo raspi-gpio set 14 ip'   # release its transmitter

That last one matters and is easy to skip: stopping the software leaves GPIO14
in ALT0 still actively driving pin 8. Setting it to an input makes it
high-impedance. It reverts on reboot.

For `listen` you want the opposite -- leave mycobot_server RUNNING, since the
traffic you are trying to see is the traffic it generates.
"""

import argparse
import sys
import time


HEADER = 0xFE
FOOTER = 0xFA
# From pymycobot's own framing: [FE FE len cmd payload... FA], where the
# length byte counts cmd + payload + footer. Total frame is therefore len + 3.
FRAME_OVERHEAD = 3

COMMANDS = {
    0x02: 'POWER_ON', 0x03: 'POWER_OFF', 0x10: 'IS_POWER_ON',
    0x18: 'RELEASE_ALL_SERVOS', 0x20: 'GET_ANGLES', 0x21: 'SEND_ANGLE',
    0x22: 'SEND_ANGLES', 0x23: 'GET_COORDS', 0x25: 'SEND_COORDS',
    0x29: 'STOP', 0x30: 'IS_IN_POSITION', 0x31: 'IS_MOVING',
    0x3A: 'JOG_ANGLE', 0x3B: 'JOG_COORD', 0x3C: 'JOG_STOP',
    0x40: 'GET_SPEED', 0x41: 'SET_SPEED',
}


def open_port(port, baud):
    try:
        import serial
    except ImportError:
        sys.exit('pyserial is missing:  pip install pyserial')
    try:
        return serial.Serial(port, baud, timeout=0.2)
    except Exception as e:
        name = e.__class__.__name__
        if 'Permission' in name or 'Permission' in str(e):
            sys.exit(
                f'{port} exists but this user cannot open it.\n\n'
                f'That is a unix group, not a wiring fault -- nothing about\n'
                f'the adapter has been tested yet.\n\n'
                f'    sudo usermod -aG dialout $USER\n\n'
                f'then log out and back in (or `newgrp dialout` for this\n'
                f'shell only) and run this again.')
        sys.exit(f'Cannot open {port}: {name}: {e}')


def cmd_loopback(args):
    """Prove the adapter runs at this baud, with nothing else connected.

    A bridge that enumerates fine at 1000000 and still garbles bytes is the
    failure worth catching here -- it looks exactly like a robot that will not
    answer, and it sends you looking at the arm.
    """
    ser = open_port(args.port, args.baud)
    print(f'{args.port} open at {args.baud} baud.\n')
    print('Jumper the adapter\'s TX to its RX.')
    print('  On an Uno held in reset, that is D0 to D1 directly.\n')

    # Every byte value, so a marginal baud shows up: clock error corrupts the
    # bytes with the most transitions first, and 0x55/0xAA are the extremes.
    pattern = bytes(range(256)) * 4
    ser.reset_input_buffer()
    ser.write(pattern)
    ser.flush()

    got = bytearray()
    deadline = time.time() + 2.0
    while len(got) < len(pattern) and time.time() < deadline:
        chunk = ser.read(len(pattern) - len(got))
        if chunk:
            got.extend(chunk)
    ser.close()

    if not got:
        print('Nothing came back.\n')
        print('Either the jumper is not on, or TX and RX are not the pins you')
        print('think. On an Uno, check RESET is tied to GND -- without that')
        print('the 328P is still running whatever sketch was last flashed and')
        print('it owns those pins.')
        return 1

    if bytes(got) == pattern:
        print(f'{len(got)}/{len(pattern)} bytes returned, byte-exact.\n')
        print(f'The adapter is good at {args.baud} baud. Next: `listen`, to')
        print('confirm which wire is the arm talking.')
        return 0

    bad = sum(1 for a, b in zip(got, pattern) if a != b)
    print(f'{len(got)}/{len(pattern)} bytes returned, {bad} corrupted.\n')
    print('The link runs but does not carry bytes intact, which at this baud')
    print('is almost always a clock the adapter cannot divide to exactly.')
    print(f'Try --baud 115200: if that is clean, {args.baud} is out of reach')
    print('for this adapter and it cannot drive the arm, which needs 1000000.')
    return 1


def cmd_listen(args):
    """Passive tap. Reads only -- nothing is transmitted on this port."""
    ser = open_port(args.port, args.baud)
    print(f'{args.port} open at {args.baud} baud, listening for '
          f'{args.seconds:.0f}s.')
    print('Reading only -- nothing is transmitted.\n')
    print('Make the arm talk while this runs, e.g. on the Pi:')
    print("    python3 -c \"from pymycobot import MyCobot280; "
          "print(MyCobot280('/dev/ttyAMA0', '1000000').get_angles())\"\n")

    buf = bytearray()
    frames = 0
    raw = 0
    deadline = time.time() + args.seconds
    try:
        while time.time() < deadline:
            chunk = ser.read(256)
            if not chunk:
                continue
            raw += len(chunk)
            buf.extend(chunk)
            frames += _drain_frames(buf)
    except KeyboardInterrupt:
        print('\n(interrupted)')
    finally:
        ser.close()

    print()
    if raw == 0:
        print('Silence -- not one byte.\n')
        print('Nothing was transmitted on the wire you tapped. Check, in this')
        print('order: that the arm was actually commanded while this ran;')
        print('that you are on the TX line and not the RX one (they are next')
        print('to each other on the header, pin 8 vs pin 10); and that the')
        print('grounds are tied together, without which nothing decodes.')
        return 1

    print(f'{raw} bytes, {frames} valid frames.\n')
    if frames:
        print('That is the myCobot protocol at this baud, so the wire is the')
        print('arm\'s TX line and the adapter reads it correctly. This is the')
        print('measurement that makes the pin mapping a fact rather than an')
        print('inference from the docs.')
        return 0
    print('Bytes arrived but none framed as FE FE ... FA.\n')
    print('So something is transmitting and the baud is wrong -- a mismatched')
    print('rate turns real traffic into exactly this. Run `loopback` first if')
    print('you have not; it separates "wrong baud" from "broken adapter".')
    return 1


def _drain_frames(buf):
    """Pull complete frames off the front of buf, printing each. Returns the
    count. Leaves a partial frame in place for the next read."""
    found = 0
    while True:
        start = buf.find(bytes([HEADER, HEADER]))
        if start < 0:
            # Keep one byte: a header split across two reads is normal.
            del buf[:max(0, len(buf) - 1)]
            return found
        del buf[:start]
        if len(buf) < 4:
            return found
        total = buf[2] + FRAME_OVERHEAD
        if total > len(buf):
            if total > 64:      # not a real length; resync past this header
                del buf[:2]
                continue
            return found
        frame = bytes(buf[:total])
        del buf[:total]
        if frame[-1] != FOOTER:
            continue
        cmd = frame[3]
        name = COMMANDS.get(cmd, f'0x{cmd:02X}')
        payload = frame[4:-1]
        print(f'  {name:<20} {len(payload):>2}B  {frame.hex(" ")}')
        found += 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['loopback', 'listen'])
    ap.add_argument('--port', default='/dev/ttyACM0',
                    help='ttyACM0 for an Uno, ttyUSB0 for most adapters')
    ap.add_argument('--baud', type=int, default=1000000,
                    help='the arm runs at 1000000 (default)')
    ap.add_argument('--seconds', type=float, default=10.0,
                    help='listen duration')
    args = ap.parse_args()
    return cmd_loopback(args) if args.mode == 'loopback' else cmd_listen(args)


if __name__ == '__main__':
    sys.exit(main())
