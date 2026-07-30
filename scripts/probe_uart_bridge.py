#!/usr/bin/env python3
"""Test a USB-to-TTL bridge before trusting it with the arm's UART.

    ./scripts/probe_uart_bridge.py loopback          # jumper TX to RX
    ./scripts/probe_uart_bridge.py listen            # passive tap, decodes frames
    ./scripts/probe_uart_bridge.py poke              # ask the arm, dump raw bytes
    ./scripts/probe_uart_bridge.py hunt              # poke on a loop while you probe pins

This is the cheap de-risking step for putting a Jetson (or anything else) on
the arm's UART in place of the Raspberry Pi. Three questions:

  1. Does the link carry bytes intact at 1000000?      -> loopback
  2. Is the wire I tapped really the arm's TX line?    -> listen
  3. Does the arm answer me, and if not, how?          -> poke

Answer them before commanding motion. probe_usb_arm.py is the next step after
these pass; it opens the same port and asks the arm for joint angles.

Which one to reach for depends on what else is on the bus. `listen` needs some
OTHER master to be polling the arm -- with the Pi removed there is no traffic
to overhear and it reports silence whatever the wiring. `poke` generates its
own traffic, so it is the one that works on a bus you are alone on.

Note what loopback does NOT tell you: whether the baud rate is right. Both
directions are the same chip driven from the same divisor register, so a wrong
baud is wrong identically at each end and the bytes still return perfect. Nor
whether a ground is connected, since it shares its own. Only a test against
the actual arm -- `listen` or `poke` -- covers either.

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

Only D0 gets a resistor. D1 is an input and the arm only ever puts 3.3V on it,
which cannot damage anything. One wire, one resistor -- and note it is the
wire the Uno DRIVES, which is the pin silkscreened "RX".

Clones vary, so measure rather than assume: UART idles high, so D0 with the
resistor fitted and nothing transmitting should sit at ~3.3V. If it reads 5V
the on-board resistor is absent and you need a 1k in series as well.

No multimeter needed, though -- fit the resistor and re-run `loopback` with
the jumper on. That is a STRICTER test than the arm, which is what makes it
worth doing: the 16U2 reading it back is a 5V AVR wanting 0.6 x VCC = 3.00V,
while the ESP32 wants 0.75 x VDD = 2.48V. A 1k/2k divider delivers 3.33V, so
it clears the AVR by 0.33V and the ESP32 by 0.86V. If the Uno can still read
its own divided output, the arm certainly can.

Safe to run even with the arm connected: the test pattern contains 0xFE
exactly four times per 1024 bytes and always followed by 0xFF, so it can never
form the FE FE frame header and the ESP32 discards all of it.

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
import contextlib
import io
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


PATTERN = bytes(range(256)) * 4


def _echo_round(ser, payload, chunk=32):
    """Write in small bursts, draining between them.

    Sending all 1024 bytes at once and reading afterwards runs the port at full
    rate in BOTH directions simultaneously, with the adapter's own ring buffer
    (128 bytes on an Uno's 16U2) the only thing between the UART and USB. That
    measures the adapter's buffering, not the link. The arm protocol sends
    frames of a dozen-odd bytes, so bursts are also what real traffic is.
    """
    got = bytearray()
    for i in range(0, len(payload), chunk):
        want = min(i + chunk, len(payload))
        ser.write(payload[i:want])
        ser.flush()
        deadline = time.time() + 0.5
        while len(got) < want and time.time() < deadline:
            b = ser.read(want - len(got))
            if b:
                got.extend(b)
    return bytes(got)


def _measure(port, baud, rounds):
    """(sent, received, missing, bad_bytes, bad_bits) over `rounds` passes."""
    ser = open_port(port, baud)
    sent = received = missing = bad_bytes = bad_bits = 0
    try:
        for _ in range(rounds):
            ser.reset_input_buffer()
            got = _echo_round(ser, PATTERN)
            sent += len(PATTERN)
            received += len(got)
            missing += len(PATTERN) - len(got)
            for a, b in zip(got, PATTERN):
                if a != b:
                    bad_bytes += 1
                    bad_bits += bin(a ^ b).count('1')
    finally:
        ser.close()
    return sent, received, missing, bad_bytes, bad_bits


def _line(baud, stats):
    sent, received, missing, bad_bytes, bad_bits = stats
    rate = 100.0 * bad_bytes / received if received else 0.0
    out = (f'  {baud:>8} baud   {received}/{sent} back   '
           f'{missing} lost   {bad_bytes} corrupt ({rate:.2f}%)')
    if bad_bytes:
        out += f'   {bad_bits / bad_bytes:.1f} bit errors per bad byte'
    print(out)


def cmd_loopback(args):
    """Measure the link's error rate with nothing but a jumper attached.

    What this CANNOT tell you is whether the baud rate is right. Both
    directions are the same chip driven from the same divisor register, so a
    wrong baud is wrong identically at each end and the bytes still come back
    perfect. Anything this catches is signal integrity or buffering.
    """
    print(f'{args.port}, {args.rounds} rounds of {len(PATTERN)} bytes.\n')
    print('Jumper the adapter\'s TX to its RX.')
    print('  On an Uno held in reset, that is D0 to D1 directly.\n')

    stats = _measure(args.port, args.baud, args.rounds)
    _line(args.baud, stats)
    sent, received, missing, bad_bytes, bad_bits = stats

    if received == 0:
        print('\nNothing came back at all.\n')
        print('Either the jumper is not on, or TX and RX are not the pins you')
        print('think. On an Uno, check RESET is tied to GND -- without that')
        print('the 328P is still running whatever sketch was last flashed and')
        print('it owns those pins.')
        return 1

    if not missing and not bad_bytes:
        print(f'\nClean. The adapter carries bytes intact at {args.baud}.\n')
        print('Next: `listen`, which is the measurement that actually matters')
        print('-- it confirms which wire the arm transmits on.')
        return 0

    # Errors. Whether they scale with baud separates a marginal edge from a
    # bad connection, and that is the whole diagnosis.
    print('\nErrors. Re-measuring lower to see whether the rate scales:\n')
    lower = [b for b in (500000, 250000, 115200) if b < args.baud]
    scaled = []
    for baud in lower:
        try:
            s = _measure(args.port, baud, args.rounds)
        except SystemExit:
            break
        _line(baud, s)
        scaled.append((baud, s))

    clean_low = [b for b, s in scaled if not s[2] and not s[3]]
    print()
    if clean_low:
        print(f'Clean at {clean_low[0]} and not at {args.baud}, so this is an')
        print('edge-rate limit rather than a broken connection.\n')
        print('On an Uno that is partly the test\'s own fault: the loopback')
        print('path crosses the on-board 1k series resistor TWICE (16U2 TX ->')
        print('1k -> D0 -> jumper -> D1 -> 1k -> 16U2 RX). ~2k into dupont')
        print('capacitance is roughly a third of a 1us bit, and driving the')
        print('arm puts only ONE of those resistors in the path while the')
        print('ESP32 drives back with a real push-pull output. So the real')
        print('link is better than this number -- go measure it with `listen`')
        print('rather than trusting either figure.\n')
        print('Shorter jumper first, though. It is free and it is usually it.')
    else:
        print('Errors at every rate, so this is not speed -- it is the')
        print('connection. A long or unshielded jumper, a missing ground, or')
        print('on an Uno the 328P not actually held in reset and driving D1')
        print('against the loopback.')

    if bad_bytes and bad_bits / bad_bytes < 2.0:
        print('\nMostly single-bit errors, which is sampling margin rather')
        print('than framing collapse -- consistent with the above.')

    print('\nDo not drive the arm until this is clean: the myCobot protocol')
    print('carries NO checksum, so a flipped bit in a SEND_ANGLES payload is')
    print('a joint angle the arm accepts and moves to.')
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


REPLY_LEN = 17     # a GET_ANGLES reply: FE FE 0E 20 + 12 payload + FA


def _poke_once(port, baud, request, wait=0.25, read=512):
    ser = open_port(port, baud)
    try:
        ser.reset_input_buffer()
        ser.write(request)
        ser.flush()
        time.sleep(wait)
        return ser.read(read)
    finally:
        ser.close()


def _echo_test(port, baud, request, copies):
    """Tell a real reply from your own transmission coming back.

    The trap this exists for: bytes arriving looks like success, so it gets
    read as "the wiring is fine, now fix the baud". But a floating receive line
    running alongside a transmitting one picks up enough crosstalk to frame as
    bytes, and two wires bridged echo outright. Both produce traffic that
    arrives only while you are sending, which is exactly when you are looking.

    Lengths separate them, because the two sides disagree about who decides it.
    GET_ANGLES is 5 bytes out and 17 back, so N copies means an echo returns 5N
    and the arm 17N. Sending a read command N times is safe -- it commands
    nothing however many times it lands.
    """
    return len(_poke_once(port, baud, request * copies, wait=0.4, read=1024))


def cmd_poke(args):
    """Send a real request and dump whatever comes back, raw, at each baud.

    The gap every other test leaves. pymycobot reports -1 for "no valid
    angles", which is the same answer whether nothing came back at all, or
    bytes came back at the wrong rate, or a well-formed frame arrived with an
    unexpected command id. Those are three different faults with three
    different fixes, and the raw bytes separate them in one run.

    GET_ANGLES only. Commands no motion.
    """
    request = bytes([HEADER, HEADER, 0x02, 0x20, FOOTER])
    print(f'{args.port}: sending {request.hex(" ")} (GET_ANGLES) and reading '
          f'the raw reply.\nNo motion is commanded.\n')

    bauds = [args.baud] + [b for b in (115200, 921600, 230400)
                           if b != args.baud]
    seen = []
    for baud in bauds:
        got = _poke_once(args.port, baud, request)
        # _drain_frames prints as it decodes; hold that until after the
        # summary line so each baud reads top-down.
        detail = io.StringIO()
        with contextlib.redirect_stdout(detail):
            framed = _drain_frames(bytearray(got)) if got else 0
        note = f'{len(got):>3} bytes'
        if got:
            note += f'   {got[:24].hex(" ")}'
            if len(got) > 24:
                note += ' ...'
        print(f'  {baud:>8} baud   {note}')
        if framed:
            print(f'{" ":>12}^ framed correctly:')
            print(detail.getvalue().rstrip())
        seen.append((baud, len(got), framed))

    print()
    good = [b for b, n, f in seen if f]
    noisy = [b for b, n, f in seen if n and not f]
    if good:
        print(f'The arm answers at {good[0]} baud. Wiring and rate are both')
        print('right -- go to serial_move_test.py.')
        return 0
    if noisy:
        baud, copies = noisy[0], 4
        sent = len(request) * copies
        print(f'Bytes came back at {baud}, but nothing framed as FE FE ... FA.')
        print('Before calling that a baud problem, rule out its more common')
        print(f'cause: your own transmission returning. Sending {copies} '
              f'copies makes\nthe two cases disagree about length.\n')
        got = _echo_test(args.port, baud, request, copies)
        print(f'  sent {sent} bytes, {got} came back')
        print(f'  an echo returns {sent}; the arm returns '
              f'{REPLY_LEN * copies}\n')

        if abs(got - sent) <= 2:
            print('That is an echo, not a reply. Nothing on the arm is')
            print('answering you -- you are hearing yourself.\n')
            print('  - Are both wires in the same hole, or touching? Check')
            print('    continuity between them with the adapter unplugged.')
            print('  - If not shorted, the receive wire is floating and')
            print('    picking up crosstalk from the transmit wire beside it.')
            print('    A floating line sits high and glitches low, which is')
            print('    why the bytes come back as mostly 0xFF. Separate the')
            print('    two wires and it goes away.')
            print('  - Either way the receive wire is not on a driven pin, so')
            print('    it is still on the wrong one.')
            return 1

        if abs(got - REPLY_LEN * copies) <= 2:
            print('Reply-length traffic, so the arm IS answering and the')
            print('bytes are being mangled in flight. Now it is worth')
            print('suspecting rate and signal quality: check the baud, then')
            print('shorten the wires and lose any divider above ~2k.')
            return 1

        print('Neither length. Something is transmitting that is neither you')
        print('nor a clean reply -- suspect a second device on the bus, or a')
        print('rate so far off that framing is arbitrary.')
        return 1

    print('Not one byte at any rate, so nothing is transmitting toward you.')
    print('Rate is not the issue -- a wrong baud produces garbage, not')
    print('silence.\n')
    print('Zero is also more specific than it looks. A receive line held LOW')
    print('is a continuous framing error, which the tty layer delivers as a')
    print('flood of 0x00 -- ~100KB/s of it at 1000000 baud. Getting none of')
    print('that means your RX line is sitting high or floating, so it is not')
    print('shorted to ground and not on a stuck-low pin. What remains:\n')
    print('  1. SWAP THE TWO SIGNAL WIRES. Free, and it is the single most')
    print('     likely cause. Remember the Uno\'s labels invert when the 328P')
    print('     is held in reset, so "wire TX to RX" is exactly the mistake')
    print('     here -- but if you followed that, swapping is the fix.')
    print('  2. Is RESET actually strapped to GND? Without it the 328P runs')
    print('     whatever sketch it last had, and a sketch that calls')
    print('     Serial.begin() drives D1 against the arm. Loopback passes')
    print('     either way, so a clean loopback does NOT prove this.')
    print('  3. Are these the right two wires? With the Pi gone, the arm\'s')
    print('     TX idles HIGH -- so with a meter to ground, the wire sitting')
    print('     at a steady 3.3V is the one your D1 wants.')
    return 1


def cmd_hunt(args):
    """Poke twice a second and report live, so you can probe pins by hand.

    For when the transmit side works and nothing answers -- the receive wire is
    on the wrong pin and the question is which. Running the whole test per
    guess is too slow to search with; this leaves the port open, asks
    continuously, and names what it hears each time, so you can walk a jumper
    down the header and watch for the line to change.

    GET_ANGLES only, so it commands nothing however long it runs.
    """
    request = bytes([HEADER, HEADER, 0x02, 0x20, FOOTER])
    print(f'{args.port} at {args.baud} baud. Asking for angles twice a '
          f'second.\nNo motion is commanded -- leave it running.\n')
    print('Move the RECEIVE wire from pin to pin. Ground and the transmit')
    print('wire stay put. Watch for REPLY.\n')
    print('  silent = nothing driving that pin (or not seated)')
    print('  echo   = hearing your own transmit wire couple across\n')

    ser = open_port(args.port, args.baud)
    n = 0
    try:
        while True:
            n += 1
            ser.reset_input_buffer()
            ser.write(request)
            ser.flush()
            time.sleep(0.2)
            got = ser.read(512)
            with contextlib.redirect_stdout(io.StringIO()):
                framed = _drain_frames(bytearray(got)) if got else 0
            if framed:
                print(f'  [{n:>4}]  REPLY  {len(got)}B  {got[:20].hex(" ")}')
                print('\n' + '=' * 68)
                print('  That pin is the arm\'s TX. Leave the wire there.')
                print('=' * 68)
                print('\n    ./scripts/serial_move_test.py '
                      f'--port {args.port}\n')
                return 0
            if not got:
                state = 'silent'
            elif len(got) == len(request):
                state = f'echo    {got.hex(" ")}'
            else:
                state = f'{len(got)}B      {got[:20].hex(" ")}'
            print(f'  [{n:>4}]  {state}')
            time.sleep(0.3)
    except KeyboardInterrupt:
        print(f'\nStopped after {n} attempts, no reply.')
        return 1
    finally:
        ser.close()


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
    ap.add_argument('mode', choices=['loopback', 'listen', 'poke', 'hunt'])
    ap.add_argument('--port', default='/dev/ttyACM0',
                    help='ttyACM0 for an Uno, ttyUSB0 for most adapters')
    ap.add_argument('--baud', type=int, default=1000000,
                    help='the arm runs at 1000000 (default)')
    ap.add_argument('--seconds', type=float, default=10.0,
                    help='listen duration')
    ap.add_argument('--rounds', type=int, default=5,
                    help='loopback passes; one is too small a sample to read '
                         'an error rate off')
    args = ap.parse_args()
    return {'loopback': cmd_loopback, 'listen': cmd_listen,
            'poke': cmd_poke, 'hunt': cmd_hunt}[args.mode](args)


if __name__ == '__main__':
    sys.exit(main())
