#!/usr/bin/env python3
"""Find out whether the arm's ESP32 can be driven directly over USB.

    ./scripts/probe_usb_arm.py

Read-only. It asks the arm for its joint angles and nothing else -- no motion
is commanded, so this is safe to run with the arm in any pose.

WHICH PORT

The one on the ATOM -- the small module with the LED matrix, up at the head
near the end effector. That is the ESP32 sitting on the servo bus, and it is
the only one that could speak the robot protocol.

NOT the USB-C on the base. On a 280 Pi that is the Raspberry Pi's power
input, and the Pi's USB device controller is not enabled anyway
(/sys/class/udc/ is empty), so it cannot present itself as a serial port even
if you wanted it to.

If you are unsure which you have plugged into, this script prints the USB
descriptor of whatever it finds. An Atom shows up through a USB-serial bridge
-- expect a vendor string like Silicon Labs, wch.cn, or QinHeng, and a device
name of the CP210x / CH9102 family. Anything mentioning Raspberry Pi means you
are on the wrong port.

WHY THIS MATTERS

The 280 Pi has an M5Stack Atom (ESP32) driving the servo bus, and the
Raspberry Pi reaches it over the GPIO UART at 1000000 baud. Every arm command
therefore travels: this machine -> network -> server.py -> UART -> ESP32.

If the Atom's USB-C port is a second serial interface into the same firmware,
that whole chain collapses to one cable. It would remove the arm-side network
leg, which is the part of the ~250ms round trip nothing in this project has
managed to reduce, and it is the topology commercial arms of this class use
(compute -> USB -> MCU -> servos).

BEFORE RUNNING

Stop the Pi's server, or two masters will be driving one bus:

    ssh er@192.168.0.15 'sudo systemctl stop mycobot_server'

Afterwards, put it back if you want the network path again:

    ssh er@192.168.0.15 'sudo systemctl start mycobot_server'
"""

import argparse
import glob
import sys
import time


CANDIDATE_GLOBS = ('/dev/ttyUSB*', '/dev/ttyACM*')
# 1000000 first: it is what pi/server.py opens the arm's UART at, so it is the
# firmware's rate rather than a property of the cable. The rest are common
# fallbacks in case the USB side is bridged differently.
BAUDS = (1000000, 115200, 921600, 230400)


def find_ports():
    ports = []
    for pattern in CANDIDATE_GLOBS:
        ports.extend(sorted(glob.glob(pattern)))
    return ports


def describe(port):
    """USB descriptor of the device behind this tty.

    Walks UP from the tty until a level carrying idVendor is found, and stops
    at the first one. Starting several levels up instead reported the xHCI
    root hub -- "product=xHCI Host Controller, manufacturer=Linux" -- which is
    true of the machine's USB controller and says nothing about what is
    plugged into it.
    """
    import os
    base = os.path.basename(port)
    for depth in ('', '..', '../..', '../../..', '../../../..'):
        root = os.path.join(f'/sys/class/tty/{base}/device', depth)
        try:
            with open(os.path.join(root, 'idVendor')) as fh:
                vendor = fh.read().strip()
        except OSError:
            continue
        bits = [f'idVendor={vendor}']
        for attr in ('idProduct', 'product', 'manufacturer'):
            try:
                with open(os.path.join(root, attr)) as fh:
                    bits.append(f'{attr}={fh.read().strip()}')
            except OSError:
                pass
        return ', '.join(bits)
    return 'no USB descriptor available'


def readable(port):
    """(ok, why-not). A permission problem is not a protocol problem, and
    reporting it as one sends you looking in completely the wrong place."""
    import grp
    import os
    if not os.access(port, os.R_OK | os.W_OK):
        try:
            owner = grp.getgrgid(os.stat(port).st_gid).gr_name
        except (OSError, KeyError):
            owner = 'dialout'
        return False, owner
    return True, None


def try_port(port, baud, timeout=6.0):
    """Return joint angles if the arm answers on this port/baud, else None."""
    try:
        from pymycobot import MyCobot280
    except ImportError as e:
        print(f'  pymycobot missing: {e}')
        return None
    try:
        mc = MyCobot280(port, str(baud))
    except Exception as e:
        print(f'    {baud:>8}  could not open: {e.__class__.__name__}: {e}')
        return None
    time.sleep(0.5)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            angles = mc.get_angles()
        except Exception as e:
            print(f'    {baud:>8}  error: {e.__class__.__name__}: {e}')
            return None
        # A cold link commonly answers -1 or [] for the first command or two,
        # exactly as it does over TCP, so this retries rather than concluding.
        if isinstance(angles, list) and len(angles) == 6:
            return angles
        time.sleep(0.3)
    print(f'    {baud:>8}  opened, but no valid reply in {timeout:.0f}s')
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', help='test only this port')
    ap.add_argument('--baud', type=int, help='test only this baud')
    args = ap.parse_args()

    ports = [args.port] if args.port else find_ports()
    if not ports:
        print('No USB serial device found.\n')
        print('Checked: ' + ', '.join(CANDIDATE_GLOBS))
        print('\nSo either the cable is not connected, or that USB-C port is')
        print('power-only. Both are possible and neither is something')
        print('software can work around -- if a data-capable port were')
        print('present it would appear here within a second of plugging in.')
        print('\nWorth confirming with:  dmesg | tail -20   (right after')
        print('plugging the cable in -- a data port always logs something)')
        return 1

    print(f'Found {len(ports)} serial device(s):\n')
    for p in ports:
        print(f'  {p}  ({describe(p)})')
    print()

    # Check access before touching bauds. Every baud fails identically on a
    # permission error, and four identical failures followed by "the arm did
    # not answer" points at the firmware when the problem is a unix group.
    blocked = [(p, g) for p, g in ((p, readable(p)[1]) for p in ports)
               if g is not None]
    if blocked and len(blocked) == len(ports):
        port, group = blocked[0]
        print('=' * 68)
        print(f'  {port} exists, but this user cannot open it.')
        print('=' * 68)
        print('\nThe device is there -- which is the interesting half of the')
        print(f'question already answered -- but it is owned by group '
              f'"{group}"')
        print('and you are not in it. Nothing about the arm has been tested')
        print('yet.\n')
        print(f'    sudo usermod -aG {group} $USER\n')
        print('Then either log out and back in, or for this shell only:\n')
        print(f'    newgrp {group}\n')
        print('and run this again.')
        return 1

    bauds = [args.baud] if args.baud else BAUDS
    for port in ports:
        print(f'{port}:')
        for baud in bauds:
            angles = try_port(port, baud)
            if angles is not None:
                print(f'    {baud:>8}  ARM ANSWERED: {angles}')
                print()
                print('=' * 68)
                print('  The arm is reachable directly over USB.')
                print('=' * 68)
                print('\nRun the stack without the Pi in the command path:\n')
                print(f'    ./run.py connection:=serial serial_port:={port} '
                      f'serial_baud:={baud} \\')
                print('        source:=device device:=0 '
                      'device_auto_exposure:=false device_exposure:=50\n')
                print('Keep mycobot_server stopped on the Pi while you do --')
                print('two masters on one serial bus behaves erratically.')
                return 0
        print()

    print('=' * 68)
    print('  A serial device exists, but the arm did not answer on it.')
    print('=' * 68)
    print('\nPermissions are fine, so this is a real negative. Most likely')
    print('that port is the ESP32 bootloader/console rather')
    print('than the robot protocol, or the firmware only bridges the GPIO')
    print('UART. Check that mycobot_server is stopped on the Pi -- if it is')
    print('still running it owns the bus and will win.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
