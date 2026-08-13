#!/usr/bin/env python3
"""Does a LONGER command crash the Atom more often than a short one?

The question
------------
`cmd_len error` is the Atom reporting a frame whose length field does not
match its payload, and repeated ones precede the LoadProhibited panic. One way
to produce that is losing a byte on the way in. If the host->arm path drops
bytes at some rate per BYTE, then a long frame is mangled proportionally more
often than a short one -- and this arm's commands differ by a factor of three
and a half:

    get_angles     fe fe 02 20 fa                      5 bytes
    set_color      fe fe 05 6a R G B fa                8 bytes
    send_angles    fe fe 0f 22 <12 bytes> <speed> fa  18 bytes

That predicts reads mostly working while motion commands crash it, which is
exactly what this link has done all along.

Why it is worth separating from load
------------------------------------
Unloading the arm took the link from 0% to 100% on 2026-08-13, which makes
mechanical load look like the whole story. But every earlier test of "does
commanding it break the link" also MOVED the arm, so load and frame length
have never been varied independently -- longer commands were always the ones
that also made the servos work.

This holds the arm still. `send_angles` is sent to the pose the arm is
already in, so the 18-byte frame goes out with no motion and no change in
load. If the long frame still crashes it, length is a cause in its own right
and no amount of parking will fix motion commands. If all three are equal,
frame length is exonerated and load stands as the explanation.

`set_color` is the control that makes this a ladder rather than a pair: it is
longer than a read but drives only the Atom's own LED -- no servo, no current,
nothing mechanical -- so it separates "longer frame" from "does something".

Reboots are counted from the kernel's framing-error tally. The ESP32's boot
output comes out at the ROM's rate, not this port's, so it cannot frame at
1000000 and lands as a burst of ~556 framing errors. See
docs/atom_firmware_crash.md.

    ./scripts/probe_frame_length.py
    ./scripts/probe_frame_length.py --rounds 4 --per-cell 25
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src', 'mycobot_driver'))

from arm_broker import SOCK_PATH  # noqa: E402
from mycobot_driver.collision_guard import CollisionGuard  # noqa: E402

FRAME_ERRORS_PER_REBOOT = 556.0


def ask(sock_path, req, timeout=90.0):
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
    except (ConnectionRefusedError, FileNotFoundError) as e:
        raise SystemExit(f'no broker on {sock_path} ({e}); '
                         'start ./scripts/arm_broker.py')
    with s:
        s.sendall((json.dumps(req) + '\n').encode())
        return json.loads(s.makefile().readline())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rounds', type=int, default=3,
                    help='times through the three cells (default 3)')
    ap.add_argument('--per-cell', type=int, default=20,
                    help='commands per cell per round (default 20)')
    ap.add_argument('--gap', type=float, default=0.25,
                    help='seconds between commands (default 0.25)')
    ap.add_argument('--speed', type=int, default=20,
                    help='speed for the send_angles cell (default 20)')
    ap.add_argument('--sock', default=SOCK_PATH)
    args = ap.parse_args()

    st = ask(args.sock, {'cmd': 'state'})
    hold = st.get('angles')
    link = st.get('link', {})
    if hold is None or st.get('age_ms', 1e9) > 5000:
        print('refusing: need a fresh pose to hold. The whole design of this '
              'test is that send_angles commands the pose the arm is ALREADY '
              f'in; without a current reading it would command a stale one.')
        return 2
    ok, why = CollisionGuard().check(hold)
    if not ok:
        print(f'refusing: the arm is at a pose the guard rejects -- {why}')
        return 2
    if link.get('fraction', 0) < 0.8:
        print(f'warning: link is {link.get("fraction", 0):.0%} before starting. '
              'This test attributes DEGRADATION to frame length, which needs a '
              'clean baseline to degrade from.\n')

    print(f'holding {[round(a, 1) for a in hold]} -- no motion is commanded\n')
    print(f'link {link.get("fraction", 0):.0%} at the start, '
          f'{args.per_cell} commands per cell, {args.rounds} rounds\n')

    cells = [
        ('get_angles   5B', {'cmd': 'call', 'method': 'get_angles'}),
        ('set_color    8B', {'cmd': 'call', 'method': 'set_color',
                             'args': [0, 0, 255]}),
        ('send_angles 18B', {'cmd': 'send_angles', 'angles': hold,
                             'speed': args.speed, 'force': True}),
    ]
    totals = {name: [0, 0, 0.0] for name, _ in cells}   # sent, refused, reboots

    for rnd in range(args.rounds):
        print(f'round {rnd + 1}')
        for name, req in cells:
            before = ask(args.sock, {'cmd': 'icount'})['icount']
            refused = 0
            t0 = time.monotonic()
            for _ in range(args.per_cell):
                r = ask(args.sock, req)
                if not r.get('ok'):
                    refused += 1
                time.sleep(args.gap)
            after = ask(args.sock, {'cmd': 'icount'})['icount']
            elapsed = time.monotonic() - t0
            fe = after['frame'] - before['frame']
            reboots = fe / FRAME_ERRORS_PER_REBOOT
            tx = after['tx'] - before['tx']
            h = ask(args.sock, {'cmd': 'health'}).get('link', {})
            t = totals[name]
            t[0] += args.per_cell
            t[1] += refused
            t[2] += reboots
            print(f'  {name}  tx={tx:5d}B  frame_err={fe:5d}  '
                  f'reboots={reboots:4.1f}  link={h.get("fraction", 0):4.0%}'
                  + (f'  refused={refused}' if refused else ''))
            # Let it recover between cells so a cell is not scored on the
            # damage the previous one did.
            time.sleep(3.0)
        print()

    print('totals')
    print(f'  {"cell":16s} {"commands":>9s} {"reboots":>8s} {"per 100 cmds":>13s}')
    for name, _ in cells:
        sent, refused, reboots = totals[name]
        print(f'  {name:16s} {sent:9d} {reboots:8.1f} '
              f'{reboots / sent * 100:13.1f}')
    print('\nIf reboots scale with frame length, the host->arm path is losing '
          'bytes and\nlong frames simply have more chances to lose one. If '
          'the three are equal,\nframe length is not a cause and load stands.')

    end = ask(args.sock, {'cmd': 'state'})
    if end.get('angles') and end.get('age_ms', 1e9) < 4000:
        drift = max(abs(a - b) for a, b in zip(end['angles'], hold))
        print(f'\narm moved {drift:.1f}deg over the run '
              '(it was commanded to hold; anything large is a corrupted frame '
              'being obeyed, which is the fault this is looking for)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
