#!/bin/bash
# Rebind the Jetson UART and test WRITES inside the short window that follows.
#
#   sudo ./scripts/rebind_and_write.sh
#
# Observed 2026-08-10: after a serial-tegra unbind/rebind the arm answers for
# a few attempts and then goes silent again. Every write test so far has been
# run from a separate process started after that window had already closed,
# which is why they all found a dead bus. This does the rebind and the test in
# one go so the commands land while the arm is still talking.
#
# The test is set_color, not motion: it touches no servo, draws no meaningful
# current, and cannot be blocked mechanically, so the LED changing or not is a
# clean answer to "does this controller execute writes at all".
#
# WATCH THE ATOM LED while this runs.

set -u
DEV="3100000.serial"
DRIVER="/sys/bus/platform/drivers/serial-tegra"
PORT="/dev/ttyTHS1"
REAL_USER="${SUDO_USER:-$(whoami)}"

if [ "$(id -u)" -ne 0 ]; then
    echo "needs sudo: sudo $0" >&2
    exit 1
fi

echo "=== rebinding $DEV ==="
echo "$DEV" > "$DRIVER/unbind" 2>/dev/null || true
sleep 1
echo "$DEV" > "$DRIVER/bind" 2>/dev/null || true
echo "  rebound, waiting 4s for the ESP32"
sleep 4

echo "=== testing inside the window -- WATCH THE LED ==="
sudo -u "$REAL_USER" python3 -u - "$PORT" <<'PY'
import sys, time
import serial

port = sys.argv[1]
sp = serial.Serial(port, 1000000, timeout=0.4)
time.sleep(0.5)


def frame(cmd, payload=b''):
    """myCobot frame: fe fe <len> <cmd> <payload> fa, len counts cmd+payload+1."""
    body = bytes([cmd]) + payload
    return bytes([0xfe, 0xfe, len(body) + 1]) + body + bytes([0xfa])


GET_ANGLES = frame(0x20)
# SET_COLOR is 0x6a with three RGB bytes.
COLORS = [('RED', 255, 0, 0), ('BLUE', 0, 0, 255), ('GREEN', 0, 255, 0)]

alive = 0
for trial in range(6):
    sp.reset_input_buffer()
    sp.write(GET_ANGLES)
    sp.flush()
    time.sleep(0.25)
    data = sp.read(4096)
    if data:
        alive += 1
        ok = data.find(bytes([0xfe, 0xfe, 0x0e, 0x20]))
        txt = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)
        print(f'  read {trial}: {len(data)} bytes, '
              f'valid reply {"yes" if ok >= 0 else "NO"}')
        if 'error' in txt:
            print(f'    firmware said: {txt.strip()[:70]}')
    else:
        print(f'  read {trial}: silent')
    time.sleep(0.2)

if not alive:
    print('\nThe window had already closed -- nothing answered. Run again.')
    sys.exit(1)

print(f'\narm answered {alive}/6 reads; sending LED writes now')
for name, r, g, b in COLORS:
    print(f'  LED -> {name}', flush=True)
    for _ in range(3):
        sp.write(frame(0x6a, bytes([r, g, b])))
        sp.flush()
        time.sleep(0.15)
    time.sleep(1.8)
    # Anything the firmware says back about the write is the point.
    junk = sp.read(4096)
    if junk:
        txt = ''.join(chr(c) if 32 <= c < 127 else '.' for c in junk)
        print(f'    replied {len(junk)} bytes: {txt.strip()[:70]}')

print()
print('Did the LED change red -> blue -> green?')
print('  yes -> the controller executes writes; earlier failures were')
print('         pymycobot framing, not the hardware')
print('  no  -> it answers reads and executes nothing, even a write that')
print('         touches no servo')
PY
