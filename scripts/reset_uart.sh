#!/bin/bash
# Reset the Jetson's ttyTHS1 UART controller and wait for the arm to respond.
# Requires sudo for the sysfs unbind/rebind.
#
# Usage:  sudo ./scripts/reset_uart.sh

DEV="3100000.serial"
DRIVER="/sys/bus/platform/drivers/serial-tegra"
PORT="/dev/ttyTHS1"
REAL_USER="${SUDO_USER:-$(whoami)}"

for cycle in $(seq 1 5); do
    echo "=== Reset cycle $cycle ==="
    echo "$DEV" > "$DRIVER/unbind" 2>/dev/null || true
    sleep 1
    echo "$DEV" > "$DRIVER/bind" 2>/dev/null || true
    echo "  UART rebound, waiting 5s for ESP32..."
    sleep 5

    # Use pymycobot directly with a longer timeout instead of poke
    RESULT=$(sudo -u "$REAL_USER" python3 -c "
import serial, time, sys
try:
    s = serial.Serial('$PORT', 1000000, timeout=1.0)
    for attempt in range(8):
        s.reset_input_buffer()
        s.write(bytes([0xfe, 0xfe, 0x02, 0x20, 0xfa]))
        s.flush()
        time.sleep(0.5)
        data = s.read(512)
        if len(data) >= 17 and data[0] == 0xfe and data[1] == 0xfe:
            angles = []
            for i in range(6):
                hi = data[4 + i*2]
                lo = data[5 + i*2]
                val = (hi << 8 | lo)
                if val > 32767: val -= 65536
                angles.append(val / 100.0)
            print(f'OK attempt={attempt+1} angles={angles}')
            sys.exit(0)
        elif len(data) > 0:
            print(f'PARTIAL attempt={attempt+1} len={len(data)} hex={data.hex(\" \")}')
        else:
            print(f'SILENT attempt={attempt+1}')
        time.sleep(0.5)
    s.close()
except Exception as e:
    print(f'ERROR {e}')
sys.exit(1)
" 2>&1) || true

    echo "  $RESULT"

    if echo "$RESULT" | grep -q "^OK "; then
        echo "  === ARM IS ALIVE ==="
        exit 0
    fi
done

echo "=== ARM DID NOT RESPOND after 5 reset cycles ==="
echo "Power-cycle the arm and try again."
exit 1
