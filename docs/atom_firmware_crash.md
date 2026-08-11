# Atom firmware crash-loop (myCobot 280 Pi, atomMain 6.2)

Captured 2026-08-10 on `/dev/ttyTHS1` at 1000000 baud, Jetson Orin Nano driving
the arm's UART directly (no Raspberry Pi in the path). Firmware freshly
reflashed to 6.2 via myStudio; behaviour unchanged by the reflash.

## Symptom

The ESP32 rejects a command as malformed, then panics and reboots, in a loop:

```
cmd_len error cmd_len error cmd_len error
Guru Meditation Error: Core  1 panic'ed (LoadProhibited). Exception was unhandled.

Core 1 register dump:
PC      : 0x400d9028  PS      : 0x00060230  A0      : 0x800da571  A1      : 0x3ffb1db0
A2      : 0x00000000  A3      : 0x400d9024  A4      : 0x3f401870  A5      : 0x000000fa
A6      : 0x3ffb1f60  A7      : 0x3ffb89ec  A8      : 0x00000000  A9      : 0x3ffb1ee0
A10     : 0x3ffb89ec  A11     : 0x00000000  A12     : 0x00000000  A13     : 0x00000001
A14     : 0x00060420  A15     : 0x00000000  SAR     : 0x0000000a  EXCCAUSE: 0x0000001c
EXCVADDR: 0x00000001  LBEG    : 0x4000c28c  LEND    : 0x4000c296  LCOUNT  : 0x00000000

Backtrace: 0x400d9028:0x3ffb1db0 0x400da56e:0x3ffb1f50 0x400d1734:0x3ffb1f90
           0x400e1f81:0x3ffb1fb0 0x40088701:0x3ffb1fd0

Rebooting...
```

`EXCCAUSE 0x1c` is LoadProhibited; `EXCVADDR 0x00000001` with `A2 = 0` is a
null-pointer dereference. A malformed command should be rejected, not crash the
controller.

## How it presents to a user

Everything downstream of the crash looks like a different fault, which is why
this took a session to find:

- the arm "goes silent" -- it is rebooting
- brief windows of responsiveness between crashes
- `send_angles` accepted and ignored
- `set_color` ignored
- one erratic large motion, presumably a partially executed command
- `get_angles` returning -1 roughly half the time -- pymycobot cannot parse a
  stream that also carries the panic text, ESP32 boot output and the internal
  servo bus

## What the host UART carries

Not just protocol replies:

- `Guru Meditation ...` panic dumps, ~2.4kB of ASCII
- ~277 bytes of nulls during reboot
- the ESP32's INTERNAL Feetech servo bus, e.g. `ff ff 01 11 00 ...` for servo
  IDs 1-6 -- these are the arm's own servo transactions leaking onto the host
  link
- valid `fe fe 0e 20 ... fa` replies mixed in among all of the above

## Ruled out

- **Jetson UART**: PIO mode active (jetsonhacks DMA->PIO overlay), driver bound,
  port uncontended, and it carries valid frames.
- **Wiring**: reads work; the same wires carry replies.
- **Servos**: temps, voltages and encoders all read live and vary; all six
  respond on the internal bus.
- **Torque/power under load**: link stability is the same with servos released
  (5/25 good) as energised (6/25) -- and `set_color`, which drives no servo,
  also fails.
- **Calibration**: was lost and has been redone; encoders now 2048 and angles 0.
  Unrelated to the crash.

## Reproduction

Poll `GET_ANGLES` (`fe fe 02 20 fa`) at ~2Hz and read the raw stream rather than
going through pymycobot, which hides the panic text:

```bash
python3 - <<'PY'
import serial, time
sp = serial.Serial('/dev/ttyTHS1', 1000000, timeout=1.0)
time.sleep(2)
for _ in range(25):
    sp.write(bytes([0xfe,0xfe,0x02,0x20,0xfa])); sp.flush()
    time.sleep(0.5)
    d = sp.read(16384)
    if b'Guru Meditation' in d:
        print(d.decode('ascii', 'ignore')); break
PY
```

## Related

`elephantrobotics/myCobot` issue #48 reports motors unresponsive after a power
cycle on a 280 Pi, fixed only temporarily by reflashing. Same class of symptom;
this capture adds the actual crash.
