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

## Reflashing does not fix it

atomMain 6.2 was flashed **twice** with myStudio, power-cycling between. The
crash survives both.

| | valid replies | crashes | silent |
|---|---|---|---|
| after first flash | 1/20 | 2 | 13 |
| after second flash | 19/60 | 2 | 35 |

The second attempt improved the proportion of good replies but did not stop the
panic, so the ratio is a measure of how often it manages a reply between
reboots rather than of any real recovery.

Servo zero calibration DOES survive: after recalibrating, encoders read 2048
and joint6 still reported 2048 across subsequent power cycles and reflashes.

## Recurrence, 2026-08-13 — and the UART formally cleared

The same crash is back, and this time the Jetson side was measured rather than
argued about. `{"cmd": "counters"}` reads the kernel's own `TIOCGICOUNT` tally
either side of each transaction, which is the one view of the link that does
not depend on anything in userspace parsing correctly.

Captured live, `cmd_len error` and the full `Guru Meditation ... LoadProhibited`
dump, unchanged from the 2026-08-10 capture above.

**What the port is doing, over 40 transactions:**

| counter | value | reading |
|---|---|---|
| `tx` | 200 | exactly 5 per poke. Every write reached the wire |
| `rx` on a failed poll | **0** | not corrupted, not mis-parsed — absent |
| `frame` | in bursts of ~556, else 0 | see below |
| `parity`, `brk` | 0 | |

The framing errors are not a link fault. They arrive in exact multiples of
~556 and only alongside a reboot: that is the ESP32's boot output, emitted at
the ROM's own rate rather than this port's, so it *cannot* frame at 1000000.
It therefore doubles as a **reboot counter**, which hit rate alone cannot
separate from ordinary silence.

**The baud is right.** Poking at 1000000 and reading back at neighbouring
rates, framing errors per character stayed at zero throughout and the reply
rate did not improve:

| read baud | 960000 | 980769 | 1000000 | 1020000 | 1041667 |
|---|---|---|---|---|---|
| replies | 10/16 | 8/16 | 13/16 | 13/16 | 12/16 |

980769 and 1020000 are the two rates a 408MHz parent clock would land on if
the divisor were being rounded. Neither is better, so it is not.

**Waiting longer buys nothing; asking again does.** A reply that is coming has
arrived within 30ms:

| read wait | 30ms | 60ms | 120ms | 250ms | 500ms | 1000ms |
|---|---|---|---|---|---|---|
| replies | 67% | 80% | 57% | 73% | 80% | 70% |

| pokes per attempt | 1 | 2 | 3 |
|---|---|---|---|
| replies | 76% | 86% | 92% |

So the 1.5s poll window was spending a second and a half per failure to learn
that nothing was coming. `_poll_once` now re-pokes instead — five attempts of
50ms — which leaves the valid fraction where it was but makes a failed poll
cost 250ms rather than 1500ms.

**The break is not what crashes it.** It was worth checking, since the break
added on 2026-08-13 is sent 4x a second and a break condition is exactly the
kind of malformed input this firmware dies on. 60 trials per cell:

| | replies | reboots |
|---|---|---|
| reconfigure + break | 40/60 (67%) | 2.0 |
| reconfigure only | 37/60 (62%) | 3.2 |
| neither | 25/60 (42%) | 1.1 |

Break slightly reduces reboots rather than causing them, and the reconfigure
is confirmed load-bearing at a larger cell size than the original 12.

**The crash is provoked by traffic, not free-running.** A passive listen —
3s, twice, no poke — returns zero bytes and no crash text. It only panics
while being asked.

So on 2026-08-13 the Jetson UART is clean by every measurement available:
writes leave intact, received characters carry no line errors, the rate is
correct, the DMA→PIO overlay is live in the running device tree, and the port
has a single owner. What is left is the Atom, doing what this document
already describes.

## Related

`elephantrobotics/myCobot` issue #48 reports motors unresponsive after a power
cycle on a 280 Pi, fixed only temporarily by reflashing. Same class of symptom;
this capture adds the actual crash.
