# JNAReach-RoboticArm

ROS 2 Humble workspace driving a **myCobot 280** with an eye-in-hand camera
doing visual servoing. The long-term goal is autonomously **pressing elevator
buttons**; hand-following is the stepping stone that proves the
perception-to-motion loop works.

## Layout

**The target topology is a Jetson Orin Nano doing everything** — RealSense on
USB, arm on the Jetson's own UART at `/dev/ttyTHS1`, no network in the control
loop. [README.md](README.md) is written for that and is the install guide.

```bash
./run.py connection:=serial serial_port:=/dev/ttyTHS1 serial_baud:=1000000 \
    source:=realsense
```

**The UART link is verified on real hardware as of 2026-08-01.** `poke`
returned a well-formed `GET_ANGLES` reply at 1000000 baud —
`fe fe 0e 20 ff 48 cb 21 fc be 0e dd 25 0b f1 b8 fa`, decoding to six
plausible joint angles with joint2 resting a third of a degree past its −135°
software limit, which is an arm hanging under gravity and not something noise
produces. `serial_move_test.py` then commanded motion successfully. Jetson →
UART → ESP32 → servos, no Pi and no network in the command path.

**The launch defaults still point at the Raspberry Pi**, so the long form
above is needed until someone flips them — a two-line change in
`servo_demo.launch.py`. Both paths therefore remain live, and confusing which
one is running is a standing source of wasted time:




| | Where | Holds |
|---|---|---|
| Dev / runtime | Ubuntu 22.04, `~/JNAReach-RoboticArm` | the ROS workspace, everything in `src/` |
| Robot *(network path)* | Raspberry Pi in the arm base, `~/JON/mycobot_project`, default `192.168.0.15` | `pi/server.py` (arm TCP, port 9000) and `pi/camera_stream.py` (MJPEG, port 8080) |
| Arm | myCobot 280 | pymycobot, from whichever host is master |

Anything below about the Pi, MJPEG transit, or `redeploy_pi.sh` applies to the
network path only. The measured servo results apply to both — they are about
the control law, not the transport — **except** that they were all fitted
against a narrow-FOV webcam, which a RealSense D405 is not.

**`pi/` is not part of the ROS build.** Editing those files does nothing until
`./scripts/redeploy_pi.sh` copies them over and restarts the units. `run.py`
checksums them against the Pi and refuses to start quietly out of sync — if it
says the Pi is not running current code, that is what it means.

```bash
python3 pi/test_camera_stream.py     # runs without a Pi or a camera
```

**"The camera lags" now has a number.** The Pi stamps each MJPEG part with its
capture time (`X-Capture-Us`) and `camera_node` reports:

```
camera transit: +4ms mean, +230ms worst, over best case
```

The two clocks are not synchronised, so the absolute difference is
meaningless — but the offset is constant, so the smallest difference seen in a
run is taken as the zero and everything is reported above it. That is enough
to catch a spike, which is the whole question. **This is the only measurement
of the Pi→host leg**; the servo's `frames Nms old` counts from `camera_node`'s
*publish* stamp, written after capture, encode, network and decode, so a stall
out there is invisible to it.

Read it against the other two: a high transit figure is frames delayed *in
flight* (Pi stalling, link congested, host not draining the socket). A slow
camera is a low `streaming N FPS` on the Pi and does **not** raise transit —
different fault, different fix.

**An overloaded Pi does not look like an overloaded Pi.** It shows up as three
apparently unrelated faults at once: the driver taking ten seconds to connect,
homing timing out, and `camera_node` failing to connect for half a minute
while `run.py`'s preflight said the camera was fine moments earlier. The Pi
runs both `server.py` and `camera_stream.py`, so anything that eats a core
starves the other. Check the camera server's own FPS/client log line before
believing the fault is where it appears to be.

Override the address with `MYCOBOT_IP`; every script and launch file reads it.

## Running it

```bash
./run.py                 # one script: preflight, build check, launch, menu
```

Extra args pass through to `ros2 launch` (`./run.py gain:=0.9`). The menu keys
are documented at the top of `run.py`; `1` starts a hunt, `2` homes, `q` quits.
Prefer `./run.py` over a bare `ros2 launch` — it catches the four kinds of
stale copy (Pi files, `install/`, leftover processes, wrong workspace) that
have each cost a debugging session, and it validates launch argument names,
which `ros2 launch` silently ignores when misspelled.

`pip install -r requirements.txt` — **the upper bounds are load-bearing**, see
the file. NumPy 2, MediaPipe 1.0 and OpenCV 5 each break ROS Humble's
compiled `cv_bridge` in ways whose error messages point somewhere else.

## Visual servoing — read before retuning

`src/mycobot_perception/mycobot_perception/visual_servo_node.py` has a long
module docstring explaining the control law. The short version, because this
is the part that has been got wrong repeatedly:

**The binding constraint is dead time, not gain.** Frames go Pi → JPEG →
network → decode → MediaPipe on CPU, then the jog takes time to execute. Two
or three detections arrive still reporting the old error while a correction is
in flight. A loop that re-commands that correction overshoots by construction —
that is the "flicking left and right" symptom, and no amount of PID fixes it.
An integral term makes it strictly worse, because it winds up during exactly
that interval.

So: proportional only, and `gain` is a **fraction of a full correction**
(0.7), not degrees. The node remembers every jog it sends and adds back the
image motion not yet visible, correcting where the hand *will* be. That
compensation is what makes 0.7 safe where 0.35+ oscillates without it.

**Overshoot and trailing are different failures.** The above fixes overshoot
on a hand held still. It does nothing for a hand that is *moving*, because a
proportional loop tracks a moving target at a constant distance behind it —
`speed * (detection_interval / gain + round_trip_lag)`, about 70px of a 640
frame at a moderate pace. That term is mostly dead time, so raising `gain`
barely touches it, and the symptom is "it keeps my hand in view but never
centres it". `lead_time` (0.15s) is the fix: the node measures how fast the
hand is crossing the image, subtracts its own jogs from that, and aims ahead.
Worth ~a third off the offset on smooth motion; little help on a fast wave,
and it costs ~1s of extra settling on a hand that appears suddenly.
`lead_time:=0.0` restores proportional-only exactly.

Two traps:

- **`command_lag` (0.15) must stay BELOW the true lag (~0.25).** The error is
  asymmetric. Too low just corrects slightly hard. Too high sweeps in jogs
  that already landed, double-counts them, concludes it overshot, and
  reverses — reproducing the flicking. A slow pipeline is a reason to fix the
  pipeline, not to raise this.
- **The gain is a profile, and the useful half is the near half.** `gain`
  (0.45) is the gain at the *centre*; `progressive_gain` (2.0) raises it with
  distance, meeting 0.7 exactly where `max_step_deg` starts clamping (error
  0.29 = 91px of a 640 frame, 69px vertically). Past that point every error
  commands the same 5° and every jog logs `+5.00`, so "chase a far hand
  harder" is not available — the arm's own joint speed binds before the clamp
  does. Backing off near the centre is, and it is worth more: against the old
  flat 0.7, acquisition 1.18s → 0.67s, residual 6px → 2px, sign reversals in
  the tail 8.9 → 0. **If you retune either number keep `gain * (1 +
  progressive_gain * 0.29)` near 0.7**, or the far field changes too. Adding
  the curve on top of 0.7 instead of reprofiling around it makes the near
  field hotter and rings — that mistake was made once already.
- **The vertical axis is not the horizontal one.** Error is normalised per
  axis, so a unit error means "at the edge" both ways, but on a 640x480 sensor
  those edges are ~25 and ~19 degrees away — hence `assumed_v_deg_per_error`
  (0 = use the horizontal number). If tilting under-shoots where panning does
  not, that is the knob; if the camera is mounted rotated the axes are
  cross-coupled and no per-axis scalar helps, so measure the real 2x2 with
  `skip_probe:=false`. The `responds Nx as strongly as assumed` line says
  which case you are in.
- **Keep the servo's `rate` at or above the camera's.** The loop acts once
  per new sighting, so a lower rate delays every correction by up to `1/rate`
  AND silently discards the detections arriving in between — at 15Hz against
  a 30fps camera that was 67ms of added latency and half the frames thrown
  away. The old justification (the driver rate-limited jogs anyway) stopped
  being true when `jog_profile` landed: a jog message now only moves a goal.
- **Auto-exposure is a frame rate control, and it is the least obvious limit
  in the system.** A UVC camera in dim light lengthens its exposure to
  brighten the image, and frame time cannot be shorter than exposure time — so
  it silently caps the rate while still *reporting* 30fps when asked.
  Measured on this webcam at 640x480 MJPG: **10.2 fps on auto, 30.2 fps with a
  short manual exposure**, same camera, same everything else. Since detection
  rate is the ceiling on the whole servo loop, the room lighting was setting
  the tracking performance. `device_auto_exposure:=false` on the host,
  `--no-auto-exposure` on the Pi — or just add light. Too short an exposure
  makes the image dark enough to break detection outright, so measure rather
  than assume; `camera_node` reports the rate it actually captures and names
  this cause when it looks like this.
- **`CAP_PROP_BUFFERSIZE=1` halves the frame rate on this driver** — 30.0 fps
  untouched against 18.5 with it set, measured back to back. It looks like the
  obvious setting for a servo loop, since a queued frame is a stale frame, but
  it only pays off when the reader is SLOWER than the camera. Both readers here
  drain continuously, so the queue never forms and the cost is pure. Neither
  path sets it now (`device_buffersize`, `--buffersize`, both 0 = leave alone).
- **Detection rate is the real ceiling.** Below ~6/s nothing tuned in the
  servo helps. Read the `tracker:` and `pipeline:` log lines before touching
  any gain; `model_complexity:=0` and a smaller camera frame move that number
  more than anything else. Raising `max_step_deg` looks like the fix for slow
  tracking and is not — 5 to 9 moves the simulated error under 3%, because
  the clamp only binds when the hand is already far out and dead time
  dominates there. It also must not exceed the driver's `max_jog_deg` (5.0),
  which silently clips it while the compensator still credits the full jog.

Signs are worked out automatically (`auto_sign`) by correlating what each jog
was predicted to do to the image against what it did. `assumed_h_sign` /
`assumed_v_sign` still exist but should not need reaching for.

**Tune it live, not by relaunching.** The servo takes runtime parameter
changes (`add_on_set_parameters_callback`); structural ones — topics, joint
names, search geometry — are refused with a reason rather than silently
ignored, which is what `ros2 param set` used to do for everything since every
value was read once into an attribute at construction.

```bash
./scripts/tune_servo.py          # with ./run.py already running
```

Scores the loop while it tracks: mean distance from centre, worst miss, and
**sign flips per second**. The last one is the point — a mean alone rewards a
loop that has given up, since a servo parked off-centre scores the same as one
buzzing evenly around it. Per-axis errors are separate because the two axes
have separate scales and a weak vertical is invisible in a combined figure.

```bash
python3 src/mycobot_perception/test/test_servo_math.py
python3 src/mycobot_driver/test/test_jog_profile.py
```
Pins the sign conventions and the auto_sign detector. Runs without ROS — it
loads the functions out of the node by source. **Run it after any change to
the control law**; a flipped sign does not crash, it drives the target out of
frame, which is indistinguishable from a badly mounted camera.

**Only joint1, joint5 and joint3 are ever commanded.** The servo names
`horizontal_joint` (joint1, pan), `vertical_joint` (joint5, tilt) and
`approach_joint` (joint3, only when `approach_enabled`). joints 2, 4 and 6 are
untouched by design — two DOF centre a target in an image and a third changes
range; more would need the full 6-DOF image Jacobian and a pseudo-inverse
rather than this 2x2. Small deltas on the untouched joints do appear in driver
logs, but those are the profiler reconciling its commanded pose with the
measured one after a divergence resync, not the servo asking.

**Driving the arm over USB does not work on this variant, by design.** The
vendor's own port table settles it:

| model | serial port | baud |
|---|---|---|
| 280 M5 | Linux: `/dev/ttyUSB` | 115200 |
| 280 AR | Linux: `/dev/ttyUSB` | 1000000 |
| **280 PI** | **`/dev/ttyAMA0`** | **1000000** |
| 280 JetsonNano | `/dev/ttyTHS1` | 115200 |

The M5 and AR variants list a USB port because on those the host computer *is*
the master over USB. The **PI variant has no USB entry at all** — its
documented interface is the Pi's own GPIO UART, which is exactly what
`pi/server.py` opens, at exactly the baud it uses. The Pi is not a hop in
front of the arm's interface; it is the arm's interface.

Confirmed experimentally before the table was found: `connection:=serial` is
plumbed correctly, and the Atom's USB-C is a real FTDI FT232 (`0403:6001`,
`product=M5stack`), but it answers **-1 to every command at every baud** with
the bus completely free — `mycobot_server` stopped *and* the Bluetooth bridge
below killed. It is the ESP32 console. `connection:=serial` and
`scripts/probe_usb_arm.py` stay for the day a firmware update changes that,
and for the M5/AR variants where the same code would just work.

So the Pi stays in the arm command path and the ~250ms round trip is a
property of this hardware. Ethernet between the Jetson and the Pi is the
remaining improvement.

**That table also validates the Jetson plan.** The `280 JetsonNano` row shows
Elephant shipping a Jetson as the onboard computer, reaching the arm over the
Jetson's own hardware UART (`/dev/ttyTHS1`) rather than USB. So "the Jetson
drives the arm directly" is a supported topology rather than a modification —
UART pins, not a USB cable.

### Putting a Jetson on the arm's UART

The end state, and the software for it is already written:

```bash
./run.py connection:=serial serial_port:=/dev/ttyTHS1 serial_baud:=1000000 \
    source:=device device:=0
```

`connection:=serial` was built for the USB attempt that failed. It is exactly
what this needs — only the port name changes.

**Which pins.** `pi/server.py` opens `/dev/ttyAMA0`, which is the Pi's PL011
UART. On a Pi 4 that lands on **GPIO14 (TXD) = physical pin 8** and **GPIO15
(RXD) = physical pin 10**, plus any ground (6, 9, 14, 20, 25, 30, 34, 39).

Elephant documents the arm's 40-pin connector with the stock Pi pinout, which
confirms this rather than leaving it inferred from `/dev/ttyAMA0`. The labels
are named from the **Pi's** point of view — `GPIO14 (UART TX)` is the *Pi*
transmitting — so a host standing in for the Pi is not crossed: its output goes
to pin 8 exactly where the Pi's output went.

**The three wires you need are adjacent.** Counting down the even row, ground,
TX and RX are the 3rd, 4th and 5th positions:

```
 1  3v3 Power     |  2  5V Power        <- 5V, and only two along from pin 6
 3  GPIO2  SDA    |  4  5V Power        <- 5V
 5  GPIO3  SCL    |  6  Ground          <- ground
 7  GPIO4         |  8  GPIO14 UART TX  <- host's TX out
 9  Ground        | 10  GPIO15 UART RX  <- host's RX in
```

That closeness is also the trap: **a wire reading ~5V is pin 2 or 4, not pin
10.** Measured exactly that way on the bench — D0 correctly on pin 8 (3.6V,
clamped) while D1 sat on a 5V rail, so the adapter transmitted fine, the arm
answered on pin 10, and nothing was listening. It presents as total silence
from `poke`, i.e. identical to a dead bus.

It has to be the PL011 rather than the mini-UART: `/dev/ttyS0` derives its
clock from the core clock and is not dependable at 1000000 baud, which is the
rate this arm runs at. That also means the Pi is configured with BT moved off
the PL011 (`dtoverlay=disable-bt` or `miniuart-bt`). Confirm on the Pi with:

```bash
ls -l /dev/serial*                 # serial0 -> ttyAMA0 means it is on the header
grep -E 'enable_uart|dtoverlay' /boot/firmware/config.txt
sudo raspi-gpio get 14,15          # expect ALT0 = TXD0/RXD0
```

Convenient for the port: **the Orin Nano's 40-pin header carries its UART on
pins 8 and 10 too** (UART1, `/dev/ttyTHS1`), so a board-for-board substitution
is pin-for-pin. Confirmed against NVIDIA's docs, not assumed. 3.3V both sides,
1/3v3, 2 and 4/5V, 6/GND all match as well.

What does **not** carry over is the naming. `GPIO14` is a Broadcom number and
means nothing on Tegra — the Linux GPIO names and numbers are unrelated — and
the device is `ttyTHS1`, not `ttyAMA0`. Only the power, ground and UART
positions are guaranteed; SPI/I2C/PWM assignments merely resemble the Pi's.

Two live risks worth checking before committing hardware:

- **A reported JetPack 7 / L4T R39.2 bug leaves the ttyTHS1 TX pad not
  driving** on the Orin Nano Super devkit — the exact board planned here. It
  would present as transmitting into silence with everything apparently
  correct, i.e. indistinguishable from the wiring faults above. Verify the UART
  loops back on the Jetson alone before wiring it to the arm.
- Corrupted data on ttyTHS1 has also been reported on JetPack 6.2.2 (Orin NX,
  Waveshare carrier).

And do NOT cross TX/RX if you are substituting the Jetson for the Pi — the
arm's harness already crosses them to suit the Pi's header, so the Jetson's
pin 8 goes where the Pi's pin 8 went. Crossing applies only when adding a
second host alongside. Check continuity before powering anything.

Electrically it is undemanding otherwise: Pi GPIO, Jetson GPIO and the ESP32
are all 3.3V logic, so no level shifting. **Use 1000000 baud, not the 115200 in
the JetsonNano row** — that column reflects that variant's own firmware, and
this arm's Atom runs at 1000000 (see `pi/server.py`).

Four things to get right:

- **Free the UART from the serial console.** On Jetson, the 40-pin header UART
  is claimed by a getty by default, which will fight for the port exactly like
  the Bluetooth bridge did: `systemctl disable --now nvgetty` (or
  `serial-getty@ttyTHS*`). Same class of bug as the `dialout` group — the port
  exists, opens, and does not work.
- **One master.** The Pi must not be driving the same lines. Either remove it
  or keep `mycobot_server` and the `rc.local` bridge disabled.
- **Power.** The Jetson wants its own supply (19V for an Orin) and will not fit
  where the Pi does. Expect it outside the base with a short cable to the UART.
- **De-risk it for the price of a cable first.** A USB-to-TTL adapter on those
  same UART lines, driven from the PC with `connection:=serial
  serial_port:=/dev/ttyUSB0 serial_baud:=1000000`, tests the entire approach
  without relocating anything. If the protocol works over a raw UART from an
  external computer, the Jetson version is the same thing on different pins.

  ```bash
  ./scripts/probe_uart_bridge.py loopback   # does the link carry bytes intact?
  ./scripts/probe_uart_bridge.py listen     # is this wire really the arm's TX?
  ./scripts/probe_uart_bridge.py poke       # ask the arm, dump the raw reply
  ./scripts/probe_uart_bridge.py hunt       # poke on a loop while you probe pins
  ```

  **Bytes arriving is not the arm answering.** A floating receive wire running
  alongside a transmitting one couples enough to frame as bytes, and it only
  does so while you are sending — which is exactly when you are looking. The
  tell is length: `GET_ANGLES` is 5 bytes out and 17 back, so N copies return
  5N from an echo and 17N from the arm. `poke` runs that automatically.

  Measured here: 5 sent, 5 back, `ff ff fb bf fb`, byte-identical over repeated
  trials while a jumpered loopback on the same adapter was 5120/5120 perfect. A
  real short would echo perfectly; mangled-but-deterministic is capacitive
  coupling, which passes edges and not levels — the receiver takes the first
  falling spike as a start bit and then samples a line already back high, hence
  bytes that are nearly all 1s. Repeatable because the same data makes the same
  edges. `hunt` exists for the search that follows: it asks twice a second and
  names what it hears, so the receive wire can be walked down the header by
  hand instead of guessing a pin and re-running everything.

  **A loopback cannot detect a wrong baud rate, ever.** Both directions are the
  same chip driven from the same divisor register, so a wrong rate is wrong
  identically at each end and the bytes still return byte-perfect. Only
  `listen`, against traffic some *other* device generated, tests the rate. What
  loopback does catch is signal integrity — and on an Uno it is pessimistic
  about the real link, because the loopback path crosses the on-board 1k series
  resistor **twice** (16U2 TX → 1k → D0 → jumper → D1 → 1k → 16U2 RX). ~2k into
  dupont capacitance is about a third of a 1µs bit; driving the arm puts only
  one of those resistors in the path. So a few tenths of a percent of corrupt
  bytes in loopback at 1000000 is expected on an Uno and does not condemn the
  adapter. The script re-measures at lower rates and says whether the error rate
  scales (edge-rate limit) or not (bad connection).

  Either way, **do not command motion until the link is clean**: the myCobot
  protocol carries no checksum, so a flipped bit in a `SEND_ANGLES` payload is
  simply a joint angle the arm accepts and moves to.

  Answer both before `probe_usb_arm.py`, which commands nothing but does
  assume the link works. **An Arduino Uno R3 serves as the adapter**, with
  three traps that have nothing to do with the arm:

  - **A bridging sketch is not available.** The Uno has one hardware UART and
    it is already wired to the USB chip; SoftwareSerial tops out around
    115200, against the 1000000 this arm needs. Tie **RESET to GND** instead —
    that parks the ATmega328P and leaves the ATmega16U2 wired straight
    through, i.e. a plain USB-to-TTL adapter. It divides 16MHz to exactly
    1000000 in double-speed mode, better than some dedicated adapters.
  - **D0/D1 invert when you do that**, and this is the evening-waster. The
    labels describe the 328P, which is now switched off. `D0` ("RX") becomes
    an **output** — the USB chip's TX — and `D1` ("TX") an **input**. So wire
    label-to-same-label, *not* crossed, which is the opposite of the reflex.
  - **The Uno is 5V and the ESP32 is not 5V tolerant.** Listening on D1 is
    free; driving D0 into the arm's RX at 5V can destroy it. Fit **one**
    resistor, 2k from D0 to ground: the board already has 1k in series to the
    D0 pin, so that completes a 1k/2k divider at 3.33V. A second resistor in
    series gives 2.5V against the ESP32's 2.475V threshold — a 25mV margin,
    which is a line that reads as neither high nor low. Measure it (idle UART
    sits high, so D0 should read ~3.3V); 5V means the board resistor is absent.
    `listen` needs no divider at all and answers the pin-mapping question on
    its own, so do that one first.

  - **Three wires, not two.** TX and RX alone cannot work, and this cost a
    session. A receiver decides high-or-low by comparing the incoming voltage
    against *its own* ground, so with no shared reference the two boards float
    relative to each other and the arm reads constant idle, constant break, or
    garbage — never valid frames. It presents exactly as `-1` from
    `get_angles`, i.e. indistinguishable from wrong pins or a dead bus. A
    loopback will not catch it either: the loopback shares the adapter's ground
    with itself. Removing the Pi makes this newly load-bearing, since the Pi
    was the common reference for everything in the base.

  - **Identify the wires by their idle voltage, with the adapter unplugged.**
    UART idles high, so with the arm powered and nothing else connected, a
    meter to the arm's ground tells you which is which:

    | reading | what it is | connect |
    |---|---|---|
    | steady ~3.3V | the arm's **TX** — an output idling high | adapter's RX (D1) |
    | ~0V, or floating/drifting | the arm's **RX** — an input | adapter's TX (D0) |
    | ~5V | not a UART line. A power rail — wrong connector |
    | **3.6V** | a 5V driver already on it, clamped (see below) |

    **A pin labelled "TX" does not settle anything** — whose TX? Two opposite
    conventions are both in use: named for the *host's* signal, which is what
    the Pi pinout does (pin 8 is "UART TX" because the *Pi* transmits, so a
    replacement host wires straight through), or named for *that board's* own
    port, which means crossed. The arm's connector is documented the first way
    but a silkscreen elsewhere on the board may be the second.

    **A divider is a speed limit as well as a level shift**, and 1000000 baud
    is not forgiving. Source impedance drives the wire capacitance, and a bit
    is 1µs: 10k/10k is 5kΩ and takes ~600ns to settle — 60% of a bit, so the
    line never arrives before it is sampled. 1k/2k is 667Ω and 8%. Reach for
    values in the low kΩ; 10k resistors are the ones most likely to be in a
    parts drawer and they do not work here. 10k/10k also outputs 2.5V, which is
    barely over the ESP32's 2.475V threshold before any loading at all.

    And the divider belongs **only on the line the adapter drives**. On the
    receive line it attenuates the arm's reply, which is the signal you are
    trying to read.

    Settle it by which pin is **driven**, which is not a labelling question.
    Adapter off, arm powered, then hang a 10k from the pin to ground: an output
    idling high barely moves, a floating input collapses toward 0. The arm's TX
    is the output, and the adapter's RX goes there.

    That 3.6V case is worth knowing on sight: 5V through the Uno's on-board 1k
    into an ESP32 input pin is held by the pin's protection diode at 3.3 + Vf ≈
    3.6V, drawing ~1.4mA. So **3.6V means your D0 is on the arm's RX and the
    divider is not working** — the orientation is right and the level shift is
    missing. A correct 1k/2k reads 3.33V. It is also the ESP32's absolute
    maximum (VDD+0.3), i.e. surviving on the diode rather than by design.

  Note the Uno's 3.3V pin is irrelevant to any of this — it is a ~50mA
  regulator output for powering peripherals, not a logic-level selector. Both
  AVRs run at VCC = 5V and their pins swing 0–5V regardless. A CP2102/CH340
  breakout with a 3.3V jumper switches real VCCIO and costs about $3, which is
  the better tool for anything past one evening.

  Stopping `mycobot_server` is **not** enough before transmitting: GPIO14 stays
  in ALT0 actively driving pin 8, so two outputs fight. `sudo raspi-gpio set 14
  ip` makes it high-impedance (reverts on reboot). For `listen` you want
  `mycobot_server` *running* — its traffic is what you are trying to see.

**What it buys is mostly reliability, not speed.** `send_angles` is not in
`server.py`'s `has_return` table so commanding is already cheap, and the
dominant term in the round trip is the servos physically moving. What goes away
is a whole class of failure: the Pi unreachable, WiFi dropping mid-run,
port 9000 single-client lockouts, and factory services stealing the UART —
three separate sessions were lost to that layer, and none of it was latency.

**The original hope, for the record:** The 280 Pi
drives its servos through an M5Stack Atom (ESP32) that the Raspberry Pi
reaches over the GPIO UART at 1000000 baud. The Atom has its own USB-C port,
and if that is a data port into the same firmware then `connection:=serial`
puts this machine directly on the bus: no `server.py`, no TCP, no network in
any arm command. That is the leg of the ~250ms round trip nothing else here
has been able to reduce, and it is the topology comparable arms use.

```bash
./scripts/probe_usb_arm.py     # read-only, no motion commanded
```

**Stop `mycobot_server` on the Pi first** — it drives the same firmware over
the UART, and two masters on one bus behaves erratically. Untested against
hardware as of this writing; the port may be power-only.

## Gotchas

- **The home pose `[0, 90, -90, 0, 0, 0]` is defined in five places** and they
  must agree: the driver node default, `robot_bringup.launch.py`,
  `moveit_bringup.launch.py`, `driver.launch.py`, and
  `src/mycobot_moveit_config/config/mycobot_280pi.srdf` (in radians).
  Consolidating this is outstanding work.
- `colcon build --symlink-install` — without the symlink flag, edits to Python
  nodes do not take effect and `ros2 param get` keeps reporting old values.
- The arm keeps moving after the stack dies unless `mc.stop()` runs; the
  driver's `destroy_node` handles it. Do not add `respawn=True` to
  `mycobot_hardware_node` — combined with `home_on_start` it re-homes the arm
  after a deliberate shutdown.
- Port 9000 is single-client (`listen(1)`). Probe it by connecting and closing
  at once; anything that holds the slot locks out the driver.
- **The Pi ships with a Bluetooth bridge that holds the arm's UART, and it
  starts on every boot.** `/etc/rc.local` runs
  `/home/er/mycobot_pi_bluetooth/bt_auto_start.sh`, which runs
  `uart_peripheral_serial.py`, which opens `/dev/ttyAMA0` — the same port
  `server.py` uses to reach the arm's ESP32. Stopping `mycobot_server` does
  **not** release it; it was found holding the port after four hours with the
  unit inactive.
  ```bash
  ssh er@<pi> 'sudo pkill -f uart_peripheral_serial'          # now
  ssh er@<pi> "sudo sed -i 's|^\./bt_auto_start.sh|#&|' /etc/rc.local"   # and at boot
  ```
  Two readers on one tty split the incoming byte stream, so this can corrupt
  the arm's *replies* as well as its commands — `get_angles` returning
  nonsense makes the driver's measured pose wrong, which is what the
  divergence guard and `auto_sign` both reason from. Suspect it whenever the
  arm moves but the image does not respond as predicted.
- **`send_angles` stops on arrival, so the commanded point must lead.** The
  jog profiler commands `jog_lookahead` (0.12s) of its own velocity ahead of
  the profiled position; without that the arm reaches each commanded angle,
  stops, and waits out the rest of the `command_interval`, which is felt as
  motion arriving in pulses with a pause between each. The trajectory streamer
  has used the same trick at the same value since long before the jog path
  existed — the jog path simply never got it. The lookahead is clamped to the
  goal so it cannot overshoot, and self-cancels as velocity goes to zero.
- **The jog profile trades lock-on time for smoothness, and the exchange rate
  is bad.** `jog_profile` streams to the jog goal on a trapezoid instead of
  commanding it outright; `max_jog_accel_deg_s2` (1200) sets how gently.
  Simulated through the real servo maths from a hand at the frame edge: 1200
  acquires in 0.36s, 600 in 2.02s, 300 fails to converge more often than not.
  Steadiness once locked (3px) and moving-hand tracking are unchanged
  throughout — a gentle ramp costs only the getting there. 1200 is therefore
  nearly a no-op by design, reaching cruise in one command; what it still buys
  is a ceiling of 72°/s on velocity *change*, which is the step joint1 could
  not follow. Dropping the deceleration planning was tried, on the theory that
  stopping at each streamed goal was the cost — it is not, and it came out
  slightly worse.
- **`/arm/jog_applied` is diagnostics, not a feedback path.** The driver
  publishes what it really commanded, which is the fastest way to see a jog
  being clipped. Feeding it back into the servo's lag compensation was tried
  and is much worse: it reports motion as it is *commanded*, one arm-response
  earlier than the compensator needs, so already-landed jogs re-enter the
  window and get counted twice — residual 3px → 40-62px, and no convergence at
  all in ten runs of twelve. The servo records its own requests instead.
- **A joint that is commanded but does not move poisons everything above it.**
  Both the lag compensator and the velocity feedforward subtract jogs they
  assume executed. When one does not, the servo books the missing image motion
  as the *target* moving fast, leads harder, and a stuck axis becomes a
  runaway. The tells, in order of directness: the driver warning `Jog target
  pinned to the measured pose`, a commanded angle that stops advancing in the
  driver log (`-> [15.0, ...]` repeating), `vel=` pegged at 3.0/s in the servo
  log, and `aim=` further from zero than `seen=`. The servo now gives up the
  lead after 8 saturated readings and says so, but that is damage control —
  find out why the joint is not turning.

## Known-unfinished

- ~~`speed_at_100_deg_s` is an unmeasured guess~~ — **measured 2026-08-01 and
  now 52.0**, joint1 over `/dev/ttyTHS1` with a D405 on the flange: 51.6 deg/s
  at speed=100, 40.6 at 60, 29.0 at 30. It had been 120, so the driver asked
  for 43% of the speed it intended and the arm trailed its own commanded goal.
  Re-measure after any payload change:
  `./scripts/measure_arm.py --serial-port /dev/ttyTHS1`.
- No camera intrinsics — `scripts/calibrate_camera.py` unused, so
  `/hand/point_cam` stays silent and depth is unavailable.
- `obstacles.yaml` is empty; `src/mycobot_bringup/config/network.yaml` is read
  by nothing and disagrees with the defaults.
- Approach (closing in on the hand) is implemented but off by default
  (`approach_enabled:=false`) until tracking is solid. **While it is off, the
  toward/away axis genuinely never moves** — `_approach_sign` is pinned to 0,
  so joint3 gets no command at all. "Some motors are not contributing" is that,
  not a fault. Only joint1 (pan), joint5 (tilt) and joint3 (approach, when
  enabled) are ever driven; joints 2, 4 and 6 are untouched by design.
- **The hand position is already EMA-smoothed** in `hand_tracker_node`
  (`smoothing`, 0.6, applied in pixel space before publishing). Do not add a
  second filter in the servo: the published point carries the *frame's*
  timestamp while representing a blend of older frames, so the smoothing lag
  is invisible to the lag compensator, and doubling it adds dead time to the
  loop whose binding constraint is dead time. Tune the existing one instead.
- Planned hardware move: Jetson Orin Nano Super + RealSense D405, with the Pi
  on Ethernet rather than WiFi. Work for it lands on the **`Jetson` branch**.
  Groundwork already in place:
  - `camera_node` takes `source:=device` to open a local V4L2 or CSI camera
    (a non-numeric `device` is handed to OpenCV as a GStreamer pipeline, which
    is how `nvarguscamerasrc` gets in) instead of the Pi's MJPEG stream.
  - **`source:=realsense`** drives a RealSense through librealsense.

    ```bash
    ./run.py source:=realsense                    # hand detection only
    ./run.py source:=realsense rs_depth:=true     # + /camera/depth_raw
    ```

    Not the same as pointing `source:=device` at it. A RealSense does
    enumerate UVC nodes, so OpenCV can sometimes grab *something* — but which
    `/dev/videoN` carries colour is not stable across replugs, the formats are
    Y8/Y16/Z16 rather than the MJPG that path requests, and it discards depth
    and the factory intrinsics, which are the whole reason to own one.

    **This is what finally supplies camera intrinsics.** `CameraInfo` has
    always gone out with width and height and empty `k`/`p`/`d` — the reason
    `/hand/point_cam` is silent and depth unavailable. librealsense hands over
    the factory calibration with the stream, so `source:=realsense` fills in a
    real projection matrix without `calibrate_camera.py` ever being run.

    **Three D405-specific things:**

    - **Depth is only valid from ~7cm to 50cm.** That is exactly right for
      pressing buttons and wrong for following a hand across a room — but
      **hand detection is unaffected either way**, because MediaPipe works on
      the colour image and never looks at depth. A hand at 2m tracks as well
      as it ever did; only the range reading goes away.
    - **Its colour comes from the same stereo imagers as depth**, so there is
      no separate RGB module and the two are natively registered.
      `rs_align_depth_to_color` is therefore off by default; on a D435/D455 it
      would be mandatory.
    - **The lens is much wider, and that invalidates the servo gains.** Error
      is normalised per axis, so 1.0 means "at the edge" on any camera — but
      the edge is ~25° away on the current webcam and ~43° on a D405. The same
      normalised error therefore commands nearly twice the rotation, and the
      loop will over-command and ring. `camera_node` computes the real FOV from
      the intrinsics and warns when it is this much wider, naming
      `skip_probe:=false` as the fix. **Re-measure before trusting any tuning
      in this file.**

    `pyrealsense2` is left optional in `requirements.txt` because the mjpeg
    and device sources do not need it; the node reports it missing and keeps
    running. **PyPI does serve an aarch64 wheel** (checked at 2.58.3), so
    `pip install pyrealsense2` works on a Jetson with no source build. An
    earlier note here claimed the opposite and was wrong.

    ```bash
    python3 src/mycobot_camera/test/test_realsense_source.py
    ```
    Stubs librealsense, so it runs with no camera attached.
  - `hand_tracker_node` takes `delegate:=gpu`, switching from
    `mp.solutions.hands` — which is CPU-only and has no delegate option at all
    — to the Tasks API `HandLandmarker`. Needs a `hand_landmarker.task` bundle
    (not shipped in the wheel) **and** a mediapipe build with GPU support; the
    stock Linux wheels are CPU-only, so it falls back and says so. The
    backend actually in use is logged at startup — do not assume.
  - **Temper expectations on GPU inference.** At 30fps the budget is 33ms a
    frame and the CPU path measures 8-19ms, so there is no shortage of
    inference capacity to relieve. The Orin's value here is latency and
    topology, not FLOPS, and the biggest single win available is the Jetson
    driving the arm as well as watching it — that is the leg the servo's
    ~250ms round trip actually lives in.
  - **MediaPipe on ARM64 is not the integration risk this file used to claim.**
    Checked against PyPI: every version the pin allows (0.10.9 through 0.10.18)
    publishes a `cp310 manylinux_2_17_aarch64` wheel, so `pip install -r
    requirements.txt` completes on a Jetson with nothing built from source.
    NumPy, OpenCV and pyrealsense2 are the same. The earlier claim that Google
    ships no Jetson wheels was wrong, and it was steering toward a day of
    building from source that is not needed.

    What remains true is narrower: the stock wheels are **CPU-only**, so
    `delegate:=gpu` still needs a custom build. Given the CPU path measures
    8-19ms against a 33ms budget, that is worth little — TensorRT or Isaac ROS
    are the answer if inference ever does become the constraint.
