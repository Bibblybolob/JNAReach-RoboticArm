# JNAReach-RoboticArm

ROS 2 Humble workspace driving a **myCobot 280 Pi** over the network, with an
eye-in-hand camera doing visual servoing. The long-term goal is autonomously
**pressing elevator buttons**; hand-following is the stepping stone that
proves the perception-to-motion loop works.

## Layout

Three machines, and confusing them is the most common source of wasted time.

| | Where | Holds |
|---|---|---|
| Dev / runtime | Ubuntu 22.04 VM, `~/JNAReach-RoboticArm` | the ROS workspace, everything in `src/` |
| Robot | Raspberry Pi in the arm base, `~/JON/mycobot_project`, default `192.168.0.15` | `pi/server.py` (arm TCP, port 9000) and `pi/camera_stream.py` (MJPEG, port 8080) |
| Arm | myCobot 280 Pi | driven by pymycobot from the Pi |

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
pins 8 and 10 too**, so a board-for-board substitution is pin-for-pin.

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
  ./scripts/probe_uart_bridge.py loopback   # does the adapter do 1000000 baud?
  ./scripts/probe_uart_bridge.py listen     # is this wire really the arm's TX?
  ```

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

- `speed_at_100_deg_s: 120.0` in the driver is an unmeasured guess.
  `scripts/measure_arm.py` exists to measure it and has never been run.
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
  - MediaPipe on ARM64/Jetson is an integration risk worth checking early:
    Google publishes no Jetson wheels, so it is community builds or building
    from source. TensorRT or Isaac ROS are the native alternatives.
