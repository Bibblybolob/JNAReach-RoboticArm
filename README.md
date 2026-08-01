# JNAReach — myCobot 280 on a Jetson Orin Nano

A ROS 2 Humble workspace that drives a myCobot 280 arm with an eye-in-hand
camera. The arm finds a hand, centres it in view, and closes in — no camera
calibration required. Working toward pressing elevator buttons autonomously.

Everything runs on one board. The Jetson watches through a RealSense on USB
and drives the arm through the Jetson's own UART pins, so there is no network
anywhere in the control loop.

> **The Raspberry-Pi-over-WiFi setup this replaced is archived in
> [legacy/README_pi_network.md](legacy/README_pi_network.md).** The code still
> supports it in full and it is still what the launch files default to — see
> [If the UART gives trouble](#if-the-uart-gives-trouble).

This guide assumes a **freshly flashed Jetson Orin Nano and nothing else.**

---

## What talks to what

```
Jetson Orin Nano (JetPack 6 / Ubuntu 22.04, aarch64)
┌────────────────────────────────────────────┐
│  camera_node        ◄── USB 3 ── RealSense D405
│  hand_tracker_node       (colour + depth + factory intrinsics)
│  visual_servo_node                          │
│  mycobot_hardware_node                      │
│         │                                   │
│         └── /dev/ttyTHS1, 1000000 baud      │
└─────────┬───────────────────────────────────┘
          │  3 wires: pin 8 TX, pin 10 RX, pin 6 GND
          ▼
   M5Stack Atom (ESP32) ──► servo bus ──► 6 joints
```

Two things follow from this shape and are worth holding onto:

**The Jetson is the master on the arm's serial bus.** Nothing else may drive
those lines at the same time — not a Raspberry Pi still wired in, not a getty,
not a second probe script. Two masters on one UART behaves erratically rather
than failing cleanly.

**The camera is the control rate.** The servo loop acts once per detection, so
whatever rate MediaPipe achieves is the rate the arm is told anything. Below
about 6 detections/second, nothing tuned in the servo helps.

---

## Before you start

| | |
|---|---|
| **Jetson Orin Nano** (Super) | with its own 19V supply. JetPack 6 or newer |
| **myCobot 280** | arm, Atom/ESP32 and servos, on its own power supply |
| **3 jumper wires** | female-to-female, for TX / RX / GND |
| **RealSense D405** | and a **USB 3** cable. USB 2 silently halves the frame rate |
| **microSD or NVMe** | flashed with JetPack 6 |

> ### One honest warning
>
> **The UART link has not yet been verified end to end on this hardware.** The
> software for it is written and tested, the pin mapping is confirmed against
> both vendors' documentation, and the wiring is three wires — but at the time
> of writing no arm has answered over it. [Step 4](#4-prove-the-link-before-you-command-anything)
> exists to tell you whether yours does, and to tell you *how* it is failing
> if it does not. Do not skip it.

---

## 1. Confirm JetPack and Ubuntu

```bash
lsb_release -d && uname -m && cat /etc/nv_tegra_release
```

You want **Ubuntu 22.04** and **aarch64**. That is JetPack 6, and it matters
more than anything else in this document: ROS 2 Humble is built for 22.04, and
there is no supported way to get it onto the 20.04 of JetPack 5 or the 18.04 of
the original Jetson Nano. If you see either, reflash before going further —
every later step assumes 22.04.

```bash
sudo apt update && sudo apt full-upgrade -y
```

## 2. Free the UART

The 40-pin header's UART is claimed by a serial console on a fresh JetPack, and
it will fight you for the port. This is the same class of problem as the
Bluetooth bridge that used to hold the Pi's UART: the device exists, opens
cleanly, and does not work.

```bash
sudo systemctl disable --now nvgetty
```

```bash
sudo usermod -aG dialout $USER
```

**Log out and back in** — group membership only applies to new sessions. Then:

```bash
ls -l /dev/ttyTHS*
```

`/dev/ttyTHS1` should be listed and your user should be able to open it. If
`ls` shows it but a script says permission denied, the logout did not happen.

## 3. Wire the arm

Power **off** both the Jetson and the arm before wiring.

Three wires, straight through — Jetson pin *N* to arm pin *N*:

| Jetson 40-pin | | arm's 40-pin connector |
|---|---|---|
| **pin 8** — UART1 TX | → | **pin 8** |
| **pin 10** — UART1 RX | ← | **pin 10** |
| **pin 6** — GND | ↔ | **pin 6** (or any ground) |

**Do not cross TX and RX.** The usual advice is to cross them, and it is wrong
here. The arm's connector is documented with the Raspberry Pi's pinout, where
"UART TX" names the *Pi* transmitting — so the labels describe the host, not
the arm. The Jetson is standing in for the Pi, so its transmit goes exactly
where the Pi's transmit went.

**Ground is not optional.** A receiver decides high-or-low against its own
ground; with no shared reference the two boards float and the arm reads
garbage or nothing. Two wires cannot work, and the failure looks identical to
wrong pins.

Both boards are 3.3V, so no level shifting is needed. Counting down the even
row, the three pins you want are the **3rd, 4th and 5th** positions — and the
1st and 2nd are 5V, so a two-position miscount puts a signal wire on a power
rail.

## 4. Prove the link before you command anything

The protocol has **no checksum**. A flipped bit in a joint angle is not a
dropped message — it is a different angle, which the arm accepts and drives
to. So the link gets tested before it gets trusted.

**First, the Jetson alone.** Disconnect the arm and jumper pin 8 to pin 10:

```bash
./scripts/probe_uart_bridge.py loopback --port /dev/ttyTHS1
```

Byte-exact is a pass. This catches a UART that is not driving at all — there is
a known JetPack 7 / L4T R39.2 bug where the ttyTHS1 TX pad does not drive on
the Orin Nano Super, and it presents as transmitting into silence with
everything apparently correct.

**Then wire the arm and ask it something.** Read-only, commands no motion:

```bash
./scripts/probe_uart_bridge.py poke --port /dev/ttyTHS1
```

| what you see | what it means |
|---|---|
| **frames decoded** | wiring and baud are both right — go to step 5 |
| **an echo** (same byte count you sent) | you are hearing yourself. Wires bridged, or the receive wire is floating and picking up crosstalk |
| **bytes but no frames** | something is transmitting — a rate or signal-quality problem, not orientation |
| **silence at every rate** | nothing is driving your receive line. Swap the two signal wires first |

If you need to hunt for the right pin, this asks twice a second while you move
a wire down the header:

```bash
./scripts/probe_uart_bridge.py hunt --port /dev/ttyTHS1
```

## 5. Install ROS 2 Humble

Identical to the desktop instructions — the apt source line derives the
architecture, so it serves arm64 packages without modification.

```bash
sudo apt install -y software-properties-common curl && sudo add-apt-repository -y universe
```

```bash
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
```

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

```bash
sudo apt update && sudo apt install -y ros-humble-desktop ros-humble-moveit python3-colcon-common-extensions python3-rosdep
```

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && source ~/.bashrc
```

> On a Jetson, `ros-humble-desktop` pulls in RViz and a lot of graphics stack.
> If the board is headless and you never plan to plan with MoveIt on it,
> `ros-humble-ros-base` plus `ros-humble-cv-bridge` is enough for the servo
> loop and saves a few GB.

## 6. Get the code and build

```bash
git clone https://github.com/Bibblybolob/JNAReach-RoboticArm.git ~/mycobot_project && cd ~/mycobot_project
```

```bash
pip install -r requirements.txt
```

**Every dependency has an aarch64 wheel** — MediaPipe, NumPy, OpenCV and
pyrealsense2 all install from PyPI on the Jetson with no source build. That
was not always true and is worth knowing, because building MediaPipe or
librealsense from source on ARM is most of a day.

**The upper bounds in `requirements.txt` are load-bearing.** NumPy 2,
MediaPipe 1.0 and OpenCV 5 each break ROS Humble's compiled `cv_bridge`, in
ways whose error messages point somewhere else entirely. If a node dies on
import after you upgrade something, check those three first.

```bash
colcon build --symlink-install && source install/setup.bash
```

**Use `--symlink-install`.** Without it colcon *copies* Python files, so edits
have no effect until you rebuild — and you will lose an hour to a change that
"didn't apply" when it was simply never installed.

```bash
echo "source ~/mycobot_project/install/setup.bash" >> ~/.bashrc
```

## 7. The camera

Plug the D405 into a **USB 3** port and check librealsense sees it:

```bash
python3 -c "import pyrealsense2 as rs; ctx=rs.context(); print([d.get_info(rs.camera_info.name) for d in ctx.devices])"
```

An empty list with the camera plugged in is almost always udev permissions:

```bash
sudo apt install -y librealsense2-udev-rules 2>/dev/null || echo "see librealsense/scripts/setup_udev_rules.sh"
```

Three things about the D405 specifically:

- **Depth is valid from about 7cm to 50cm.** Right for pressing buttons, wrong
  for following a hand across a room — but **hand detection does not care**,
  because MediaPipe works on the colour image and never looks at depth. A hand
  at 2m tracks exactly as well; only the range reading goes away.
- **Its colour comes from the same stereo imagers as depth**, so the two are
  natively registered. `rs_align_depth_to_color` stays off; on a D435/D455 it
  would be mandatory.
- **The lens is much wider than the webcam the servo was tuned against** — the
  frame edge is roughly 43° out instead of 25°. The same normalised error
  therefore commands nearly twice the rotation, so **expect to retune.**
  `camera_node` measures the real field of view from the intrinsics and warns
  when it looks like this.

This is also what finally supplies **camera intrinsics**. `CameraInfo` used to
go out with width and height and nothing else, which is why `/hand/point_cam`
was always silent. librealsense hands the factory calibration over with the
stream, so no chequerboard is needed.

## 8. First motion

One joint, small, and verified — with the link measured before anything moves:

```bash
./scripts/serial_move_test.py --port /dev/ttyTHS1
```

It reads the arm's angles 20 times with the arm still, checks every reply
agrees, and only then moves joint 1 by 15° and puts it back. **If the replies
disagree it refuses to move**, because corruption that shows up as a bad
reading would show up in a command as an angle the arm drives to.

Keep clear of the arm. It is stiff when powered — that is normal, the servos
hold position — and forcing a joint by hand against a powered servo is how
gearbox teeth strip.

---

## Running it

```bash
./run.py connection:=serial serial_port:=/dev/ttyTHS1 serial_baud:=1000000 source:=realsense
```

That is the whole thing: preflight, build check, launch, and a menu.

```
  1) Search for a hand      2) Home the arm       3) Stop servoing
  4) Resume servoing        5/6) Jogging ON/OFF
  7) Status                 8) Pipeline rates     9) Health check
  l) Live log               q) Quit (shuts the stack down)
```

The arm stays **idle at home** until you press `1`. Home is
`[0, 90, -90, 0, 0, 0]` degrees, commanded from wherever the arm happens to
be — the largest single move it makes, so check the path is clear.

> **That command line is long because the launch defaults still point at the
> Raspberry Pi.** They have deliberately not been changed: until the UART link
> is proven on your hardware, the network path is the fallback that works, and
> silently removing it would leave you with nothing. Once step 4 passes,
> flipping the defaults in `servo_demo.launch.py` is a two-line change. In the
> meantime, an alias earns its keep:
>
> ```bash
> echo "alias jnareach='~/mycobot_project/run.py connection:=serial serial_port:=/dev/ttyTHS1 serial_baud:=1000000 source:=realsense'" >> ~/.bashrc
> ```

Prefer `./run.py` over a bare `ros2 launch`: it catches the kinds of stale copy
that have each cost a debugging session, and it validates launch argument
names, which `ros2 launch` silently ignores when misspelled.

`8` samples the camera and detection rates. Since the servo acts once per
detection, **the detection rate is your control rate** — read it before
touching any gain.

### Tuning

Live, without relaunching:

```bash
./scripts/tune_servo.py
```

It scores the loop while it tracks: mean distance from centre, worst miss, and
**sign flips per second**. The last is the point — a mean alone rewards a loop
that has given up, since a servo parked off-centre scores like one buzzing
evenly around centre.

The values that matter, as they actually are in the code:

| Argument | Default | Effect |
|---|---|---|
| `gain` | 0.45 | **fraction** of the full correction at the centre, not degrees |
| `progressive_gain` | 2.0 | raises gain with distance; keep `gain * (1 + progressive_gain * 0.29)` near 0.7 |
| `lead_time` | 0.15 | aims ahead of a moving hand; `0.0` is proportional-only |
| `command_lag` | 0.15 | seconds from sending a jog to seeing it — **keep below the true lag** |
| `max_step_deg` | 5.0 | biggest single jog; must not exceed the driver's `max_jog_deg` |
| `deadband` | 0.04 | image error below which it stops correcting |
| `rate` | 30.0 | keep at or above the camera's frame rate |
| `assumed_deg_per_error` | 25.0 | **wrong for a D405** — see step 7, and measure with `skip_probe:=false` |
| `approach_enabled` | false | close in as well as centring |
| `show_window` | false | OpenCV window from the tracker |

**There is no PID, and adding one makes it worse.** The binding constraint is
dead time, not gain: two or three detections arrive still reporting the old
error while a correction is in flight, so a loop that re-commands that
correction overshoots by construction. An integral term winds up across
exactly that interval. Instead the node remembers every jog it sent and adds
back the image motion not yet visible.

Full reasoning, and the traps, are in
[CLAUDE.md](CLAUDE.md#visual-servoing--read-before-retuning) and the module
docstring of `visual_servo_node.py`. **Read them before retuning** — most of
what looks like an obvious improvement has already been tried and measured.

```bash
python3 src/mycobot_perception/test/test_servo_math.py
python3 src/mycobot_driver/test/test_jog_profile.py
python3 src/mycobot_camera/test/test_realsense_source.py
```

Run these after any change to the control law. A flipped sign does not crash —
it drives the target out of frame, which is indistinguishable from a badly
mounted camera.

---

## Topics and services

| Name | Type | Purpose |
|---|---|---|
| `/joint_states` | JointState | 6 arm joints |
| `/arm/home` | SetBool | move to the fixed home pose |
| `/arm/jog` | JointJog | relative joint moves, in **degrees** |
| `/arm/jog_enable` | SetBool | gate for jogging |
| `/arm/jog_applied` | JointJog | what the driver really commanded — **diagnostics only** |
| `/servo/search` | Trigger | start hunting for a hand |
| `/servo/enable` | SetBool | master stop for servoing |
| `/camera/image_raw` | Image | colour feed |
| `/camera/camera_info` | CameraInfo | now carries real intrinsics under `source:=realsense` |
| `/camera/depth_raw` | Image | 16UC1 depth, only with `rs_depth:=true` |
| `/hand/point_px` | PointStamped | x,y = pixel; **z = palm width in pixels** |
| `/hand/annotated` | Image | landmarks drawn, for debugging |

**Only joint1, joint5 and joint3 are ever commanded** — pan, tilt, and
approach when enabled. Joints 2, 4 and 6 are untouched by design: two DOF
centre a target in an image and a third changes range. "Some motors are not
contributing" is that, not a fault.

---

## Troubleshooting

**The arm does not answer, `-1` from everything.** Work through
[step 4](#4-prove-the-link-before-you-command-anything) rather than guessing —
`-1` means "no valid angles" and covers nothing arriving, bytes at the wrong
rate, and a frame with an unexpected command id. Those want opposite fixes.

**It answers sometimes.** Shorten the wires. Anything above about 2kΩ of
series resistance will not carry 1000000 baud — a bit is 1µs and a 10k divider
takes 600ns to settle.

**Permission denied on `/dev/ttyTHS1`.** `dialout` membership, and you did not
log out. `id | grep dialout` settles it.

**The port exists but nothing works.** `nvgetty` came back, or something else
holds it: `sudo fuser -v /dev/ttyTHS1`.

**The camera is not found.** USB 3 port, USB 3 cable, and udev rules. A D405
on USB 2 will often still enumerate and then run at a fraction of the rate,
which caps the whole servo loop — `camera_node` warns when it sees this.

**Tracking is slow or trails.** Read the `tracker:` and `pipeline:` log lines
before touching a gain. Below ~6 detections/s nothing in the servo helps;
`model_complexity:=0` and a smaller frame move that number more than anything
else.

**It oscillates after switching to the D405.** Expected — the lens is nearly
twice as wide as the one the gains were fitted to. Measure the real response
with `skip_probe:=false`.

**A joint is commanded but does not move.** Stop and find out why. Both the
lag compensator and the velocity feedforward subtract jogs they assume
executed, so a stuck axis becomes a runaway: the servo books the missing image
motion as the target moving fast and leads harder.

### If the UART gives trouble

The Raspberry Pi path is unchanged and still works. It is what the launch
files default to, so it is one command away:

```bash
MYCOBOT_IP=192.168.0.15 ./run.py
```

Setup for it is in [legacy/README_pi_network.md](legacy/README_pi_network.md).

---

## Known-unfinished

- **The UART link is unproven on real hardware.** Everything else here follows
  from it working.
- `speed_at_100_deg_s: 120.0` in the driver is an unmeasured guess.
  `scripts/measure_arm.py` exists to measure it and has never been run.
- The home pose is defined in five places that must agree; consolidating it is
  outstanding.
- `obstacles.yaml` is empty, and `src/mycobot_bringup/config/network.yaml` is
  read by nothing and disagrees with the defaults.
- Approach is implemented but off by default until tracking is solid.
- The 2x2 image Jacobian has never been measured with `skip_probe:=false`,
  which matters more with the D405's wider lens than it did before.

`librealsense/` is gitignored. It was cloned to build `pyrealsense2` from
source before it turned out PyPI serves an aarch64 wheel — if you have that
827MB directory and did not build anything from it, you can delete it.

Development notes, measured results, and the reasoning behind the control law
are in [CLAUDE.md](CLAUDE.md). It is written for whoever touches this next,
and most of the traps in it were found the expensive way.
