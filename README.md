# JNAReach — myCobot 280 on a Jetson Orin Nano

A ROS 2 Humble workspace that drives a myCobot 280 with an eye-in-hand camera.
The arm finds a hand, centres it in view, and closes in — that loop is also
what finds an elevator button, centres on it, and approaches to within a
couple inches using the D405's depth. See
[Elevator buttons](#elevator-buttons) below.

Everything runs on one board: the Jetson watches through a RealSense on USB
and drives the arm through its own UART pins, so **there is no network
anywhere in the control loop.**

```
Jetson Orin Nano (JetPack 6 / Ubuntu 22.04, aarch64)
┌────────────────────────────────────────────┐
│  camera_node        ◄── USB 3 ── RealSense D405
│  hand_tracker_node                          │
│  visual_servo_node                          │
│  mycobot_hardware_node                      │
│         └── /dev/ttyTHS1, 1000000 baud      │
└─────────┬───────────────────────────────────┘
          │  3 wires: pin 8 TX, pin 10 RX, pin 6 GND
          ▼
   M5Stack Atom (ESP32) ──► servo bus ──► 6 joints
```

---

## Run it

```bash
./run.py connection:=serial serial_port:=/dev/ttyTHS1 serial_baud:=1000000 source:=realsense
```

Long, because the launch defaults still point at the old Raspberry Pi setup.
Worth an alias:

```bash
echo "alias jnareach='~/mycobot_project/run.py connection:=serial serial_port:=/dev/ttyTHS1 serial_baud:=1000000 source:=realsense'" >> ~/.bashrc
```

That one script does preflight, a build check, the launch, and then a menu:

```
  1) Search for a hand      2) Home the arm       3) Stop servoing
  4) Resume servoing        5/6) Jogging ON/OFF
  7) Status                 8) Pipeline rates     9) Health check
  l) Live log               q) Quit (shuts the stack down)
```

The arm stays **idle at home until you press `1`.** Home is
`[0, 90, -150, 55, 0, 0]` degrees, commanded from wherever it happens to be —
the largest single move it makes, so check the path is clear.

Prefer `./run.py` over a bare `ros2 launch`: it catches the kinds of stale copy
that have each cost a debugging session, and it validates launch argument
names, which `ros2 launch` silently ignores when misspelled.

**Run it under tmux.** An SSH session that dies takes its children with it,
including a stack mid-motion:

```bash
tmux new -s arm
```

### The two numbers to read first

Press `8`. The servo loop acts once per detection, so **the detection rate is
your control rate.** Below about 6/s nothing tuned in the servo helps, and
`model_complexity:=0` or a smaller frame will move that number more than any
gain will.

---

## Check the link if the arm misbehaves

The protocol has **no checksum** — a flipped bit in a joint angle is not a
dropped message, it is a different angle that the arm accepts and drives to.
So when something is odd, test the link rather than the tuning.

```bash
./scripts/probe_uart_bridge.py poke --port /dev/ttyTHS1
```

Read-only, commands no motion.

| what you see | what it means |
|---|---|
| **frames decoded** | wiring and baud are both right |
| **an echo** (same byte count you sent) | hearing yourself — wires bridged, or the receive wire is floating and picking up crosstalk |
| **bytes but no frames** | something is transmitting: a rate or signal-quality problem, not orientation |
| **silence at every rate** | nothing is driving your receive line — swap the two signal wires first |

```bash
./scripts/serial_move_test.py --port /dev/ttyTHS1
```

Moves joint 1 by 15° and puts it back, but only after reading the angles 20
times with the arm still and confirming every reply agrees. **If they disagree
it refuses to move**, which is the right behaviour on a protocol with no
checksum.

Other modes: `loopback` (jumper pin 8 to pin 10, arm disconnected — proves the
Jetson's own UART), and `hunt` (asks twice a second while you move a wire down
the header).

---

## Tuning

Live, without relaunching — the servo takes runtime parameter changes:

```bash
./scripts/tune_servo.py
```

It scores the loop while it tracks: mean distance from centre, worst miss, and
**sign flips per second**. That last one is the point — a mean alone rewards a
loop that has given up, since a servo parked off-centre scores like one buzzing
evenly around centre.

| Argument | Default | Effect |
|---|---|---|
| `gain` | 0.45 | **fraction** of the full correction at the centre, not degrees |
| `progressive_gain` | 2.0 | raises gain with distance; keep `gain * (1 + progressive_gain * 0.29)` near 0.7 |
| `lead_time` | 0.15 | aims ahead of a moving hand; `0.0` is proportional-only |
| `command_lag` | 0.15 | seconds from sending a jog to seeing it — **keep below the true lag** |
| `max_step_deg` | 5.0 | biggest single jog; must not exceed the driver's `max_jog_deg` |
| `deadband` | 0.04 | image error below which it stops correcting |
| `rate` | 30.0 | keep at or above the camera's frame rate |
| `assumed_deg_per_error` | 25.0 | **wrong for a D405** — see below |
| `approach_enabled` | false | close in as well as centring |
| `show_window` | false | OpenCV window from the tracker |

**There is no PID, and adding one makes it worse.** The binding constraint is
dead time, not gain: two or three detections arrive still reporting the old
error while a correction is in flight, so a loop that re-commands that
correction overshoots by construction. An integral term winds up across exactly
that interval. Instead the node remembers every jog it sent and adds back the
image motion not yet visible.

**The D405's lens invalidates the gains above.** Error is normalised per axis,
so 1.0 means "at the frame edge" on any camera — but that edge is ~25° away on
the webcam these were fitted against and ~43° on a D405. The same normalised
error commands nearly twice the rotation, so expect ringing until you
re-measure:

```bash
./run.py connection:=serial serial_port:=/dev/ttyTHS1 source:=realsense skip_probe:=false
```

`camera_node` computes the real field of view from the camera's intrinsics and
warns when it looks like this.

Full reasoning and the traps are in
[CLAUDE.md](CLAUDE.md#visual-servoing--read-before-retuning). **Read it before
retuning** — most of what looks like an obvious improvement has been tried and
measured.

```bash
python3 src/mycobot_perception/test/test_servo_math.py
python3 src/mycobot_driver/test/test_jog_profile.py
python3 src/mycobot_camera/test/test_realsense_source.py
```

Run these after any change to the control law. A flipped sign does not crash —
it drives the target out of frame, which is indistinguishable from a badly
mounted camera.

---

## Elevator buttons

Same servo loop, a different target. `button_detector_node` runs a
YOLOv11n model trained on elevator buttons instead of MediaPipe, and
`detection_bridge_node` turns its detections into the same PointStamped
contract the hand tracker publishes — the servo does not know the
difference. The one thing that does change is the approach axis: instead of
using apparent target size as a range proxy, it reads real depth off the
D405 and closes in until the button is ~50mm away.

```bash
ros2 launch mycobot_bringup button_servo.launch.py
```

The arm homes, then sits idle until a floor is selected:

```bash
ros2 param set /detection_bridge_node target_label "button-3"
```

`target_label` has to match a class name from the trained model exactly
(case-insensitive). The currently trained `elevator_buttons.pt` was fitted
on a public dataset and knows: `alarm`, `button-1/2/3/up/down/g`,
`close`, `closed-door`, `down`, `floor-1/2/3/ground`, `key`, `open`, `up` —
list them yourself against whatever model is actually loaded with
`python3 -c "from ultralytics import YOLO; print(YOLO('elevator_buttons.pt').names)"`,
since a retrained model may use different names.

Any input method can set that parameter — a CLI, a limit-switch morse
decoder, eventually a microphone. The bridge node does not care which. Leave
`target_label` empty to have it centre on whichever button it is most
confident about, useful for testing detection without wiring up selection.

```bash
ros2 service call /servo/enable std_srvs/srv/SetBool "{data: false}"
```

stops it at any point, same as hand tracking.

| Argument | Default | Effect |
|---|---|---|
| `target_depth_mm` | 50.0 | stop this many mm from the button (~2 inches) |
| `depth_approach` | true | use D405 depth instead of apparent-size proxy |
| `model_path` | `elevator_buttons.pt` | trained YOLOv11n weights |
| `confidence_threshold` | 0.3 | lower than hand tracking — button glyphs are small |
| `lead_time` | 0.0 | buttons do not move; velocity prediction is pure noise here |
| `deadband` | 0.02 | tighter than hand tracking for precise centring |

### Training the detector

`button_detector_node` expects a model whose classes are the button labels
themselves (`"button-3"`, `"floor-2"`, `"open"`, …) — that is what makes
floor selection a label match rather than an image-space guess. The exact
label spelling is whatever the training dataset used; there is no
normalisation between e.g. `"3"` and `"floor-3"`.

```bash
python3 -m venv --system-site-packages .venv-train
.venv-train/bin/pip install roboflow ultralytics
.venv-train/bin/python3 scripts/train_button_detector.py \
    --workspace <roboflow-workspace> --project <roboflow-project> --version <n>
```

**Use a venv, not the system Python.** `ultralytics` pulls in a NumPy 2 /
OpenCV 5 `torch` dependency chain, and this project's `cv_bridge` needs
NumPy < 2 and OpenCV < 5 — installing training deps system-wide silently
breaks the ROS stack the next time anyone runs `pip install -r
requirements.txt`. `--system-site-packages` keeps the venv able to see the
already-installed ROS Python packages while isolating the conflicting ones.

**Expect CPU-only training on this board.** The generic PyPI `torch` wheel
does not see the Jetson's GPU — `torch.cuda.is_available()` is `False`, even
though the hardware has one. NVIDIA does publish a JetPack-matched wheel
(`developer.download.nvidia.com/compute/redist/jp/v61/pytorch/` for
JetPack 6.1), but it in turn wants cuDNN 9 while the stock JetPack image
ships cuDNN 8.9 — getting real GPU training working means chasing that too.
For a few hundred images, CPU training is workable: ~8 minutes/epoch on a
YOLOv11n at 640px on this board, so budget several hours for 50 epochs. Rerun
against an already-downloaded dataset with `--data-yaml
path/to/data.yaml` instead of `--workspace/--project/--version` to skip
re-fetching from Roboflow.

Once trained, `elevator_buttons.pt` is NOT checked into this repo (see
`.gitignore`) — copy it somewhere durable and point at it with
`model_path:=/path/to/elevator_buttons.pt`, or drop it in the workspace root
where the launch default expects it.

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
| `/camera/camera_info` | CameraInfo | real intrinsics under `source:=realsense` |
| `/camera/depth_raw` | Image | 16UC1 depth, only with `rs_depth:=true` |
| `/hand/point_px` | PointStamped | x,y = pixel; **z = palm width in pixels** |
| `/hand/annotated` | Image | landmarks drawn, for debugging |
| `/perception/button_detections` | Detection2DArray | YOLOv11n button detections, class label = floor/button |
| `/button/point_px` | PointStamped | x,y = pixel; **z = depth in mm** when `depth_approach:=true` |
| `target_label` (param on `/detection_bridge_node`) | string | floor to target; empty = highest-confidence any button |

**Only joint1, joint5 and joint3 are ever commanded** — pan, tilt, and approach
when enabled. Joints 2, 4 and 6 are untouched by design: two DOF centre a
target in an image and a third changes range. "Some motors are not
contributing" is that, not a fault.

---

## Troubleshooting

**Permission denied on `/dev/ttyTHS1`.** `dialout` membership, and you did not
log out after adding yourself. `id | grep dialout` settles it.

**The port exists but nothing works.** Something else holds it:
`sudo fuser -v /dev/ttyTHS1`.

**The arm answers sometimes.** Shorten the wires. Above about 2kΩ of series
resistance nothing carries 1000000 baud — a bit is 1µs and a 10k divider takes
600ns to settle.

**The camera is not found.** USB 3 port, USB 3 cable, udev rules. A D405 on USB
2 often still enumerates and then runs at a fraction of the rate, which caps
the whole servo loop; `camera_node` warns when it sees this.

```bash
python3 -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().devices])"
```

**Tracking is slow or trails.** Read the `tracker:` and `pipeline:` log lines
before touching a gain.

**It oscillates.** If you just switched to the D405, that is the lens — see
Tuning above.

**A joint is commanded but does not move.** Stop and find out why. Both the lag
compensator and the velocity feedforward subtract jogs they assume executed, so
a stuck axis becomes a runaway: the servo books the missing image motion as the
target moving fast and leads harder.

**The arm is stiff and will not move by hand.** Normal — the servos hold
position when powered. Do not force a joint against a powered servo; that is
how gearbox teeth strip. Power-cycle the arm to make it limp.

---

## Getting a shell somewhere else

Taking the board to a lab, or to a building with a real lift, and needing to
reach it once you are there — with no Ethernet and no monitor. Four routes,
and the first needs no network at all.

**1. USB-C works anywhere.** The Jetson presents itself as a USB network
adapter on a fixed address, so a laptop and the cable you already have is a
full shell regardless of what WiFi exists:

```bash
ssh <user>@192.168.55.1
```

From there, join whatever is local:

```bash
sudo nmcli device wifi connect "THEIR_SSID" --ask
```

**Rely on this one.** Every WiFi route below can be defeated by a network you
do not control — client isolation alone breaks laptop-to-board SSH on most
guest networks, and a captive portal or a venue with no WiFi defeats the rest.
The cable cannot be.

**2. Add the network before you go.** NetworkManager stores a profile for a
network it cannot currently see and joins the moment it is in range, so the
destination can be configured while you still have a working shell. The same
script that seeds a card works against the live filesystem:

```bash
sudo ./scripts/seed_jetson_wifi.sh / "THEIR_SSID"
```

```bash
sudo nmcli connection reload
```

It prompts for the passphrase with echo off, so nothing lands in shell history.

**3. A phone hotspot as a standing fallback.** Seeded the same way, it becomes
a network you carry rather than one you have to be granted:

```bash
sudo ./scripts/seed_jetson_wifi.sh / "YOUR_PHONE_HOTSPOT"
```

Turn the hotspot on, the board joins it, your laptop joins it, you have a
shell. This is the one that works in places you have no say over.

**4. The Jetson as its own access point.** No infrastructure needed at all,
from a shell you already have:

```bash
sudo nmcli device wifi hotspot ifname wlan0 ssid jetson password "<choose one>"
```

Join `jetson` from your laptop, then `ssh <user>@10.42.0.1`. No internet on the
board in this mode, so it is for control rather than for `apt`.

---

## If you rebuild the board

Condensed, because it is a one-off. The long-form version with every trap is in
the git history of this file.

1. **JetPack 6 / Ubuntu 22.04 / aarch64.** `lsb_release -d && uname -m`.
   Humble is built for 22.04 and there is no supported way onto anything older.
2. **Resize the rootfs.** `df -h /` should show ~57G of a 64GB card, not 22G.
   If the first-boot wizard was skipped, so was the resize:
   `sudo /usr/lib/nvidia/resizefs/nvresizefs.sh`. Otherwise apt runs out of
   space partway through installing ROS and blames its own cache.
3. **`sudo usermod -aG dialout $USER`**, then log out and back in. You do
   *not* need to disable `nvgetty` on an Orin — that advice is for the original
   Jetson Nano, and here it would disable the `ttyTCU0` console instead.
4. **ROS 2 Humble** from apt, exactly as on a desktop; the source line derives
   the architecture. `ros-humble-ros-base` plus `ros-humble-cv-bridge` is
   enough for the servo loop and saves ~4GB over `desktop`.
5. **`pip install -r requirements.txt`** plus `pip install pyrealsense2`.
   Everything has an aarch64 wheel — nothing needs building from source. The
   upper bounds are load-bearing: NumPy 2, MediaPipe 1.0 and OpenCV 5 each
   break `cv_bridge` in ways whose error messages point elsewhere.
6. **`colcon build --symlink-install`.** Without the symlink flag colcon
   *copies* Python files and your edits do nothing until you rebuild.

**Headless first boot**, if there is no monitor: seed the WiFi and the user
account onto the card before booting it, then it comes up on the network with a
working login.

```bash
sudo ./scripts/seed_jetson_wifi.sh /mnt/jetson "YOUR_SSID"
sudo ./scripts/seed_jetson_user.sh /mnt/jetson <username>
```

The second one exists because a fresh JetPack boots into `nv-oem-config.target`,
which declares `Conflicts=multi-user.target` — so `ssh.service` can never start
and the USB serial console has no getty behind it. The board pings, serves
DHCP, and refuses SSH forever. Both scripts explain themselves at the top.

**Wiring**, if it ever comes apart: Jetson pin 8 → arm pin 8, pin 10 → pin 10,
pin 6 → ground. **Not crossed** — the arm's connector is labelled with the
Pi's pinout, where "UART TX" means the *host* transmits, and the Jetson stands
in for the host. Ground is not optional; without a shared reference the two
boards float and the arm reads nothing.

---

## Known-unfinished

- `speed_at_100_deg_s: 120.0` in the driver is an unmeasured guess.
  `scripts/measure_arm.py` exists to measure it and has never been run.
- The home pose is defined in five places that must agree.
- `obstacles.yaml` is empty; `src/mycobot_bringup/config/network.yaml` is read
  by nothing and disagrees with the defaults.
- Approach is implemented but off by default until tracking is solid.
- **The 2×2 image Jacobian has never been measured** with `skip_probe:=false`,
  which matters more with the D405's wider lens than it did before.
- **The elevator button pipeline has not run against real hardware yet.**
  `button_detector_node`, `detection_bridge_node`, `depth_approach`, and
  `button_servo.launch.py` are new and untested end-to-end. `elevator_buttons.pt`
  is trained (mAP50 0.929, mAP50-95 0.546 on the held-out validation set, 50
  epochs on the public `entc-elevator-button-detection` dataset) but will need
  retuning against the real panel it targets — different lighting and button
  styling than the training images almost always costs accuracy on a small
  (~440 image) dataset. `approach_gain` and `target_depth_mm` are unverified
  against a real D405 depth reading close to the sensor's near limit (~70mm).
- **GPU training on this Jetson is unresolved.** The JetPack-matched PyTorch
  wheel needs cuDNN 9; the board ships cuDNN 8.9. Training currently runs on
  CPU — 50 epochs on ~440 images took 6.75 hours. See
  [Training the detector](#training-the-detector).

The Raspberry-Pi-over-network setup this replaced still works and is what the
launch files default to; it is archived in
[legacy/README_pi_network.md](legacy/README_pi_network.md).

Development notes, measured results, and the reasoning behind the control law
are in [CLAUDE.md](CLAUDE.md). Most of the traps in it were found the
expensive way.
