# JNAReach — myCobot 280 on a Jetson Orin Nano

A ROS 2 Humble workspace that drives a myCobot 280 with an eye-in-hand camera.
The arm finds a hand, centres it in view, and closes in. Working toward
pressing elevator buttons autonomously.

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

## Move the arm from a terminal

These go through `scripts/arm_broker.py`, not ROS, so they work with the stack
down. **The broker must be running** — it owns `/dev/ttyTHS1`, and two writers
on one tty produce a length field that does not match its payload, which the
Atom reports as a firmware crash:

```bash
python3 scripts/arm_broker.py
```

```bash
./scripts/jog_joints.py
```

One joint at a time: `2 +5` moves joint2 by 5°, `2 =90` drives it to 90°, `p`
prints the pose, `r` releases the servos — **the arm will sag, hold it.**

```bash
./scripts/park_arm.py
```

Parks at the least-loaded pose. This is a real diagnostic step rather than
tidying up: unloading the arm alone took valid replies from 18% to 96% on
2026-08-12, with no reflash and no power cycle.

---

## Press a button

```bash
./scripts/press_button.py --look          # detect only, no motion
./scripts/press_button.py up --dry-run    # plan and print, command nothing
./scripts/press_button.py 5 --tool-mm 27.7
./scripts/press_button.py B1 --tool-mm 27.7
```

The whole chain in one command:

```
detector -> depth -> target_in_base -> plan_press -> arm_motion -> send_angles
```

**The button is named, not numbered.** `up`, `down`, `help`, `open`, `close`,
or a floor legend — `5`, `B1`, `G`, `LG`. The hall call is the whole point:
`up` and `down` are the project's top priority and were not expressible at all
while this argument took an integer.

### Which detector

`--detector auto` (the default) runs the **trained two-stage model** and falls
back to the geometric finder. They solve different panels and neither
supersedes the other:

| | good for | fails on |
|---|---|---|
| **model** (stage A + reader) | real lifts, any layout, reads legends | the lab's printed panel |
| **geometric** (`keypad_finder`) | the lab's printed 12-button panel | anything else — the lattice is hard-coded |

Force one with `--detector model` or `--detector geometric`.

Falling back is safe in the direction that matters: `keypad_finder` refuses a
frame it cannot fit rather than mislabelling one, so the worst case is a retry.

**A button whose legend was not read confidently is shown as `floor?` and is
deliberately unaskable.** It is a real button, found and located, and it still
counts toward the panel plane fit — but `5` will not match it, so the arm
cannot press it believing it is 5. Raise or lower that line with `--read-min`
(0.95; at 0.75 the reader was confidently wrong on 11.7% of legends).

Keyholes and emergency stops are detected and then **withheld** — they never
become press targets at any confidence.

Every stage **refuses rather than guessing**, because each can produce a
confident wrong answer that ends with the arm driving somewhere real. The
finder rejects a frame whose row count does not match the panel layout, since
a missed row shifts every number and presses the wrong floor; `target_in_base`
rejects a depth outside the D405's usable band; `plan_press` rejects a
standoff/touch pair that changes arm configuration, which would swing the arm
through the panel on the way. A refusal costs a retry, never a wrong button.

**Why the geometric finder still exists.** The *old* `elevator_buttons.pt`
could not see the lab's printed panel at all — measured 2026-08-14, zero
detections at 0.5 and nothing above 0.12 even at 0.05, and no class above
`button-3` against a 12-floor keypad. YOLO-World zero-shot found nothing
either. `keypad_finder.py` uses geometry instead — equal-sized ellipses on a
lattice — and gets all twelve at **10/12 frames, ~400ms, 0.26px jitter**.

The two-stage model replaces it for *real* panels (`floor` recall 0.952,
`up`/`down` 0.674/0.744 on 202 held-out real lifts, 51.5ms on the GPU), but it
has not been shown to read the inkjet print, so the fallback stays.

**Resolution is load-bearing.** 1280×720 is the default because at 640×480 the
buttons are r≈12px and rows drop out: **10/12 frames against 2/6.**

### Two limits, both printed rather than hidden

- **`--tool-mm` defaults to 0**, so the *flange origin* is driven onto the
  button and anything protruding past it contacts off by that much. The fitted
  presser is **27.7mm** — pass it every time; see step 1 below.
- **The tool cannot generally be held square to the panel.** Exact aim costs
  ~25mm of position at most placements, with no joint at a limit — it is
  dexterity, not limits. The planner aims as squarely as the arm allows while
  still hitting the button, and prints what it got (~21° is typical, against
  70° when orientation was left unconstrained).

**Where a square press is actually possible**, position error while holding
the tool exactly on the panel normal:

| button height | 120mm | 150mm | 180mm | 210mm | 240mm |
|---|---|---|---|---|---|
| 150mm | **0.0** | **0.0** | 24.9 | 53.1 | 72.4 |
| 200mm | **0.0** | 21.4 | 46.7 | 72.8 | 99.5 |

So put the panel **120–150mm away with the buttons 150–200mm up**. Further out
the script will plan a standoff the arm cannot reach and refuse — which is
what it should do.

Needs a verified calibration at `~/hand_eye/eye_to_hand.json`; see
`calibrate_hand_eye.py` and `verify_calibration.py`.

---

## Making the arm precise and smooth — do these in order

This is the hardware procedure. It is ordered deliberately: each step needs
the one before it, and step 3 decides which of the later ones are worth doing
at all. Nothing here can be done from a desk.

**Precision and smoothness are different faults with different fixes.** A
press that lands 5mm off is precision. A press that lunges and stops in jerks
is smoothness. Do not tune one expecting the other to improve.

### Where the error actually is

Measured on this arm, not estimated:

| source | contribution at ~300mm reach |
|---|---|
| **backlash, 0.79°** | **~4.1mm** — the dominant term |
| hand-eye calibration | 3.2mm (5.0mm via depth) |
| depth noise | ~1–2mm |
| path bow between waypoints | 0.31mm (was 2.88mm) |
| IK | 0.03mm |

Root-sum-square is about **5–6mm**. Against a 20mm button that works; against
a 10mm button it is marginal. **Do not spend time on the IK** — at 0.03mm it
is four orders of magnitude below the term that matters.

### 1. Measure the tool — currently **27.7mm**

The fitted presser is a cone on a mounting plate, **27.7mm** from the flange
face, and that is the number to pass until the tool is refitted.

```bash
./scripts/measure_tool.py        # confirms it AS MOUNTED
```

`--tool-mm` defaults to **0**, which drives the *flange origin* onto the
button, so anything protruding contacts short by its own length. Every command
below needs this number.

**A CAD length and a mounted length are not the same measurement.** 27.7mm is
the designed protrusion; what the arm actually has depends on how deep the
tool seats, whether the plate stands proud, and whether it sits on the tool
axis. Run the script once to confirm — it also reports the OFF-AXIS spread,
which catches a presser mounted slightly off-centre. That fault otherwise
appears much later as a constant sideways miss that reads like a calibration
error.

The script **refuses** rather than printing a number when the poses disagree
by more than 8mm — a bare arm gives 146mm of spread, because with nothing
slender fitted the answer is whatever bit of flange body falls near the axis.
A refusal means no tool is fitted, not that the script failed.

### 2. Park the arm where it can actually aim

Measured for a 27.7mm tool, as `aim off normal / body clearance`:

| base→panel | button 120mm up | 160mm up | 200mm up |
|---|---|---|---|
| **120mm** | refused | 18.5° / 78mm | 17.3° / 100mm |
| **150mm** | refused | 18.6° / 77mm | 20.0° / 102mm |
| **180mm** | 19.2° / 48mm | 20.5° / 79mm | 24.5° / 128mm |
| **210mm** | 21.5° / 53mm | 24.5° / 83mm | 37.8° / 128mm |

**Park with the button 120–180mm from the base and 160–200mm up.** Low buttons
at close range do not plan at all, and the far corner degrades to 37.8°. On a
wheelchair this is a *mounting-height specification*, not a preference: if the
chair cannot reliably reach that band, fix the mount before tuning anything.

### 3. Dry run, then one real press

```bash
./scripts/press_button.py 5 --tool-mm <measured> --dry-run
./scripts/press_button.py 5 --tool-mm <measured> --speed 20
```

Three printed lines decide whether the rest is meaningful:

- **`panel normal from N buttons:`** — if absent, it fell back to a
  radial-from-base approach, which is ~30° off the truth. Everything
  downstream inherits that error.
- **`aim: X deg off the panel normal`** — if absent, the full-pose solve
  failed and the tool arrives pointing an arbitrary way.
- **`contact: pressed / NONE / BLOCKED`** — `NONE` means the tool reached the
  touch pose without resistance, so the button was not where depth said it
  was. `BLOCKED` means something that is not a button stopped the arm.

Keep a hand near the stop. The closed-loop waiting, straight-line approach,
lookahead, blending and contact detection are verified in simulation and by
~100 tests, but this is the first time they move a real arm.

### 4. The repeatability test — the one that decides everything after

Press the **same button 10 times from the same parking spot** and record where
the tip lands.

- **Tight cluster in the wrong place** → systematic error (calibration, tool
  length). It is *subtractable*: go to step 7.
- **Scattered cluster** → random error (backlash, depth noise). Not
  subtractable: go to step 6.

Do not skip this. The two outcomes have opposite remedies, and without it any
further tuning is guesswork.

### 5. Fit a domed, compliant tip

A flat face at ~20° off normal contacts on one edge and can skid. A
hemispherical tip contacts identically at any angle — the only effect is a
lateral shift of `r·sin(20°)`, which for a 2mm tip radius is **0.7mm**.

**Compliance matters as much as the dome.** A rigid tip drives any depth
over-estimate straight into the panel as load, and load is what makes this
arm's Atom reboot. A spring-loaded shaft or a soft cap turns a force spike
into travel, and it sharpens the stall signal the contact detection reads.

### 6. Correct visually at the standoff *(if step 4 said scattered)*

At the 40mm standoff, re-detect the button **and** the tool tip, measure the
residual offset in the image, and null it before the final leg.

This is the largest single precision win available. It replaces a *predicted*
3.2mm calibration with a *measured* one, and it cancels mount drift from
wheelchair vibration and parking variation at the same time — none of those
survive a measurement taken 40mm from contact.

### 7. Subtract the measured offset *(if step 4 said tight-but-wrong)*

Take the mean miss over the 10 presses and apply it as a constant correction.
Only works once the error is repeatable, which is what step 5 and the
no-reversal planning below are for.

### 8. Re-verify after any Atom reflash

Reflashing wipes the servo zeros. Wrong zeros corrupt FK, which corrupts the
hand-eye calibration underneath it even though nothing touched the camera.

```bash
./scripts/calibrate_zero.py
./scripts/verify_calibration.py
```

### What NOT to do

- **Do not lower `max_jog_accel_deg_s2`.** It is the intuitive fix for jerk
  and it is wrong here. Measured: 1200 acquires in 0.36s, 600 in 2.02s, 300
  often fails to converge — and steadiness once locked is **3px in all
  three**. A gentler ramp costs only the getting there. The jerk is
  stop-start *between* waypoints, which the lookahead already fixes.
- **Do not tighten the arrival tolerance below 1.0°.** That is the 0.79°
  backlash floor. The arm settles there at 4.5s and is flat out to 21s;
  waiting longer buys nothing and every move starts reporting a failure to
  arrive.
- **Do not touch the IK.** 0.03mm.
- **Do not let a joint reverse on the final approach.** `plan_press` now
  measures this (`reversal_deg`) and re-solves to remove it. A reversal
  re-opens that joint's backlash a few millimetres from the button, turning a
  repeatable offset into random slop that step 7 cannot subtract.

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

- ~~`speed_at_100_deg_s` is an unmeasured guess~~ — measured 2026-08-01 and
  now 52.0. Re-measure after any payload change with `scripts/measure_arm.py`.
- The up/down call buttons are not detected yet; only the 12 numbered ones are.
- The home pose is defined in five places that must agree.
- `obstacles.yaml` is empty; `src/mycobot_bringup/config/network.yaml` is read
  by nothing and disagrees with the defaults.
- Approach is implemented but off by default until tracking is solid.
- **The 2×2 image Jacobian has never been measured** with `skip_probe:=false`,
  which matters more with the D405's wider lens than it did before.

The Raspberry-Pi-over-network setup this replaced still works and is what the
launch files default to; it is archived in
[legacy/README_pi_network.md](legacy/README_pi_network.md).

Development notes, measured results, and the reasoning behind the control law
are in [CLAUDE.md](CLAUDE.md). Most of the traps in it were found the
expensive way.
