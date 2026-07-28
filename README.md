# JNAReach — myCobot 280 Pi

A ROS 2 workspace for driving a myCobot 280 Pi over the network, with
eye-in-hand visual servoing. The arm finds a hand with a flange-mounted camera,
centres it in view, and closes in — no camera calibration required.

Working toward pressing elevator buttons autonomously.

This guide assumes **a freshly installed Ubuntu 22.04** and nothing else. If
you already have ROS 2 Humble, skip to [Get the code](#3-get-the-code).

Documentation for the older gripper/food-handling version of this workspace is
in [legacy/README_ros2_workspace.md](legacy/README_ros2_workspace.md).

---

## How the pieces fit together

```
Desktop / VM (Ubuntu 22.04)          Raspberry Pi (on the arm)
┌──────────────────────────┐         ┌──────────────────────────┐
│ mycobot_hardware_node    │◄──TCP──►│ server.py       :9000    │──► arm servos
│   joint states           │  9000   │  (bridges TCP to serial) │
│   trajectories, jogging  │         │                          │
│                          │         │                          │
│ camera_node              │◄──HTTP──│ camera_stream.py  :8080  │◄── USB webcam
│ hand_tracker_node        │  8080   │  (MJPEG server)          │
│ visual_servo_node        │         └──────────────────────────┘
│ move_group (MoveIt2)     │
└──────────────────────────┘
```

**The Pi's TCP server accepts exactly one client.** Only one thing may talk to
the arm at a time — the driver, or a script like `measure_arm.py`, never both.
This constraint shapes a lot of the design.

---

## 1. Install ROS 2 Humble

```bash
sudo apt update && sudo apt install -y software-properties-common curl && sudo add-apt-repository -y universe
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

Source ROS in every new shell — put it in `.bashrc` so you stop thinking about it:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && source ~/.bashrc
```

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

`ultralytics` is only needed for the YOLO food/object detector, which the
elevator work does not use:

```bash
pip install ultralytics
```

## 3. Get the code

```bash
git clone https://github.com/Bibblybolob/JNAReach-RoboticArm.git ~/mycobot_project
```

## 4. Build

```bash
cd ~/mycobot_project && colcon build --symlink-install && source install/setup.bash
```

**Use `--symlink-install`.** Without it colcon *copies* Python files, so edits
have no effect until you rebuild — and you will lose time to a change that
"didn't apply" when in fact it was never installed.

Add the workspace to your shell too:

```bash
echo "source ~/mycobot_project/install/setup.bash" >> ~/.bashrc
```

## 5. Find the Pi and set its address

The Pi may have **both** WiFi and Ethernet, on different subnets. You need the
address your desktop can actually reach — check your router, or on the Pi:

```bash
hostname -I
```

Then, on the desktop:

```bash
export MYCOBOT_IP=192.168.0.15
```

Every launch file reads `$MYCOBOT_IP`. Put it in `.bashrc`. A `robot_ip:=...`
launch argument overrides it for one run.

If the address comes from DHCP it will change on reboot; a router reservation
saves repeating this.

## 6. Set up the Pi

One-time, if the Pi has never been configured:

```bash
ssh er@$MYCOBOT_IP 'mkdir -p ~/JON/mycobot_project/pi' && scp -r pi/* er@$MYCOBOT_IP:~/JON/mycobot_project/pi/ && ssh er@$MYCOBOT_IP 'bash ~/JON/mycobot_project/pi/setup_pi.sh'
```

Afterwards, and whenever anything in `pi/` changes, this deploys the scripts,
installs the systemd units, frees ports 9000/8080 and restarts both services:

```bash
./scripts/redeploy_pi.sh
```

## 7. Check both services before going further

Camera — expect `HTTP 200`:

```bash
curl -s -m 3 -o /dev/null -w 'camera HTTP %{http_code}\n' "http://$MYCOBOT_IP:8080/?action=snapshot"
```

Arm — expect a list of six joint angles. This only reads, it does not move:

```bash
python3 scripts/measure_arm.py --ip $MYCOBOT_IP --skip-motion
```

Both working means the hard part is done. If either fails, see
[Troubleshooting](#troubleshooting) — every failure listed there is one that
actually happened during development.

---

## Running it

### Finger following (one command)

```bash
./run.py
```

That is the whole thing. It checks the Pi is actually serving both ports,
launches the stack (output to `/tmp/mycobot_stack.log`), waits for the nodes,
then gives you a menu:

```
  1) Search for a hand      2) Home the arm       3) Stop servoing
  4) Resume servoing        5/6) Jogging ON/OFF
  7) Status                 8) Pipeline rates     9) Health check
  l) Live log               q) Quit (shuts the stack down)
```

The arm stays **idle at home** until you press `1`.

> **The Pi runs its own copy of the code.** `pi/server.py` and
> `pi/camera_stream.py` live on the robot, not in the workspace, so editing
> them here changes nothing until `./scripts/redeploy_pi.sh` pushes them
> across. Several arm-side fixes — the client idle timeout most of all — exist
> only in that copy, which makes it possible to "fix" a disconnect, rebuild,
> and see no change whatsoever. `run.py` now checksums those files against the
> robot at startup and offers to redeploy if they differ.

Checking the ports up front turns two confusing ROS-level symptoms —
`Waiting for /arm/jog_enable` and a servo node that never sees a frame — into
one clear message naming the service that is down.

`7` shows which nodes are up and live joint angles. `8` samples the camera and
detection rates; the gap tells you whether MediaPipe is the bottleneck, and
since the servo loop acts once per detection, the detection rate *is* your
control rate. `l` tails the launch output, so you keep the diagnostics without
them drowning the menu. `q` shuts the stack down cleanly, so nothing is left
holding the arm's single client slot.

Extra arguments pass through, e.g. `./run.py gain:=1.5`. If a stack is already
running it attaches to it rather than starting a second one.

The raw equivalents still work if you prefer them:

```bash
ros2 launch mycobot_bringup servo_demo.launch.py
ros2 service call /servo/search std_srvs/srv/Trigger
```

It sweeps until it sees a hand, twitches a few joints to learn how the camera
is mounted (**hold your hand still for this**), then centres and closes in.
After 15s with no sighting it returns home. Stop it at any time:

```bash
ros2 service call /servo/enable std_srvs/srv/SetBool "{data: false}"
```

Useful overrides:

| Argument | Default | Effect |
|---|---|---|
| `gain` | 0.7 | **fraction** of the full centring correction per sighting, not degrees; raise toward 0.9 to follow harder |
| `max_step_deg` | 5.0 | biggest single jog; times the detection rate, this caps how fast the camera can slew |
| `command_lag` | 0.15 | seconds from sending a jog to seeing it; **keep below the true lag** — see below |
| `lag_compensation` | true | predict where the hand will be once sent jogs land; turn off only with `gain:=0.3` |
| `auto_sign` | true | detect and flip an inverted axis from the tracking motion itself |
| `deadband` | 0.04 | image error it stops correcting below; lower to sit nearer dead centre |
| `assumed_deg_per_error` | 25.0 | degrees that would fully centre a frame-edge target ≈ half the camera FOV; geometry, not tuning |
| `target_landmark` | 9 | palm centre; `8` steers at the index fingertip instead |
| `max_frame_age` | 0.12 | drop camera frames already staler than this rather than tracking on them |
| `lost_timeout` | 15.0 | seconds before giving up and homing |
| `target_size_fraction` | 0.45 | how close to get; higher is closer |
| `approach_enabled` | false | set true to close in as well as centring |
| `search_on_start` | false | start hunting without the trigger |
| `show_window` | false | OpenCV window from the tracker |

**There is no PID.** Tracking is proportional control plus a model of the
loop's own delay. Each time the hand is seen, the camera moves a fixed
*fraction* of the way to having it centred; nothing accumulates between
sightings.

The reason is dead time. Capture on the Pi, JPEG over the network, decode,
MediaPipe on CPU, then a jog the arm takes time to execute — several tenths of
a second pass between an observation and the camera finishing its response to
it, and detections keep arriving during that gap still reporting the *old*
error. Re-commanding a correction that is already on its way is overshoot by
construction: the arm sails past centre, comes back, and oscillates. An
integral term winds up across exactly that interval and makes it worse.

Backing the gain off to 0.3 stops the oscillation but makes the arm trail a
moving hand. So instead the node remembers every jog it sends and, on each
detection, adds back the image motion those jogs have not produced yet —
correcting where the hand *will* be rather than where it was. Simulating the
pipeline (`src/mycobot_perception/test/`, and the notes in
`visual_servo_node.py`):

| | settles |
|---|---|
| gain 1.44, no compensation — the original PID | never |
| gain 0.30, no compensation | 1.3 s |
| gain 0.70, no compensation | never |
| gain 0.70, with compensation | 0.7 s |

**`command_lag` is asymmetric — set it low.** Too low is harmless: some
in-flight motion goes uncounted and the loop corrects a little harder than
needed. Too high is not: the window sweeps in jogs that have *already* landed
and are already visible in the measurement, the compensator counts them twice,
decides it overshot, and reverses — which is the flicking it exists to
prevent. In simulation, 0.35 s and above oscillates at every gain above 0.3,
while 0.10–0.15 is stable from gain 0.5 to 1.1. A slow pipeline is not a
reason to raise it; it is a reason to fix the pipeline.

**If it still lags,** read the two report lines rather than guessing:

```
tracker:  8.4 detections/s, 91ms per frame, 40ms old on arrival, dropped 12 stale of 118
pipeline: 8.4 detections/s, frames 63ms old when acted on
```

Below ~6 detections/s nothing tuned in the servo will help — the arm simply
is not being told where the hand is often enough. `model_complexity:=0` and a
smaller camera frame are the two things that move that number.

### MoveIt2 planning and RViz

```bash
ros2 launch mycobot_bringup moveit_bringup.launch.py
```

Heavier, and unnecessary for servoing — visual servoing bypasses MoveIt
entirely, jogging joints straight from image error.

### Homing

```bash
ros2 service call /arm/home std_srvs/srv/SetBool "{data: true}"
```

Moves to a fixed pose, `[0, 90, -90, -90, 0, 0]` degrees. This is commanded
from wherever the arm happens to be, which makes it the largest single move the
arm makes — check the path is clear.

---

## Topics and services

| Name | Type | Purpose |
|---|---|---|
| `/joint_states` | JointState | 6 arm joints |
| `/arm_controller/follow_joint_trajectory` | FollowJointTrajectory | MoveIt execution |
| `/arm/home` | SetBool | move to the fixed home pose |
| `/arm/jog` | JointJog | relative joint moves, in **degrees** |
| `/arm/jog_enable` | SetBool | gate for jogging |
| `/servo/search` | Trigger | start hunting for a hand |
| `/servo/enable` | SetBool | master stop for servoing |
| `/camera/image_raw` | Image | camera feed |
| `/hand/point_px` | PointStamped | x,y = fingertip pixel; **z = palm width in pixels** |
| `/hand/annotated` | Image | landmarks drawn, for debugging |

---

## Calibration (optional)

Servoing needs none of this. These unlock metric 3D work later.

**Arm speed** — replaces two guessed constants with measured values. Moves the
arm; stop the driver first, since only one client may connect:

```bash
python3 scripts/measure_arm.py --ip $MYCOBOT_IP
```

**Camera intrinsics** — needed before any pixel can become a ray:

```bash
python3 scripts/calibrate_camera.py --stream "http://$MYCOBOT_IP:8080/?action=stream"
```

---

## Troubleshooting

**"Connection refused" on port 9000, but the camera on 8080 works.**
The Pi has two interfaces and the old `server.py` bound only to the wlan0
address, so it was listening on an address the desktop could not route to.
Fixed by binding `0.0.0.0` — but the fix must be deployed:
`./scripts/redeploy_pi.sh`.

**The arm was reachable, then stopped accepting connections.**
The server accepts one client, and older versions had no receive timeout, so a
client that died without closing blocked it forever. Redeploy, then check
nothing else holds the socket — a leftover `measure_arm.py` or a second driver
will lock everything else out.

**The Pi's IP changed and nothing connects.**
It will, eventually — the Pi takes a DHCP lease, and a router reboot or a
lease expiry reassigns it. This repo has already chased that address three
times. Short-term fix is `export MYCOBOT_IP=<new address>`; every launch file,
script and node default reads it.

Three ways to stop it mattering, best first:

1. **DHCP reservation on the router.** Bind the Pi's MAC to a fixed address.
   Nothing on the Pi or in this repo changes, and it survives reflashing.
2. **mDNS.** `MYCOBOT_IP` accepts a hostname, and every consumer of it
   (`ping`, `ssh`, `socket.create_connection`, the MJPEG URL, pymycobot)
   resolves names fine. Find the Pi's hostname with
   `ssh er@<current-ip> hostname`, then `export MYCOBOT_IP=<hostname>.local`
   and the address can move freely.
3. **Static IP on the Pi**, or a direct Ethernet link with static addresses on
   both ends — which sidesteps DHCP entirely and is worth doing anyway.

**A service is dead after a reboot with no error.**
The units used to start before the network had an address, fail, exhaust
systemd's start limit, and be abandoned permanently. Fixed in the current
units; `redeploy_pi.sh` installs them.

**Config changes have no effect.**
Stale build. Rebuild with `--symlink-install`, and check what is actually
installed rather than what is in `src/`:

```bash
grep -rn "192.168" ~/mycobot_project/install/*/lib/python3*/site-packages/*/camera_node.py
```

**"Waiting for /arm/jog_enable (is the driver running?)"**
The driver could not reach the arm. It now stays up and retries every 5s
instead of dying, and prints the address it failed on.

**A node dies on import: `_ARRAY_API not found`, `KeyError: 16`, or
`module 'mediapipe' has no attribute 'solutions'`.**
Python dependency versions, not ROS. `pip` installs into `~/.local`, which
shadows the system packages ROS 2 Humble's compiled extensions were built
against — so upgrading numpy, opencv or mediapipe breaks nodes that worked,
with errors that point at ROS instead of at the upgrade. `cv_bridge` is the
usual casualty, and it takes both `camera_node` and `hand_tracker_node` with
it. Fix:

```bash
pip install -r requirements.txt
```

That pins `numpy<2`, `mediapipe<1.0` and `opencv-contrib-python<5`. The
reasoning for each bound is in the file.

**Repeated "Lost connection to the arm ... Broken pipe" and camera read
timeouts.**
The link is not flaky — the host is stalling. `server.py` drops a client that
has been silent too long (to stop a dead client locking out the single-client
server forever), and a loaded desktop VM freezes every ROS node for tens of
seconds at a time, which looks identical to a dead client from the Pi's side.
The tell is that *both* the driver and the servo node go completely silent for
the same window, despite independent timers that log every 2–3s.

Mitigated on three fronts: the server's idle timeout is now 120s with TCP
keepalive tuned to reap a genuinely dead peer in ~60s, the driver sends a
keepalive read every 5s so the link rarely goes idle, and the camera's read
timeout is 30s. If it still happens, fix the stall rather than the timeouts —
run `free -h` and `vmstat 1` while the stack is up, and if it is swapping give
the VM more RAM (≥4 GB) and ≥2 vCPUs.

**Servo enabled but the arm does not move.**
Both consoles now say why — the driver names the reason a jog was rejected, and
the servo says whether it is inside the deadband, has no target, or has no
Jacobian. `On target` means it is already centred: move your hand toward the
frame edge.

**Probe fails with "Jacobian is singular".**
From that pose the two joints move the image in nearly the same direction, so
they cannot steer independently. Move the arm elsewhere and re-trigger.

**MediaPipe sees nothing.**
Check `/camera/image_raw` first — with a flange-mounted camera it is often
pointing somewhere unexpected. MediaPipe also needs the *whole* hand, so back
off to 30–50cm.

---

## Notes and current state

**No collision geometry.** `obstacles.yaml` is empty. Self-collision checking
still applies, but nothing in the environment is modelled — the planner will
happily drive through your bench. Add real measured geometry before working
near anything you care about.

**Two constants are still guesses**, both flagged in their files:
`speed_at_100_deg_s` in the driver, and `max_velocity` in `joint_limits.yaml`.
`measure_arm.py` replaces both with measured values in about five minutes.

**Depth is approximate.** With one camera there is no true range — apparent
palm size stands in for it. An OAK-D Pro would replace that step; the code is
structured so only the pixel→3D conversion changes.

**The gripper has been removed** from the URDF, SRDF, controllers, and driver.
Those four must stay consistent: a controller or SRDF referencing a joint the
URDF does not define stops `move_group` from starting.
