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

Two traps:

- **`command_lag` (0.15) must stay BELOW the true lag (~0.25).** The error is
  asymmetric. Too low just corrects slightly hard. Too high sweeps in jogs
  that already landed, double-counts them, concludes it overshot, and
  reverses — reproducing the flicking. A slow pipeline is a reason to fix the
  pipeline, not to raise this.
- **Detection rate is the real ceiling.** Below ~6/s nothing tuned in the
  servo helps. Read the `tracker:` and `pipeline:` log lines before touching
  any gain; `model_complexity:=0` and a smaller camera frame move that number
  more than anything else.

Signs are worked out automatically (`auto_sign`) by correlating what each jog
was predicted to do to the image against what it did. `assumed_h_sign` /
`assumed_v_sign` still exist but should not need reaching for.

```bash
python3 src/mycobot_perception/test/test_servo_math.py
```
Pins the sign conventions and the auto_sign detector. Runs without ROS — it
loads the functions out of the node by source. **Run it after any change to
the control law**; a flipped sign does not crash, it drives the target out of
frame, which is indistinguishable from a badly mounted camera.

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

## Known-unfinished

- `speed_at_100_deg_s: 120.0` in the driver is an unmeasured guess.
  `scripts/measure_arm.py` exists to measure it and has never been run.
- No camera intrinsics — `scripts/calibrate_camera.py` unused, so
  `/hand/point_cam` stays silent and depth is unavailable.
- `obstacles.yaml` is empty; `src/mycobot_bringup/config/network.yaml` is read
  by nothing and disagrees with the defaults.
- Approach (closing in on the hand) is implemented but off by default
  (`approach_enabled:=false`) until tracking is solid.
- Planned hardware move: Jetson Orin Nano + RealSense D405, which removes most
  of the latency the servo currently compensates for.
