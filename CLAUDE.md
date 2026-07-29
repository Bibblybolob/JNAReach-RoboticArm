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
- Planned hardware move: Jetson Orin Nano + RealSense D405, which removes most
  of the latency the servo currently compensates for.
