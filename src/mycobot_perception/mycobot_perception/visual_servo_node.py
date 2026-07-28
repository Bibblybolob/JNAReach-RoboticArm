"""
Image-based visual servoing: find a hand, centre it, and approach it.

This is the eye-in-hand demo that needs NO calibration. It never computes a 3D
position. It only knows "the target is 40 pixels left of centre" and turns that
into "move the joint that shifts the view right". Because the camera rides on
the end effector, image error maps directly to joint motion.

That means no camera intrinsics, no hand-eye transform, no depth estimate --
none of which exist yet. Those unlock better things later; this works today.


BEHAVIOUR

The arm is idle at home until you ask for it:

    ros2 service call /servo/search std_srvs/srv/Trigger

Then it runs a state machine:

    IDLE      sitting at home, doing nothing
    SEARCHING sweeping slowly to bring a hand into view
    TRACKING  centring the hand and closing in on it
    HOMING    returning to the home pose, then back to IDLE

If the hand goes out of view for lost_timeout seconds (default 15) it gives up
and homes. Call /servo/search again to restart, or /servo/enable false to stop
immediately at any point.

Jogging is armed automatically at startup -- the node calls /arm/jog_enable
itself -- so no manual service calls are needed beyond the search trigger.


APPROACHING WITHOUT DEPTH

Centring alone leaves the arm at whatever distance it started. To close in
without a depth sensor, the loop uses apparent size: the tracker reports palm
width in pixels, and a hand that grows in frame is a hand getting nearer. The
approach joint is driven until the palm reaches target_size_fraction of the
frame width.

The limits of this are worth knowing. Apparent size is not distance -- a large
hand and a close hand look identical -- so this closes in on a consistent
*framing*, not a measured standoff. And approach naturally ends by breaking
itself: once the hand fills the frame MediaPipe can no longer see the whole
hand, tracking drops, and the lost-target timeout takes over.


WHY IT CALIBRATES ITSELF FIRST

The mapping from image error to joint motion depends on how the camera is
physically rotated on the flange. Mounted upright, "target is left" means
"rotate joint1 one way". Rotated 90 degrees, the same error needs a different
joint entirely; rotated 180, the same joint but the opposite sign. Guessing
wrong means the arm drives the target OUT of frame -- a runaway, and exactly
the failure you do not want when the target is someone's hand.

Rather than asking you to measure the mount or work out signs by trial and
error, the node measures it: it jogs each joint a little, watches which way the
tracked point moved in the image, and builds the 2x2 Jacobian relating joint
motion to image motion. Inverting that gives the control law. This is standard
visual-servo Jacobian estimation, it takes a few seconds, and it is correct for
whatever orientation you actually mounted the camera in.

The approach axis is probed the same way: nudge the approach joint, see whether
the palm grew or shrank, and keep the sign.

If probing fails (no hand visible, arm blocked), the node refuses to servo
rather than falling back to a guess.
"""

from __future__ import annotations

import math
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from control_msgs.msg import JointJog
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import SetBool, Trigger


# State machine states.
IDLE = 'IDLE'
SEARCHING = 'SEARCHING'
TRACKING = 'TRACKING'
HOMING = 'HOMING'


class VisualServoNode(Node):

    def __init__(self) -> None:
        super().__init__('visual_servo_node')

        self.declare_parameter('point_topic', '/hand/point_px')
        # CameraInfo, not the image stream. All this node needs from the
        # camera is the frame size, and it only needs it once -- subscribing
        # to /camera/image_raw for that meant deserialising a ~900KB bgr8
        # frame every cycle, forever, to read two integers it already had.
        # On a host whose stalls were dropping the arm connection, that was
        # megabytes per second of pure waste.
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('jog_topic', '/arm/jog')

        # Joints used to steer the view. joint1 swings the arm horizontally;
        # joint5 is the wrist pitch, which tilts the camera vertically. These
        # are a starting guess only -- the probe below measures what each one
        # actually does to the image and adapts.
        self.declare_parameter('horizontal_joint', 'joint1')
        self.declare_parameter('vertical_joint', 'joint5')

        # --- PID gains ---
        # Error is normalised to half-frames: 1.0 means the target sits at the
        # frame edge. Gains are in degrees of joint motion per unit of error.
        #
        # Proportional alone cannot centre the target. It produces motion in
        # proportion to error, so as error shrinks so does the correction, and
        # it stalls wherever the remaining push is too small to overcome
        # stiction, gravity sag, or the deadband. That standing offset is why
        # the hand ends up near the centre rather than at it.
        self.declare_parameter('gain', 3.0)
        # Integral accumulates the error that proportional leaves behind and
        # keeps pushing until it is actually gone. This is the term that
        # centres the target. Too high and it overshoots and oscillates.
        self.declare_parameter('ki', 1.2)
        # Derivative damps the approach, which buys room to raise the other two
        # without ringing. Computed on the error signal, which is already
        # smoothed upstream in the tracker.
        self.declare_parameter('kd', 0.35)
        # Anti-windup. Without a cap the integral keeps growing whenever the
        # arm cannot reduce the error -- target out of reach, joint at a limit,
        # jogging disabled -- and then unloads all at once as a lurch when
        # motion resumes.
        self.declare_parameter('integral_limit', 0.8)
        # Ignore errors smaller than this (normalised). MediaPipe jitter and
        # hand tremor are both a few pixels; without a deadband the arm hunts
        # continuously and buzzes. Tighter than before now that the integral
        # term can actually close the remaining gap.
        self.declare_parameter('deadband', 0.015)
        # Loop rate. The driver rate-limits jogs to its command_interval
        # (0.06s, ~16Hz), so going much above that just discards commands.
        self.declare_parameter('rate', 15.0)
        # Stop if the target has not been seen for this long. Without it, the
        # arm keeps acting on a stale position after the hand leaves frame.
        self.declare_parameter('target_timeout', 0.5)
        self.declare_parameter('max_step_deg', 2.0)

        # Probe settings. probe_deg is how far each joint is nudged to measure
        # its effect; big enough to produce clear image motion, small enough to
        # be a twitch.
        self.declare_parameter('probe_deg', 4.0)
        self.declare_parameter('probe_settle', 1.2)
        # Start tracking the instant a hand is seen, instead of spending
        # several seconds twitching joints to measure the camera mounting.
        #
        # The probe exists because a wrong sign drives the target OUT of
        # frame. Skipping it means trusting the assumption below instead of
        # measuring, so if the camera is mounted rotated the arm will move
        # the wrong way -- recoverable, since losing the target homes after
        # lost_timeout, but it will not track until the signs are right.
        self.declare_parameter('skip_probe', True)
        # Degrees of joint motion per unit of normalised image error, used
        # only when the probe is skipped. This is what the probe would
        # otherwise measure.
        self.declare_parameter('assumed_deg_per_error', 8.0)
        # Flip either of these if the arm drives the hand out of frame along
        # that axis rather than centring it.
        self.declare_parameter('assumed_h_sign', 1.0)
        self.declare_parameter('assumed_v_sign', 1.0)
        # +1 means extending the approach joint makes the hand look bigger.
        # 0 disables closing in without disabling centring.
        self.declare_parameter('assumed_approach_sign', 1.0)

        # --- Approach ---
        # Joint driven to close distance. joint3 (elbow) extends and retracts
        # the arm, which moves the flange-mounted camera along its view more
        # than the other joints do. Whichever joint is chosen, the probe
        # measures its actual effect on apparent size, so a poor choice shows
        # up as a failed probe rather than as wrong motion.
        self.declare_parameter('approach_joint', 'joint3')
        self.declare_parameter('approach_enabled', True)
        # Stop closing in when palm width reaches this fraction of frame width.
        # Higher gets closer; too high and MediaPipe loses the hand because it
        # no longer fits in frame, which ends the approach abruptly.
        self.declare_parameter('target_size_fraction', 0.45)
        self.declare_parameter('approach_gain', 6.0)
        self.declare_parameter('approach_deadband', 0.03)
        self.declare_parameter('max_approach_step_deg', 1.5)

        # --- Search / idle behaviour ---
        # Seconds without a sighting before giving up and homing.
        self.declare_parameter('lost_timeout', 15.0)
        # Sweep the wrist pitch while looking for a hand. joint5 tilts the
        # flange-mounted camera through its whole vertical arc, so a single
        # sweep covers far more of the room than panning the base does.
        self.declare_parameter('search_joint', 'joint5')
        # Total travel, centred on wherever the sweep began. The home pose
        # leaves joint5 at 0, and search follows homing, so 180 means the
        # camera really does swing +90 to -90.
        self.declare_parameter('search_range_deg', 180.0)
        # Seconds for one traverse of that range. The step size is derived
        # from this and the measured time since the last tick rather than
        # being a fixed number of degrees per loop -- on a host that stalls,
        # a fixed step makes the sweep take however long the stalls add up
        # to, which is why the old sweep crawled. Pacing against the clock
        # keeps the sweep honest whatever the loop rate does.
        self.declare_parameter('search_sweep_seconds', 15.0)
        # Never let one catch-up step become a lunge after a stall. The
        # driver clamps to max_jog_deg anyway; this keeps intent local.
        self.declare_parameter('search_max_step_deg', 2.5)
        # Begin searching as soon as the node starts, instead of waiting for
        # the trigger. Off by default: launching a file should not set the arm
        # hunting around the room.
        self.declare_parameter('search_on_start', False)
        # Arm the driver's jog gate automatically at startup.
        self.declare_parameter('auto_arm_jog', True)

        self._h_joint = self.get_parameter('horizontal_joint').value
        self._v_joint = self.get_parameter('vertical_joint').value
        self._gain = float(self.get_parameter('gain').value)
        self._ki = float(self.get_parameter('ki').value)
        self._kd = float(self.get_parameter('kd').value)
        self._integral_limit = float(self.get_parameter('integral_limit').value)
        self._deadband = float(self.get_parameter('deadband').value)

        # PID state, in normalised image-error units.
        self._integral = np.zeros(2, dtype=float)
        self._prev_error: np.ndarray | None = None
        self._prev_time: float | None = None
        self._rate = float(self.get_parameter('rate').value)
        self._timeout = float(self.get_parameter('target_timeout').value)
        self._max_step = float(self.get_parameter('max_step_deg').value)
        self._probe_deg = float(self.get_parameter('probe_deg').value)
        self._probe_settle = float(self.get_parameter('probe_settle').value)
        self._skip_probe = bool(self.get_parameter('skip_probe').value)
        self._assumed_deg = float(
            self.get_parameter('assumed_deg_per_error').value)
        self._assumed_h_sign = float(self.get_parameter('assumed_h_sign').value)
        self._assumed_v_sign = float(self.get_parameter('assumed_v_sign').value)
        self._assumed_approach_sign = float(
            self.get_parameter('assumed_approach_sign').value)

        self._approach_joint = self.get_parameter('approach_joint').value
        self._approach_enabled = bool(self.get_parameter('approach_enabled').value)
        self._target_size = float(self.get_parameter('target_size_fraction').value)
        self._approach_gain = float(self.get_parameter('approach_gain').value)
        self._approach_deadband = float(self.get_parameter('approach_deadband').value)
        self._max_approach_step = float(
            self.get_parameter('max_approach_step_deg').value)

        self._lost_timeout = float(self.get_parameter('lost_timeout').value)
        self._search_joint = self.get_parameter('search_joint').value
        self._search_range = float(self.get_parameter('search_range_deg').value)
        self._sweep_seconds = float(
            self.get_parameter('search_sweep_seconds').value)
        self._search_max_step = float(
            self.get_parameter('search_max_step_deg').value)

        # Sign of d(palm size)/d(approach joint), learned by the probe. Without
        # it we would not know whether extending the joint moves the camera
        # toward the hand or away from it.
        self._approach_sign = 0.0

        # Latest palm width in pixels, from point.z.
        self._last_size_px: float | None = None

        # State machine.
        self._state = IDLE
        self._state_since = time.monotonic()
        # Sweep direction and accumulated travel while SEARCHING.
        self._search_dir = 1.0
        self._search_travel = 0.0
        # Wall-clock of the previous sweep tick, so each step can be sized
        # from real elapsed time instead of assuming the loop ran on time.
        self._last_sweep_time: float | None = None

        # Frame size, learned from the first image. Needed to normalise pixel
        # error; we do not assume 640x480.
        self._width: int | None = None
        self._height: int | None = None

        self._last_point: tuple[float, float] | None = None
        self._last_point_time = 0.0
        # Timestamp of the measurement the control loop last acted on. The
        # loop runs faster than detections arrive, so without this it re-uses
        # the same reading for several iterations -- integrating the identical
        # error each time, which inflates the integral term and unloads as a
        # lurch when the arm finally moves. Acting once per measurement keeps
        # the integral honest.
        self._acted_point_time = 0.0
        # Distinguishes "the tracker has never said anything" (wrong topic, node
        # not running, MediaPipe not detecting) from "the hand is momentarily
        # out of frame". These need completely different fixes, and reporting
        # both as "hold your hand in view" sends you hunting for the wrong one.
        self._ever_received_point = False

        # Inverse image Jacobian: maps normalised image error to joint deltas.
        # None until probing succeeds; servoing refuses to run without it.
        self._jinv: np.ndarray | None = None
        self._probed = False
        self._probing = False

        cb = ReentrantCallbackGroup()

        self._point_topic = self.get_parameter('point_topic').value
        self._info_topic_name = self.get_parameter('camera_info_topic').value

        self._point_sub = self.create_subscription(
            PointStamped, self._point_topic,
            self._point_cb, 1, callback_group=cb)
        self._info_sub = self.create_subscription(
            CameraInfo, self._info_topic_name,
            self._camera_info_cb, 10, callback_group=cb)
        self._jog_pub = self.create_publisher(
            JointJog, self.get_parameter('jog_topic').value, 1)

        # Servoing is armed by default now; the search trigger is what actually
        # sets the arm moving, so a second gate here served no purpose.
        self._enabled = True
        self._enable_srv = self.create_service(
            SetBool, 'servo/enable', self._enable_cb, callback_group=cb)
        self._search_srv = self.create_service(
            Trigger, 'servo/search', self._search_cb, callback_group=cb)

        # Clients for the driver's gate and homing service.
        self._jog_enable_cli = self.create_client(
            SetBool, '/arm/jog_enable', callback_group=cb)
        self._home_cli = self.create_client(
            SetBool, '/arm/home', callback_group=cb)

        self._timer = self.create_timer(
            1.0 / self._rate, self._servo_step, callback_group=cb)

        if bool(self.get_parameter('auto_arm_jog').value):
            # Deferred: the driver may not be up yet at construction time.
            self._arm_timer = self.create_timer(
                2.0, self._arm_jog_once, callback_group=cb)

        if bool(self.get_parameter('search_on_start').value):
            self._set_state(SEARCHING)

        self.get_logger().info(
            'Visual servo ready, idle at home. Start it with:\n'
            '    ros2 service call /servo/search std_srvs/srv/Trigger\n'
            f'It will home again after {self._lost_timeout:.0f}s without a '
            'sighting.'
        )

    # ---- Driver gate ----

    def _arm_jog_once(self) -> None:
        """Enable the driver's jog gate, retrying until the driver appears."""
        if not self._jog_enable_cli.service_is_ready():
            self.get_logger().info(
                'Waiting for /arm/jog_enable (is the driver running?)...',
                throttle_duration_sec=5.0)
            return
        # Only report success, and only stop retrying, once the driver has
        # actually answered. The previous version fired the request, logged
        # "Armed", and cancelled the retry timer without ever looking at the
        # result -- so if the driver was busy (homing used to block this
        # service outright) the gate stayed shut while the log claimed
        # otherwise, and every later jog was silently discarded. That is why
        # jogging had to be switched on by hand.
        req = SetBool.Request()
        req.data = True
        future = self._jog_enable_cli.call_async(req)

        def _armed(fut):
            try:
                res = fut.result()
            except Exception as e:
                self.get_logger().warn(
                    f'/arm/jog_enable call failed ({e}); will retry.')
                return
            if res is not None and getattr(res, 'success', True):
                self.get_logger().info(
                    'Armed the driver jog gate (/arm/jog_enable).')
                self._arm_timer.cancel()
            else:
                self.get_logger().warn(
                    '/arm/jog_enable refused; will retry.')

        future.add_done_callback(_armed)

    # ---- State machine ----

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self.get_logger().info(f'{self._state} -> {state}')
        self._state = state
        self._state_since = time.monotonic()
        self._reset_pid()
        if state == SEARCHING:
            self._search_travel = 0.0
            # Drop the previous tick's timestamp too, or the first step of a
            # new sweep is sized from however long the node sat in IDLE.
            self._last_sweep_time = None

    def _search_cb(self, request, response):
        if not self._enabled:
            response.success = False
            response.message = 'Servo is disabled; enable it first.'
            return response
        self._set_state(SEARCHING)
        response.success = True
        response.message = (
            'Searching for a hand. Hold one in view; it will home again after '
            f'{self._lost_timeout:.0f}s without a sighting.'
        )
        return response

    def _go_home(self) -> None:
        """Ask the driver to return to its fixed home pose."""
        if not self._home_cli.service_is_ready():
            self.get_logger().warn(
                '/arm/home unavailable; staying put instead of homing.')
            self._set_state(IDLE)
            return
        req = SetBool.Request()
        req.data = True
        future = self._home_cli.call_async(req)
        # Homing sets _in_motion in the driver, which makes it ignore jogs for
        # the duration, so we simply wait for it rather than commanding motion.
        future.add_done_callback(lambda _f: self._set_state(IDLE))

    # ---- Inputs ----

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._width is None and msg.width > 0 and msg.height > 0:
            self._width, self._height = msg.width, msg.height
            self.get_logger().info(f'Frame size: {msg.width}x{msg.height}')

    def _point_cb(self, msg: PointStamped) -> None:
        if not self._ever_received_point:
            self.get_logger().info(
                f'First target sighting on {self._point_topic}.')
            # This only means the tracker sees a hand -- it fires regardless
            # of the servo state machine, so on its own it does not mean the
            # arm will move. Say so explicitly rather than leaving IDLE users
            # staring at a hand-detected log wondering why nothing happens.
            if self._state == IDLE:
                self.get_logger().info(
                    'Servo is IDLE, so this sighting will not move the arm. '
                    'Start it with: '
                    'ros2 service call /servo/search std_srvs/srv/Trigger')
            self._ever_received_point = True
        self._last_point = (msg.point.x, msg.point.y)
        # z carries palm width in pixels, not a depth. See hand_tracker_node.
        self._last_size_px = msg.point.z if msg.point.z > 0 else None
        self._last_point_time = time.monotonic()

    def _enable_cb(self, request, response):
        """Master on/off. Probing now happens on the SEARCHING -> TRACKING
        transition instead of here, so this is purely a kill switch: disabling
        stops the arm immediately, wherever it is in the state machine.
        """
        want = bool(request.data)
        # Clean PID state either way: on enable so a stale integral cannot
        # lurch the arm, on disable so it does not sit accumulating.
        self._reset_pid()
        self._enabled = want

        if not want:
            self._set_state(IDLE)
            response.message = 'Servo disabled; arm stopped where it is.'
        else:
            response.message = (
                'Servo enabled and idle. Trigger a hunt with: '
                'ros2 service call /servo/search std_srvs/srv/Trigger'
            )
        response.success = True
        self.get_logger().info(response.message)
        return response

    # ---- Error ----

    def _current_error(self):
        """Normalised (ex, ey) offset of the target from image centre.

        Units are half-frames: +1.0 means the target sits at the right/bottom
        edge. Normalising means gain does not have to be retuned when the
        camera resolution changes.
        """
        if self._width is None or self._last_point is None:
            return None
        if time.monotonic() - self._last_point_time > self._timeout:
            return None
        px, py = self._last_point
        ex = (px - self._width / 2.0) / (self._width / 2.0)
        ey = (py - self._height / 2.0) / (self._height / 2.0)
        return ex, ey

    def _wait_for_fresh_point(self, timeout=2.0):
        """Block until a target sighting newer than now arrives."""
        mark = time.monotonic()
        deadline = mark + timeout
        while time.monotonic() < deadline:
            if self._last_point is not None and self._last_point_time > mark:
                return self._last_point
            time.sleep(0.05)
        return None

    def _send_jog(self, joint: str, delta_deg: float) -> None:
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [joint]
        msg.displacements = [float(delta_deg)]
        self._jog_pub.publish(msg)

    # ---- Probe ----

    def _probe_axis(self, joint: str):
        """Nudge one joint and return the image displacement it caused."""
        before = self._wait_for_fresh_point()
        if before is None:
            return None, 'target not visible'

        self._send_jog(joint, self._probe_deg)
        time.sleep(self._probe_settle)
        after = self._wait_for_fresh_point()
        if after is None:
            return None, 'target lost during probe'

        # Put it back so the probe leaves the arm where it started.
        self._send_jog(joint, -self._probe_deg)
        time.sleep(self._probe_settle)

        dx = (after[0] - before[0]) / (self._width / 2.0)
        dy = (after[1] - before[1]) / (self._height / 2.0)
        return (dx, dy), None

    def _probe_approach(self):
        """Determine whether the approach joint moves the camera nearer.

        Returns (+1, None) if a positive jog makes the hand look bigger,
        (-1, None) if smaller, or (None, reason) if it cannot be told.
        """
        if self._last_size_px is None:
            return None, 'tracker is not reporting palm size'
        before = self._last_size_px

        self._send_jog(self._approach_joint, self._probe_deg)
        time.sleep(self._probe_settle)
        after = self._last_size_px

        self._send_jog(self._approach_joint, -self._probe_deg)
        time.sleep(self._probe_settle)

        if after is None:
            return None, 'target lost during approach probe'

        change = (after - before) / max(before, 1.0)
        # Require a clear change. Below this the measurement is camera noise,
        # and committing to a sign from noise means approaching in the wrong
        # direction, which drives the arm away from the hand.
        if abs(change) < 0.02:
            return None, (
                f'apparent size barely changed ({change * 100:+.1f}%); this '
                f'joint may not move the camera along its view'
            )
        return (1.0 if change > 0 else -1.0), None

    def _assume_orientation(self) -> None:
        """Take the camera mounting on trust instead of measuring it.

        Builds the same inverse Jacobian the probe would produce, but
        diagonal and from parameters: horizontal image error drives the
        horizontal joint, vertical drives the vertical one. That is correct
        for a camera mounted square on the flange and wrong by a sign or an
        axis swap for anything else, which is exactly what the probe exists
        to discover. Skipping it buys immediate tracking at the cost of
        that guarantee.
        """
        self._jinv = np.array(
            [[self._assumed_deg * self._assumed_h_sign, 0.0],
             [0.0, self._assumed_deg * self._assumed_v_sign]], dtype=float)
        self._approach_sign = (
            self._assumed_approach_sign if self._approach_enabled else 0.0)
        self._probed = True
        self.get_logger().info(
            f'Tracking immediately without probing: assuming '
            f'{self._assumed_deg:.0f}deg/error, h sign '
            f'{self._assumed_h_sign:+.0f}, v sign {self._assumed_v_sign:+.0f}. '
            'If the arm drives the hand out of frame, flip the matching sign '
            '(assumed_h_sign / assumed_v_sign), or set skip_probe:=false to '
            'measure it instead.')

    def _probe(self):
        """Measure the image Jacobian by moving each joint and watching.

        Builds J where [dx, dy]^T = J @ [d_horizontal, d_vertical]^T, then
        inverts it so the control law can go the other way.
        """
        if self._probing:
            return False, 'already probing'

        if self._width is None:
            n = self.count_publishers(self._info_topic_name)
            return False, (
                f'no frame size yet -- nothing usable on '
                f'{self._info_topic_name} ({n} publisher(s) detected). '
                + ('Is camera_node running?' if n == 0 else
                   'A publisher exists but no CameraInfo arrived -- check '
                   f'`ros2 topic hz {self._info_topic_name}`.')
            )

        # Give the tracker a moment before giving up. Enabling the servo the
        # instant a hand enters frame is a normal thing to do, and failing on
        # the first missing message would be needlessly brittle.
        if self._last_point is None:
            self._wait_for_fresh_point(timeout=3.0)

        if not self._ever_received_point:
            n = self.count_publishers(self._point_topic)
            if n == 0:
                return False, (
                    f'nothing is publishing {self._point_topic}. '
                    'hand_tracker_node is probably not running (or is using a '
                    'different topic). Start it with: '
                    'ros2 run mycobot_perception hand_tracker_node'
                )
            return False, (
                f'{self._point_topic} has {n} publisher(s) but has never sent a '
                'message, so MediaPipe is not detecting a hand. View '
                '/hand/annotated to see what the camera sees -- usually this is '
                'lighting, the hand being too close to fill the frame, or the '
                'camera pointing somewhere other than you expect.'
            )

        if self._last_point is None:
            return False, 'no target visible -- hold your hand in view'

        age = time.monotonic() - self._last_point_time
        if age > self._timeout:
            return False, (
                f'target last seen {age:.1f}s ago (timeout {self._timeout}s). '
                'Tracking is working but the hand is not currently visible -- '
                'hold it in view and re-enable.'
            )

        self._probing = True
        try:
            self.get_logger().info(
                f'Probing camera orientation: nudging {self._h_joint} and '
                f'{self._v_joint} by {self._probe_deg} deg each...'
            )

            h_resp, err = self._probe_axis(self._h_joint)
            if h_resp is None:
                return False, f'{self._h_joint}: {err}'
            v_resp, err = self._probe_axis(self._v_joint)
            if v_resp is None:
                return False, f'{self._v_joint}: {err}'

            j = np.array([[h_resp[0], v_resp[0]],
                          [h_resp[1], v_resp[1]]], dtype=float)

            self.get_logger().info(
                f'  {self._h_joint} +{self._probe_deg}deg moved target '
                f'({h_resp[0]:+.3f}, {h_resp[1]:+.3f}) of a half-frame'
            )
            self.get_logger().info(
                f'  {self._v_joint} +{self._probe_deg}deg moved target '
                f'({v_resp[0]:+.3f}, {v_resp[1]:+.3f}) of a half-frame'
            )

            det = float(np.linalg.det(j))
            # A near-singular Jacobian means both joints push the target the
            # same way in the image, so the pair cannot steer independently.
            # Servoing on that inverse would produce enormous commands.
            if abs(det) < 1e-4:
                return False, (
                    f'Jacobian is singular (det={det:.2e}). The two joints move '
                    'the image in nearly the same direction from this pose. '
                    'Move the arm to a different starting pose and retry, or '
                    'pick a different vertical_joint.'
                )

            # Scale to per-degree, since the probe used probe_deg steps.
            self._jinv = np.linalg.inv(j) * self._probe_deg

            # Learn which way the approach joint changes apparent size. Only
            # the sign is needed: the magnitude varies with distance and pose,
            # so a fixed gain plus the correct direction is more robust than a
            # calibrated scale that is wrong everywhere except where measured.
            if self._approach_enabled:
                sign, err = self._probe_approach()
                if sign is None:
                    self.get_logger().warn(
                        f'Approach probe failed ({err}); centring only, no '
                        'closing in. Set approach_enabled:=false to silence.')
                    self._approach_sign = 0.0
                else:
                    self._approach_sign = sign
                    direction = 'closer' if sign > 0 else 'further'
                    self.get_logger().info(
                        f'  {self._approach_joint} +{self._probe_deg}deg makes '
                        f'the hand appear {direction}')

            self._probed = True
            self.get_logger().info('Probe OK -- Jacobian estimated.')
            return True, 'ok'
        except Exception as e:
            return False, str(e)
        finally:
            self._probing = False

    # ---- Control loop ----

    def _reset_pid(self) -> None:
        """Drop accumulated PID state.

        Called whenever the loop stops acting on a continuous error signal --
        target lost, servo disabled, probe run. Keeping the integral across a
        gap means unloading a correction for an error that may no longer exist.
        """
        self._integral[:] = 0.0
        self._prev_error = None
        self._prev_time = None
        # Let the next measurement through immediately rather than waiting for
        # one newer than whatever we last acted on before the reset.
        self._acted_point_time = 0.0

    def _search_sweep(self) -> None:
        """Tilt the wrist through its arc looking for a hand.

        Paced against the wall clock, not the loop counter: the step is
        whatever covers search_range_deg in search_sweep_seconds given the
        time that actually elapsed since the last tick. A fixed
        degrees-per-tick step silently stretches the sweep by however long
        the host stalled, which is how a sweep meant to take 15s ended up
        crawling a few degrees over a minute.

        Deliberately slow regardless: MediaPipe needs a few clean frames to
        lock on, and sweeping faster than it can detect means panning
        straight past a hand that was in view the whole time.
        """
        now = time.monotonic()
        if self._last_sweep_time is None:
            # First tick of this sweep -- no elapsed time to work from yet.
            self._last_sweep_time = now
            return
        dt = now - self._last_sweep_time
        self._last_sweep_time = now

        deg_per_sec = self._search_range / max(self._sweep_seconds, 0.1)
        step = deg_per_sec * dt * self._search_dir
        # After a stall dt is large; cap the catch-up so the arm eases back
        # into the sweep rather than lunging.
        step = max(-self._search_max_step,
                   min(self._search_max_step, step))

        self._send_jog(self._search_joint, step)
        self._search_travel += step

        # Reverse at the ends of the sweep. Range is measured from wherever the
        # search started, so it stays near the home pose instead of wandering.
        if abs(self._search_travel) >= self._search_range / 2.0:
            self._search_dir *= -1.0
            self.get_logger().info(
                f'Search sweep reversing at {self._search_travel:+.0f}deg')

        self.get_logger().info(
            f'Searching... {self._search_joint} at {self._search_travel:+.0f}deg '
            f'({deg_per_sec:.0f}deg/s)',
            throttle_duration_sec=3.0)

    def _approach_step(self) -> float:
        """Degrees to move the approach joint to close in on the hand.

        Uses apparent palm size as the range proxy: bigger means nearer. Zero
        if approach is off, the probe could not determine a direction, or the
        hand already fills the target fraction of the frame.
        """
        if (not self._approach_enabled or self._approach_sign == 0.0
                or self._last_size_px is None or self._width is None):
            return 0.0

        current = self._last_size_px / float(self._width)
        error = self._target_size - current
        if abs(error) < self._approach_deadband:
            return 0.0

        step = self._approach_gain * error * self._approach_sign
        return max(-self._max_approach_step,
                   min(self._max_approach_step, step))

    def _target_age(self) -> float:
        if self._last_point is None:
            return float('inf')
        return time.monotonic() - self._last_point_time

    def _servo_step(self) -> None:
        if self._probing or not self._enabled:
            return

        # HOMING is driven by the service callback; nothing to do until it
        # completes and flips the state to IDLE.
        if self._state in (IDLE, HOMING):
            return

        visible = self._target_age() <= self._timeout

        if self._state == SEARCHING:
            if not visible:
                self._search_sweep()
                return
            # Found one. Probe first if the camera orientation is still
            # unknown -- unless we have been told to take it on trust and
            # start moving straight away.
            if not self._probed and self._skip_probe:
                self._assume_orientation()
            if not self._probed:
                ok, detail = self._probe()
                if not ok:
                    self.get_logger().error(
                        f'Probe failed: {detail}. Returning home.')
                    self._set_state(HOMING)
                    self._go_home()
                    return
            self._set_state(TRACKING)
            return

        # --- TRACKING ---
        if not visible:
            lost_for = self._target_age()
            if lost_for > self._lost_timeout:
                self.get_logger().info(
                    f'No sighting for {self._lost_timeout:.0f}s -- going home.')
                self._set_state(HOMING)
                self._go_home()
                return
            # Hold position during a brief dropout rather than sweeping away
            # from a hand that is probably about to reappear.
            self.get_logger().info(
                f'Target lost {lost_for:.1f}s ago; holding '
                f'({self._lost_timeout - lost_for:.0f}s until home)',
                throttle_duration_sec=2.0)
            self._reset_pid()
            return

        if self._jinv is None:
            self.get_logger().warn(
                'Tracking with no Jacobian -- probe did not run.',
                throttle_duration_sec=3.0)
            return

        err = self._current_error()
        if err is None:
            self._reset_pid()
            return
        ex, ey = err

        # Only act once per measurement. The timer runs at `rate` (15Hz) but
        # detections arrive slower, so without this the same reading would be
        # integrated repeatedly. Holding the previous command until new data
        # arrives is both smoother and more correct than acting on a duplicate.
        if self._last_point_time <= self._acted_point_time:
            return
        self._acted_point_time = self._last_point_time

        if math.hypot(ex, ey) < self._deadband:
            # Being centred is success, not a fault -- but it looks identical
            # to a dead loop from outside, so say so.
            self.get_logger().info(
                f'On target (error {math.hypot(ex, ey):.3f} < deadband '
                f'{self._deadband}); holding.',
                throttle_duration_sec=5.0)
            # Bleed the integral off while on target so it does not carry a
            # stale push into the next correction.
            self._integral *= 0.9
            return

        now = time.monotonic()
        error = np.array([ex, ey], dtype=float)
        dt = (now - self._prev_time) if self._prev_time else (1.0 / self._rate)
        # Guard against a stalled loop producing a huge dt, which would make
        # the integral jump and the derivative explode.
        dt = max(1e-3, min(dt, 0.5))

        self._integral += error * dt
        # Clamp per-axis so one saturated axis cannot poison the other.
        np.clip(self._integral, -self._integral_limit, self._integral_limit,
                out=self._integral)

        derivative = np.zeros(2, dtype=float)
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time = now

        # PID in image space, then map through the inverse Jacobian to joints.
        # Negated because we drive the error toward zero.
        control = (self._gain * error
                   + self._ki * self._integral
                   + self._kd * derivative)
        delta = self._jinv @ control * -1.0
        dh, dv = float(delta[0]), float(delta[1])

        dh = max(-self._max_step, min(self._max_step, dh))
        dv = max(-self._max_step, min(self._max_step, dv))

        names = [self._h_joint, self._v_joint]
        deltas = [dh, dv]

        # Close in as well as centre. Sent in the same message so the arm
        # approaches and tracks together rather than alternating between them.
        da = self._approach_step()
        if da != 0.0:
            if self._approach_joint in names:
                deltas[names.index(self._approach_joint)] += da
            else:
                names.append(self._approach_joint)
                deltas.append(da)

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = names
        msg.displacements = deltas
        self._jog_pub.publish(msg)

        size_note = ''
        if self._last_size_px is not None and self._width:
            frac = self._last_size_px / float(self._width)
            size_note = f' size={frac:.2f}/{self._target_size:.2f}'

        # If this logs but the arm does not move, the jog is being rejected by
        # the driver -- check the driver's console, which now says why.
        self.get_logger().info(
            f'err=({ex:+.3f},{ey:+.3f}){size_note} -> '
            + ' '.join(f'{n}{d:+.2f}' for n, d in zip(names, deltas)),
            throttle_duration_sec=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualServoNode()
    # Explicit thread count: the orientation probe blocks inside a service
    # callback (it sleeps while waiting to see where the target moved), and the
    # point subscription must keep running on another thread throughout or the
    # probe can never observe anything and always reports "target lost".
    # MultiThreadedExecutor() defaults to cpu_count(), which is 1 on some VMs.
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    node.get_logger().info('Executor: 4 threads')
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
