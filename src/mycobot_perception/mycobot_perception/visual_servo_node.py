"""
Image-based visual servoing: drive the arm so a tracked point stays centred.

This is the eye-in-hand demo that needs NO calibration. It never computes a 3D
position. It only knows "the target is 40 pixels left of centre" and turns that
into "move the joint that shifts the view right". Because the camera rides on
the end effector, image error maps directly to joint motion.

That means no camera intrinsics, no hand-eye transform, no depth estimate --
none of which exist yet. Those unlock better things later; this works today.

    ros2 run mycobot_perception hand_tracker_node
    ros2 run mycobot_perception visual_servo_node
    ros2 service call /arm/jog_enable std_srvs/srv/SetBool "{data: true}"

Nothing moves until that last call. Set data:false to stop.


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

If probing fails (no hand visible, arm blocked), the node refuses to servo
rather than falling back to a guess.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from control_msgs.msg import JointJog
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool


class VisualServoNode(Node):

    def __init__(self) -> None:
        super().__init__('visual_servo_node')

        self.declare_parameter('point_topic', '/hand/point_px')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('jog_topic', '/arm/jog')

        # Joints used to steer the view. joint1 swings the arm horizontally;
        # joint5 is the wrist pitch, which tilts the camera vertically. These
        # are a starting guess only -- the probe below measures what each one
        # actually does to the image and adapts.
        self.declare_parameter('horizontal_joint', 'joint1')
        self.declare_parameter('vertical_joint', 'joint5')

        # Proportional gain, in degrees of joint motion per unit of normalised
        # image error (error is in units of half-frames, so 1.0 = target at the
        # frame edge). Low by default: servo loops are much easier to tune down
        # from stable than up from oscillating.
        self.declare_parameter('gain', 2.5)
        # Ignore errors smaller than this (normalised). MediaPipe jitter and
        # your own hand tremor are both a few pixels; without a deadband the
        # arm hunts continuously and buzzes.
        self.declare_parameter('deadband', 0.04)
        self.declare_parameter('rate', 10.0)
        # Stop if the target has not been seen for this long. Without it, the
        # arm keeps acting on a stale position after the hand leaves frame.
        self.declare_parameter('target_timeout', 0.5)
        self.declare_parameter('max_step_deg', 2.0)

        # Probe settings. probe_deg is how far each joint is nudged to measure
        # its effect; big enough to produce clear image motion, small enough to
        # be a twitch.
        self.declare_parameter('auto_probe', True)
        self.declare_parameter('probe_deg', 4.0)
        self.declare_parameter('probe_settle', 1.2)

        self._h_joint = self.get_parameter('horizontal_joint').value
        self._v_joint = self.get_parameter('vertical_joint').value
        self._gain = float(self.get_parameter('gain').value)
        self._deadband = float(self.get_parameter('deadband').value)
        self._rate = float(self.get_parameter('rate').value)
        self._timeout = float(self.get_parameter('target_timeout').value)
        self._max_step = float(self.get_parameter('max_step_deg').value)
        self._auto_probe = bool(self.get_parameter('auto_probe').value)
        self._probe_deg = float(self.get_parameter('probe_deg').value)
        self._probe_settle = float(self.get_parameter('probe_settle').value)

        # Frame size, learned from the first image. Needed to normalise pixel
        # error; we do not assume 640x480.
        self._width: int | None = None
        self._height: int | None = None

        self._last_point: tuple[float, float] | None = None
        self._last_point_time = 0.0
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
        self._image_topic_name = self.get_parameter('image_topic').value

        self._point_sub = self.create_subscription(
            PointStamped, self._point_topic,
            self._point_cb, 1, callback_group=cb)
        self._image_sub = self.create_subscription(
            Image, self._image_topic_name,
            self._image_cb, 1, callback_group=cb)
        self._jog_pub = self.create_publisher(
            JointJog, self.get_parameter('jog_topic').value, 1)

        self._enabled = False
        self._enable_srv = self.create_service(
            SetBool, 'servo/enable', self._enable_cb, callback_group=cb)

        self._timer = self.create_timer(
            1.0 / self._rate, self._servo_step, callback_group=cb)

        self.get_logger().info(
            'Visual servo ready. It will probe the camera orientation on the '
            'first enable. Nothing moves until BOTH /servo/enable and '
            '/arm/jog_enable are true.'
        )

    # ---- Inputs ----

    def _image_cb(self, msg: Image) -> None:
        if self._width is None:
            self._width, self._height = msg.width, msg.height
            self.get_logger().info(f'Frame size: {msg.width}x{msg.height}')

    def _point_cb(self, msg: PointStamped) -> None:
        if not self._ever_received_point:
            self.get_logger().info(
                f'First target sighting on {self._point_topic} -- tracking is live.')
            self._ever_received_point = True
        self._last_point = (msg.point.x, msg.point.y)
        self._last_point_time = time.monotonic()

    def _enable_cb(self, request, response):
        want = bool(request.data)
        if want and not self._probed:
            if not self._auto_probe:
                response.success = False
                response.message = (
                    'auto_probe is off and no Jacobian is set; refusing to '
                    'servo with an unknown camera orientation.'
                )
                return response
            ok, detail = self._probe()
            if not ok:
                response.success = False
                response.message = f'Probe failed: {detail}'
                self.get_logger().error(f'Probe failed: {detail}')
                return response

        self._enabled = want
        response.success = True
        response.message = f'Servo {"enabled" if want else "disabled"}'
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

    def _probe(self):
        """Measure the image Jacobian by moving each joint and watching.

        Builds J where [dx, dy]^T = J @ [d_horizontal, d_vertical]^T, then
        inverts it so the control law can go the other way.
        """
        if self._probing:
            return False, 'already probing'

        if self._width is None:
            n = self.count_publishers(self._image_topic_name)
            return False, (
                f'no image received on {self._image_topic_name} '
                f'({n} publisher(s) detected). '
                + ('Is camera_node running?' if n == 0 else
                   'A publisher exists but no frames arrived -- check '
                   f'`ros2 topic hz {self._image_topic_name}`.')
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
            self._probed = True
            self.get_logger().info('Probe OK -- Jacobian estimated.')
            return True, 'ok'
        except Exception as e:
            return False, str(e)
        finally:
            self._probing = False

    # ---- Control loop ----

    def _servo_step(self) -> None:
        if self._probing:
            return
        if not self._enabled:
            return
        if self._jinv is None:
            self.get_logger().warn(
                'Enabled but no Jacobian -- the orientation probe has not run '
                'successfully. Disable and re-enable to retry.',
                throttle_duration_sec=3.0)
            return

        err = self._current_error()
        if err is None:
            # Enabled and expected to be working, so say why nothing happens
            # rather than sitting there quietly doing nothing.
            if self._width is None:
                reason = 'no image received yet'
            elif self._last_point is None:
                reason = 'no target has ever been seen'
            else:
                age = time.monotonic() - self._last_point_time
                reason = (f'target last seen {age:.1f}s ago, over the '
                          f'{self._timeout}s timeout')
            self.get_logger().warn(
                f'Not servoing: {reason}', throttle_duration_sec=3.0)
            return
        ex, ey = err

        if math.hypot(ex, ey) < self._deadband:
            # Being centred is success, not a fault -- but it looks identical
            # to a dead loop from outside, so say so.
            self.get_logger().info(
                f'On target (error {math.hypot(ex, ey):.3f} < deadband '
                f'{self._deadband}); holding. Move the target off-centre to '
                'see motion.',
                throttle_duration_sec=5.0)
            return

        # Drive the error to zero: joint delta = -Jinv @ error.
        delta = self._jinv @ np.array([ex, ey], dtype=float) * -self._gain
        dh, dv = float(delta[0]), float(delta[1])

        dh = max(-self._max_step, min(self._max_step, dh))
        dv = max(-self._max_step, min(self._max_step, dv))

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [self._h_joint, self._v_joint]
        msg.displacements = [dh, dv]
        self._jog_pub.publish(msg)

        # If this logs but the arm does not move, the jog is being rejected by
        # the driver -- check the driver's console, which now says why.
        self.get_logger().info(
            f'err=({ex:+.3f},{ey:+.3f}) -> {self._h_joint}{dh:+.2f}deg '
            f'{self._v_joint}{dv:+.2f}deg',
            throttle_duration_sec=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualServoNode()
    executor = rclpy.executors.MultiThreadedExecutor()
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
