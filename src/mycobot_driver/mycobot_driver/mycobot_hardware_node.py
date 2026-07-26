"""
ROS 2 driver node for myCobot 280 Pi.

Connects to the robot via pymycobot TCP socket and bridges to ROS 2:
  - Publishes /joint_states at a configurable rate (6 arm joints)
  - Provides a FollowJointTrajectory action server for MoveIt2 arm planning
  - Provides a fixed-pose homing service at /arm/home (SetBool)

The gripper has been removed for the elevator-button task. This must stay in
sync with show_gripper:=false in the URDF, the gripper-free SRDF, and the
gripper_controller removal in moveit_controllers.yaml — publishing a joint the
model does not define (or omitting one it does) breaks MoveIt's state monitor.

Server.py on the Pi accepts a single client, so this one TCP connection
carries everything.
"""

import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from std_srvs.srv import SetBool

from pymycobot import MyCobot280Socket


class MyCobotHardwareNode(Node):

    JOINT_NAMES = [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
    ]

    def __init__(self):
        super().__init__('mycobot_hardware_node')

        self.declare_parameter('robot_ip', '192.168.1.169')
        self.declare_parameter('robot_port', 9000)
        # NOTE ON LINK BUDGET: Server.py on the Pi accepts a single client and
        # blocks up to 100ms (its read() wait_time) on any command in its
        # has_return table. get_angles (0x20) and get_gripper_value (0x65) are
        # both in that table; send_angles (0x22) is not. So every joint-state
        # cycle that polls both can cost ~200ms of the *only* connection we
        # have, while motion commands are nearly free but must queue behind
        # them on self._lock. Polling at 20Hz oversubscribed the link ~4x and
        # was the real cause of stuttering motion. Keep this rate modest.
        self.declare_parameter('publish_rate', 10.0)
        # While a trajectory is executing, joint-state polling is throttled to
        # this rate so motion commands own the link. MoveIt's state monitor
        # tolerates a slower feed far better than the arm tolerates late
        # position commands.
        self.declare_parameter('publish_rate_during_motion', 2.0)
        self.declare_parameter('default_speed', 80)

        # --- Trajectory streaming ---
        # Interval between send_angles() commands during execution. This is the
        # knob that trades smoothness against link load: too fast and commands
        # queue up behind each other, too slow and motion visibly steps.
        # ~20ms of serial write + WiFi RTT means 60ms is a safe starting point.
        self.declare_parameter('command_interval', 0.06)
        # How far ahead of the current schedule position to aim, in seconds.
        # Commanding the arm's *current* scheduled pose makes it perpetually
        # chase a target it has already reached, which reads as stop-start.
        # Aiming slightly ahead keeps a moving target in front of it, which is
        # what produces continuous motion.
        self.declare_parameter('lookahead', 0.12)
        # Speed passed to send_angles during streaming. Lower than a point-to-
        # point move: each command is a small step along the path, so the arm
        # does not need to sprint to it.
        self.declare_parameter('trajectory_speed', 60)
        # Grace period after the planned trajectory end to let the arm settle
        # before we report success.
        self.declare_parameter('settle_timeout', 2.0)
        self.declare_parameter('settle_tolerance_deg', 4.0)

        # --- Homing ---
        # Fixed joint-angle home pose, in degrees.
        # Kept in sync with the "home" group_state in mycobot_280pi.srdf
        # ([0, 1.5708, -1.5708, -1.5708, 0, 0] rad) — change both together or
        # RViz's named "home" and this service will disagree.
        self.declare_parameter(
            'home_angles_deg', [0.0, 90.0, -90.0, -90.0, 0.0, 0.0]
        )
        # Homing runs slower than normal motion on purpose: it is commanded
        # from an arbitrary unknown starting pose, which makes it the single
        # most dangerous move the arm makes.
        self.declare_parameter('home_speed', 30)
        self.declare_parameter('home_timeout', 15.0)

        ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        port = self.get_parameter('robot_port').get_parameter_value().integer_value
        self._rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self._speed = self.get_parameter('default_speed').get_parameter_value().integer_value
        self._motion_rate = self.get_parameter(
            'publish_rate_during_motion').get_parameter_value().double_value
        self._cmd_interval = self.get_parameter('command_interval').get_parameter_value().double_value
        self._lookahead = self.get_parameter('lookahead').get_parameter_value().double_value
        self._traj_speed = self.get_parameter('trajectory_speed').get_parameter_value().integer_value
        self._settle_timeout = self.get_parameter('settle_timeout').get_parameter_value().double_value
        self._settle_tol = self.get_parameter('settle_tolerance_deg').get_parameter_value().double_value

        self._home_angles = list(
            self.get_parameter('home_angles_deg').get_parameter_value().double_array_value
        )
        self._home_speed = self.get_parameter('home_speed').get_parameter_value().integer_value
        self._home_timeout = self.get_parameter('home_timeout').get_parameter_value().double_value
        if len(self._home_angles) != 6:
            self.get_logger().warn(
                f'home_angles_deg has {len(self._home_angles)} entries, expected 6. '
                'Falling back to all-zeros.'
            )
            self._home_angles = [0.0] * 6

        # Counts joint-state cycles so polling can be throttled during motion.
        self._js_cycle = 0
        # Set while a trajectory (or homing) is executing. Joint-state polling
        # backs off to _motion_rate so the single TCP link stays available for
        # motion commands.
        self._in_motion = threading.Event()

        self.get_logger().info(f'Connecting to myCobot at {ip}:{port}')
        self._mc = MyCobot280Socket(ip, port)
        time.sleep(0.5)

        try:
            if self._mc.get_fresh_mode() != 1:
                self._mc.set_fresh_mode(1)
                self.get_logger().info('Set fresh mode (responsive movement)')
        except Exception as e:
            self.get_logger().warn(f'Could not set fresh mode: {e}')

        self._lock = threading.Lock()

        # Joint state publisher
        self._js_pub = self.create_publisher(JointState, 'joint_states', 10)
        self._timer = self.create_timer(1.0 / self._rate, self._publish_joint_states)

        # Each callback group gets its own thread in MultiThreadedExecutor,
        # preventing the 20Hz timer from starving the service/action callbacks.
        action_cb_group = ReentrantCallbackGroup()
        service_cb_group = MutuallyExclusiveCallbackGroup()

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'arm_controller/follow_joint_trajectory',
            execute_callback=self._execute_trajectory,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=action_cb_group,
        )

        # Homing. Deliberately a Trigger-style SetBool rather than something
        # that runs at startup: the move to home starts from an arbitrary
        # unknown pose, so it should be an explicit operator decision, not a
        # side effect of launching the driver.
        self._home_srv = self.create_service(
            SetBool, 'arm/home', self._home_callback,
            callback_group=service_cb_group,
        )

        self.get_logger().info(
            'myCobot hardware node ready (arm trajectory + homing service)'
        )

    # ---- Joint State Publisher ----

    def _read_angles_rad(self):
        """Read current joint angles from the robot, returns radians or None."""
        try:
            with self._lock:
                angles_deg = self._mc.get_angles()
            if not isinstance(angles_deg, list) or len(angles_deg) != 6:
                return None
            return [math.radians(a) for a in angles_deg]
        except Exception:
            return None

    def _publish_joint_states(self):
        # Back off hard while the arm is moving. get_angles() is a blocking
        # round-trip on the same single-client socket the motion commands use;
        # stealing that bandwidth mid-trajectory is what makes motion stutter.
        self._js_cycle += 1
        if self._in_motion.is_set():
            stride = max(1, int(round(self._rate / max(self._motion_rate, 0.1))))
            if self._js_cycle % stride:
                return

        angles = self._read_angles_rad()
        if angles is None:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.JOINT_NAMES)
        msg.position = angles
        self._js_pub.publish(msg)

    # ---- FollowJointTrajectory Action ----

    def _goal_callback(self, goal_request):
        self.get_logger().info('Received trajectory goal')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def _wait_until_reached(self, target_deg, tolerance_deg=5.0, timeout=8.0):
        """Poll joint angles until robot reaches target or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with self._lock:
                    current = self._mc.get_angles()
                if isinstance(current, list) and len(current) == 6:
                    max_err = max(abs(c - t) for c, t in zip(current, target_deg))
                    if max_err < tolerance_deg:
                        return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    @staticmethod
    def _point_time(point) -> float:
        """time_from_start of a trajectory point, in seconds."""
        return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9

    def _sample_trajectory(self, points, t: float):
        """Linearly interpolate the planned trajectory at time t (seconds).

        MoveIt hands us a time-parameterized path; sampling it on a wall clock
        is what lets the arm follow the *plan's* velocity profile instead of
        whatever pace we happen to issue commands at. Returns positions in
        radians, clamped to the trajectory's endpoints.
        """
        if t <= self._point_time(points[0]):
            return list(points[0].positions)
        last = points[-1]
        if t >= self._point_time(last):
            return list(last.positions)

        # Trajectories are short (hundreds of points at most) and this runs at
        # ~16Hz, so a linear scan is not worth optimizing away.
        for i in range(1, len(points)):
            t1 = self._point_time(points[i])
            if t <= t1:
                t0 = self._point_time(points[i - 1])
                span = t1 - t0
                if span <= 0.0:
                    return list(points[i].positions)
                a = (t - t0) / span
                p0 = points[i - 1].positions
                p1 = points[i].positions
                return [q0 + a * (q1 - q0) for q0, q1 in zip(p0, p1)]
        return list(last.positions)

    def _execute_trajectory(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        feedback_msg = FollowJointTrajectory.Feedback()
        points = trajectory.points
        n = len(points)

        if n == 0:
            goal_handle.succeed()
            return FollowJointTrajectory.Result()

        duration = self._point_time(points[-1])
        if duration <= 0.0:
            # Untimed trajectory (some planners emit these) — nothing to pace
            # against, so just command the endpoint.
            final_deg = [math.degrees(p) for p in points[-1].positions]
            self._in_motion.set()
            try:
                with self._lock:
                    self._mc.send_angles(final_deg, self._speed)
                self._wait_until_reached(
                    final_deg, tolerance_deg=self._settle_tol,
                    timeout=self._settle_timeout,
                )
            finally:
                self._in_motion.clear()
            goal_handle.succeed()
            return FollowJointTrajectory.Result()

        self.get_logger().info(
            f'Executing trajectory: {n} points over {duration:.2f}s, '
            f'streaming every {self._cmd_interval * 1000:.0f}ms '
            f'(lookahead {self._lookahead * 1000:.0f}ms)'
        )

        # Suppress the joint-state polling storm for the duration of the move.
        self._in_motion.set()
        result = FollowJointTrajectory.Result()
        start = time.monotonic()
        next_cmd = start

        try:
            while True:
                now = time.monotonic()
                elapsed = now - start

                if goal_handle.is_cancel_requested:
                    # Stop where we are rather than continuing to the goal.
                    try:
                        with self._lock:
                            self._mc.stop()
                    except Exception as e:
                        self.get_logger().warn(f'stop() failed on cancel: {e}')
                    goal_handle.canceled()
                    self.get_logger().info('Trajectory canceled')
                    return result

                if elapsed >= duration:
                    break

                # Aim slightly ahead of where the schedule says we should be.
                # Without this the arm is always commanded to a pose it has
                # effectively already reached, so it decelerates into every
                # command — the stop-start feel we are trying to eliminate.
                target_rad = self._sample_trajectory(points, elapsed + self._lookahead)
                target_deg = [math.degrees(p) for p in target_rad]

                try:
                    with self._lock:
                        self._mc.send_angles(target_deg, self._traj_speed)
                except Exception as e:
                    # A dropped command is recoverable: the next one is only
                    # command_interval away and supersedes it anyway.
                    self.get_logger().warn(f'send_angles failed mid-stream: {e}')

                next_cmd += self._cmd_interval
                sleep_for = next_cmd - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # We fell behind schedule (link congestion). Resync rather
                    # than accumulating drift and firing a burst of commands.
                    next_cmd = time.monotonic()

            # Command the exact endpoint, then let the arm settle into it.
            final_deg = [math.degrees(p) for p in points[-1].positions]
            try:
                with self._lock:
                    self._mc.send_angles(final_deg, self._traj_speed)
            except Exception as e:
                self.get_logger().error(f'Failed to send final angles: {e}')

            reached = self._wait_until_reached(
                final_deg,
                tolerance_deg=self._settle_tol,
                timeout=self._settle_timeout,
            )
            if not reached:
                self.get_logger().warn(
                    f'Arm did not settle within {self._settle_tol}deg '
                    f'after {self._settle_timeout}s'
                )
        finally:
            self._in_motion.clear()

        # Publish one feedback sample at the end. Publishing per-command during
        # the stream would mean an extra blocking get_angles() round-trip on
        # every cycle, which is exactly the contention we removed.
        current_angles = self._read_angles_rad()
        if current_angles:
            feedback_msg.actual.positions = current_angles
            feedback_msg.desired.positions = list(points[-1].positions)
            feedback_msg.error.positions = [
                d - a for d, a in zip(points[-1].positions, current_angles)
            ]
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        self.get_logger().info('Trajectory execution complete')
        return result

    # ---- Homing ----

    def _home_callback(self, request, response):
        """Move the arm to the configured fixed home pose.

        SetBool is used only for its trigger-with-result shape; request.data is
        ignored. Moves at home_speed (slower than normal motion) because this
        command is issued from an arbitrary starting pose and may sweep a large
        part of the workspace.

        This is a joint-space move, NOT a re-zeroing of the servos. It assumes
        the arm's encoders are already trustworthy. If the arm has lost its
        reference, this will confidently drive to the wrong place.
        """
        target = list(self._home_angles)
        self.get_logger().info(f'Homing to {target} at speed {self._home_speed}')

        self._in_motion.set()
        try:
            with self._lock:
                self._mc.send_angles(target, self._home_speed)
            reached = self._wait_until_reached(
                target,
                tolerance_deg=self._settle_tol,
                timeout=self._home_timeout,
            )
        except Exception as e:
            response.success = False
            response.message = f'Homing failed: {e}'
            self.get_logger().error(response.message)
            return response
        finally:
            self._in_motion.clear()

        response.success = bool(reached)
        if reached:
            response.message = f'Homed to {target}'
            self.get_logger().info('Homing complete')
        else:
            response.message = (
                f'Homing timed out after {self._home_timeout}s; '
                f'arm did not reach {target} within {self._settle_tol}deg'
            )
            self.get_logger().warn(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MyCobotHardwareNode()
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
