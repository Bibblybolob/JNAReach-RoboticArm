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
import os
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointJog
from std_srvs.srv import SetBool

from pymycobot import MyCobot280Socket


class MyCobotHardwareNode(Node):

    JOINT_NAMES = [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
    ]

    def __init__(self):
        super().__init__('mycobot_hardware_node')

        # Same MYCOBOT_IP convention the launch files use, so running this
        # node directly with `ros2 run` picks up the same address instead of
        # a stale literal. Accepts a hostname as readily as an IP -- see the
        # note on mDNS in the README, since the Pi's DHCP address moves.
        self.declare_parameter(
            'robot_ip', os.environ.get('MYCOBOT_IP', '192.168.0.15'))
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
        # Fallback speed for streaming if adaptive speed is disabled.
        self.declare_parameter('trajectory_speed', 60)
        # Scale send_angles speed to the size of each step.
        #
        # A fixed speed is wrong for streaming: send_angles(target, speed) is a
        # point-to-point command, so a constant 60 tells the arm to sprint to a
        # target only ~60ms away, arrive, stop, and wait for the next one. That
        # micro stop-start is felt as jerk even when commands arrive on time.
        # Instead, pick the speed that covers this step in roughly the time
        # until the next command, so the arm is still moving when it arrives.
        self.declare_parameter('adaptive_speed', True)
        # Degrees/second the fastest joint achieves at send_angles speed=100.
        # THIS IS A GUESS AND SHOULD BE MEASURED — see scripts/measure_arm.py.
        # Too high makes every step under-speed (arm lags, motion drags);
        # too low makes it over-speed (arm sprints and stops = jerk).
        self.declare_parameter('speed_at_100_deg_s', 120.0)
        # Command slightly more speed than strictly needed so the arm leads
        # rather than trails the schedule. 1.0 = exact, 1.3 = 30% margin.
        self.declare_parameter('speed_headroom', 1.3)
        self.declare_parameter('min_speed', 15)
        self.declare_parameter('max_speed', 100)
        # Grace period after the planned trajectory end to let the arm settle
        # before we report success.
        self.declare_parameter('settle_timeout', 2.0)
        self.declare_parameter('settle_tolerance_deg', 4.0)

        # --- Homing ---
        # Fixed joint-angle home pose, in degrees.
        # Kept in sync with the "home" group_state in mycobot_280pi.srdf
        # ([0, 1.5708, -1.5708, 0, 0, 0] rad) — change both together or
        # RViz's named "home" and this service will disagree.
        self.declare_parameter(
            'home_angles_deg', [0.0, 90.0, -90.0, 0.0, 0.0, 0.0]
        )
        # Homing runs slower than normal motion on purpose: it is commanded
        # from an arbitrary unknown starting pose, which makes it the single
        # most dangerous move the arm makes.
        self.declare_parameter('home_speed', 30)
        self.declare_parameter('home_timeout', 15.0)
        # Drive to home once, shortly after connecting, so the arm always
        # starts from a known pose instead of wherever it was left.
        #
        # Off by default here on purpose: this moves the arm from an
        # arbitrary unknown position as a side effect of starting a node,
        # which is not something a driver should decide on its own. The
        # bringup launch turns it on, because there the operator is
        # deliberately starting the whole stack and expects the arm to
        # settle at home before anything else happens.
        self.declare_parameter('home_on_start', False)

        # --- Jogging (visual servoing) ---
        # Largest displacement honoured in a single JointJog, in degrees. A
        # spurious detection should nudge the arm, not fling it.
        self.declare_parameter('max_jog_deg', 3.0)
        self.declare_parameter('jog_speed', 40)
        # Size each jog's speed to the size of that jog, exactly as trajectory
        # streaming does. A fixed speed is wrong here for the same reason it is
        # wrong there: send_angles is point-to-point, so commanding speed 40 to
        # a target 0.5deg away makes the arm sprint the gap, stop, and wait for
        # the next command ~60ms later. At servo rates that micro stop-start is
        # the dominant source of visible jerk. Set false to go back to a fixed
        # jog_speed.
        self.declare_parameter('adaptive_jog_speed', True)
        # Seconds of no jogging before the jog base is resynced from a hardware
        # read. Jogs chain off the *commanded* pose so they compound; a
        # measurement taken while the arm is still travelling to the last
        # target is behind it, and adopting that as the new base throws the
        # accumulated lead away. Do that a few times a second and the arm
        # creeps instead of sweeping -- the servo believes it commanded 60deg
        # of search travel while the joint moves ~9deg. Only resync once
        # jogging has actually stopped and the arm has settled.
        self.declare_parameter('jog_resync_after', 0.5)
        # ...but never let the commanded pose run away from reality. If the
        # arm is blocked or saturated it will not reach the target, and
        # without this the base would keep advancing past where the arm can
        # physically go. Beyond this divergence, trust the hardware.
        self.declare_parameter('jog_max_divergence_deg', 10.0)
        # Per-joint travel in degrees, used to clamp jog targets. Defaults
        # match the URDF limits for the myCobot 280 Pi. Holding a servo against
        # its hard stop damages it, so this clamp is about the hardware, not
        # about being conservative with the workspace.
        self.declare_parameter('joint_limits_deg', [
            -168.0, 168.0,
            -140.0, 140.0,
            -150.0, 150.0,
            -150.0, 150.0,
            -155.0, 160.0,
            -180.0, 180.0,
        ])

        ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        port = self.get_parameter('robot_port').get_parameter_value().integer_value
        self._rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self._speed = self.get_parameter('default_speed').get_parameter_value().integer_value
        self._motion_rate = self.get_parameter(
            'publish_rate_during_motion').get_parameter_value().double_value
        self._cmd_interval = self.get_parameter('command_interval').get_parameter_value().double_value
        self._lookahead = self.get_parameter('lookahead').get_parameter_value().double_value
        self._traj_speed = self.get_parameter('trajectory_speed').get_parameter_value().integer_value
        self._adaptive_speed = self.get_parameter('adaptive_speed').get_parameter_value().bool_value
        self._speed_at_100 = self.get_parameter('speed_at_100_deg_s').get_parameter_value().double_value
        self._speed_headroom = self.get_parameter('speed_headroom').get_parameter_value().double_value
        self._min_speed = self.get_parameter('min_speed').get_parameter_value().integer_value
        self._max_speed = self.get_parameter('max_speed').get_parameter_value().integer_value
        self._settle_timeout = self.get_parameter('settle_timeout').get_parameter_value().double_value
        self._settle_tol = self.get_parameter('settle_tolerance_deg').get_parameter_value().double_value

        self._home_angles = list(
            self.get_parameter('home_angles_deg').get_parameter_value().double_array_value
        )
        self._home_speed = self.get_parameter('home_speed').get_parameter_value().integer_value
        self._home_timeout = self.get_parameter('home_timeout').get_parameter_value().double_value
        self._home_on_start = self.get_parameter(
            'home_on_start').get_parameter_value().bool_value
        self._start_homed = False
        if len(self._home_angles) != 6:
            self.get_logger().warn(
                f'home_angles_deg has {len(self._home_angles)} entries, expected 6. '
                'Falling back to all-zeros.'
            )
            self._home_angles = [0.0] * 6

        # Counts joint-state cycles so polling can be throttled during motion.
        self._js_cycle = 0
        # Set while a trajectory (or homing) is executing. While set, the
        # joint-state timer stops touching the hardware entirely.
        self._in_motion = threading.Event()
        # Most recent commanded joint positions (radians), published in place
        # of a hardware read while _in_motion is set. See _publish_joint_states.
        self._cmd_positions = None
        # Last measured joint angles (radians), refreshed by the joint-state
        # timer. Jogging applies its deltas to this rather than paying a
        # blocking read per command.
        self._last_angles_rad = None
        self._last_jog_time = 0.0
        # Monotonic time of the last command actually put on the socket. The
        # keepalive timer uses this to tell "quiet because nothing needs
        # sending" from "quiet because the link is genuinely unused".
        self._last_tx_time = 0.0
        # Jogging is off until explicitly enabled via /arm/jog_enable.
        self._jog_enabled = False

        self._max_jog_deg = self.get_parameter('max_jog_deg').get_parameter_value().double_value
        self._jog_speed = self.get_parameter('jog_speed').get_parameter_value().integer_value
        self._adaptive_jog_speed = self.get_parameter(
            'adaptive_jog_speed').get_parameter_value().bool_value
        self._jog_resync_after = self.get_parameter(
            'jog_resync_after').get_parameter_value().double_value
        self._jog_max_divergence = self.get_parameter(
            'jog_max_divergence_deg').get_parameter_value().double_value
        flat = list(
            self.get_parameter('joint_limits_deg').get_parameter_value().double_array_value
        )
        if len(flat) != 12:
            self.get_logger().warn(
                f'joint_limits_deg has {len(flat)} entries, expected 12 '
                '(lo,hi per joint). Falling back to +/-160 for all joints.'
            )
            flat = [v for _ in range(6) for v in (-160.0, 160.0)]
        self._joint_limits_deg = [(flat[i], flat[i + 1]) for i in range(0, 12, 2)]

        self._ip, self._port = ip, port
        self._lock = threading.Lock()

        # The connection is attempted here but NOT allowed to kill the node.
        # Previously MyCobot280Socket() was constructed inline, so an
        # unreachable arm raised straight out of __init__ and the node died
        # before advertising anything. Every client then waited forever on
        # services that would never exist -- with no clue that the real fault
        # was one unreachable IP. Coming up disconnected and retrying makes the
        # failure visible and recoverable without a restart.
        self._mc = None
        self._connect()

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

        # Jogging: relative joint moves for visual servoing. Gated behind
        # /arm/jog_enable so a running servo node cannot move the arm until
        # someone deliberately turns it on.
        self._jog_sub = self.create_subscription(
            JointJog, 'arm/jog', self._jog_callback, 1,
            callback_group=action_cb_group,
        )
        self._jog_enable_srv = self.create_service(
            SetBool, 'arm/jog_enable', self._jog_enable_callback,
            callback_group=service_cb_group,
        )

        # Reconnect in the background if the arm was unreachable at startup or
        # drops out later, so a power cycle of the Pi does not require
        # restarting the whole stack.
        self._reconnect_timer = self.create_timer(
            5.0, self._retry_connect, callback_group=service_cb_group,
        )

        # Shares service_cb_group (MutuallyExclusive) so it can never re-enter
        # itself or run concurrently with the reconnect timer.
        self._keepalive_timer = self.create_timer(
            2.0, self._keepalive, callback_group=service_cb_group,
        )

        if self._home_on_start:
            # A timer rather than a call in __init__: homing blocks for up to
            # home_timeout, and the arm may not even be connected yet. This
            # waits for a connection, homes once, then cancels itself.
            self._start_home_timer = self.create_timer(
                1.0, self._home_on_start_once,
                callback_group=service_cb_group,
            )

        if self._mc is None:
            self.get_logger().warn(
                'myCobot hardware node ready but NOT CONNECTED. Services and '
                'topics are advertised (so clients will not hang waiting for '
                'them), and it will keep retrying the arm every 5s.'
            )
        else:
            self.get_logger().info(
                'myCobot hardware node ready (arm trajectory + homing + jog)'
            )

    # ---- Connection ----

    def _connect(self) -> bool:
        """Try to open the socket to the arm. Never raises."""
        try:
            self.get_logger().info(
                f'Connecting to myCobot at {self._ip}:{self._port} ...')
            mc = MyCobot280Socket(self._ip, self._port)
            time.sleep(0.5)
            # Prove the link works rather than trusting that constructing the
            # socket succeeded; a dead server can still accept a connection.
            # The very first command on a fresh socket commonly comes back -1
            # even on a healthy arm: server.py accepts the TCP connection
            # before its serial link to the arm's controller has necessarily
            # synced, so a few retries here save the 5s wait for the next
            # reconnect timer tick on what is usually just a cold-start blip.
            angles = None
            for attempt in range(5):
                angles = mc.get_angles()
                if (isinstance(angles, list) and len(angles) == 6
                        and self._angles_plausible(angles)):
                    break
                time.sleep(0.3)
            if not isinstance(angles, list) or len(angles) != 6:
                raise RuntimeError(
                    f'connected but get_angles() returned {angles!r}')
            if not self._angles_plausible(angles):
                raise RuntimeError(
                    f'connected but get_angles() returned {angles!r}, which '
                    f'is not a physically possible pose')
            # Being parked outside the configured travel is worth saying, but
            # it is not a reason to refuse the arm -- see _angles_plausible.
            self._warn_if_outside_limits(angles)
            self._mc = mc
            self.get_logger().info(f'Connected. Joint angles: '
                                   f'{[round(a, 1) for a in angles]}')
            try:
                if self._mc.get_fresh_mode() != 1:
                    self._mc.set_fresh_mode(1)
                    self.get_logger().info('Set fresh mode (responsive movement)')
            except Exception as e:
                self.get_logger().warn(f'Could not set fresh mode: {e}')
            return True
        except Exception as e:
            self._mc = None
            # Throttled: the reconnect timer retries every 5s, and repeating
            # this six-line block that often buries everything else in the
            # log and makes the terminal crawl. The first one prints
            # immediately; after that it is a reminder, not news.
            self.get_logger().error(
                f'Cannot reach the arm at {self._ip}:{self._port} -- {e}\n'
                '  The node is running but every command will be refused '
                'until this is fixed. Check that:\n'
                f'    - the Pi is powered and on the network (ping {self._ip})\n'
                '    - server.py is running on it (port 9000)\n'
                '    - nothing else holds the connection; Server.py accepts '
                'ONE client, so a stray script or a second driver locks it out',
                throttle_duration_sec=30.0,
            )
            return False

    def _retry_connect(self) -> None:
        if self._mc is None:
            self._connect()

    def _keepalive(self) -> None:
        """Keep the TCP link warm so the Pi never sees it as idle.

        server.py closes a client that has said nothing for CLIENT_IDLE_TIMEOUT
        seconds, which is correct for a dead peer but fires on a merely stalled
        one too. The joint-state timer normally keeps the link busy, but it
        backs off hard while jogging is armed (see _publish_joint_states), and
        the whole process can stall on a loaded host -- so there are real
        windows where nothing is sent for long enough to be disconnected.

        A cheap periodic read closes those windows. This is belt-and-braces
        alongside the raised timeout on the server: if the host stalls badly
        enough, this timer stalls with it and only the server-side change
        saves the connection.
        """
        if self._mc is None or self._in_motion.is_set():
            return
        if time.monotonic() - self._last_tx_time < 5.0:
            return

        # Never block a jog waiting for the link. If the lock is held, the
        # socket is busy by definition, which is exactly the state this timer
        # exists to create -- so there is nothing to do.
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._mc.get_angles()
            self._last_tx_time = time.monotonic()
        except OSError as e:
            self._handle_link_error(e, 'keepalive')
        except Exception as e:
            self.get_logger().warn(
                f'keepalive read failed: {e}', throttle_duration_sec=10.0)
        finally:
            self._lock.release()

    # No joint on this arm can reach beyond this, so anything past it is a
    # decoding failure rather than a pose. Kept well clear of joint6's +/-180
    # so a real reading is never mistaken for garbage.
    SANITY_LIMIT_DEG = 200.0

    def _home_on_start_once(self) -> None:
        """Drive to home once, as soon as the arm is actually reachable.

        Waits rather than giving up if the arm is not connected yet -- at
        startup the driver may still be retrying, and homing is exactly what
        should happen the moment it succeeds.
        """
        if self._start_homed or not rclpy.ok():
            self._start_home_timer.cancel()
            return
        if self._mc is None:
            return  # not connected yet; try again on the next tick

        self._start_homed = True
        self._start_home_timer.cancel()
        self.get_logger().info(
            f'Homing on startup to {self._home_angles} '
            '(set home_on_start:=false to skip)')
        # Reuse the service path so startup homing and /arm/home cannot drift
        # apart in behaviour.
        self._home_callback(SetBool.Request(), SetBool.Response())

    def _angles_plausible(self, angles_deg) -> bool:
        """Reject a reading that cannot be a real pose at all.

        This catches a garbled response that still decodes to six floats. It
        deliberately does NOT enforce joint_limits_deg: those are two
        different jobs, and conflating them is a mistake that cost a session.

        An arm parked slightly outside the configured travel is reporting the
        truth -- it happens after being moved by hand while released, or when
        the configured limits are simply tighter than the hardware's real
        range. Treating that as a bad read refused to connect at all, which
        strands the driver with no way to command the arm back, even though
        the jog clamp would have walked it into range on the first move.
        """
        return all(
            -self.SANITY_LIMIT_DEG <= a <= self.SANITY_LIMIT_DEG
            for a in angles_deg
        )

    def _warn_if_outside_limits(self, angles_deg) -> None:
        """Say so when the arm sits outside its configured travel.

        Worth knowing -- it means joint_limits_deg disagrees with reality, and
        jog targets on that joint will be clamped back into range -- but it is
        information, not a fault, so it never blocks anything.
        """
        outside = [
            f'joint{i + 1}={a:.1f} (limit {lo:.0f}..{hi:.0f})'
            for i, (a, (lo, hi)) in enumerate(
                zip(angles_deg, self._joint_limits_deg))
            if not (lo <= a <= hi)
        ]
        if outside:
            self.get_logger().warn(
                'Arm is outside its configured travel: ' + ', '.join(outside)
                + '. Jogs will clamp these joints back into range.',
                throttle_duration_sec=30.0)

    def _handle_link_error(self, exc: Exception, context: str) -> None:
        """Mark the arm disconnected after a socket-level failure.

        pymycobot resurfaces raw socket exceptions (BrokenPipeError,
        ConnectionResetError, ...) from its own send()/recv(). Once one of
        those fires, the underlying socket is unusable for anything else --
        retrying on the same object just reproduces the identical error on
        every subsequent call. Without this, that meant every jog and every
        joint-state poll (10-16Hz) re-failing identically forever, flooding
        the log and never actually recovering. Treat it as a full disconnect
        instead: this stops calling into the dead socket, and the existing
        5s reconnect timer opens a fresh one.
        """
        was_connected = self._mc is not None
        self._mc = None
        if was_connected:
            self.get_logger().error(
                f'Lost connection to the arm during {context}: {exc}. '
                'Will retry every 5s.'
            )

    # ---- Joint State Publisher ----

    def _read_angles_rad(self):
        """Read current joint angles from the robot, returns radians or None."""
        if self._mc is None:
            return None
        try:
            with self._lock:
                angles_deg = self._mc.get_angles()
            self._last_tx_time = time.monotonic()
            if not isinstance(angles_deg, list) or len(angles_deg) != 6:
                return None
            if not self._angles_plausible(angles_deg):
                self.get_logger().warn(
                    f'Ignoring impossible angle read {angles_deg} -- '
                    'treating as a garbled response.',
                    throttle_duration_sec=2.0)
                return None
            self._warn_if_outside_limits(angles_deg)
            return [math.radians(a) for a in angles_deg]
        except Exception as e:
            self._handle_link_error(e, 'joint-state read')
            return None

    def _publish_joint_states(self):
        self._js_cycle += 1

        if self._in_motion.is_set():
            # Do NOT read the hardware while moving. get_angles() blocks up to
            # 100ms in the Pi's server while holding self._lock, which stalls
            # the command stream for the duration. Even at 2Hz that is a hitch
            # twice a second, which is felt directly as jerk.
            #
            # We already know where the arm was told to go, so publish the
            # commanded setpoint instead: MoveIt's state monitor gets a smooth,
            # continuous feed and the link stays entirely free for motion.
            # The cost is that /joint_states reports intent rather than measured
            # position during the move; the first post-motion cycle reads real
            # hardware again and corrects any discrepancy.
            cmd = self._cmd_positions
            if cmd is None:
                return
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(self.JOINT_NAMES)
            msg.position = list(cmd)
            self._js_pub.publish(msg)
            return

        # While jogging is armed, back the polling off for the same reason it
        # backs off during a trajectory: _read_angles_rad blocks up to 100ms
        # holding _lock, and at a 100ms timer period that keeps the lock busy
        # nearly all the time. Jog callbacks then block on the lock and, with a
        # queue depth of 1, get dropped -- the arm sits still while the servo
        # loop happily publishes commands nobody executes.
        if self._jog_enabled:
            stride = max(1, int(round(self._rate / max(self._motion_rate, 0.1))))
            if self._js_cycle % stride:
                return

        angles = self._read_angles_rad()
        if angles is None:
            return

        # Cache for jogging, which needs a starting point to apply a delta to
        # without paying its own blocking read on every command.
        #
        # Do NOT adopt this reading as the jog base while jogs are actively
        # flowing. send_angles is a move, not a teleport, so a read taken
        # mid-travel sits behind the commanded target, and taking it as the
        # new base silently discards however far the arm still had to go.
        # Resync only once jogging has stopped, or if the commanded pose has
        # drifted implausibly far from where the arm actually is (blocked
        # joint, saturated servo) -- at which point reality wins.
        if self._last_angles_rad is None:
            self._last_angles_rad = angles
        else:
            quiet = (time.monotonic() - self._last_jog_time
                     >= self._jog_resync_after)
            divergence = max(
                abs(math.degrees(c - m))
                for c, m in zip(self._last_angles_rad, angles)
            )
            if quiet or divergence > self._jog_max_divergence:
                if not quiet:
                    self.get_logger().warn(
                        f'Jog target has diverged {divergence:.1f}deg from the '
                        'measured pose -- the arm is not keeping up. '
                        'Resyncing to hardware.',
                        throttle_duration_sec=3.0)
                self._last_angles_rad = angles

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.JOINT_NAMES)
        msg.position = angles
        self._js_pub.publish(msg)

    # ---- Jogging (for visual servoing) ----

    def _jog_callback(self, msg: JointJog):
        """Apply a relative joint displacement immediately.

        This exists because Server.py accepts a single client: a servo node
        cannot open its own connection to the arm, so incremental commands
        have to come through the driver that already owns the socket.

        Displacements are in DEGREES here (JointJog does not fix units, and
        degrees match what send_angles takes, avoiding a conversion that is
        easy to get wrong in a fast loop).

        Deliberately ignored while a trajectory is executing -- two things
        commanding the arm at once produces motion neither one intended.
        """
        # Every rejection below is throttled-logged rather than silent. A servo
        # loop that publishes correctly into a driver that quietly discards
        # everything is indistinguishable from a broken servo loop, and that is
        # a miserable thing to debug.
        if self._in_motion.is_set():
            self.get_logger().warn(
                'Ignoring jog: a trajectory is executing.',
                throttle_duration_sec=2.0)
            return

        if not self._jog_enabled:
            self.get_logger().warn(
                'Ignoring jog: jogging is disabled. Enable it with: '
                'ros2 service call /arm/jog_enable std_srvs/srv/SetBool '
                '"{data: true}"',
                throttle_duration_sec=2.0)
            return

        if self._mc is None:
            self.get_logger().warn(
                'Ignoring jog: not connected to the arm.',
                throttle_duration_sec=5.0)
            return

        base = self._last_angles_rad
        if base is None:
            # No successful joint-state read yet, so there is no starting pose
            # to apply a delta to. Usually means the arm is unreachable.
            self.get_logger().warn(
                'Ignoring jog: no joint angles read yet -- cannot apply a '
                'relative move. Is the arm reachable?',
                throttle_duration_sec=2.0)
            return

        # Rate-limit to the same interval the trajectory streamer uses. A servo
        # loop publishing faster than the link can carry just builds a backlog.
        now = time.monotonic()
        if now - self._last_jog_time < self._cmd_interval:
            return

        target_deg = [math.degrees(a) for a in base]
        # Clamp every joint here, not just the one(s) this message actually
        # jogs. Only the touched joints get re-clamped below after their
        # delta is applied -- an untouched joint's slot is just carried over
        # from base, so if base ever picked up one bad value (a garbled
        # read on a flaky link, say), it would be resent unchecked on every
        # future jog indefinitely, since nothing here ever looks at it again
        # until the arm itself finally refuses it.
        for i, (lo, hi) in enumerate(self._joint_limits_deg):
            target_deg[i] = max(lo, min(hi, target_deg[i]))
        # Where the arm is starting from, for sizing this jog's speed below.
        from_deg = list(target_deg)

        names = list(msg.joint_names)
        deltas = list(msg.displacements)
        if len(names) != len(deltas):
            self.get_logger().warn(
                f'JointJog has {len(names)} names but {len(deltas)} '
                'displacements; ignoring'
            )
            return

        moved = False
        for name, delta in zip(names, deltas):
            if name not in self.JOINT_NAMES:
                continue
            idx = self.JOINT_NAMES.index(name)
            # Cap any single step. A bad detection should nudge the arm, not
            # fling it across the workspace.
            delta = max(-self._max_jog_deg, min(self._max_jog_deg, float(delta)))
            lo, hi = self._joint_limits_deg[idx]
            # Clamp to the joint's travel. This is not about being cautious
            # with the workspace -- driving a servo into its hard stop and
            # holding it there is how you damage it.
            target_deg[idx] = max(lo, min(hi, target_deg[idx] + delta))
            moved = True

        if not moved:
            self.get_logger().warn(
                f'Ignoring jog: none of {names} are arm joints '
                f'(expected any of {self.JOINT_NAMES})',
                throttle_duration_sec=2.0)
            return

        speed = self._step_speed(
            from_deg, target_deg, self._cmd_interval,
            enabled=self._adaptive_jog_speed, fallback=self._jog_speed,
        )

        try:
            with self._lock:
                self._mc.send_angles(target_deg, speed)
            self._last_tx_time = time.monotonic()
            self.get_logger().info(
                f'jog -> {[round(d, 1) for d in target_deg]}',
                throttle_duration_sec=2.0)
            self._last_jog_time = now
            # Track the commanded pose so successive jogs compound instead of
            # each one being applied to a stale reading.
            self._last_angles_rad = [math.radians(d) for d in target_deg]
        except OSError as e:
            # A real socket-level failure -- the link is actually dead.
            self._handle_link_error(e, 'jog')
        except Exception as e:
            # pymycobot validates the target locally before it ever touches
            # the socket, so a rejected value (e.g. "invalid angle value")
            # raises here too but has nothing to do with the connection.
            # Treating it as a lost link would tear down a perfectly healthy
            # socket over a bad number -- just drop this jog and log it.
            self.get_logger().warn(
                f'jog rejected: {e}', throttle_duration_sec=2.0)

    def _jog_enable_callback(self, request, response):
        """Deadman for jogging. Servoing does nothing until this is enabled."""
        self._jog_enabled = bool(request.data)
        state = 'ENABLED' if self._jog_enabled else 'disabled'
        response.success = True
        response.message = f'Jogging {state}'
        self.get_logger().info(f'Jogging {state}')
        return response

    # ---- FollowJointTrajectory Action ----

    def _goal_callback(self, goal_request):
        if self._mc is None:
            self.get_logger().error(
                'Rejecting trajectory: not connected to the arm.')
            return GoalResponse.REJECT
        self.get_logger().info('Received trajectory goal')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def _wait_until_reached(self, target_deg, tolerance_deg=5.0, timeout=8.0):
        """Poll joint angles until robot reaches target or timeout.

        Reads here feed _cmd_positions as a side effect, so /joint_states can
        publish real measured angles during this phase without spending a
        second round-trip on the link (hardware reads in the joint-state timer
        are suppressed while _in_motion is set).
        """
        if self._mc is None:
            return False
        deadline = time.monotonic() + timeout
        # rclpy.ok() so Ctrl-C is not ignored for up to home_timeout while the
        # arm crawls toward a pose nobody is waiting for any more. Without it,
        # shutting down mid-home kept commanding the arm for another 15s,
        # which reads as the stack refusing to die.
        while time.monotonic() < deadline and rclpy.ok():
            try:
                with self._lock:
                    current = self._mc.get_angles()
                if isinstance(current, list) and len(current) == 6:
                    self._cmd_positions = [math.radians(c) for c in current]
                    max_err = max(abs(c - t) for c, t in zip(current, target_deg))
                    if max_err < tolerance_deg:
                        return True
            except Exception as e:
                self._handle_link_error(e, 'position poll')
                return False
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

    def _step_speed(self, from_deg, to_deg, dt, enabled=None, fallback=None):
        """Pick a send_angles speed (0..100) that covers this step in ~dt.

        send_angles is point-to-point: it drives to the target at the given
        speed and stops. During streaming each target is only one command
        interval away, so a fixed speed makes the arm sprint the tiny gap and
        wait — micro stop-start that reads as jerk. Sizing speed to the step
        keeps it moving continuously into the next command.

        `enabled`/`fallback` let jogging reuse this with its own toggle and
        default speed; omitted, they take the trajectory settings.
        """
        enabled = self._adaptive_speed if enabled is None else enabled
        fallback = self._traj_speed if fallback is None else fallback
        if not enabled or dt <= 0.0:
            return fallback
        max_delta = max(abs(a - b) for a, b in zip(from_deg, to_deg))
        if max_delta <= 0.0:
            return self._min_speed
        needed_deg_s = (max_delta / dt) * self._speed_headroom
        if self._speed_at_100 <= 0.0:
            return self._traj_speed
        speed = int(round(needed_deg_s / self._speed_at_100 * 100.0))
        return max(self._min_speed, min(self._max_speed, speed))

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
            self._cmd_positions = list(points[-1].positions)
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
                self._cmd_positions = None
            goal_handle.succeed()
            return FollowJointTrajectory.Result()

        self.get_logger().info(
            f'Executing trajectory: {n} points over {duration:.2f}s, '
            f'streaming every {self._cmd_interval * 1000:.0f}ms '
            f'(lookahead {self._lookahead * 1000:.0f}ms)'
        )

        # Seed the commanded position before suppressing hardware reads, so
        # /joint_states has something valid to publish from the first cycle.
        self._cmd_positions = list(points[0].positions)
        prev_deg = [math.degrees(p) for p in points[0].positions]

        # Stop reading the hardware for the duration of the move.
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

                # Size the speed to this specific step so the arm flows into
                # the next command instead of sprinting and stopping.
                speed = self._step_speed(prev_deg, target_deg, self._cmd_interval)

                try:
                    with self._lock:
                        self._mc.send_angles(target_deg, speed)
                    self._last_tx_time = time.monotonic()
                except OSError as e:
                    # A dropped command on an otherwise-live link is
                    # recoverable: the next one is only command_interval away
                    # and supersedes it anyway. A socket-level failure is not
                    # -- without marking it, every remaining step of this
                    # trajectory would re-fail identically on the same dead
                    # socket instead of the 5s reconnect timer ever getting a
                    # chance to open a new one.
                    self._handle_link_error(e, 'trajectory streaming')
                except Exception as e:
                    # Local validation rejecting this target has nothing to
                    # do with the link -- do not tear down a healthy socket
                    # over it.
                    self.get_logger().warn(
                        f'send_angles rejected mid-stream: {e}',
                        throttle_duration_sec=2.0)

                # Feeds /joint_states while hardware reads are suppressed.
                self._cmd_positions = list(target_rad)
                prev_deg = target_deg

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
            self._cmd_positions = list(points[-1].positions)
            try:
                with self._lock:
                    self._mc.send_angles(
                        final_deg,
                        self._step_speed(prev_deg, final_deg, self._cmd_interval),
                    )
            except OSError as e:
                self._handle_link_error(e, 'trajectory final position')
            except Exception as e:
                self.get_logger().warn(f'send_angles rejected for final '
                                       f'position: {e}')

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
            # Fall back to real hardware reads now that the link is free.
            self._cmd_positions = None

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
        if self._mc is None:
            response.success = False
            response.message = 'Not connected to the arm; cannot home.'
            self.get_logger().error(response.message)
            return response

        target = list(self._home_angles)
        self.get_logger().info(f'Homing to {target} at speed {self._home_speed}')

        # Homing is a single point-to-point move, not a command stream, so
        # there is no burst of commands to protect. _in_motion is still set to
        # stop the joint-state timer from adding a *second* concurrent read on
        # top of the one _wait_until_reached is already doing; that polling
        # feeds _cmd_positions with real measured angles, so /joint_states
        # keeps reporting the arm's actual position as it travels.
        self._in_motion.set()
        try:
            with self._lock:
                self._mc.send_angles(target, self._home_speed)
            reached = self._wait_until_reached(
                target,
                tolerance_deg=self._settle_tol,
                timeout=self._home_timeout,
            )
        except OSError as e:
            self._handle_link_error(e, 'homing')
            response.success = False
            response.message = f'Homing failed: {e}'
            self.get_logger().error(response.message)
            return response
        except Exception as e:
            response.success = False
            response.message = f'Homing failed: {e}'
            self.get_logger().error(response.message)
            return response
        finally:
            self._in_motion.clear()
            self._cmd_positions = None

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
    # Thread count is set explicitly. MultiThreadedExecutor() defaults to
    # multiprocessing.cpu_count(), and this often runs on a VM with one or two
    # vCPUs. This node has a timer that blocks on TCP reads for up to 100ms at
    # a time; with only one thread that timer monopolises the executor and the
    # jog subscription, action server and services are starved -- the arm
    # simply stops responding to anything while appearing healthy.
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
