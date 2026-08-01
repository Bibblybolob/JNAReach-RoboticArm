"""
Bring up the myCobot 280 Pi driver and camera nodes.

Launches:
  - robot_state_publisher (publishes URDF TF tree)
  - mycobot_hardware_node (arm joint states + trajectory action + homing service)
  - camera_node (MJPEG stream -> sensor_msgs/Image)

Usage:
  ros2 launch mycobot_bringup robot_bringup.launch.py
  ros2 launch mycobot_bringup robot_bringup.launch.py robot_ip:=192.168.0.15
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


# Robot address, overridable without editing anything:
#   export MYCOBOT_IP=192.168.0.50      (whole shell session)
#   ros2 launch ... robot_ip:=1.2.3.4   (one run, wins over the env var)
DEFAULT_ROBOT_IP = os.environ.get('MYCOBOT_IP', '192.168.0.15')


# Launch arguments arrive as strings and are YAML-parsed on the way into a
# node, so `device_exposure:=50` becomes an INTEGER and a node declaring a
# float refuses it -- the node dies at startup over a value the user typed
# entirely reasonably. Every numeric or boolean parameter below is therefore
# given an explicit type rather than left to inference. Three separate
# startup crashes came from not doing this.
def _f(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _i(name):
    return ParameterValue(LaunchConfiguration(name), value_type=int)


def _b(name):
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def _s(name):
    return ParameterValue(LaunchConfiguration(name), value_type=str)


def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip', default_value=DEFAULT_ROBOT_IP,
    )
    robot_port_arg = DeclareLaunchArgument(
        'robot_port', default_value='9000',
    )
    camera_port_arg = DeclareLaunchArgument(
        'camera_port', default_value='8080',
    )
    stream_read_timeout_arg = DeclareLaunchArgument(
        'stream_read_timeout', default_value='30.0',
        description='Seconds without a new MJPEG frame before reconnecting; '
                    'raise this if the stream reconnects during brief Wi-Fi '
                    'or server stalls that would otherwise recover on their own',
    )
    # Local-camera options. Declared here because camera_node lives in this
    # file, and forwarded from servo_demo so one command can move the camera
    # off the Pi entirely.
    source_arg = DeclareLaunchArgument(
        'source', default_value='mjpeg', choices=['mjpeg', 'device', 'realsense'],
        description="'mjpeg' reads the Pi's stream; 'device' opens a camera "
                    'plugged into THIS machine, which removes the encode, '
                    'the network hop and the decode rather than speeding '
                    'them up')
    device_arg = DeclareLaunchArgument(
        'device', default_value='0',
        description='V4L2 index for source:=device (0 = /dev/video0), or a '
                    'GStreamer pipeline string for a CSI camera')
    device_auto_exposure_arg = DeclareLaunchArgument(
        'device_auto_exposure', default_value='true',
        description='Auto-exposure caps the frame rate in dim light: this '
                    'webcam measured 10.2 fps on auto and 30.2 with a short '
                    'manual exposure. false trades image brightness for rate; '
                    'adding light is the better fix if you can')
    device_exposure_arg = DeclareLaunchArgument(
        'device_exposure', default_value='0.0',
        description='Manual exposure value when device_auto_exposure is '
                    'false. 0 keeps the driver default')
    device_fps_arg = DeclareLaunchArgument('device_fps', default_value='30.0')
    device_width_arg = DeclareLaunchArgument('device_width', default_value='640')
    # Driver timing. Both were hardcoded here, which silently overrode the
    # node's own declared defaults -- changing the node did nothing.
    max_jog_deg_arg = DeclareLaunchArgument('max_jog_deg', default_value='5.0')
    max_jog_speed_arg = DeclareLaunchArgument(
        'max_jog_speed_deg_s', default_value='80.0')
    command_interval_arg = DeclareLaunchArgument(
        'command_interval', default_value='0.06')
    speed_at_100_arg = DeclareLaunchArgument(
        'speed_at_100_deg_s', default_value='52.0')
    rs_width_arg = DeclareLaunchArgument('rs_width', default_value='640')
    rs_height_arg = DeclareLaunchArgument('rs_height', default_value='480')
    rs_fps_arg = DeclareLaunchArgument('rs_fps', default_value='30')
    rs_depth_arg = DeclareLaunchArgument('rs_depth', default_value='false')
    rs_serial_arg = DeclareLaunchArgument('rs_serial', default_value='')
    rs_align_arg = DeclareLaunchArgument(
        'rs_align_depth_to_color', default_value='false')
    device_height_arg = DeclareLaunchArgument('device_height', default_value='480')
    connection_arg = DeclareLaunchArgument(
        'connection', default_value='tcp', choices=['tcp', 'serial'],
        description="How to reach the arm. 'tcp' goes through the Pi's "
                    "server.py over the network. 'serial' drives the arm's "
                    'ESP32 directly over USB, removing the Pi, TCP and the '
                    'network from every arm command. Stop mycobot_server on '
                    'the Pi first -- two masters on one bus is erratic. Test '
                    'with scripts/probe_usb_arm.py before relying on it')
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyUSB0',
        description='Serial device for connection:=serial')
    serial_baud_arg = DeclareLaunchArgument(
        'serial_baud', default_value='1000000',
        description="Baud for connection:=serial. 1000000 matches what the "
                    "Pi's server.py opens the arm's UART at -- it is the "
                    'firmware rate, not a property of the cable')
    home_on_start_arg = DeclareLaunchArgument(
        'home_on_start', default_value='true',
        description='Drive to the home pose once on startup so the arm always '
                    'begins from a known position. Set false to leave it '
                    'wherever it was',
    )
    robot_ip = LaunchConfiguration('robot_ip')
    robot_port = LaunchConfiguration('robot_port')
    camera_port = LaunchConfiguration('camera_port')

    description_dir = get_package_share_directory('mycobot_description')
    xacro_file = os.path.join(description_dir, 'urdf', 'mycobot_280pi.urdf.xacro')

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
    )

    hardware_node = Node(
        package='mycobot_driver',
        executable='mycobot_hardware_node',
        name='mycobot_hardware_node',
        parameters=[{
            'robot_ip': robot_ip,
            'robot_port': robot_port,
            # See mycobot_hardware_node.py for why this is 10 and not 20:
            # the Pi's TCP server is single-client and blocks ~100ms per
            # read, so fast polling starves the motion commands.
            'publish_rate': 10.0,
            'publish_rate_during_motion': 2.0,
            'default_speed': 80,
            'command_interval': _f('command_interval'),
            'max_jog_deg': _f('max_jog_deg'),
            'max_jog_speed_deg_s': _f('max_jog_speed_deg_s'),
            'lookahead': 0.12,
            'trajectory_speed': 60,
            # Scale each streaming step's speed to its size rather than using
            # a fixed value. speed_at_100_deg_s was MEASURED on 2026-08-01 at
            # 52 deg/s; re-measure after a payload change with
            # scripts/measure_arm.py --serial-port /dev/ttyTHS1.
            'adaptive_speed': True,
            'speed_at_100_deg_s': _f('speed_at_100_deg_s'),
            'speed_headroom': 1.3,
            'home_angles_deg': [0.0, 90.0, -90.0, 0.0, 0.0, 0.0],
            'home_speed': 30,
            'home_on_start': _b('home_on_start'),
            'connection': _s('connection'),
            'serial_port': _s('serial_port'),
            'serial_baud': _i('serial_baud'),
        }],
        output='screen',
        # Deliberately NOT respawned, unlike the other nodes. Two reasons,
        # both specific to this one: it holds the arm's single client slot,
        # so a respawn racing the dying instance can lock itself out; and
        # with home_on_start every respawn drives the arm to home, which
        # means a crash silently moves the robot. A driver restarting itself
        # into motion is worse than a driver that stays down and says so.
        # It already survives losing the arm on its own -- see the 5s
        # reconnect timer -- so respawn was buying very little here.
    )

    camera_node = Node(
        package='mycobot_camera',
        executable='camera_node',
        name='camera_node',
        parameters=[{
            'camera_url': PythonExpression([
                "'http://'", " + '", robot_ip,
                "' + ':'", " + '", camera_port,
                "' + '/?action=stream'",
            ]),
            # Keep this at or just above the Pi's capture rate (--fps in
            # pi/mjpg_streamer.service, now 30). It was once 30 against a
            # 10fps source, which just re-sent each frame ~3x and burned host
            # CPU the stack cannot spare. RAISE BOTH TOGETHER, NOT ONE: this
            # number alone does not make the Pi send faster, it only makes
            # this node republish the same frame more often.
            'frame_rate': 31.0,
            'stream_read_timeout': _f('stream_read_timeout'),
            'source': _s('source'),
            # Forced to str: launch YAML-parses '0' into an integer, but this
            # parameter is a string so that a GStreamer pipeline can go in the
            # same field. Without this the node dies at startup on a type
            # mismatch for the most ordinary value anyone would pass.
            'device': ParameterValue(LaunchConfiguration('device'),
                                     value_type=str),
            'device_auto_exposure': _b('device_auto_exposure'),
            'device_exposure': _f('device_exposure'),
            'device_fps': _f('device_fps'),
            'device_width': _i('device_width'),
            'rs_width': _i('rs_width'),
            'rs_height': _i('rs_height'),
            'rs_fps': _i('rs_fps'),
            'rs_depth': _b('rs_depth'),
            'rs_serial': _s('rs_serial'),
            'rs_align_depth_to_color': _b('rs_align_depth_to_color'),
            'device_height': _i('device_height'),
        }],
        output='screen',
        respawn=True,
        respawn_delay=3.0,
    )

    return LaunchDescription([
        robot_ip_arg,
        robot_port_arg,
        camera_port_arg,
        stream_read_timeout_arg,
        home_on_start_arg,
        connection_arg,
        serial_port_arg,
        serial_baud_arg,
        source_arg,
        device_arg,
        device_auto_exposure_arg,
        device_exposure_arg,
        device_fps_arg,
        device_width_arg,
        max_jog_deg_arg,
        max_jog_speed_arg,
        command_interval_arg,
        speed_at_100_arg,
        rs_width_arg,
        rs_height_arg,
        rs_fps_arg,
        rs_depth_arg,
        rs_serial_arg,
        rs_align_arg,
        device_height_arg,
        robot_state_publisher,
        hardware_node,
        camera_node,
    ])
