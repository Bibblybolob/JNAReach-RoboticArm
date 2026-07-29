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
        'source', default_value='mjpeg', choices=['mjpeg', 'device'],
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
    device_height_arg = DeclareLaunchArgument('device_height', default_value='480')
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
            'command_interval': 0.06,
            'lookahead': 0.12,
            'trajectory_speed': 60,
            # Scale each streaming step's speed to its size instead of using a
            # fixed value. speed_at_100_deg_s is a GUESS — measure it with
            # scripts/measure_arm.py and set the real number here.
            'adaptive_speed': True,
            'speed_at_100_deg_s': 120.0,
            'speed_headroom': 1.3,
            'home_angles_deg': [0.0, 90.0, -90.0, 0.0, 0.0, 0.0],
            'home_speed': 30,
            'home_on_start': LaunchConfiguration('home_on_start'),
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
            'stream_read_timeout': LaunchConfiguration('stream_read_timeout'),
            'source': LaunchConfiguration('source'),
            # Forced to str: launch YAML-parses '0' into an integer, but this
            # parameter is a string so that a GStreamer pipeline can go in the
            # same field. Without this the node dies at startup on a type
            # mismatch for the most ordinary value anyone would pass.
            'device': ParameterValue(LaunchConfiguration('device'),
                                     value_type=str),
            'device_auto_exposure': LaunchConfiguration('device_auto_exposure'),
            'device_exposure': LaunchConfiguration('device_exposure'),
            'device_fps': LaunchConfiguration('device_fps'),
            'device_width': LaunchConfiguration('device_width'),
            'device_height': LaunchConfiguration('device_height'),
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
        source_arg,
        device_arg,
        device_auto_exposure_arg,
        device_exposure_arg,
        device_fps_arg,
        device_width_arg,
        device_height_arg,
        robot_state_publisher,
        hardware_node,
        camera_node,
    ])
