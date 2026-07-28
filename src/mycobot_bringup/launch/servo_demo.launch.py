"""
Everything needed for finger-following, in one command.

    ros2 launch mycobot_bringup servo_demo.launch.py

Starts, in one terminal:
  - robot_state_publisher   (URDF TF tree)
  - mycobot_hardware_node   (arm driver: joint states, trajectories, jogging)
  - camera_node             (Pi MJPEG stream -> /camera/image_raw)
  - hand_tracker_node       (MediaPipe -> /hand/point_px)
  - visual_servo_node       (image error -> /arm/jog)

move_group and RViz are deliberately NOT included. Visual servoing bypasses
MoveIt entirely -- it jogs joints directly from image error -- so planning is
dead weight here, and it is a lot of dead weight on a VM. Use
moveit_bringup.launch.py when you want planning.

On startup the arm drives to its home pose ([0, 90, -90, 0, 0, 0] degrees) so
it always begins from a known position rather than wherever it was left. Pass
home_on_start:=false to skip that.

Servoing and jogging are armed automatically. The arm then sits at home until
you ask it to hunt:

    ros2 service call /servo/search std_srvs/srv/Trigger

It then sweeps to find a hand, centres on it, and closes in. After 15s with no
sighting it returns home and waits for the next trigger. To stop it at any
point:

    ros2 service call /servo/enable std_srvs/srv/SetBool "{data: false}"

The first time it finds a hand it runs a one-time orientation probe, twitching
a few joints to learn how the camera is mounted -- hold your hand still for it.

Useful arguments:
    robot_ip:=192.168.0.15         Pi address
    gain:=1.5                      lower if the arm oscillates
    ki:=0.6                        lower if it overshoots and hunts
    lost_timeout:=30.0             longer grace before homing
    target_size_fraction:=0.55     closer approach (0.45 default)
    approach_enabled:=false        centre only, do not close in
    search_on_start:=true          start hunting without the trigger
    home_on_start:=false           do not home on startup
    show_window:=true              OpenCV window (needs a display)
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# Robot address, overridable without editing anything:
#   export MYCOBOT_IP=192.168.0.50      (whole shell session)
#   ros2 launch ... robot_ip:=1.2.3.4   (one run, wins over the env var)
DEFAULT_ROBOT_IP = os.environ.get('MYCOBOT_IP', '192.168.0.15')


def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value=DEFAULT_ROBOT_IP)
    robot_port_arg = DeclareLaunchArgument('robot_port', default_value='9000')
    camera_port_arg = DeclareLaunchArgument('camera_port', default_value='8080')
    stream_read_timeout_arg = DeclareLaunchArgument(
        'stream_read_timeout', default_value='30.0',
        description='Seconds without a new MJPEG frame before reconnecting. '
                    'Generous on purpose: a loaded host can stall for tens of '
                    'seconds, and reconnecting mid-stall makes it worse')
    home_on_start_arg = DeclareLaunchArgument(
        'home_on_start', default_value='true',
        description='Drive to the home pose once on startup, so the arm sits '
                    'at a known position until you trigger a hunt')
    gain_arg = DeclareLaunchArgument(
        'gain', default_value='3.0',
        description='Servo proportional gain; halve it if the arm oscillates')
    ki_arg = DeclareLaunchArgument(
        'ki', default_value='1.2',
        description='Integral gain; this is the term that actually centres '
                    'the target. Lower it if the arm overshoots and hunts')
    kd_arg = DeclareLaunchArgument(
        'kd', default_value='0.35',
        description='Derivative gain; damps the approach')
    deadband_arg = DeclareLaunchArgument(
        'deadband', default_value='0.015',
        description='Image error below which the arm holds still')
    model_complexity_arg = DeclareLaunchArgument(
        'model_complexity', default_value='0',
        description='MediaPipe hand model: 0 is ~2x faster than 1. On a '
                    'CPU-bound host fresher detections smooth the servo loop '
                    'more than extra landmark precision does. Use 1 if you '
                    'have GPU inference')
    show_window_arg = DeclareLaunchArgument(
        'show_window', default_value='false',
        description='Open an OpenCV window from the tracker (needs a display)')
    lost_timeout_arg = DeclareLaunchArgument(
        'lost_timeout', default_value='15.0',
        description='Seconds without a sighting before returning home')
    target_size_arg = DeclareLaunchArgument(
        'target_size_fraction', default_value='0.45',
        description='Palm width as a fraction of frame width to close in to')
    approach_enabled_arg = DeclareLaunchArgument(
        'approach_enabled', default_value='true',
        description='Close in on the hand as well as centring it')
    search_on_start_arg = DeclareLaunchArgument(
        'search_on_start', default_value='false',
        description='Begin hunting immediately instead of waiting for the '
                    '/servo/search trigger')

    bringup_dir = get_package_share_directory('mycobot_bringup')

    # Reuse robot_bringup rather than restating the driver and camera
    # parameters, so tuning values live in exactly one place.
    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'robot_bringup.launch.py')),
        launch_arguments={
            'robot_ip': LaunchConfiguration('robot_ip'),
            'robot_port': LaunchConfiguration('robot_port'),
            'camera_port': LaunchConfiguration('camera_port'),
            'stream_read_timeout': LaunchConfiguration('stream_read_timeout'),
            'home_on_start': LaunchConfiguration('home_on_start'),
        }.items(),
    )

    hand_tracker = Node(
        package='mycobot_perception',
        executable='hand_tracker_node',
        name='hand_tracker_node',
        parameters=[{
            'show_window': LaunchConfiguration('show_window'),
            # Encoding and publishing an annotated frame costs real CPU per
            # frame, and nothing subscribes to it in this launch. Host load is
            # what stalls the stack and drops both links, so this is off unless
            # you are actually looking at the window.
            'publish_annotated': LaunchConfiguration('show_window'),
            'model_complexity': LaunchConfiguration('model_complexity'),
        }],
        output='screen',
        respawn=True,
        respawn_delay=3.0,
    )

    servo = Node(
        package='mycobot_perception',
        executable='visual_servo_node',
        name='visual_servo_node',
        parameters=[{
            'gain': LaunchConfiguration('gain'),
            'ki': LaunchConfiguration('ki'),
            'kd': LaunchConfiguration('kd'),
            'deadband': LaunchConfiguration('deadband'),
            'lost_timeout': LaunchConfiguration('lost_timeout'),
            'target_size_fraction': LaunchConfiguration('target_size_fraction'),
            'approach_enabled': LaunchConfiguration('approach_enabled'),
            'search_on_start': LaunchConfiguration('search_on_start'),
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
        gain_arg,
        ki_arg,
        kd_arg,
        deadband_arg,
        model_complexity_arg,
        show_window_arg,
        lost_timeout_arg,
        target_size_arg,
        approach_enabled_arg,
        search_on_start_arg,
        robot,
        hand_tracker,
        servo,
    ])
