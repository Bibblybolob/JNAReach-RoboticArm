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

NOTHING MOVES until both gates are opened, in either order:

    ros2 service call /arm/jog_enable  std_srvs/srv/SetBool "{data: true}"
    ros2 service call /servo/enable    std_srvs/srv/SetBool "{data: true}"

Setting either to false stops the arm. Enabling the servo triggers a one-time
orientation probe: it twitches two joints to learn which way the camera is
mounted, so hold your hand in view and still while it runs.

Pass auto_enable:=true to open both gates automatically a few seconds after
startup. Off by default: the arm should not start moving merely because you
launched a file.

Useful arguments:
    robot_ip:=192.168.1.46      Pi address
    gain:=1.5                   lower if the arm oscillates
    show_window:=true           OpenCV window of the tracker (needs a display)
    auto_enable:=true           skip the manual service calls
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value='192.168.1.46')
    robot_port_arg = DeclareLaunchArgument('robot_port', default_value='9000')
    camera_port_arg = DeclareLaunchArgument('camera_port', default_value='8080')
    gain_arg = DeclareLaunchArgument(
        'gain', default_value='2.5',
        description='Servo proportional gain; halve it if the arm oscillates')
    deadband_arg = DeclareLaunchArgument(
        'deadband', default_value='0.04',
        description='Image error below which the arm holds still')
    show_window_arg = DeclareLaunchArgument(
        'show_window', default_value='false',
        description='Open an OpenCV window from the tracker (needs a display)')
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable', default_value='false',
        description='Open both safety gates automatically after startup')

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
        }.items(),
    )

    hand_tracker = Node(
        package='mycobot_perception',
        executable='hand_tracker_node',
        name='hand_tracker_node',
        parameters=[{
            'show_window': LaunchConfiguration('show_window'),
        }],
        output='screen',
    )

    servo = Node(
        package='mycobot_perception',
        executable='visual_servo_node',
        name='visual_servo_node',
        parameters=[{
            'gain': LaunchConfiguration('gain'),
            'deadband': LaunchConfiguration('deadband'),
        }],
        output='screen',
    )

    # Delayed so the driver has read joint angles and the tracker has seen a
    # hand before the probe runs. Enabling immediately would probe against a
    # driver that cannot jog yet.
    auto_enable = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'service', 'call', '/arm/jog_enable',
                     'std_srvs/srv/SetBool', '{data: true}'],
                output='screen',
            ),
            ExecuteProcess(
                cmd=['ros2', 'service', 'call', '/servo/enable',
                     'std_srvs/srv/SetBool', '{data: true}'],
                output='screen',
            ),
        ],
        condition=IfCondition(LaunchConfiguration('auto_enable')),
    )

    return LaunchDescription([
        robot_ip_arg,
        robot_port_arg,
        camera_port_arg,
        gain_arg,
        deadband_arg,
        show_window_arg,
        auto_enable_arg,
        robot,
        hand_tracker,
        servo,
        auto_enable,
    ])
