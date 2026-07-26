"""
Bring up the myCobot 280 Pi driver and camera nodes.

Launches:
  - robot_state_publisher (publishes URDF TF tree)
  - mycobot_hardware_node (arm joint states + trajectory action + homing service)
  - camera_node (MJPEG stream -> sensor_msgs/Image)

Usage:
  ros2 launch mycobot_bringup robot_bringup.launch.py
  ros2 launch mycobot_bringup robot_bringup.launch.py robot_ip:=192.168.1.46
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip', default_value='192.168.1.46',
    )
    robot_port_arg = DeclareLaunchArgument(
        'robot_port', default_value='9000',
    )
    camera_port_arg = DeclareLaunchArgument(
        'camera_port', default_value='8080',
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
            'home_angles_deg': [0.0, 90.0, -90.0, -90.0, 0.0, 0.0],
            'home_speed': 30,
        }],
        output='screen',
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
            'frame_rate': 30.0,
        }],
        output='screen',
    )

    return LaunchDescription([
        robot_ip_arg,
        robot_port_arg,
        camera_port_arg,
        robot_state_publisher,
        hardware_node,
        camera_node,
    ])
