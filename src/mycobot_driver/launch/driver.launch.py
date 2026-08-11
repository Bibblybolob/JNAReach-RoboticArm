"""
Launch the myCobot 280 Pi hardware driver node.

This node handles arm control (joint states + trajectory execution + homing)
over a TCP connection to the Pi's pymycobot Server.py. The gripper has been
removed for the elevator-button task.

Usage:
  ros2 launch mycobot_driver driver.launch.py
  ros2 launch mycobot_driver driver.launch.py robot_ip:=192.168.0.15 robot_port:=9000
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Robot address, overridable without editing anything:
#   export MYCOBOT_IP=192.168.0.50      (whole shell session)
#   ros2 launch ... robot_ip:=1.2.3.4   (one run, wins over the env var)
DEFAULT_ROBOT_IP = os.environ.get('MYCOBOT_IP', '192.168.0.15')


def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip', default_value=DEFAULT_ROBOT_IP,
        description='IP address of the myCobot 280 Pi',
    )
    robot_port_arg = DeclareLaunchArgument(
        'robot_port', default_value='9000',
        description='TCP port of the pymycobot server',
    )

    robot_ip = LaunchConfiguration('robot_ip')
    robot_port = LaunchConfiguration('robot_port')

    # Home pose in degrees, one entry per arm joint.
    # Mirrored by the "home" group_state in mycobot_280pi.srdf (in radians).
    # Change both together.
    home_angles = [0.0, 90.0, -150.0, 55.0, 0.0, 0.0]

    hardware_node = Node(
        package='mycobot_driver',
        executable='mycobot_hardware_node',
        name='mycobot_hardware_node',
        parameters=[{
            'robot_ip': robot_ip,
            'robot_port': robot_port,
            # Server.py on the Pi is single-client and blocks ~100ms on any
            # command that returns data. Polling joint states too fast starves
            # the motion commands sharing that socket, which is what made
            # motion stutter. 10Hz leaves headroom; during trajectory
            # execution the driver drops to publish_rate_during_motion.
            'publish_rate': 10.0,
            'publish_rate_during_motion': 2.0,
            'default_speed': 80,
            # Trajectory streaming. See mycobot_hardware_node.py for the
            # reasoning behind each of these.
            'command_interval': 0.06,
            'lookahead': 0.12,
            'trajectory_speed': 60,
            # See scripts/measure_arm.py — speed_at_100_deg_s is a guess.
            'adaptive_speed': True,
            'speed_at_100_deg_s': 120.0,
            'speed_headroom': 1.3,
            'settle_timeout': 2.0,
            'settle_tolerance_deg': 4.0,
            # Fixed home pose, degrees. Override for your workspace.
            'home_angles_deg': home_angles,
            'home_speed': 30,
            'home_timeout': 40.0,
        }],
        output='screen',
    )

    return LaunchDescription([
        robot_ip_arg,
        robot_port_arg,
        hardware_node,
    ])
