"""
Full MoveIt2 bringup for myCobot 280 Pi.

Launches:
  - robot_state_publisher (publishes URDF TF tree)
  - mycobot_hardware_node (arm joint states + trajectory action + homing service)
  - camera_node (MJPEG stream -> sensor_msgs/Image for RViz2)
  - move_group (MoveIt2 motion planning via OMPL/RRTConnect)
  - rviz2 (visualization with MotionPlanning panel)

Usage:
  ros2 launch mycobot_bringup moveit_bringup.launch.py
  ros2 launch mycobot_bringup moveit_bringup.launch.py robot_ip:=192.168.0.15
"""

import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    full_path = os.path.join(get_package_share_directory(package_name), file_path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


# Robot address, overridable without editing anything:
#   export MYCOBOT_IP=192.168.0.50      (whole shell session)
#   ros2 launch ... robot_ip:=1.2.3.4   (one run, wins over the env var)
DEFAULT_ROBOT_IP = os.environ.get('MYCOBOT_IP', '192.168.0.15')


def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value=DEFAULT_ROBOT_IP)
    robot_port_arg = DeclareLaunchArgument('robot_port', default_value='9000')
    camera_port_arg = DeclareLaunchArgument('camera_port', default_value='8080')

    # This file had no serial arguments at all, so it could only ever reach
    # the arm through the Pi's TCP server -- on a Jetson driving the UART
    # directly it came up permanently disconnected, with MoveIt planning
    # happily against a robot it could not command.
    connection_arg = DeclareLaunchArgument(
        'connection', default_value='serial', choices=['tcp', 'serial'],
        description="How to reach the arm. 'serial' drives the ESP32 directly "
                    "over the Jetson's UART; 'tcp' goes via the Pi.")
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyTHS1',
        description='Serial device for connection:=serial')
    serial_baud_arg = DeclareLaunchArgument(
        'serial_baud', default_value='1000000',
        description="Baud for connection:=serial -- the arm's firmware rate")

    # Off by default HERE specifically, unlike the other bringups. This file
    # exists to plan moves by hand in RViz, and homing on startup drives the
    # arm before the operator has looked at it -- which with a payload the
    # shoulder is marginal on is exactly the move worth NOT making
    # automatically. Use the "home" named state in the MotionPlanning panel.
    home_on_start_arg = DeclareLaunchArgument(
        'home_on_start', default_value='false',
        description='Drive to the home pose on startup. Off here so planning '
                    'sessions begin from wherever the arm actually is')

    source_arg = DeclareLaunchArgument(
        'source', default_value='realsense',
        choices=['mjpeg', 'device', 'realsense'],
        description="'realsense' opens a D405 locally; 'mjpeg' reads the Pi's "
                    "stream")

    robot_ip = LaunchConfiguration('robot_ip')
    robot_port = LaunchConfiguration('robot_port')
    camera_port = LaunchConfiguration('camera_port')

    description_dir = get_package_share_directory('mycobot_description')
    moveit_dir = get_package_share_directory('mycobot_moveit_config')
    xacro_file = os.path.join(description_dir, 'urdf', 'mycobot_280pi.urdf.xacro')

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str,
    )

    with open(os.path.join(moveit_dir, 'config', 'mycobot_280pi.srdf'), 'r') as f:
        robot_description_semantic = f.read()

    kinematics_yaml = load_yaml('mycobot_moveit_config', 'config/kinematics.yaml')
    joint_limits_yaml = load_yaml('mycobot_moveit_config', 'config/joint_limits.yaml')
    controllers_yaml = load_yaml('mycobot_moveit_config', 'config/moveit_controllers.yaml')

    ompl_yaml = load_yaml('mycobot_moveit_config', 'config/ompl_planning.yaml')
    planning_pipelines_config = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_yaml,
    }

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
            'connection': LaunchConfiguration('connection'),
            'serial_port': LaunchConfiguration('serial_port'),
            'serial_baud': ParameterValue(
                LaunchConfiguration('serial_baud'), value_type=int),
            'home_on_start': ParameterValue(
                LaunchConfiguration('home_on_start'), value_type=bool),
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
            # fixed value. 52.0 is MEASURED (joint1 over ttyTHS1 with a D405 on
            # the flange, 2026-08-01); this file still said 120, the old guess,
            # so it asked for 43% of the speed it intended and the arm trailed
            # its own commanded goal.
            'adaptive_speed': True,
            'speed_at_100_deg_s': 52.0,
            'speed_headroom': 1.3,
            'home_angles_deg': [0.0, 90.0, -150.0, 55.0, 0.0, 0.0],
            'home_speed': 30,
        }],
        output='screen',
    )

    camera_node = Node(
        package='mycobot_camera',
        executable='camera_node',
        name='camera_node',
        parameters=[{
            'source': LaunchConfiguration('source'),
            # Only read when source:=mjpeg, but harmless to pass always.
            'camera_url': PythonExpression([
                "'http://'", " + '", robot_ip,
                "' + ':'", " + '", camera_port,
                "' + '/?action=stream'",
            ]),
            'frame_rate': 15.0,
            'rs_width': 640,
            'rs_height': 480,
            'rs_fps': 30,
            'rs_depth': True,
        }],
        output='screen',
    )

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description,
                'robot_description_semantic': robot_description_semantic,
                'robot_description_planning': joint_limits_yaml,
                'use_sim_time': False,
                'trajectory_execution.allowed_execution_duration_scaling': 4.0,
                'trajectory_execution.allowed_goal_duration_margin': 10.0,
                'trajectory_execution.execution_duration_monitoring': False,
            },
            kinematics_yaml,
            controllers_yaml,
            planning_pipelines_config,
        ],
    )

    obstacles_file = os.path.join(moveit_dir, 'config', 'obstacles.yaml')

    scene_objects_node = Node(
        package='mycobot_driver',
        executable='scene_objects',
        name='scene_objects',
        parameters=[{'obstacles_file': obstacles_file}],
        output='screen',
    )

    rviz_config = os.path.join(moveit_dir, 'rviz', 'moveit.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[
            {
                'robot_description': robot_description,
                'robot_description_semantic': robot_description_semantic,
            },
            kinematics_yaml,
            planning_pipelines_config,
        ],
    )

    return LaunchDescription([
        robot_ip_arg,
        robot_port_arg,
        camera_port_arg,
        connection_arg,
        serial_port_arg,
        serial_baud_arg,
        home_on_start_arg,
        source_arg,
        robot_state_publisher,
        hardware_node,
        camera_node,
        move_group_node,
        scene_objects_node,
        rviz_node,
    ])
