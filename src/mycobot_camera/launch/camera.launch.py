"""
Launch the myCobot camera node.

Connects to the Pi's MJPEG HTTP stream and publishes ROS 2 Image messages.

Usage:
  ros2 launch mycobot_camera camera.launch.py
  ros2 launch mycobot_camera camera.launch.py camera_url:=http://192.168.0.15:8080/?action=stream
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
    camera_url_arg = DeclareLaunchArgument(
        'camera_url',
        default_value=f'http://{DEFAULT_ROBOT_IP}:8080/?action=stream',
        description='MJPEG stream URL from camera_stream.py on the Pi',
    )
    stream_read_timeout_arg = DeclareLaunchArgument(
        'stream_read_timeout', default_value='15.0',
        description='Seconds without a new MJPEG frame before reconnecting; '
                    'raise this if the stream reconnects during brief Wi-Fi '
                    'or server stalls that would otherwise recover on their own',
    )

    camera_node = Node(
        package='mycobot_camera',
        executable='camera_node',
        name='camera_node',
        parameters=[{
            'camera_url': LaunchConfiguration('camera_url'),
            'frame_rate': 30.0,
            'frame_id': 'camera_link',
            'stream_read_timeout': LaunchConfiguration('stream_read_timeout'),
        }],
        output='screen',
    )

    return LaunchDescription([
        camera_url_arg,
        stream_read_timeout_arg,
        camera_node,
    ])
