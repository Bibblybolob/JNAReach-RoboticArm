"""
Elevator button detection and approach, in one command.

    ros2 launch mycobot_bringup button_servo.launch.py

Starts:
  - robot_state_publisher   (URDF TF tree)
  - mycobot_hardware_node   (arm driver over UART)
  - camera_node             (RealSense D405 with depth)
  - button_detector_node    (YOLOv11n -> Detection2DArray)
  - detection_bridge_node   (Detection2DArray + depth -> PointStamped)
  - visual_servo_node       (image error -> /arm/jog, depth-based approach)

The arm homes, then sits idle until a button label is selected:

    ros2 param set /detection_bridge_node target_label "3"

It then centres the selected button and approaches to target_depth_mm (~50mm
= ~2 inches). To stop at any point:

    ros2 service call /servo/enable std_srvs/srv/SetBool "{data: false}"

Defaults are set for the Jetson Orin Nano + myCobot 280 PI direct UART
topology with a RealSense D405 eye-in-hand.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


DEFAULT_ROBOT_IP = os.environ.get('MYCOBOT_IP', '192.168.0.15')


def _f(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _i(name):
    return ParameterValue(LaunchConfiguration(name), value_type=int)


def _b(name):
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def _s(name):
    return ParameterValue(LaunchConfiguration(name), value_type=str)


def generate_launch_description():
    # --- Robot / camera args (forwarded to robot_bringup) ---
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value=DEFAULT_ROBOT_IP)
    robot_port_arg = DeclareLaunchArgument('robot_port', default_value='9000')
    camera_port_arg = DeclareLaunchArgument('camera_port', default_value='8080')
    stream_read_timeout_arg = DeclareLaunchArgument(
        'stream_read_timeout', default_value='30.0')
    home_on_start_arg = DeclareLaunchArgument(
        'home_on_start', default_value='true')
    connection_arg = DeclareLaunchArgument(
        'connection', default_value='serial', choices=['tcp', 'serial'])
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyTHS1')
    serial_baud_arg = DeclareLaunchArgument(
        'serial_baud', default_value='1000000')
    source_arg = DeclareLaunchArgument(
        'source', default_value='realsense',
        choices=['mjpeg', 'device', 'realsense'])
    device_arg = DeclareLaunchArgument('device', default_value='0')
    device_auto_exposure_arg = DeclareLaunchArgument(
        'device_auto_exposure', default_value='true')
    device_exposure_arg = DeclareLaunchArgument(
        'device_exposure', default_value='0.0')
    device_fps_arg = DeclareLaunchArgument('device_fps', default_value='30.0')
    device_width_arg = DeclareLaunchArgument('device_width', default_value='640')
    device_height_arg = DeclareLaunchArgument('device_height', default_value='480')
    command_interval_arg = DeclareLaunchArgument(
        'command_interval', default_value='0.03',
        description='Over UART a send is 0.1ms, so 0.03 is free')
    max_jog_deg_arg = DeclareLaunchArgument('max_jog_deg', default_value='5.0')
    max_jog_speed_arg = DeclareLaunchArgument(
        'max_jog_speed_deg_s', default_value='80.0')
    speed_at_100_arg = DeclareLaunchArgument(
        'speed_at_100_deg_s', default_value='52.0')
    rs_auto_exposure_arg = DeclareLaunchArgument(
        'rs_auto_exposure', default_value='true')
    rs_constant_fps_arg = DeclareLaunchArgument(
        'rs_constant_fps', default_value='true')
    rs_exposure_arg = DeclareLaunchArgument('rs_exposure', default_value='0.0')
    rs_width_arg = DeclareLaunchArgument('rs_width', default_value='640')
    rs_height_arg = DeclareLaunchArgument('rs_height', default_value='480')
    rs_fps_arg = DeclareLaunchArgument('rs_fps', default_value='30')
    rs_depth_arg = DeclareLaunchArgument(
        'rs_depth', default_value='true',
        description='Depth is needed for approach; on by default here')
    rs_serial_arg = DeclareLaunchArgument('rs_serial', default_value='')
    rs_align_arg = DeclareLaunchArgument(
        'rs_align_depth_to_color', default_value='false')

    # --- Button detector args ---
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='elevator_buttons.pt',
        description='Path to the trained YOLOv11n elevator button model')
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold', default_value='0.3')
    detector_device_arg = DeclareLaunchArgument(
        'detector_device', default_value='cuda:0',
        description='Inference device for the button detector')

    # --- Detection bridge args ---
    target_label_arg = DeclareLaunchArgument(
        'target_label', default_value='',
        description="Floor label to target, e.g. '3'. Empty = any button")

    # --- Servo args ---
    gain_arg = DeclareLaunchArgument('gain', default_value='0.45')
    command_lag_arg = DeclareLaunchArgument('command_lag', default_value='0.10',
        description='Lower than the TCP path: no network in the loop')
    lag_comp_arg = DeclareLaunchArgument('lag_compensation', default_value='true')
    max_step_arg = DeclareLaunchArgument('max_step_deg', default_value='5.0')
    lead_time_arg = DeclareLaunchArgument(
        'lead_time', default_value='0.0',
        description='Buttons do not move; velocity prediction is noise')
    velocity_smoothing_arg = DeclareLaunchArgument(
        'velocity_smoothing', default_value='0.6')
    auto_sign_arg = DeclareLaunchArgument('auto_sign', default_value='true')
    rate_arg = DeclareLaunchArgument('rate', default_value='30.0')
    progressive_gain_arg = DeclareLaunchArgument(
        'progressive_gain', default_value='2.0')
    deg_per_error_arg = DeclareLaunchArgument(
        'assumed_deg_per_error', default_value='25.0')
    v_deg_per_error_arg = DeclareLaunchArgument(
        'assumed_v_deg_per_error', default_value='0.0')
    deadband_arg = DeclareLaunchArgument(
        'deadband', default_value='0.02',
        description='Tighter than hand tracking: want precise button centering')
    approach_enabled_arg = DeclareLaunchArgument(
        'approach_enabled', default_value='true')
    depth_approach_arg = DeclareLaunchArgument(
        'depth_approach', default_value='true',
        description='Use real depth from the D405 instead of size proxy')
    target_depth_mm_arg = DeclareLaunchArgument(
        'target_depth_mm', default_value='50.0',
        description='Stop this many mm from the button (~2 inches)')
    target_size_arg = DeclareLaunchArgument(
        'target_size_fraction', default_value='0.45')
    skip_probe_arg = DeclareLaunchArgument('skip_probe', default_value='true')
    h_sign_arg = DeclareLaunchArgument('assumed_h_sign', default_value='1.0')
    v_sign_arg = DeclareLaunchArgument('assumed_v_sign', default_value='1.0')
    search_on_start_arg = DeclareLaunchArgument(
        'search_on_start', default_value='true',
        description='Begin hunting immediately')
    sweep_seconds_arg = DeclareLaunchArgument(
        'search_sweep_seconds', default_value='15.0')
    search_range_arg = DeclareLaunchArgument(
        'search_range_deg', default_value='180.0')
    lost_timeout_arg = DeclareLaunchArgument('lost_timeout', default_value='15.0')
    show_window_arg = DeclareLaunchArgument('show_window', default_value='false')
    ki_arg = DeclareLaunchArgument('ki', default_value='0.0')
    kd_arg = DeclareLaunchArgument('kd', default_value='0.0')
    integral_limit_arg = DeclareLaunchArgument('integral_limit', default_value='0.5')
    max_frame_age_arg = DeclareLaunchArgument('max_frame_age', default_value='0.2')

    bringup_dir = get_package_share_directory('mycobot_bringup')

    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'robot_bringup.launch.py')),
        launch_arguments={
            'robot_ip': LaunchConfiguration('robot_ip'),
            'robot_port': LaunchConfiguration('robot_port'),
            'camera_port': LaunchConfiguration('camera_port'),
            'stream_read_timeout': LaunchConfiguration('stream_read_timeout'),
            'home_on_start': LaunchConfiguration('home_on_start'),
            'connection': LaunchConfiguration('connection'),
            'serial_port': LaunchConfiguration('serial_port'),
            'serial_baud': LaunchConfiguration('serial_baud'),
            'source': LaunchConfiguration('source'),
            'device': LaunchConfiguration('device'),
            'device_auto_exposure': LaunchConfiguration('device_auto_exposure'),
            'device_exposure': LaunchConfiguration('device_exposure'),
            'device_fps': LaunchConfiguration('device_fps'),
            'device_width': LaunchConfiguration('device_width'),
            'device_height': LaunchConfiguration('device_height'),
            'command_interval': LaunchConfiguration('command_interval'),
            'max_jog_deg': LaunchConfiguration('max_jog_deg'),
            'max_jog_speed_deg_s': LaunchConfiguration('max_jog_speed_deg_s'),
            'speed_at_100_deg_s': LaunchConfiguration('speed_at_100_deg_s'),
            'rs_auto_exposure': LaunchConfiguration('rs_auto_exposure'),
            'rs_constant_fps': LaunchConfiguration('rs_constant_fps'),
            'rs_exposure': LaunchConfiguration('rs_exposure'),
            'rs_width': LaunchConfiguration('rs_width'),
            'rs_height': LaunchConfiguration('rs_height'),
            'rs_fps': LaunchConfiguration('rs_fps'),
            'rs_depth': LaunchConfiguration('rs_depth'),
            'rs_serial': LaunchConfiguration('rs_serial'),
            'rs_align_depth_to_color': LaunchConfiguration(
                'rs_align_depth_to_color'),
        }.items(),
    )

    button_detector = Node(
        package='mycobot_perception',
        executable='button_detector_node',
        name='button_detector_node',
        parameters=[{
            'model_path': _s('model_path'),
            'confidence_threshold': _f('confidence_threshold'),
            'device': _s('detector_device'),
            'show_window': _b('show_window'),
        }],
        output='screen',
        respawn=True,
        respawn_delay=3.0,
    )

    detection_bridge = Node(
        package='mycobot_perception',
        executable='detection_bridge_node',
        name='detection_bridge_node',
        parameters=[{
            'target_label': _s('target_label'),
            'use_depth': _b('rs_depth'),
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
            'point_topic': '/button/point_px',
            'rate': _f('rate'),
            'gain': _f('gain'),
            'progressive_gain': _f('progressive_gain'),
            'assumed_deg_per_error': _f('assumed_deg_per_error'),
            'assumed_v_deg_per_error': LaunchConfiguration(
                'assumed_v_deg_per_error'),
            'deadband': _f('deadband'),
            'lag_compensation': _b('lag_compensation'),
            'command_lag': _f('command_lag'),
            'max_step_deg': _f('max_step_deg'),
            'lead_time': _f('lead_time'),
            'velocity_smoothing': _f('velocity_smoothing'),
            'auto_sign': _b('auto_sign'),
            'lost_timeout': _f('lost_timeout'),
            'target_size_fraction': _f('target_size_fraction'),
            'approach_enabled': _b('approach_enabled'),
            'depth_approach': _b('depth_approach'),
            'target_depth_mm': _f('target_depth_mm'),
            'search_on_start': _b('search_on_start'),
            'search_sweep_seconds': _f('search_sweep_seconds'),
            'search_range_deg': _f('search_range_deg'),
            'skip_probe': _b('skip_probe'),
            'assumed_h_sign': _f('assumed_h_sign'),
            'assumed_v_sign': _f('assumed_v_sign'),
            'ki': _f('ki'),
            'kd': _f('kd'),
            'integral_limit': _f('integral_limit'),
            'max_frame_age': _f('max_frame_age'),
        }],
        output='screen',
        respawn=True,
        respawn_delay=3.0,
    )

    return LaunchDescription([
        # Robot / camera
        robot_ip_arg, robot_port_arg, camera_port_arg,
        stream_read_timeout_arg, home_on_start_arg,
        connection_arg, serial_port_arg, serial_baud_arg,
        source_arg, device_arg,
        device_auto_exposure_arg, device_exposure_arg,
        device_fps_arg, device_width_arg, device_height_arg,
        command_interval_arg,
        max_jog_deg_arg, max_jog_speed_arg, speed_at_100_arg,
        rs_auto_exposure_arg, rs_constant_fps_arg, rs_exposure_arg,
        rs_width_arg, rs_height_arg, rs_fps_arg,
        rs_depth_arg, rs_serial_arg, rs_align_arg,
        # Detector
        model_path_arg, confidence_threshold_arg, detector_device_arg,
        # Bridge
        target_label_arg,
        # Servo
        gain_arg, command_lag_arg, lag_comp_arg, max_step_arg,
        lead_time_arg, velocity_smoothing_arg, auto_sign_arg,
        rate_arg, progressive_gain_arg,
        deg_per_error_arg, v_deg_per_error_arg, deadband_arg,
        approach_enabled_arg, depth_approach_arg, target_depth_mm_arg,
        target_size_arg, skip_probe_arg,
        h_sign_arg, v_sign_arg, search_on_start_arg,
        sweep_seconds_arg, search_range_arg, lost_timeout_arg,
        show_window_arg, ki_arg, kd_arg, integral_limit_arg,
        max_frame_age_arg,
        # Nodes
        robot,
        button_detector,
        detection_bridge,
        servo,
    ])
