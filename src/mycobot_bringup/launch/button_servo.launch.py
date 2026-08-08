"""
Everything needed for elevator-button approach, in one command.

    ros2 launch mycobot_bringup button_servo.launch.py

Starts, in one terminal:
  - robot_state_publisher   (URDF TF tree)
  - mycobot_hardware_node   (arm driver: joint states, trajectories, jogging)
  - camera_node             (RealSense -> /camera/image_raw, with depth)
  - button_detector_node    (button model -> detections)
  - detection_bridge_node   (detections -> /button/point_px)
  - visual_servo_node       (image error -> /arm/jog)

This is servo_demo.launch.py's topology retargeted at a fixed button instead
of a moving hand, so the defaults differ in the ways that follow from that:

  - connection:=serial, serial_port:=/dev/ttyTHS1, serial_baud:=1000000 --
    the Jetson-on-the-arm's-own-UART path (see CLAUDE.md), not the Pi/TCP
    path servo_demo defaults to.
  - source:=realsense, rs_depth:=true -- buttons need range to know when to
    stop approaching, and the RealSense is what supplies real intrinsics.
  - approach_enabled:=true -- centring alone is not the job; closing in on
    the button is.
  - lead_time:=0.0 -- lead_time predicts where a MOVING target will be.
    Buttons don't move, so predicting motion only adds noise; proportional
    control off the current error is all this target needs.
  - search_on_start:=true -- start hunting immediately rather than waiting
    on a service trigger, since this is meant to run unattended as part of
    a larger approach sequence.

point_topic is hardcoded to /button/point_px -- there is no analogous
track:=hand/color switch here, only the one detector.

move_group and RViz are deliberately NOT included, for the same reason as
servo_demo: this bypasses MoveIt and jogs joints directly from image error.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


# Robot address, overridable without editing anything:
#   export MYCOBOT_IP=192.168.0.50      (whole shell session)
#   ros2 launch ... robot_ip:=1.2.3.4   (one run, wins over the env var)
DEFAULT_ROBOT_IP = os.environ.get('MYCOBOT_IP', '192.168.0.15')


# Launch arguments arrive as strings and are YAML-parsed on the way into a
# node, so an untyped numeric/boolean argument becomes whatever type the
# YAML parser guesses and a node declaring a different type refuses it at
# startup. Every numeric or boolean parameter below is therefore given an
# explicit type rather than left to inference.
def _f(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _b(name):
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def _s(name):
    return ParameterValue(LaunchConfiguration(name), value_type=str)


def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value=DEFAULT_ROBOT_IP)
    robot_port_arg = DeclareLaunchArgument('robot_port', default_value='9000')
    home_on_start_arg = DeclareLaunchArgument(
        'home_on_start', default_value='true',
        description='Drive to the home pose once on startup, so the arm '
                    'starts from a known position rather than wherever it '
                    'was left')
    connection_arg = DeclareLaunchArgument(
        'connection', default_value='serial', choices=['tcp', 'serial'],
        description='serial drives the arm over the Jetson\'s own UART '
                    '(/dev/ttyTHS1), with no Pi and no network in the arm '
                    'command path -- the target topology for this launch. '
                    'tcp falls back to the Pi at robot_ip')
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyTHS1')
    serial_baud_arg = DeclareLaunchArgument(
        'serial_baud', default_value='1000000')
    source_arg = DeclareLaunchArgument(
        'source', default_value='realsense',
        choices=['mjpeg', 'device', 'realsense'],
        description='Where frames come from. realsense is the default here '
                    'because it is what supplies both depth for the '
                    'approach and real camera intrinsics')
    command_interval_arg = DeclareLaunchArgument(
        'command_interval', default_value='0.06',
        description='Seconds between streamed arm commands. Safe on the '
                    'TCP path; over serial a send measures 0.1ms so 0.03 '
                    'is free.')
    max_jog_deg_arg = DeclareLaunchArgument(
        'max_jog_deg', default_value='5.0',
        description="Driver ceiling on a single jog. Must be >= the servo's "
                    'max_step_deg or steps are clipped while the servo '
                    'still credits them in full.')
    max_jog_speed_arg = DeclareLaunchArgument(
        'max_jog_speed_deg_s', default_value='80.0',
        description='Ceiling on jog velocity. The arm measures 52 deg/s at '
                    'speed=100, so above that is headroom rather than speed.')
    speed_at_100_arg = DeclareLaunchArgument(
        'speed_at_100_deg_s', default_value='52.0',
        description='Measured 2026-08-01. Sizes every streamed step, so a '
                    'value that is too high makes the arm trail its own goal.')
    rs_auto_exposure_arg = DeclareLaunchArgument(
        'rs_auto_exposure', default_value='true',
        description='Auto-exposure is a FRAME RATE control: in dim light '
                    'the sensor lengthens exposure past the frame period '
                    'and silently delivers a fraction of the requested rate.')
    rs_constant_fps_arg = DeclareLaunchArgument(
        'rs_constant_fps', default_value='true',
        description='Hold the requested rate even on auto-exposure, '
                    'accepting a darker image instead.')
    rs_exposure_arg = DeclareLaunchArgument(
        'rs_exposure', default_value='0.0',
        description='Microseconds. Only used with rs_auto_exposure:=false.')
    rs_width_arg = DeclareLaunchArgument('rs_width', default_value='640')
    rs_height_arg = DeclareLaunchArgument('rs_height', default_value='480')
    rs_fps_arg = DeclareLaunchArgument('rs_fps', default_value='30')
    rs_depth_arg = DeclareLaunchArgument(
        'rs_depth', default_value='true',
        description='Publish /camera/depth_raw. On by default here -- '
                    'approaching a button needs range, unlike plain '
                    'centring which is 2D')
    rs_serial_arg = DeclareLaunchArgument('rs_serial', default_value='')
    rs_align_arg = DeclareLaunchArgument(
        'rs_align_depth_to_color', default_value='false',
        description='Needed on a D435/D455, a no-op on a D405 whose colour '
                    'and depth come from the same imagers.')

    gain_arg = DeclareLaunchArgument(
        'gain', default_value='0.45',
        description='Fraction of the full centring correction applied at '
                    'the CENTRE of the frame. progressive_gain raises it '
                    'with distance, reaching 0.7 where the step clamp '
                    'takes over')
    command_lag_arg = DeclareLaunchArgument(
        'command_lag', default_value='0.15',
        description='Seconds from sending a jog to seeing it in a frame. '
                    'Keep BELOW the true lag (~0.25) -- too high makes the '
                    'compensator double-count landed jogs and reverse')
    lag_comp_arg = DeclareLaunchArgument(
        'lag_compensation', default_value='true',
        description='Predict where the target will be once jogs already '
                    'sent have landed. If disabled, drop gain to 0.3 as well')
    max_step_arg = DeclareLaunchArgument(
        'max_step_deg', default_value='5.0',
        description='Biggest single jog; must stay at or below the '
                    "driver's max_jog_deg (5.0), which clamps it anyway")
    lead_time_arg = DeclareLaunchArgument(
        'lead_time', default_value='0.0',
        description='Seconds to aim ahead of a MOVING target. Buttons '
                    "don't move, so this defaults off here, unlike "
                    'servo_demo where it tracks a moving hand')
    progressive_gain_arg = DeclareLaunchArgument(
        'progressive_gain', default_value='2.0',
        description='Scale the gain with distance from centre: effective '
                    'gain = gain * (1 + k * |error|). Keep '
                    'gain * (1 + k * 0.29) near 0.7 if you change either '
                    'number; 0 restores flat proportional')
    deadband_arg = DeclareLaunchArgument(
        'deadband', default_value='0.04',
        description='Image error below which the arm holds still.')
    rate_arg = DeclareLaunchArgument(
        'rate', default_value='30.0',
        description='How often the servo checks for a new sighting. Keep '
                    'at or above the camera rate')
    skip_probe_arg = DeclareLaunchArgument(
        'skip_probe', default_value='true',
        description='Track a target the moment it is seen, assuming the '
                    'camera mounting instead of measuring it. false runs '
                    'the orientation probe first')
    h_sign_arg = DeclareLaunchArgument(
        'assumed_h_sign', default_value='1.0',
        description='Flip to -1.0 if the arm drives the target '
                    'horizontally out of frame instead of centring it')
    v_sign_arg = DeclareLaunchArgument(
        'assumed_v_sign', default_value='1.0',
        description='Flip to -1.0 if the arm drives the target vertically '
                    'out of frame instead of centring it')
    search_on_start_arg = DeclareLaunchArgument(
        'search_on_start', default_value='true',
        description='Begin hunting immediately rather than waiting for the '
                    '/servo/search trigger -- this launch is meant to run '
                    'unattended')
    approach_enabled_arg = DeclareLaunchArgument(
        'approach_enabled', default_value='true',
        description='Close in on the button as well as centring it -- the '
                    'point of this launch, unlike servo_demo where it '
                    'defaults off')
    show_window_arg = DeclareLaunchArgument(
        'show_window', default_value='false',
        description='Open an OpenCV window from the detector (needs a '
                    'display)')

    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='elevator_buttons.pt',
        description='Button detection model weights')
    button_confidence_arg = DeclareLaunchArgument(
        'button_confidence', default_value='0.5',
        description='Minimum detector confidence to accept a detection')
    button_device_arg = DeclareLaunchArgument(
        'button_device', default_value='cuda:0',
        description='Inference device for the button detector')
    target_label_arg = DeclareLaunchArgument(
        'target_label', default_value='',
        description='Which detected label to steer at; empty follows '
                    'whatever the bridge picks by default')
    depth_approach_arg = DeclareLaunchArgument(
        'depth_approach', default_value='true',
        description='Use measured depth to gate/drive the approach rather '
                    'than image size alone')
    target_depth_mm_arg = DeclareLaunchArgument(
        'target_depth_mm', default_value='70.0',
        description='Depth at which the approach is considered complete. '
                    'A D405 is only valid from ~70mm to 500mm, so this sits '
                    'at the near edge of that range')
    lost_timeout_arg = DeclareLaunchArgument(
        'lost_timeout', default_value='15.0',
        description='Seconds without a sighting before returning home')

    bringup_dir = get_package_share_directory('mycobot_bringup')

    # Reuse robot_bringup rather than restating the driver and camera
    # parameters, so tuning values live in exactly one place.
    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'robot_bringup.launch.py')),
        # Plain LaunchConfiguration here, NOT the typed helpers -- these are
        # launch ARGUMENTS being forwarded to another launch file
        # (substitutions resolving to strings); the typing belongs where the
        # value reaches a node's `parameters`, which for all of these is
        # robot_bringup itself.
        launch_arguments={
            'robot_ip': LaunchConfiguration('robot_ip'),
            'robot_port': LaunchConfiguration('robot_port'),
            'home_on_start': LaunchConfiguration('home_on_start'),
            'connection': LaunchConfiguration('connection'),
            'serial_port': LaunchConfiguration('serial_port'),
            'serial_baud': LaunchConfiguration('serial_baud'),
            'source': LaunchConfiguration('source'),
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
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': _f('button_confidence'),
            'device': _s('button_device'),
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
            'deadband': _f('deadband'),
            'lag_compensation': _b('lag_compensation'),
            'command_lag': _f('command_lag'),
            'max_step_deg': _f('max_step_deg'),
            'lead_time': _f('lead_time'),
            'lost_timeout': _f('lost_timeout'),
            'approach_enabled': _b('approach_enabled'),
            'search_on_start': _b('search_on_start'),
            'skip_probe': _b('skip_probe'),
            'assumed_h_sign': _f('assumed_h_sign'),
            'assumed_v_sign': _f('assumed_v_sign'),
            'depth_approach': _b('depth_approach'),
            'target_depth_mm': _f('target_depth_mm'),
        }],
        output='screen',
        respawn=True,
        respawn_delay=3.0,
    )

    return LaunchDescription([
        robot_ip_arg,
        robot_port_arg,
        home_on_start_arg,
        connection_arg,
        serial_port_arg,
        serial_baud_arg,
        source_arg,
        command_interval_arg,
        max_jog_deg_arg,
        max_jog_speed_arg,
        speed_at_100_arg,
        rs_auto_exposure_arg,
        rs_constant_fps_arg,
        rs_exposure_arg,
        rs_width_arg,
        rs_height_arg,
        rs_fps_arg,
        rs_depth_arg,
        rs_serial_arg,
        rs_align_arg,
        gain_arg,
        command_lag_arg,
        lag_comp_arg,
        max_step_arg,
        lead_time_arg,
        progressive_gain_arg,
        deadband_arg,
        rate_arg,
        skip_probe_arg,
        h_sign_arg,
        v_sign_arg,
        search_on_start_arg,
        approach_enabled_arg,
        show_window_arg,
        model_path_arg,
        button_confidence_arg,
        button_device_arg,
        target_label_arg,
        depth_approach_arg,
        target_depth_mm_arg,
        lost_timeout_arg,
        robot,
        button_detector,
        detection_bridge,
        servo,
    ])
