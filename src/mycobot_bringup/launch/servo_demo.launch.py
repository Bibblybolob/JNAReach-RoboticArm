"""
Everything needed for finger-following, in one command.

    ros2 launch mycobot_bringup servo_demo.launch.py

Starts, in one terminal:
  - robot_state_publisher   (URDF TF tree)
  - mycobot_hardware_node   (arm driver: joint states, trajectories, jogging)
  - camera_node             (Pi MJPEG stream -> /camera/image_raw)
  - hand_tracker_node       (MediaPipe -> /hand/point_px)     track:=hand
    or color_tracker_node   (LAB threshold -> /color/point_px) track:=color
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

It then sweeps joint5 (wrist pitch) through +90 to -90 over 15s looking for a
hand and follows it, keeping it in the middle of the frame. After 15s with no
sighting it returns home and waits for the next trigger. To stop it at any
point:

    ros2 service call /servo/enable std_srvs/srv/SetBool "{data: false}"

By default it starts tracking the moment it sees a hand, assuming the camera
is mounted square on the flange. If the arm drives your hand OUT of frame
instead of centring it, flip the offending axis with assumed_h_sign:=-1.0 or
assumed_v_sign:=-1.0. Pass skip_probe:=false to measure the mounting instead,
which is slower (a few seconds of twitching, hold your hand still) but correct
for any orientation.

Tracking is proportional control with lag compensation, and no PID. Each
sighting moves the camera a FRACTION of the way to having the hand centred --
`gain`, not degrees. That fraction is not flat: it is 0.45 at the centre of
the frame and rises with distance (`progressive_gain`), reaching 0.7 at the
point where the step clamp takes over, so the arm is gentle when the hand is
nearly centred and at full authority when it is not. On its own a fraction
that large would
oscillate, because the camera pipeline is slow enough that two or three more
detections arrive reporting the old error while a correction is still in
flight, and the loop would re-command it every time. So the node remembers
every jog it sends and adds back the image motion those jogs have not
produced yet, correcting where the hand WILL be rather than where it was.
That is what makes 0.7 safe where the old PID at an effective 1.4 flicked
back and forth forever.

That stops it overshooting a hand held still. Centring a hand that is MOVING
is a different problem: a proportional loop tracks a moving target at a fixed
distance behind it, however high the gain, because the gap it settles at is
set mostly by dead time and dead time does not care about gain. At a moderate
pace that offset is around 70px of a 640-wide frame -- the arm shadows your
hand and never quite sits on it. So the node also measures how fast the hand
is crossing the image and aims `lead_time` (0.15s) ahead of it, which takes
about a third off that offset. If the arm follows a moving hand but always
sits behind it, that is the knob.

It does not do much for a hand waved quickly back and forth, which reverses
faster than any prediction can be right about, and it costs about a second of
extra settling on a hand that appears out of nowhere. lead_time:=0.0 restores
the old behaviour exactly.

Signs are worked out automatically now: auto_sign correlates what each jog was
predicted to do to the image against what it did, and flips an inverted axis
on its own. assumed_h_sign / assumed_v_sign are still there, but you should
not have to reach for them.

Closing in on the hand is OFF by default while tracking is being tuned. Turn
it back on with approach_enabled:=true.

If it still lags, watch the two report lines -- `tracker:` and `pipeline:` --
which give detections per second and how stale each one is. Below ~6/s
nothing tuned in the servo will help; use model_complexity:=0 and a smaller
camera frame.

"Move faster the farther the hand is from centre, slower as it gets close"
is what the gain profile does, but the useful half of it is the near half.
The loop asks for gain * error * assumed_deg_per_error degrees and
max_step_deg caps that at 5, so everything past error 0.29 -- 91px of a
640-wide frame, 69px vertically -- already commands the maximum and every jog
reads `+5.00`. Raising the cap does not speed up acquisition either, because
the arm's own top joint speed binds first: the error falls 0.80, 0.60, 0.40,
0.20 over the first half second at every gain and every clamp tried. So the
far field cannot go faster, and the win is in backing OFF near the centre.
The shipped profile does exactly that -- 0.4 deg at 10px out, 3.1 at 64px,
the full 5.0 from 91px -- and against a flat 0.7 it acquires a hand in 0.67s
rather than 1.18s, holds it 2px from centre rather than 6, and stops the error
alternating sign entirely.

If tilting works less well than panning, suspect the model before the tuning.
`assumed_deg_per_error` was one number for both axes, but the frame is wider
than it is tall, so the vertical edge is a smaller angle away --
`assumed_v_deg_per_error:=19.0` is the honest figure for a 640x480 sensor.
And if the camera is mounted rotated at all, the two axes are cross-coupled,
which a diagonal model cannot express at any scale: use skip_probe:=false to
measure the real 2x2. The `responds Nx as strongly as assumed` log line tells
you which case you are in.

Useful arguments:
    robot_ip:=192.168.0.15         Pi address
    connection:=serial             drive the arm over USB, no Pi in the path
    track:=color                   follow a colour blob, not a hand
    target_color:=red              which colour (red default)
    lead_time:=0.25                sit on a moving hand, not behind it (0.15)
    lead_time:=0.0                 back to proportional only
    gain:=0.6                      hotter near the centre (0.45 default)
    progressive_gain:=0.0          flat gain, ignoring distance (2.0 default)
    command_lag:=0.10              if it overshoots; NEVER raise it far
    lag_compensation:=false        back to plain P (then use gain:=0.3)
    deadband:=0.02                 sit nearer dead centre (0.04 default)
    target_landmark:=8             steer at the index fingertip, not the palm
    lost_timeout:=30.0             longer grace before homing
    approach_enabled:=true         close in as well as centring
    target_size_fraction:=0.55     closer approach (0.45 default)
    search_sweep_seconds:=25.0     slower sweep, more chance to lock on
    search_range_deg:=90.0         narrower sweep (+45 to -45)
    delegate:=gpu                  Tasks API on the GPU (needs a model +
                                   a GPU-capable mediapipe build)
    skip_probe:=false              measure the camera mounting first
    assumed_v_deg_per_error:=19.0  if tilting is weaker than panning
    assumed_h_sign:=-1.0           flip if it drives the hand out sideways
    assumed_v_sign:=-1.0           flip if it drives the hand out vertically
    search_on_start:=true          start hunting without the trigger
    home_on_start:=false           do not home on startup
    show_window:=true              OpenCV window (needs a display)
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
        'gain', default_value='0.45',
        description='Fraction of the full centring correction applied at the '
                    'CENTRE of the frame. Not degrees: 0.45 means "move '
                    'forty-five percent of the way to centred". '
                    'progressive_gain raises it with distance, reaching 0.7 '
                    'where the step clamp takes over')
    command_lag_arg = DeclareLaunchArgument(
        'command_lag', default_value='0.15',
        description='Seconds from sending a jog to seeing it in a frame. '
                    'Keep BELOW the true lag: too low just corrects a little '
                    'harder, too high makes the compensator double-count '
                    'landed jogs and reverse, which is the flicking it '
                    'exists to prevent')
    lag_comp_arg = DeclareLaunchArgument(
        'lag_compensation', default_value='true',
        description='Predict where the hand will be once jogs already sent '
                    'have landed, instead of correcting where it was. If you '
                    'turn this off, drop gain to 0.3 as well')
    max_step_arg = DeclareLaunchArgument(
        'max_step_deg', default_value='5.0',
        description='Biggest single jog. Times the detection rate, this is '
                    'the top speed the camera can slew. Raising it is rarely '
                    'the fix for slow tracking (5 to 9 moves the error under '
                    "3%); it must also stay at or below the driver's "
                    'max_jog_deg (5.0), which clamps it anyway')
    lead_time_arg = DeclareLaunchArgument(
        'lead_time', default_value='0.15',
        description='Seconds to aim AHEAD of a moving hand, using its '
                    'measured image speed. This is what centres a moving '
                    'hand instead of trailing it at a fixed distance. Raise '
                    'to track a moving hand more tightly, lower if a hand '
                    'that appears suddenly overshoots; 0 disables it')
    velocity_smoothing_arg = DeclareLaunchArgument(
        'velocity_smoothing', default_value='0.6',
        description='Noise filter on the hand-speed estimate that lead_time '
                    'multiplies. Velocity comes from differencing sightings, '
                    'which amplifies jitter, so lowering this makes the arm '
                    'chase noise')
    auto_sign_arg = DeclareLaunchArgument(
        'auto_sign', default_value='true',
        description='Work out from the tracking motion itself whether an '
                    'axis is inverted, and flip it. Replaces guessing at '
                    'assumed_h_sign / assumed_v_sign by hand')
    rate_arg = DeclareLaunchArgument(
        'rate', default_value='30.0',
        description='How often the servo checks for a new sighting. Keep at '
                    'or above the camera rate: the loop acts once per new '
                    'detection, so a lower rate both delays every correction '
                    'by up to 1/rate and silently discards the detections '
                    'that arrive in between')
    progressive_gain_arg = DeclareLaunchArgument(
        'progressive_gain', default_value='2.0',
        description='Scale the gain with distance from centre: effective '
                    'gain = gain * (1 + k * |error|). Gentle close in, full '
                    'authority far out. Against a flat gain 0.7 this acquires '
                    'a hand in 0.67s instead of 1.18s, holds it 2px from '
                    'centre instead of 6, and stops the error alternating '
                    'sign. Keep gain * (1 + k * 0.29) near 0.7 if you change '
                    'either number; 0 restores flat proportional')
    v_deg_per_error_arg = DeclareLaunchArgument(
        'assumed_v_deg_per_error', default_value='0.0',
        description='assumed_deg_per_error for the vertical axis only; 0 '
                    'means use the same value for both. The frame is wider '
                    'than it is tall, so the vertical edge is a smaller angle '
                    'away -- ~19 against 25 on a 640x480 sensor. Raise to '
                    'make tilting move less per unit error, lower for more')
    deg_per_error_arg = DeclareLaunchArgument(
        'assumed_deg_per_error', default_value='25.0',
        description='Degrees of joint motion that would fully centre a target '
                    'at the frame edge -- about half the camera field of '
                    'view. Geometry, not tuning; change it only if the lens '
                    'is unusually wide or narrow')
    deadband_arg = DeclareLaunchArgument(
        'deadband', default_value='0.04',
        description='Image error below which the arm holds still. 0.05 is a '
                    'hand comfortably in the middle of the picture. Raise it '
                    'if the arm buzzes, lower it to sit nearer dead centre')
    connection_arg = DeclareLaunchArgument(
        'connection', default_value='tcp', choices=['tcp', 'serial'],
        description="'serial' drives the arm's ESP32 directly over USB, "
                    'removing the Pi and the network from the arm command '
                    'path. Probe it first with scripts/probe_usb_arm.py')
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyUSB0')
    serial_baud_arg = DeclareLaunchArgument(
        'serial_baud', default_value='1000000')
    max_jog_deg_arg = DeclareLaunchArgument(
        'max_jog_deg', default_value='5.0',
        description="Driver ceiling on a single jog. Must be >= the servo's "
                    'max_step_deg or steps are clipped while the servo still '
                    'credits them in full.')
    max_jog_speed_arg = DeclareLaunchArgument(
        'max_jog_speed_deg_s', default_value='80.0',
        description='Ceiling on jog velocity. The arm measures 52 deg/s at '
                    'speed=100, so above that is headroom rather than speed.')
    command_interval_arg = DeclareLaunchArgument(
        'command_interval', default_value='0.06',
        description='Seconds between streamed arm commands. 0.06 is safe on '
                    'the TCP path, where each command crosses the network; '
                    'over serial a send measures 0.1ms, so 0.03 is free.')
    speed_at_100_arg = DeclareLaunchArgument(
        'speed_at_100_deg_s', default_value='52.0',
        description='Measured 2026-08-01. Sizes every streamed step, so a '
                    'value that is too high makes the arm trail its own goal.')
    rs_auto_exposure_arg = DeclareLaunchArgument(
        'rs_auto_exposure', default_value='true',
        description='Auto-exposure is a FRAME RATE control: in dim light the '
                    'sensor lengthens exposure past the frame period and '
                    'silently delivers a fraction of the requested rate.')
    rs_constant_fps_arg = DeclareLaunchArgument(
        'rs_constant_fps', default_value='true',
        description='Hold the requested rate even on auto-exposure, accepting '
                    'a darker image instead. Usually enough on its own.')
    rs_exposure_arg = DeclareLaunchArgument(
        'rs_exposure', default_value='0.0',
        description='Microseconds. Only used with rs_auto_exposure:=false. '
                    'Too short breaks detection outright, so measure.')
    rs_width_arg = DeclareLaunchArgument('rs_width', default_value='640')
    rs_height_arg = DeclareLaunchArgument('rs_height', default_value='480')
    rs_fps_arg = DeclareLaunchArgument('rs_fps', default_value='30')
    rs_depth_arg = DeclareLaunchArgument(
        'rs_depth', default_value='false',
        description='Publish /camera/depth_raw. Off by default: hand '
                    'detection is 2D and depth costs USB bandwidth.')
    rs_serial_arg = DeclareLaunchArgument('rs_serial', default_value='')
    rs_align_arg = DeclareLaunchArgument(
        'rs_align_depth_to_color', default_value='false',
        description='Needed on a D435/D455, a no-op on a D405 whose colour '
                    'and depth come from the same imagers.')

    source_arg = DeclareLaunchArgument(
        'source', default_value='mjpeg',
        choices=['mjpeg', 'device', 'realsense'],
        description='Where frames come from. mjpeg reads the Pi over HTTP; '
                    'device opens a camera plugged into THIS machine, which '
                    'removes the encode, the network hop and the decode '
                    'rather than making them faster')
    device_arg = DeclareLaunchArgument(
        'device', default_value='0',
        description='V4L2 index for source:=device, or a GStreamer pipeline')
    device_auto_exposure_arg = DeclareLaunchArgument(
        'device_auto_exposure', default_value='true',
        description='false trades brightness for frame rate -- auto-exposure caps the rate in dim light (10.2 vs 30.2 fps measured on this webcam)')
    device_exposure_arg = DeclareLaunchArgument(
        'device_exposure', default_value='0.0',
        description='manual exposure value; 0 keeps the default')
    device_fps_arg = DeclareLaunchArgument(
        'device_fps', default_value='30.0',
        description='frame rate requested from a local camera')
    device_width_arg = DeclareLaunchArgument(
        'device_width', default_value='640',
        description='local camera width')
    device_height_arg = DeclareLaunchArgument(
        'device_height', default_value='480',
        description='local camera height')
    track_arg = DeclareLaunchArgument(
        'track', default_value='hand', choices=['hand', 'color'],
        description="What to follow. 'color' runs color_tracker_node instead "
                    'of MediaPipe and points the servo at it. The control '
                    'loop is identical -- only the measurement changes, and '
                    'it changes for the better: a blob centroid is steadier '
                    'than a hand landmark and costs 0.2ms against 8-19ms')
    target_color_arg = DeclareLaunchArgument(
        'target_color', default_value='red',
        description='Colour to follow when track:=color (red, green, blue, '
                    'yellow). Tune with lab_bounds if the lighting fights it')
    delegate_arg = DeclareLaunchArgument(
        'delegate', default_value='cpu',
        description="Hand detection backend: 'cpu' is mp.solutions.hands, "
                    "which has no GPU option at all. 'gpu' switches to the "
                    "Tasks API and needs both a hand_landmarker.task bundle "
                    "and a mediapipe build with GPU support (the stock Linux "
                    "wheels are CPU-only). 'auto' tries GPU and falls back "
                    'quietly. The backend actually used is logged at startup')
    hand_model_arg = DeclareLaunchArgument(
        'hand_model_path', default_value='',
        description='Path to hand_landmarker.task for delegate:=gpu. Empty '
                    'uses ~/hand_landmarker.task')
    model_complexity_arg = DeclareLaunchArgument(
        'model_complexity', default_value='0',
        description='MediaPipe hand model: 0 is ~2x faster than 1. On a '
                    'CPU-bound host fresher detections smooth the servo loop '
                    'more than extra landmark precision does. Use 1 if you '
                    'have GPU inference')
    target_landmark_arg = DeclareLaunchArgument(
        'target_landmark', default_value='9',
        description='MediaPipe landmark to steer at. 9 is the middle-finger '
                    'knuckle, i.e. the centre of the palm, and is the '
                    'steadiest point on a hand because it does not move when '
                    'fingers flex. 8 is the index fingertip -- more precise '
                    'to point with, but it makes the arm chase finger jitter')
    ki_arg = DeclareLaunchArgument(
        'ki', default_value='0.0',
        description='Integral gain. 0 by default because this loop is dead-'
                    'time limited: the integral winds up across the interval '
                    'between commanding a jog and seeing it, which is exactly '
                    'the interval that causes overshoot. Measured, not '
                    'assumed -- see CLAUDE.md.')
    kd_arg = DeclareLaunchArgument(
        'kd', default_value='0.0',
        description='Derivative gain. 0 by default; lead_time already does '
                    'the anticipating, from measured target velocity rather '
                    'than from differentiating a noisy error.')
    integral_limit_arg = DeclareLaunchArgument(
        'integral_limit', default_value='0.5',
        description='Windup clamp on the integral vector magnitude.')
    reprobe_after_arg = DeclareLaunchArgument(
        'reprobe_after', default_value='20',
        description='Consecutive growing updates before re-measuring the '
                    'Jacobian. A moving hand produces six routinely, so this '
                    'has to sit well above the sign heuristic.')
    reprobe_cooldown_arg = DeclareLaunchArgument(
        'reprobe_cooldown', default_value='20.0')
    aim_down_arg = DeclareLaunchArgument(
        'aim_offset_down_m', default_value='0.0',
        description='Hold the camera this many METRES below the target rather '
                    'than centred on it. 0.0254 is one inch.')
    aim_right_arg = DeclareLaunchArgument(
        'aim_offset_right_m', default_value='0.0')
    aim_range_arg = DeclareLaunchArgument(
        'aim_range_m', default_value='0.1',
        description='Standoff the offsets are computed at. A D405 cannot '
                    'measure below ~70mm, so this is told, not read.')
    max_reprobes_arg = DeclareLaunchArgument(
        'max_reprobes', default_value='4',
        description='Re-measure the Jacobian when tracking stops converging. '
                    'A rotated camera gives a pose-dependent Jacobian, so one '
                    'probe is only locally valid.')
    probe_retries_arg = DeclareLaunchArgument(
        'probe_retries', default_value='3',
        description='Probes to attempt before falling back to the assumed '
                    'orientation. One momentary loss of the target should not '
                    'end the run.')
    max_frame_age_arg = DeclareLaunchArgument(
        'max_frame_age', default_value='0.2',
        description='Drop camera frames already older than this instead of '
                    'tracking on them. A stale detection describes a place '
                    'the hand has left; skipping it costs one detection and '
                    'recovers the whole delay. 0 disables')
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
        'approach_enabled', default_value='false',
        description='Close in on the hand as well as centring it. Off until '
                    'tracking is solid: it is a second loop on a second axis '
                    'driven by a much noisier signal, and it moves the camera '
                    'the centring loop is trying to hold steady')
    sweep_seconds_arg = DeclareLaunchArgument(
        'search_sweep_seconds', default_value='15.0',
        description='Seconds for one traverse of the search sweep. Slower '
                    'gives MediaPipe more clean frames to lock on')
    search_range_arg = DeclareLaunchArgument(
        'search_range_deg', default_value='180.0',
        description='Total sweep travel, centred where the search began. '
                    'Home leaves joint5 at 0, so 180 swings +90 to -90')
    skip_probe_arg = DeclareLaunchArgument(
        'skip_probe', default_value='true',
        description='Track a hand the moment it is seen, assuming the camera '
                    'mounting instead of measuring it. false runs the '
                    'orientation probe first, which is slower but correct for '
                    'any mounting')
    h_sign_arg = DeclareLaunchArgument(
        'assumed_h_sign', default_value='1.0',
        description='Flip to -1.0 if the arm drives the hand horizontally out '
                    'of frame instead of centring it')
    v_sign_arg = DeclareLaunchArgument(
        'assumed_v_sign', default_value='1.0',
        description='Flip to -1.0 if the arm drives the hand vertically out '
                    'of frame instead of centring it')
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
        # Plain LaunchConfiguration here, NOT the typed helpers. These are
        # launch ARGUMENTS being forwarded to another launch file, which are
        # substitutions resolving to strings; the typing belongs where the
        # value reaches a node's `parameters`, which for all of these is
        # robot_bringup. Passing ParameterValue here fails at load with
        # "'ParameterValue' object is not iterable", which does not point
        # anywhere near the mistake.
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

    color_tracker = Node(
        package='mycobot_perception',
        executable='color_tracker_node',
        name='color_tracker_node',
        condition=LaunchConfigurationEquals('track', 'color'),
        parameters=[{
            'target_color': _s('target_color'),
            'show_window': _b('show_window'),
            'publish_annotated': LaunchConfiguration('show_window'),
        }],
        output='screen',
        respawn=True,
        respawn_delay=3.0,
    )

    hand_tracker = Node(
        package='mycobot_perception',
        executable='hand_tracker_node',
        condition=LaunchConfigurationEquals('track', 'hand'),
        name='hand_tracker_node',
        parameters=[{
            'show_window': _b('show_window'),
            # Encoding and publishing an annotated frame costs real CPU per
            # frame, and nothing subscribes to it in this launch. Host load is
            # what stalls the stack and drops both links, so this is off unless
            # you are actually looking at the window.
            'publish_annotated': LaunchConfiguration('show_window'),
            'model_complexity': _i('model_complexity'),
            'delegate': _s('delegate'),
            'target_landmark': _i('target_landmark'),
            'max_frame_age': _f('max_frame_age'),
            'probe_retries': _i('probe_retries'),
            'max_reprobes': _i('max_reprobes'),
            'ki': _f('ki'),
            'kd': _f('kd'),
            'integral_limit': _f('integral_limit'),
            'reprobe_after': _i('reprobe_after'),
            'reprobe_cooldown': _f('reprobe_cooldown'),
            'aim_offset_down_m': _f('aim_offset_down_m'),
            'aim_offset_right_m': _f('aim_offset_right_m'),
            'aim_range_m': _f('aim_range_m'),
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
            # Whichever tracker is running publishes the contract the servo
            # consumes, so switching targets is a topic change and nothing
            # more -- the control law does not know the difference.
            'point_topic': PythonExpression([
                "'/color/point_px' if '", LaunchConfiguration('track'),
                "' == 'color' else '/hand/point_px'"]),
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
            'search_on_start': _b('search_on_start'),
            'search_sweep_seconds': _f('search_sweep_seconds'),
            'search_range_deg': _f('search_range_deg'),
            'skip_probe': _b('skip_probe'),
            'assumed_h_sign': _f('assumed_h_sign'),
            'assumed_v_sign': _f('assumed_v_sign'),
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
        command_lag_arg,
        lag_comp_arg,
        max_step_arg,
        lead_time_arg,
        velocity_smoothing_arg,
        auto_sign_arg,
        deg_per_error_arg,
        v_deg_per_error_arg,
        rate_arg,
        progressive_gain_arg,
        deadband_arg,
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
        rs_auto_exposure_arg,
        rs_constant_fps_arg,
        rs_exposure_arg,
        rs_width_arg,
        rs_height_arg,
        rs_fps_arg,
        rs_depth_arg,
        rs_serial_arg,
        rs_align_arg,
        device_height_arg,
        track_arg,
        target_color_arg,
        model_complexity_arg,
        delegate_arg,
        hand_model_arg,
        target_landmark_arg,
        ki_arg,
        kd_arg,
        integral_limit_arg,
        reprobe_after_arg,
        reprobe_cooldown_arg,
        aim_down_arg,
        aim_right_arg,
        aim_range_arg,
        max_reprobes_arg,
        probe_retries_arg,
        max_frame_age_arg,
        show_window_arg,
        lost_timeout_arg,
        target_size_arg,
        approach_enabled_arg,
        sweep_seconds_arg,
        search_range_arg,
        skip_probe_arg,
        h_sign_arg,
        v_sign_arg,
        search_on_start_arg,
        robot,
        color_tracker,
        hand_tracker,
        servo,
    ])
