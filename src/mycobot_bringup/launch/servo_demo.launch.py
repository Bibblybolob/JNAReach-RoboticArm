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
`gain`, 0.7 by default, not degrees. On its own a fraction that large would
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

If the arm seems not to chase a far-out hand any harder than a nearly-centred
one, that is the step clamp rather than the gain. The loop asks for
gain * error * assumed_deg_per_error degrees and max_step_deg caps it at 5, so
everything past error 0.29 -- 91px of a 640-wide frame -- is already commanding
the maximum, and every jog reads `+5.00`. Raising the cap does not speed up
acquisition either, because the arm's own top joint speed binds first: in
simulation the error falls 0.80, 0.60, 0.40, 0.20 over the first half second
at every gain and every clamp tried. What a higher gain does change is the
endgame, where it rings instead of settling.

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
    lead_time:=0.25                sit on a moving hand, not behind it (0.15)
    lead_time:=0.0                 back to proportional only
    gain:=0.9                      follow harder still (0.7 default)
    command_lag:=0.10              if it overshoots; NEVER raise it far
    lag_compensation:=false        back to plain P (then use gain:=0.3)
    deadband:=0.02                 sit nearer dead centre (0.04 default)
    target_landmark:=8             steer at the index fingertip, not the palm
    lost_timeout:=30.0             longer grace before homing
    approach_enabled:=true         close in as well as centring
    target_size_fraction:=0.55     closer approach (0.45 default)
    search_sweep_seconds:=25.0     slower sweep, more chance to lock on
    search_range_deg:=90.0         narrower sweep (+45 to -45)
    skip_probe:=false              measure the camera mounting first
    assumed_v_deg_per_error:=19.0  if tilting is weaker than panning
    progressive_gain:=1.0          chase a far-out hand superlinearly
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
        'gain', default_value='0.7',
        description='Fraction of the full centring correction to apply each '
                    'time the hand is seen. Not degrees: 0.7 means "move '
                    'seventy percent of the way to centred". Safe this high '
                    'only because of lag compensation; turn that off and '
                    'anything above 0.35 oscillates')
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
    progressive_gain_arg = DeclareLaunchArgument(
        'progressive_gain', default_value='0.0',
        description='Make the correction grow faster than the error does '
                    '(effective gain = gain * (1 + k * |error|)). Plain '
                    'proportional is already linear in the error; this makes '
                    'it superlinear. 0 because measurement says it does not '
                    'help -- past 91px from centre max_step_deg is already '
                    'commanding the maximum, so this only adds ringing near '
                    'the centre. Try it and watch whether seen= settles')
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
    max_frame_age_arg = DeclareLaunchArgument(
        'max_frame_age', default_value='0.12',
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
            'target_landmark': LaunchConfiguration('target_landmark'),
            'max_frame_age': LaunchConfiguration('max_frame_age'),
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
            'progressive_gain': LaunchConfiguration('progressive_gain'),
            'assumed_deg_per_error': LaunchConfiguration('assumed_deg_per_error'),
            'assumed_v_deg_per_error': LaunchConfiguration(
                'assumed_v_deg_per_error'),
            'deadband': LaunchConfiguration('deadband'),
            'lag_compensation': LaunchConfiguration('lag_compensation'),
            'command_lag': LaunchConfiguration('command_lag'),
            'max_step_deg': LaunchConfiguration('max_step_deg'),
            'lead_time': LaunchConfiguration('lead_time'),
            'velocity_smoothing': LaunchConfiguration('velocity_smoothing'),
            'auto_sign': LaunchConfiguration('auto_sign'),
            'lost_timeout': LaunchConfiguration('lost_timeout'),
            'target_size_fraction': LaunchConfiguration('target_size_fraction'),
            'approach_enabled': LaunchConfiguration('approach_enabled'),
            'search_on_start': LaunchConfiguration('search_on_start'),
            'search_sweep_seconds': LaunchConfiguration('search_sweep_seconds'),
            'search_range_deg': LaunchConfiguration('search_range_deg'),
            'skip_probe': LaunchConfiguration('skip_probe'),
            'assumed_h_sign': LaunchConfiguration('assumed_h_sign'),
            'assumed_v_sign': LaunchConfiguration('assumed_v_sign'),
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
        progressive_gain_arg,
        deadband_arg,
        model_complexity_arg,
        target_landmark_arg,
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
        hand_tracker,
        servo,
    ])
