"""
Image-based visual servoing: find a hand, centre it, and approach it.

This is the eye-in-hand demo that needs NO calibration. It never computes a 3D
position. It only knows "the target is 40 pixels left of centre" and turns that
into "move the joint that shifts the view right". Because the camera rides on
the end effector, image error maps directly to joint motion.

That means no camera intrinsics, no hand-eye transform, no depth estimate --
none of which exist yet. Those unlock better things later; this works today.


BEHAVIOUR

The arm is idle at home until you ask for it:

    ros2 service call /servo/search std_srvs/srv/Trigger

Then it runs a state machine:

    IDLE      sitting at home, doing nothing
    SEARCHING sweeping slowly to bring a hand into view
    TRACKING  centring the hand and closing in on it
    HOMING    returning to the home pose, then back to IDLE

Losing the target briefly holds position -- a hand that blinked out is usually
about to reappear. Past resume_search_after (4s) it goes back to sweeping, and
only after lost_timeout (15s) without any sighting does it give up and home. A
search that has never seen a hand keeps sweeping rather than homing: it was
asked to hunt. Call /servo/enable false to stop it at any point.


THE CONTROL LAW IS PROPORTIONAL, PLUS TWO FEEDFORWARD TERMS

There is no integral and no derivative term, and that is not an omission.
What there is instead is a model of the two things a bare proportional loop
gets wrong here: its own dead time, and the target's motion.

This loop has a lot of dead time in it. A frame is captured on the Pi, JPEG
encoded, pushed over the network, decoded, run through MediaPipe on a CPU, and
smoothed, before the servo ever sees it -- and then the jog it produces takes
its own time to actually move the arm. Several tenths of a second pass between
"the hand is 40% left of centre" and "the camera has finished responding to
that". Meanwhile detections keep arriving reporting the *old* error, because
the correction has not landed yet.

Each update commands some fraction of the motion that would fully centre the
target. If that fraction times the number of updates that elapse during the
dead time comes out above 1.0, the loop commands more correction than the
error deserves, sails past centre, and does the same thing coming back: a
steady oscillation that never settles. Adding an integral term to "finish the
job" makes it strictly worse, because the integral winds up during exactly the
interval when the correction is in flight but not yet visible. An early
version of this node did that, at an effective fraction near 1.4, and it
flicked back and forth without ever converging.

Backing the fraction off to 0.3 fixes the oscillation but buys stability with
speed, and the arm then visibly trails a moving hand. So instead of tiptoeing
around the dead time, the loop MODELS it. Every jog is remembered with the
time it was sent. When a detection arrives, the node adds back the image
motion produced by jogs that this frame is too old to show yet, giving an
estimate of where the hand is NOW rather than where it was. Correcting that
estimate means never re-commanding a correction already on its way -- and
without that duplication, the fraction can be 0.7 instead of 0.3.

Simulating the whole pipeline (0.25s true lag, 8 detections/s, 5 deg cap):

    gain 1.44, no compensation     oscillates forever   <- the original PID
    gain 0.30, no compensation     settles in 1.3s
    gain 0.70, no compensation     oscillates forever
    gain 0.70, with compensation   settles in 0.7s

That fixes overshoot on a hand holding still. It does NOT make the arm centre
a hand that is moving, and those are separate failures with separate causes.

A proportional loop chasing a moving target settles at a constant distance
behind it. Each cycle it removes `gain` of the error it can see while the
target opens a fresh gap, and the two balance at

    offset = target_speed * (detection_interval / gain + round_trip_lag)

At 8 detections/s, gain 0.7 and 0.25s of lag, a hand crossing the frame at
half a half-frame per second settles 0.23 out -- 73px of a 640-wide frame,
for as long as it keeps moving. Raising gain hardly touches it, because the
dead-time term does not contain gain. The arm shadows the hand at a fixed
offset, which looks precisely like "it is keeping it in view rather than
centring it", and it is the reason that complaint survives all the work above.

So the loop also aims where the hand is GOING. It measures the hand's image
velocity between sightings, subtracts the part of that motion caused by its
own jogs, and leads the target by `lead_time`. This is feedforward, not an
integral: it is recomputed from the two most recent sightings every cycle and
holds no accumulated state, so it cannot wind up during the dead time the way
the original PID's integral did.

Simulating the pipeline against these actual functions, mean distance from
centre while the hand keeps moving:

    hand speed     lead_time 0    lead_time 0.15
    0.25           39px           25px            -35%
    0.50           73px           49px            -33%
    1.00          145px           97px            -33%

It holds up whether command_lag over- or under-states the true lag: getting
that wrong makes the velocity estimate too SMALL (the own-motion subtraction
below leaves some of our own jog in the measurement), so the term under-leads
and degrades toward plain proportional rather than overshooting.

Two things it does NOT fix, both worth knowing before reaching for the knob:

  * A hand waved quickly back and forth barely improves (-13% to +7%). That
    motion reverses faster than any lead can be right about, and it is also
    the case that runs into the slew ceiling below.
  * A hand that appears suddenly takes about a second longer to centre. The
    jump reads as speed, so the loop leads a hand that is not actually going
    anywhere. Lower lead_time if that matters more than tracking does.

Above all of this sits a ceiling no control law reaches past:

    top slew = (max_step_deg / assumed_deg_per_error) / detection interval

5 deg, 25 deg-per-unit and 8 detections/s give 1.6 half-frames per second,
and a briskly waved hand exceeds that. Raising max_step_deg is the obvious
response and does almost nothing (under 3% from 5 to 9) because the clamp
only binds when the hand is already far out, where dead time dominates. The
detection rate is what actually moves it: 5/s to 15/s nearly halves the error
on that same wave. Read the `tracker:` and `pipeline:` lines first.

The one thing to be careful with is `command_lag`, and it is asymmetric. Too
LOW is harmless -- some in-flight motion goes uncounted and the loop corrects
slightly harder than it needs to. Too HIGH is not: the window then sweeps in
jogs that have already landed and are already visible in the measurement, the
compensator counts them twice, decides it has overshot, and reverses. That
reversal is a flicking oscillation, which is the thing being fixed. So it sits
at 0.15 against a true lag nearer 0.25, deliberately.

THE GAIN IS A PROFILE, NOT A NUMBER

"Move faster the farther the hand is from centre, slower as it closes in" is
what proportional control already does -- but only out to error 0.29, because
max_step_deg caps the step at 5 degrees and every error past 91px of a
640-wide frame (69px vertically) commands that same maximum. The far half of
that ask is not available: the arm's own joint speed binds before the clamp
does, and the error falls 0.80, 0.60, 0.40, 0.20 over the first half second at
every gain and every clamp setting tried.

The half that IS available is backing off near the centre, and it turns out to
be worth more than the far half would have been. `gain` is now the gain at the
centre (0.45) and `progressive_gain` (2.0) raises it with distance, meeting
0.7 -- the old flat value -- exactly where the clamp takes over:

     10px from centre   eff gain 0.47   0.4 deg
     64px               0.62            3.1
     91px               0.70            5.0   <- clamp from here out
    256px               1.16            5.0

Against that flat 0.7, from a hand appearing at error 0.8: acquisition
1.18s -> 0.67s, residual 6px -> 2px, and the error stops changing sign in the
tail altogether (8.9 reversals -> 0). Faster and steadier at once, because the
loop no longer arrives at the centre carrying more speed than it can shed
inside the dead time.

It costs a little on a hand moving steadily -- 50px -> 56px at a moderate
pace, since the near field it damps is where a tracked hand sits. `lead_time`
is the term for moving hands; this one is for arriving and staying.

The remaining knobs, in the order worth touching them:

    lead_time          0.15   how far ahead of a moving hand to aim; the one
                              that decides centred vs merely in frame
    gain               0.45   fraction of the full correction, AT THE CENTRE
    progressive_gain    2.0   how fast that fraction grows with distance;
                              keep gain * (1 + this * 0.29) near 0.7
    max_step_deg        5.0   ceiling on one jog; with the detection rate,
                              this sets the top speed the camera can slew,
                              and must not exceed the driver's max_jog_deg.
                              Raising it is rarely the answer -- see above
    command_lag        0.15   see above; lower is safe, higher is not
    velocity_smoothing  0.6   noise filter on the velocity estimate; lower
                              chases MediaPipe jitter
    assumed_deg_per_error 25  geometry, not tuning -- half the camera FOV

And the signs no longer need guessing at the command line. `auto_sign`
correlates what each jog was predicted to do to the image against what it
actually did, and flips an axis that turns out to be wired backwards, saying
so in the log. Same answer the probe gives, obtained from ordinary tracking
motion instead of a twitching calibration phase.

Jogging is armed automatically at startup -- the node calls /arm/jog_enable
itself -- so no manual service calls are needed beyond the search trigger.


APPROACHING WITHOUT DEPTH

Centring alone leaves the arm at whatever distance it started. To close in
without a depth sensor, the loop uses apparent size: the tracker reports palm
width in pixels, and a hand that grows in frame is a hand getting nearer. The
approach joint is driven until the palm reaches target_size_fraction of the
frame width.

The limits of this are worth knowing. Apparent size is not distance -- a large
hand and a close hand look identical -- so this closes in on a consistent
*framing*, not a measured standoff. And approach naturally ends by breaking
itself: once the hand fills the frame MediaPipe can no longer see the whole
hand, tracking drops, and the lost-target timeout takes over.


WHY IT CALIBRATES ITSELF FIRST

The mapping from image error to joint motion depends on how the camera is
physically rotated on the flange. Mounted upright, "target is left" means
"rotate joint1 one way". Rotated 90 degrees, the same error needs a different
joint entirely; rotated 180, the same joint but the opposite sign. Guessing
wrong means the arm drives the target OUT of frame -- a runaway, and exactly
the failure you do not want when the target is someone's hand.

Rather than asking you to measure the mount or work out signs by trial and
error, the node measures it: it jogs each joint a little, watches which way the
tracked point moved in the image, and builds the 2x2 Jacobian relating joint
motion to image motion. Inverting that gives the control law. This is standard
visual-servo Jacobian estimation, it takes a few seconds, and it is correct for
whatever orientation you actually mounted the camera in.

The approach axis is probed the same way: nudge the approach joint, see whether
the palm grew or shrank, and keep the sign.

If probing fails (no hand visible, arm blocked), the node refuses to servo
rather than falling back to a guess.
"""

from __future__ import annotations

import math
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from control_msgs.msg import JointJog
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import SetBool, Trigger


# State machine states.
IDLE = 'IDLE'
SEARCHING = 'SEARCHING'
TRACKING = 'TRACKING'
HOMING = 'HOMING'


class VisualServoNode(Node):

    def __init__(self) -> None:
        super().__init__('visual_servo_node')

        self.declare_parameter('point_topic', '/hand/point_px')
        # CameraInfo, not the image stream. All this node needs from the
        # camera is the frame size, and it only needs it once -- subscribing
        # to /camera/image_raw for that meant deserialising a ~900KB bgr8
        # frame every cycle, forever, to read two integers it already had.
        # On a host whose stalls were dropping the arm connection, that was
        # megabytes per second of pure waste.
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('jog_topic', '/arm/jog')

        # Joints used to steer the view. joint1 swings the arm horizontally;
        # joint5 is the wrist pitch, which tilts the camera vertically. These
        # are a starting guess only -- the probe below measures what each one
        # actually does to the image and adapts.
        self.declare_parameter('horizontal_joint', 'joint1')
        self.declare_parameter('vertical_joint', 'joint5')

        # --- Proportional gain (the only gain) ---
        # Error is normalised to half-frames: 1.0 means the target sits at the
        # frame edge.
        #
        # This is a FRACTION, not degrees. 0.7 means "each time you see the
        # hand, move seventy percent of the way to having it centred".
        #
        # A fraction this close to 1.0 is only safe because of the lag
        # compensation below. Without it, corrections still in flight get
        # re-commanded by every detection that arrives before they land, and
        # anything above about 0.35 oscillates. With it, the loop subtracts
        # what it has already asked for and can afford to be decisive.
        # NOT the whole gain any more -- this is the gain at the CENTRE, and
        # progressive_gain below raises it with distance. 0.45 here with
        # progressive_gain 2.0 gives an effective 0.7 at the point where
        # max_step_deg starts clamping, so the far field is unchanged from
        # when this was a flat 0.7 and only the endgame is gentler.
        self.declare_parameter('gain', 0.45)
        # Make the correction grow FASTER than the error does:
        #
        #     effective gain = gain * (1 + progressive_gain * |error|)
        #
        # 0 is plain proportional. That is already "farther out means a bigger
        # jog" -- the step is linear in the error -- but only out to error
        # 0.29 (91px of a 640-wide frame, 69px vertically), because
        # max_step_deg clamps the step at 5 degrees and everything beyond that
        # commands the same maximum. So the useful half of "far fast, near
        # slow" is not the far half. It is the near half.
        #
        # At the defaults the profile runs:
        #
        #     10px from centre   eff gain 0.47   0.4 deg
        #     32px               0.53            1.3
        #     64px               0.62            3.1
        #     91px               0.70            5.0  <- clamp from here out
        #    256px               1.16            5.0
        #
        # Measured from a hand appearing at error 0.8, against the flat
        # gain 0.7 this replaces: acquisition 1.18s -> 0.67s, residual 6px ->
        # 2px, and the sign of the error stops alternating altogether (8.9
        # reversals in the tail -> 0). Faster AND steadier, because the loop
        # no longer arrives at the centre carrying more speed than it can shed
        # inside the dead time.
        #
        # An earlier pass concluded this did not help. That pass added the
        # curve ON TOP of gain 0.7 rather than reprofiling around it, so the
        # near field got hotter instead of cooler and it rang -- the opposite
        # of the point. Keep gain * (1 + progressive_gain * 0.29) near 0.7
        # when changing either number.
        #
        # It costs a little on a hand that is moving steadily: mean offset
        # 50px -> 56px at a moderate pace, because the near field it damps is
        # exactly where a tracked hand sits. lead_time is the term that
        # addresses moving hands; this one addresses arriving and staying.
        self.declare_parameter('progressive_gain', 2.0)
        # --- Lag compensation ---
        # The one change that lets this track fast instead of carefully.
        #
        # Corrections take time to appear: the arm has to move, the Pi has to
        # capture and encode a frame showing it moved, and MediaPipe has to
        # find the hand in that frame. Detections arriving during that window
        # report the error as it was BEFORE the correction, so a naive loop
        # commands the same motion again, and again, and overshoots by however
        # many detections fit in the gap. That is the entire reason the old
        # build flicked left and right.
        #
        # So keep every jog sent, and when a measurement arrives, add back the
        # image motion the not-yet-visible jogs are about to produce. What is
        # left is an estimate of where the hand actually is NOW rather than
        # where it was, and the loop corrects that instead.
        self.declare_parameter('lag_compensation', True)
        # Seconds from publishing a jog to that motion showing up in a frame's
        # timestamp: arm response plus capture, JPEG, network and MediaPipe.
        #
        # DELIBERATELY SET BELOW THE TRUE LAG. The two directions of error are
        # not symmetric, and the asymmetry is sharp. Set it too high and the
        # window sweeps in jogs that have ALREADY landed and are already in
        # the measurement; the compensator counts them twice, concludes it has
        # overshot, and reverses -- which is a flicking oscillation, the exact
        # failure this exists to remove. Set it too low and some genuinely
        # in-flight motion goes uncounted, so the loop corrects slightly more
        # than it should, which at these gains merely converges a little
        # faster.
        #
        # Simulating the pipeline at a true lag of 0.25s: gains from 0.5 to
        # 1.1 all settle in ~0.7s at command_lag 0.10-0.15, while 0.35 and
        # above oscillate at every gain above 0.3. One notch back from that
        # cliff is the right place to sit, so 0.15 -- and raising it is the
        # wrong response to a slow pipeline. Fix the pipeline instead.
        self.declare_parameter('command_lag', 0.15)
        # Never let the compensator invent more than this much error. It is an
        # open-loop prediction; if the arm did not actually execute a jog (link
        # dropped, joint at a limit, driver rejected it) the prediction is
        # wrong, and an unbounded wrong prediction is a runaway.
        self.declare_parameter('max_compensation', 0.8)
        # --- Target velocity feedforward ---
        # What makes the difference between keeping a moving hand IN FRAME and
        # holding it AT THE CENTRE.
        #
        # Lag compensation above answers "where is the hand now, given my jogs
        # in flight". It says nothing about the hand's own motion, and a
        # proportional loop chasing a moving target settles at a constant
        # distance behind it rather than on it. That is not a tuning miss, it
        # is what proportional control does with a ramp: each cycle it removes
        # `gain` of the error it can see, while the target opens up a fresh
        # gap. Simulating the pipeline, the gap converges to
        #
        #     offset = target_speed * (detection_interval / gain + round_trip_lag)
        #
        # At 8 detections/s, gain 0.7 and 0.25s of lag, a hand crossing the
        # frame at half a half-frame per second parks 0.23 out -- about 73px
        # of a 640-wide frame, permanently, however long you wait. Raising
        # gain barely touches it: the dead-time term does not contain gain,
        # and gain is capped by stability anyway. The arm ends up shadowing
        # the hand at a fixed offset, which reads exactly as "it is keeping it
        # in view rather than centring it".
        #
        # The fix is to aim where the hand is GOING. Measure how fast it is
        # crossing the image, subtract the part of that motion which is our
        # own jogs, and lead the target by the time a correction takes to show
        # up. This is a feedforward term, not an integral: it is computed
        # fresh from the two most recent sightings and carries no accumulated
        # state, so it cannot wind up during the dead time the way the
        # original PID's integral did.
        #
        # Seconds to lead by. At 0.15, simulated against these functions, the
        # trailing offset on a moving hand falls by about a third at every
        # speed tried (73px -> 49px at 0.5 half-frames/s on a 640 frame).
        #
        # But leading is prediction, and predicting further costs step
        # response: a hand that appears out of nowhere reads as enormous
        # speed, so the loop leads a hand that is not going anywhere and takes
        # roughly a second longer to settle on it. Raising this past ~0.25
        # trades away more settling than it wins back in tracking. Set 0 to
        # disable and get the old proportional-only behaviour exactly.
        self.declare_parameter('lead_time', 0.15)
        # Smoothing on the velocity estimate. Velocity comes from differencing
        # consecutive detections, and differencing amplifies noise: a couple
        # of pixels of MediaPipe jitter across a 0.125s gap looks like real
        # speed, and lead_time multiplies it straight into the error. Lowering
        # this makes the arm chase jitter for no tracking benefit.
        self.declare_parameter('velocity_smoothing', 0.6)
        # Ceiling on the estimated target speed, in half-frames per second. A
        # dropped detection, a jump to a different hand, or a tracker glitch
        # produces one enormous difference; without a clamp that becomes one
        # enormous lead and the arm lunges.
        self.declare_parameter('max_target_speed', 3.0)
        # Ceiling on the lead itself, in half-frames. max_target_speed alone is
        # not enough protection: at 3.0 and lead_time 0.15 a saturated estimate
        # still adds 0.45, which is most of the way to the frame edge and
        # swamps the real error.
        self.declare_parameter('max_lead', 0.25)
        # How many consecutive saturated velocity estimates before the lead is
        # abandoned as unreliable.
        #
        # A real hand cannot travel at max_target_speed for seconds on end, so
        # a pinned estimate does not mean "fast hand", it means the own-motion
        # subtraction in _update_velocity is wrong -- and the usual reason is
        # that the jogs it is subtracting never actually executed. A joint held
        # by a limit, blocked, or simply not keeping up produces exactly that:
        # the node predicts image motion, none arrives, and the difference is
        # booked as the target moving at enormous speed in the direction that
        # makes the error look bigger. The lead then drives harder, which
        # sustains the saturation. That feedback ran on real hardware with a
        # joint1 that was not turning, and it turned a stuck axis into a
        # runaway.
        #
        # So treat sustained saturation as a broken model rather than a fast
        # target: drop the lead and say why. Tracking degrades to plain
        # proportional, which is the correct behaviour when the thing the lead
        # depends on cannot be trusted.
        self.declare_parameter('velocity_saturation_limit', 8)
        # Ignore errors smaller than this (normalised). MediaPipe jitter and
        # hand tremor are both a few pixels; without a deadband the arm hunts
        # continuously and buzzes. 0.04 of a half-frame is a hand sitting
        # comfortably in the middle of the picture.
        self.declare_parameter('deadband', 0.04)
        # Loop rate. The driver rate-limits jogs to its command_interval
        # (0.06s, ~16Hz), so going much above that just discards commands.
        self.declare_parameter('rate', 15.0)
        # Stop if the target has not been seen for this long. Without it, the
        # arm keeps acting on a stale position after the hand leaves frame.
        self.declare_parameter('target_timeout', 0.9)
        # Cap per cycle, and a hard ceiling on how fast the view can slew:
        #
        #     top speed = (max_step_deg / assumed_deg_per_error) / detection interval
        #
        # At 5 deg, 25 deg-per-unit and 8 detections/s that is 1.6 half-frames
        # per second. Below about 4 the arm cannot keep up with a hand moving
        # at any pace, which reads as "it follows but always lags behind".
        #
        # Raising it above 5 looks like the fix for that and is not: simulated
        # against a waved hand, 5 -> 9 moved the mean error by under 3%. The
        # clamp only binds once the hand is already far off centre (past about
        # 0.29 of a half-frame at gain 0.7), and by then the loop is limited
        # by round-trip dead time, so a bigger step overshoots rather than
        # catches up. lead_time and the detection rate are the levers that
        # move that number.
        #
        # MUST NOT EXCEED the driver's max_jog_deg (5.0), which clamps every
        # jog. Setting this higher silently caps the real motion while the lag
        # compensator still credits the full amount, so the loop believes
        # corrections landed that never did.
        self.declare_parameter('max_step_deg', 5.0)

        # Probe settings. probe_deg is how far each joint is nudged to measure
        # its effect; big enough to produce clear image motion, small enough to
        # be a twitch.
        self.declare_parameter('probe_deg', 4.0)
        self.declare_parameter('probe_settle', 1.2)
        # Start tracking the instant a hand is seen, instead of spending
        # several seconds twitching joints to measure the camera mounting.
        #
        # The probe exists because a wrong sign drives the target OUT of
        # frame. Skipping it means trusting the assumption below instead of
        # measuring, so if the camera is mounted rotated the arm will move
        # the wrong way -- recoverable, since losing the target homes after
        # lost_timeout, but it will not track until the signs are right.
        self.declare_parameter('skip_probe', True)
        # Degrees of joint motion for a FULL correction of a unit image error,
        # used only when the probe is skipped. This is what the probe would
        # otherwise measure, and it is a geometry number rather than a tuning
        # knob: with the camera on the flange, rotating the steering joint by
        # X degrees swings the view by about X degrees, so bringing a target
        # at the frame edge (error 1.0) to the centre takes about half the
        # camera's field of view. ~50 deg horizontal FOV on a typical webcam
        # puts that at 25. `gain` then applies a fraction of it.
        self.declare_parameter('assumed_deg_per_error', 25.0)
        # Same number for the VERTICAL axis, when it differs. 0 means "use
        # assumed_deg_per_error for both", which is the old behaviour.
        #
        # It usually does differ, because the frame is not square. The error is
        # normalised per axis -- horizontal by width/2, vertical by height/2 --
        # so a unit error means "at the edge" in both directions, but the edges
        # are different angular distances away. A 640x480 sensor with ~50 deg
        # horizontal FOV has only ~38 vertical, which puts the honest vertical
        # number nearer 19 than 25.
        #
        # Raise this to make the vertical axis move LESS per unit error, lower
        # it to make it move more. If tilting consistently under-shoots where
        # panning does not, this is the knob -- but check the `responds Nx as
        # strongly as assumed` log line first, which measures the ratio rather
        # than guessing it, and consider skip_probe:=false, since a camera
        # mounted rotated needs off-diagonal Jacobian terms that no per-axis
        # scalar can express.
        self.declare_parameter('assumed_v_deg_per_error', 0.0)
        # Flip either of these if the arm drives the hand out of frame along
        # that axis rather than centring it.
        self.declare_parameter('assumed_h_sign', 1.0)
        self.declare_parameter('assumed_v_sign', 1.0)
        # Check those assumed signs against reality while tracking, and flip
        # one if it is provably backwards.
        #
        # This is the probe's job done without the probe. Every jog is a tiny
        # experiment whose result shows up in the next detection: we asked the
        # image to move left by this much, did it? Correlating what was
        # commanded against what was observed answers the sign question from
        # the tracking motion itself, with no twitching phase and no guessing
        # at the command line -- which is what you were doing by hand with
        # assumed_h_sign:=-1.0.
        #
        # It also measures how far off assumed_deg_per_error is, and reports
        # that, because a mounting can be the right way round and still be
        # twice as responsive as assumed.
        self.declare_parameter('auto_sign', True)
        # Correlation samples to gather before trusting the verdict. Each
        # sample is one detection, so ~25 is about three seconds of tracking.
        self.declare_parameter('auto_sign_samples', 25)
        # +1 means extending the approach joint makes the hand look bigger.
        # 0 disables closing in without disabling centring.
        self.declare_parameter('assumed_approach_sign', 1.0)

        # --- Approach ---
        # Joint driven to close distance. joint3 (elbow) extends and retracts
        # the arm, which moves the flange-mounted camera along its view more
        # than the other joints do. Whichever joint is chosen, the probe
        # measures its actual effect on apparent size, so a poor choice shows
        # up as a failed probe rather than as wrong motion.
        self.declare_parameter('approach_joint', 'joint3')
        # Off by default. Closing in is a second control loop, on a second
        # axis, driven by a range proxy (apparent palm size) that is far
        # noisier than the position error -- and it moves the camera, which
        # perturbs the centring loop it shares the arm with. Get tracking
        # solid first, then turn this on.
        self.declare_parameter('approach_enabled', False)
        # Stop closing in when palm width reaches this fraction of frame width.
        # Higher gets closer; too high and MediaPipe loses the hand because it
        # no longer fits in frame, which ends the approach abruptly.
        self.declare_parameter('target_size_fraction', 0.45)
        self.declare_parameter('approach_gain', 6.0)
        self.declare_parameter('approach_deadband', 0.03)
        self.declare_parameter('max_approach_step_deg', 1.5)

        # --- Search / idle behaviour ---
        # Seconds without a sighting before giving up and homing.
        self.declare_parameter('lost_timeout', 15.0)
        # Seconds of a lost target to tolerate before going back to sweeping.
        # Below this it holds still, since a hand that blinked out is usually
        # about to reappear in the same place. Above it, holding is just
        # standing still while the thing you are looking for is somewhere
        # else -- and with skip_probe a single spurious detection is enough
        # to enter TRACKING, so without this the sweep froze a degree in and
        # sat there until lost_timeout expired.
        self.declare_parameter('resume_search_after', 4.0)
        # Sweep the wrist pitch while looking for a hand. joint5 tilts the
        # flange-mounted camera through its whole vertical arc, so a single
        # sweep covers far more of the room than panning the base does.
        self.declare_parameter('search_joint', 'joint5')
        # Total travel, centred on wherever the sweep began. The home pose
        # leaves joint5 at 0, and search follows homing, so 180 means the
        # camera really does swing +90 to -90.
        self.declare_parameter('search_range_deg', 180.0)
        # Seconds for one traverse of that range. The step size is derived
        # from this and the measured time since the last tick rather than
        # being a fixed number of degrees per loop -- on a host that stalls,
        # a fixed step makes the sweep take however long the stalls add up
        # to, which is why the old sweep crawled. Pacing against the clock
        # keeps the sweep honest whatever the loop rate does.
        self.declare_parameter('search_sweep_seconds', 15.0)
        # Never let one catch-up step become a lunge after a stall. The
        # driver clamps to max_jog_deg anyway; this keeps intent local.
        self.declare_parameter('search_max_step_deg', 2.5)
        # Begin searching as soon as the node starts, instead of waiting for
        # the trigger. Off by default: launching a file should not set the arm
        # hunting around the room.
        self.declare_parameter('search_on_start', False)
        # Arm the driver's jog gate automatically at startup.
        self.declare_parameter('auto_arm_jog', True)

        self._h_joint = self.get_parameter('horizontal_joint').value
        self._v_joint = self.get_parameter('vertical_joint').value
        self._gain = float(self.get_parameter('gain').value)
        self._progressive_gain = float(
            self.get_parameter('progressive_gain').value)
        self._deadband = float(self.get_parameter('deadband').value)
        self._rate = float(self.get_parameter('rate').value)
        self._lag_comp = bool(self.get_parameter('lag_compensation').value)
        self._command_lag = float(self.get_parameter('command_lag').value)
        self._max_comp = float(self.get_parameter('max_compensation').value)
        self._lead_time = float(self.get_parameter('lead_time').value)
        self._vel_smoothing = float(
            self.get_parameter('velocity_smoothing').value)
        self._max_target_speed = float(
            self.get_parameter('max_target_speed').value)
        self._max_lead = float(self.get_parameter('max_lead').value)
        self._vel_sat_limit = int(
            self.get_parameter('velocity_saturation_limit').value)
        self._auto_sign = bool(self.get_parameter('auto_sign').value)
        self._auto_sign_samples = int(
            self.get_parameter('auto_sign_samples').value)

        # Jogs published but not yet visible in a detection, as
        # (ros_time_seconds, np.array([d_horizontal, d_vertical])). Kept in ROS
        # time, not monotonic, because they get compared against image
        # timestamps, which camera_node writes from the ROS clock.
        self._sent: list[tuple[float, np.ndarray]] = []

        # Forward image Jacobian: joint degrees -> image error. The inverse of
        # _jinv, cached because the control law needs _jinv and the lag
        # compensator needs this one, every cycle.
        self._jfwd: np.ndarray | None = None

        # Correlation accumulators for auto_sign, per axis:
        #   _corr[i] = sum(predicted_change * observed_change)
        #   _pmag[i] = sum(predicted_change ** 2)
        # The ratio _corr/_pmag is a least-squares estimate of how much of the
        # predicted image motion actually happened. Near +1 means the model is
        # right; near -1 means that axis is inverted; near 0 means the joint is
        # not steering that image direction at all.
        self._corr = np.zeros(2, dtype=float)
        self._pmag = np.zeros(2, dtype=float)
        self._corr_n = 0
        self._sign_verdict = [False, False]
        # Previous accepted measurement, for computing observed image motion
        # between one detection and the next: (ros_stamp, error array).
        self._prev_meas: tuple[float, np.ndarray] | None = None

        # Estimated target velocity in half-frames per second, and the reading
        # it was last measured against. Deliberately NOT sharing _prev_meas
        # with the sign estimator: that one is cleared every auto_sign_samples
        # detections when it reaches a verdict, and a velocity estimate that
        # restarts from nothing every few seconds is worse than none.
        self._vel = np.zeros(2, dtype=float)
        self._vel_prev: tuple[float, np.ndarray] | None = None
        # Consecutive velocity estimates that came out pinned at
        # max_target_speed, and whether the lead has been given up on as a
        # result. See velocity_saturation_limit.
        self._vel_saturated = 0
        self._vel_untrusted = False

        # Rolling pipeline statistics, so "it lags" stops being a guess.
        self._lat_sum = 0.0
        self._lat_n = 0
        self._det_first = 0.0
        self._det_count = 0
        self._timeout = float(self.get_parameter('target_timeout').value)
        self._max_step = float(self.get_parameter('max_step_deg').value)
        self._probe_deg = float(self.get_parameter('probe_deg').value)
        self._probe_settle = float(self.get_parameter('probe_settle').value)
        self._skip_probe = bool(self.get_parameter('skip_probe').value)
        self._assumed_deg = float(
            self.get_parameter('assumed_deg_per_error').value)
        # 0 is the "same as horizontal" sentinel, so the single-number case
        # keeps working untouched.
        self._assumed_v_deg = float(
            self.get_parameter('assumed_v_deg_per_error').value)
        if self._assumed_v_deg <= 0.0:
            self._assumed_v_deg = self._assumed_deg
        self._assumed_h_sign = float(self.get_parameter('assumed_h_sign').value)
        self._assumed_v_sign = float(self.get_parameter('assumed_v_sign').value)
        self._assumed_approach_sign = float(
            self.get_parameter('assumed_approach_sign').value)

        self._approach_joint = self.get_parameter('approach_joint').value
        self._approach_enabled = bool(self.get_parameter('approach_enabled').value)
        self._target_size = float(self.get_parameter('target_size_fraction').value)
        self._approach_gain = float(self.get_parameter('approach_gain').value)
        self._approach_deadband = float(self.get_parameter('approach_deadband').value)
        self._max_approach_step = float(
            self.get_parameter('max_approach_step_deg').value)

        self._lost_timeout = float(self.get_parameter('lost_timeout').value)
        self._resume_search_after = float(
            self.get_parameter('resume_search_after').value)
        # Whether a lock has ever been held. A fresh search sweeps until it
        # finds something; only after having had a target and lost it does
        # the lost_timeout homing apply.
        self._had_lock = False
        self._search_joint = self.get_parameter('search_joint').value
        self._search_range = float(self.get_parameter('search_range_deg').value)
        self._sweep_seconds = float(
            self.get_parameter('search_sweep_seconds').value)
        self._search_max_step = float(
            self.get_parameter('search_max_step_deg').value)

        # Sign of d(palm size)/d(approach joint), learned by the probe. Without
        # it we would not know whether extending the joint moves the camera
        # toward the hand or away from it.
        self._approach_sign = 0.0

        # Latest palm width in pixels, from point.z.
        self._last_size_px: float | None = None

        # State machine.
        self._state = IDLE
        self._state_since = time.monotonic()
        # Sweep direction and accumulated travel while SEARCHING.
        self._search_dir = 1.0
        self._search_travel = 0.0
        # Set only by /servo/search, so a fresh hunt sweeps from zero while
        # a resume after a lost lock carries on from where it was.
        self._restart_sweep = True
        # Wall-clock of the previous sweep tick, so each step can be sized
        # from real elapsed time instead of assuming the loop ran on time.
        self._last_sweep_time: float | None = None

        # Frame size, learned from the first image. Needed to normalise pixel
        # error; we do not assume 640x480.
        self._width: int | None = None
        self._height: int | None = None

        self._last_point: tuple[float, float] | None = None
        self._last_point_time = 0.0
        # ROS-clock timestamp of the frame the last detection came from, as
        # opposed to _last_point_time, which is when the message arrived here.
        self._last_point_stamp = 0.0
        # Timestamp of the measurement the control loop last acted on. The
        # loop runs faster than detections arrive, so without this it re-uses
        # the same reading for several iterations -- and since each correction
        # is a fraction of the error it sees, re-using one reading commands
        # that fraction two or three times for a single observation, which is
        # over-correction by another name. One sighting, one command.
        self._acted_point_time = 0.0
        # Consecutive updates on which the error GREW on each axis. A control
        # law with the right sign shrinks the error; one that only ever grows
        # it is pushing the target out of frame, which on this rig means the
        # assumed camera orientation has a sign inverted.
        self._growing = [0, 0]
        # Consecutive updates spent far from centre without improving. A
        # growing error means the wrong sign; an error that simply refuses to
        # shrink while the joint travels means that axis is not steering the
        # image at all -- a swapped axis rather than a flipped one. Both look
        # like "it will not centre", so both need naming.
        self._stuck = [0, 0]
        self._best_abs_err = [float('inf'), float('inf')]
        self._prev_abs_err: tuple[float, float] | None = None
        # Distinguishes "the tracker has never said anything" (wrong topic, node
        # not running, MediaPipe not detecting) from "the hand is momentarily
        # out of frame". These need completely different fixes, and reporting
        # both as "hold your hand in view" sends you hunting for the wrong one.
        self._ever_received_point = False

        # Inverse image Jacobian: maps normalised image error to joint deltas.
        # None until probing succeeds; servoing refuses to run without it.
        self._jinv: np.ndarray | None = None
        self._probed = False
        self._probing = False

        cb = ReentrantCallbackGroup()

        self._point_topic = self.get_parameter('point_topic').value
        self._info_topic_name = self.get_parameter('camera_info_topic').value

        self._point_sub = self.create_subscription(
            PointStamped, self._point_topic,
            self._point_cb, 1, callback_group=cb)
        self._info_sub = self.create_subscription(
            CameraInfo, self._info_topic_name,
            self._camera_info_cb, 10, callback_group=cb)
        self._jog_pub = self.create_publisher(
            JointJog, self.get_parameter('jog_topic').value, 1)

        # Servoing is armed by default now; the search trigger is what actually
        # sets the arm moving, so a second gate here served no purpose.
        self._enabled = True
        self._enable_srv = self.create_service(
            SetBool, 'servo/enable', self._enable_cb, callback_group=cb)
        self._search_srv = self.create_service(
            Trigger, 'servo/search', self._search_cb, callback_group=cb)

        # Clients for the driver's gate and homing service.
        self._jog_enable_cli = self.create_client(
            SetBool, '/arm/jog_enable', callback_group=cb)
        self._home_cli = self.create_client(
            SetBool, '/arm/home', callback_group=cb)

        self._timer = self.create_timer(
            1.0 / self._rate, self._servo_step, callback_group=cb)

        if bool(self.get_parameter('auto_arm_jog').value):
            # Deferred: the driver may not be up yet at construction time.
            self._arm_timer = self.create_timer(
                2.0, self._arm_jog_once, callback_group=cb)

        if bool(self.get_parameter('search_on_start').value):
            self._set_state(SEARCHING)

        self.get_logger().info(
            'Visual servo ready, idle at home. Start it with:\n'
            '    ros2 service call /servo/search std_srvs/srv/Trigger\n'
            f'It will home again after {self._lost_timeout:.0f}s without a '
            'sighting.'
        )

    # ---- Driver gate ----

    def _arm_jog_once(self) -> None:
        """Enable the driver's jog gate, retrying until the driver appears."""
        if not self._jog_enable_cli.service_is_ready():
            self.get_logger().info(
                'Waiting for /arm/jog_enable (is the driver running?)...',
                throttle_duration_sec=5.0)
            return
        # Only report success, and only stop retrying, once the driver has
        # actually answered. The previous version fired the request, logged
        # "Armed", and cancelled the retry timer without ever looking at the
        # result -- so if the driver was busy (homing used to block this
        # service outright) the gate stayed shut while the log claimed
        # otherwise, and every later jog was silently discarded. That is why
        # jogging had to be switched on by hand.
        req = SetBool.Request()
        req.data = True
        future = self._jog_enable_cli.call_async(req)

        def _armed(fut):
            try:
                res = fut.result()
            except Exception as e:
                self.get_logger().warn(
                    f'/arm/jog_enable call failed ({e}); will retry.')
                return
            if res is not None and getattr(res, 'success', True):
                self.get_logger().info(
                    'Armed the driver jog gate (/arm/jog_enable).')
                self._arm_timer.cancel()
            else:
                self.get_logger().warn(
                    '/arm/jog_enable refused; will retry.')

        future.add_done_callback(_armed)

    # ---- State machine ----

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self.get_logger().info(f'{self._state} -> {state}')
        self._state = state
        self._state_since = time.monotonic()
        self._reset_tracking()
        if state == SEARCHING:
            # Only a hunt started from scratch sweeps from zero. Resuming
            # after a lost lock keeps its progress: zeroing it there meant
            # every flicker of a detection restarted the sweep where it
            # stood, so the arm twitched in place instead of ever getting
            # anywhere. _restart_sweep is set by the search service.
            if self._restart_sweep:
                self._search_travel = 0.0
                self._restart_sweep = False
            # Drop the previous tick's timestamp either way, or the first
            # step is sized from however long the node spent elsewhere.
            self._last_sweep_time = None

    def _search_cb(self, request, response):
        if not self._enabled:
            response.success = False
            response.message = 'Servo is disabled; enable it first.'
            return response
        self._restart_sweep = True
        self._set_state(SEARCHING)
        response.success = True
        response.message = (
            'Searching for a hand. Hold one in view; it will home again after '
            f'{self._lost_timeout:.0f}s without a sighting.'
        )
        return response

    def _go_home(self) -> None:
        """Ask the driver to return to its fixed home pose."""
        if not self._home_cli.service_is_ready():
            self.get_logger().warn(
                '/arm/home unavailable; staying put instead of homing.')
            self._set_state(IDLE)
            return
        req = SetBool.Request()
        req.data = True
        future = self._home_cli.call_async(req)
        # Homing sets _in_motion in the driver, which makes it ignore jogs for
        # the duration, so we simply wait for it rather than commanding motion.
        future.add_done_callback(lambda _f: self._set_state(IDLE))

    # ---- Inputs ----

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._width is None and msg.width > 0 and msg.height > 0:
            self._width, self._height = msg.width, msg.height
            self.get_logger().info(f'Frame size: {msg.width}x{msg.height}')

    def _point_cb(self, msg: PointStamped) -> None:
        if not self._ever_received_point:
            self.get_logger().info(
                f'First target sighting on {self._point_topic}.')
            # This only means the tracker sees a hand -- it fires regardless
            # of the servo state machine, so on its own it does not mean the
            # arm will move. Say so explicitly rather than leaving IDLE users
            # staring at a hand-detected log wondering why nothing happens.
            if self._state == IDLE:
                self.get_logger().info(
                    'Servo is IDLE, so this sighting will not move the arm. '
                    'Start it with: '
                    'ros2 service call /servo/search std_srvs/srv/Trigger')
            self._ever_received_point = True
        self._last_point = (msg.point.x, msg.point.y)
        # z carries palm width in pixels, not a depth. See hand_tracker_node.
        self._last_size_px = msg.point.z if msg.point.z > 0 else None
        self._last_point_time = time.monotonic()

        # The frame's own timestamp, carried through by the tracker. This is
        # what the lag compensator reasons in: it has to know WHEN the scene
        # in this measurement was, not when the message showed up.
        now_ros = self._ros_now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # A zero or wildly-off stamp means nothing upstream is stamping
        # properly; fall back to now, which degrades to no compensation
        # rather than to nonsense compensation.
        if not (0.0 < now_ros - stamp < 5.0):
            stamp = now_ros
        self._last_point_stamp = stamp

        self._lat_sum += now_ros - stamp
        self._lat_n += 1
        if self._det_count == 0:
            self._det_first = now_ros
        self._det_count += 1
        self._report_pipeline(now_ros)

    def _enable_cb(self, request, response):
        """Master on/off. Probing now happens on the SEARCHING -> TRACKING
        transition instead of here, so this is purely a kill switch: disabling
        stops the arm immediately, wherever it is in the state machine.
        """
        want = bool(request.data)
        # Clear the convergence diagnostics either way, so a warning about the
        # previous attempt cannot fire against the next one.
        self._reset_tracking()
        self._enabled = want

        if not want:
            self._set_state(IDLE)
            response.message = 'Servo disabled; arm stopped where it is.'
        else:
            response.message = (
                'Servo enabled and idle. Trigger a hunt with: '
                'ros2 service call /servo/search std_srvs/srv/Trigger'
            )
        response.success = True
        self.get_logger().info(response.message)
        return response

    # ---- Error ----

    def _current_error(self):
        """Normalised (ex, ey) offset of the target from image centre.

        Units are half-frames: +1.0 means the target sits at the right/bottom
        edge. Normalising means gain does not have to be retuned when the
        camera resolution changes.
        """
        if self._width is None or self._last_point is None:
            return None
        if time.monotonic() - self._last_point_time > self._timeout:
            return None
        px, py = self._last_point
        ex = (px - self._width / 2.0) / (self._width / 2.0)
        ey = (py - self._height / 2.0) / (self._height / 2.0)
        return ex, ey

    def _wait_for_fresh_point(self, timeout=2.0):
        """Block until a target sighting newer than now arrives."""
        mark = time.monotonic()
        deadline = mark + timeout
        while time.monotonic() < deadline:
            if self._last_point is not None and self._last_point_time > mark:
                return self._last_point
            time.sleep(0.05)
        return None

    def _ros_now(self) -> float:
        """ROS clock in seconds.

        The lag compensator compares command times against image timestamps,
        and camera_node stamps images from the ROS clock, so both sides of
        that comparison have to live in the same clock. monotonic() is used
        elsewhere for plain elapsed-time checks where it does not matter.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    def _report_pipeline(self, now_ros: float) -> None:
        """Say, in numbers, how fresh the data driving the arm actually is.

        Everything about how hard this loop can push comes down to two
        figures: how often a detection arrives, and how old it is when it
        does. Tuning gain without knowing them is guesswork, and "it lags"
        is not a measurement.
        """
        if self._det_count < 40:
            return
        span = now_ros - self._det_first
        rate = self._det_count / span if span > 0 else 0.0
        latency = self._lat_sum / max(self._lat_n, 1)
        note = ''
        # Note what is NOT suggested here: raising command_lag to match. A
        # command_lag above the true lag makes the compensator double-count
        # corrections that have already landed and reverse, which oscillates.
        # Latency is a pipeline problem and wants a pipeline fix.
        if rate < 6.0:
            note = (' -- too few detections to track a moving hand; '
                    'model_complexity:=0 and a smaller camera frame '
                    'help most')
        elif latency > 0.35:
            note = (' -- stale enough to limit how fast the arm can safely '
                    'follow; the fix is a faster pipeline, not a bigger '
                    'command_lag')
        self.get_logger().info(
            f'pipeline: {rate:.1f} detections/s, frames {latency * 1000:.0f}ms '
            f'old when acted on{note}')
        self._det_count = 0
        self._lat_sum = 0.0
        self._lat_n = 0

    def _record_sent(self, stamp: float, dh: float, dv: float) -> None:
        """Remember a jog so the compensator knows it is still in flight."""
        self._sent.append((stamp, np.array([dh, dv], dtype=float)))
        # Anything older than a couple of lags has certainly landed and been
        # seen; keeping it would just make the list grow without bound.
        cutoff = stamp - max(2.0 * self._command_lag, 1.0)
        if len(self._sent) > 4:
            self._sent = [s for s in self._sent if s[0] >= cutoff]

    def _sent_between(self, t0: float, t1: float) -> np.ndarray:
        """Total commanded motion issued in (t0, t1]."""
        total = np.zeros(2, dtype=float)
        for when, delta in self._sent:
            if t0 < when <= t1:
                total += delta
        return total

    def _send_jog(self, joint: str, delta_deg: float) -> None:
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = [joint]
        msg.displacements = [float(delta_deg)]
        self._jog_pub.publish(msg)

    # ---- Probe ----

    def _probe_axis(self, joint: str):
        """Nudge one joint and return the image displacement it caused."""
        before = self._wait_for_fresh_point()
        if before is None:
            return None, 'target not visible'

        self._send_jog(joint, self._probe_deg)
        time.sleep(self._probe_settle)
        after = self._wait_for_fresh_point()
        if after is None:
            return None, 'target lost during probe'

        # Put it back so the probe leaves the arm where it started.
        self._send_jog(joint, -self._probe_deg)
        time.sleep(self._probe_settle)

        dx = (after[0] - before[0]) / (self._width / 2.0)
        dy = (after[1] - before[1]) / (self._height / 2.0)
        return (dx, dy), None

    def _probe_approach(self):
        """Determine whether the approach joint moves the camera nearer.

        Returns (+1, None) if a positive jog makes the hand look bigger,
        (-1, None) if smaller, or (None, reason) if it cannot be told.
        """
        if self._last_size_px is None:
            return None, 'tracker is not reporting palm size'
        before = self._last_size_px

        self._send_jog(self._approach_joint, self._probe_deg)
        time.sleep(self._probe_settle)
        after = self._last_size_px

        self._send_jog(self._approach_joint, -self._probe_deg)
        time.sleep(self._probe_settle)

        if after is None:
            return None, 'target lost during approach probe'

        change = (after - before) / max(before, 1.0)
        # Require a clear change. Below this the measurement is camera noise,
        # and committing to a sign from noise means approaching in the wrong
        # direction, which drives the arm away from the hand.
        if abs(change) < 0.02:
            return None, (
                f'apparent size barely changed ({change * 100:+.1f}%); this '
                f'joint may not move the camera along its view'
            )
        return (1.0 if change > 0 else -1.0), None

    def _assume_orientation(self) -> None:
        """Take the camera mounting on trust instead of measuring it.

        Builds the same inverse Jacobian the probe would produce, but
        diagonal and from parameters: horizontal image error drives the
        horizontal joint, vertical drives the vertical one. That is correct
        for a camera mounted square on the flange and wrong by a sign or an
        axis swap for anything else, which is exactly what the probe exists
        to discover. Skipping it buys immediate tracking at the cost of
        that guarantee.
        """
        self._set_jacobian(np.array(
            [[self._assumed_deg * self._assumed_h_sign, 0.0],
             [0.0, self._assumed_v_deg * self._assumed_v_sign]], dtype=float))
        self._approach_sign = (
            self._assumed_approach_sign if self._approach_enabled else 0.0)
        self._probed = True
        self.get_logger().info(
            f'Tracking immediately without probing: a full correction is '
            f'{self._assumed_deg:.0f}deg horizontally and '
            f'{self._assumed_v_deg:.0f}deg vertically per unit error, h sign '
            f'{self._assumed_h_sign:+.0f}, v sign {self._assumed_v_sign:+.0f}, '
            f'applying {self._gain * 100:.0f}% of it per sighting '
            f'({self._assumed_deg * self._gain:.1f}deg at the frame edge). '
            'If the arm drives the hand out of frame, flip the matching sign '
            '(assumed_h_sign / assumed_v_sign), or set skip_probe:=false to '
            'measure it instead.')

    def _probe(self):
        """Measure the image Jacobian by moving each joint and watching.

        Builds J where [dx, dy]^T = J @ [d_horizontal, d_vertical]^T, then
        inverts it so the control law can go the other way.
        """
        if self._probing:
            return False, 'already probing'

        if self._width is None:
            n = self.count_publishers(self._info_topic_name)
            return False, (
                f'no frame size yet -- nothing usable on '
                f'{self._info_topic_name} ({n} publisher(s) detected). '
                + ('Is camera_node running?' if n == 0 else
                   'A publisher exists but no CameraInfo arrived -- check '
                   f'`ros2 topic hz {self._info_topic_name}`.')
            )

        # Give the tracker a moment before giving up. Enabling the servo the
        # instant a hand enters frame is a normal thing to do, and failing on
        # the first missing message would be needlessly brittle.
        if self._last_point is None:
            self._wait_for_fresh_point(timeout=3.0)

        if not self._ever_received_point:
            n = self.count_publishers(self._point_topic)
            if n == 0:
                return False, (
                    f'nothing is publishing {self._point_topic}. '
                    'hand_tracker_node is probably not running (or is using a '
                    'different topic). Start it with: '
                    'ros2 run mycobot_perception hand_tracker_node'
                )
            return False, (
                f'{self._point_topic} has {n} publisher(s) but has never sent a '
                'message, so MediaPipe is not detecting a hand. View '
                '/hand/annotated to see what the camera sees -- usually this is '
                'lighting, the hand being too close to fill the frame, or the '
                'camera pointing somewhere other than you expect.'
            )

        if self._last_point is None:
            return False, 'no target visible -- hold your hand in view'

        age = time.monotonic() - self._last_point_time
        if age > self._timeout:
            return False, (
                f'target last seen {age:.1f}s ago (timeout {self._timeout}s). '
                'Tracking is working but the hand is not currently visible -- '
                'hold it in view and re-enable.'
            )

        self._probing = True
        try:
            self.get_logger().info(
                f'Probing camera orientation: nudging {self._h_joint} and '
                f'{self._v_joint} by {self._probe_deg} deg each...'
            )

            h_resp, err = self._probe_axis(self._h_joint)
            if h_resp is None:
                return False, f'{self._h_joint}: {err}'
            v_resp, err = self._probe_axis(self._v_joint)
            if v_resp is None:
                return False, f'{self._v_joint}: {err}'

            j = np.array([[h_resp[0], v_resp[0]],
                          [h_resp[1], v_resp[1]]], dtype=float)

            self.get_logger().info(
                f'  {self._h_joint} +{self._probe_deg}deg moved target '
                f'({h_resp[0]:+.3f}, {h_resp[1]:+.3f}) of a half-frame'
            )
            self.get_logger().info(
                f'  {self._v_joint} +{self._probe_deg}deg moved target '
                f'({v_resp[0]:+.3f}, {v_resp[1]:+.3f}) of a half-frame'
            )

            det = float(np.linalg.det(j))
            # A near-singular Jacobian means both joints push the target the
            # same way in the image, so the pair cannot steer independently.
            # Servoing on that inverse would produce enormous commands.
            if abs(det) < 1e-4:
                return False, (
                    f'Jacobian is singular (det={det:.2e}). The two joints move '
                    'the image in nearly the same direction from this pose. '
                    'Move the arm to a different starting pose and retry, or '
                    'pick a different vertical_joint.'
                )

            # Scale to per-degree, since the probe used probe_deg steps.
            self._set_jacobian(np.linalg.inv(j) * self._probe_deg)

            # Learn which way the approach joint changes apparent size. Only
            # the sign is needed: the magnitude varies with distance and pose,
            # so a fixed gain plus the correct direction is more robust than a
            # calibrated scale that is wrong everywhere except where measured.
            if self._approach_enabled:
                sign, err = self._probe_approach()
                if sign is None:
                    self.get_logger().warn(
                        f'Approach probe failed ({err}); centring only, no '
                        'closing in. Set approach_enabled:=false to silence.')
                    self._approach_sign = 0.0
                else:
                    self._approach_sign = sign
                    direction = 'closer' if sign > 0 else 'further'
                    self.get_logger().info(
                        f'  {self._approach_joint} +{self._probe_deg}deg makes '
                        f'the hand appear {direction}')

            self._probed = True
            self.get_logger().info('Probe OK -- Jacobian estimated.')
            return True, 'ok'
        except Exception as e:
            return False, str(e)
        finally:
            self._probing = False

    # ---- Control loop ----

    def _set_jacobian(self, jinv: np.ndarray) -> None:
        """Install the image Jacobian and cache its forward direction.

        _jinv answers "what joint motion corrects this image error", which is
        what the control law wants. The lag compensator wants the opposite
        question -- "what will this joint motion do to the image" -- and asking
        it every cycle means inverting the same matrix fifteen times a second
        forever. Invert once, here.
        """
        self._jinv = jinv
        try:
            self._jfwd = np.linalg.inv(jinv)
        except np.linalg.LinAlgError:
            self._jfwd = None
            self.get_logger().warn(
                'Image Jacobian is not invertible; lag compensation and the '
                'sign check are both off for this lock.')
        self._reset_sign_estimate()

    def _reset_sign_estimate(self) -> None:
        self._corr[:] = 0.0
        self._pmag[:] = 0.0
        self._corr_n = 0
        self._prev_meas = None

    def _update_sign_estimate(self, stamp: float, error: np.ndarray) -> None:
        """Score what the last jog was predicted to do against what it did.

        Each detection closes the loop on the jogs issued one command_lag
        before it: those are the ones whose effect this frame is the first to
        show. Regressing observed image motion on predicted image motion gives
        a scale factor per axis -- about +1 if the model is right, about -1 if
        that axis is wired backwards, about 0 if the joint does not move the
        image that way at all.

        Hand motion is the noise here, and it is not small. That is why this
        accumulates over dozens of detections before saying anything: over a
        few seconds the commanded motion correlates with the image and a
        waving hand does not.
        """
        prev = self._prev_meas
        self._prev_meas = (stamp, error.copy())
        if prev is None or self._jfwd is None:
            return
        prev_stamp, prev_error = prev
        # Jogs that landed between the two frames, shifted by the lag.
        commanded = self._sent_between(prev_stamp - self._command_lag,
                                       stamp - self._command_lag)
        if not np.any(commanded):
            return
        predicted = self._jfwd @ commanded
        observed = error - prev_error

        self._corr += predicted * observed
        self._pmag += predicted * predicted
        self._corr_n += 1

        if self._corr_n < self._auto_sign_samples:
            return

        for i, (axis, param) in enumerate(
                (('horizontal', 'assumed_h_sign'),
                 ('vertical', 'assumed_v_sign'))):
            if self._pmag[i] < 1e-6 or self._sign_verdict[i]:
                continue
            ratio = self._corr[i] / self._pmag[i]
            if ratio < -0.25:
                # Backwards, and confidently so. Flip it and carry on rather
                # than printing an instruction and continuing to misbehave --
                # the whole point of measuring is not having to relaunch.
                flip = np.array([1.0, 1.0])
                flip[i] = -1.0
                self._set_jacobian(self._jinv * flip[:, None])
                self._sign_verdict[i] = True
                self.get_logger().warn(
                    f'The {axis} axis is inverted (measured response '
                    f'{ratio:+.2f}x of predicted over {self._corr_n} '
                    f'detections) -- flipping it now. Launch with '
                    f'{param}:={-(self._assumed_v_sign if i else self._assumed_h_sign):+.0f} '
                    f'to start out correct next time.')
                return
            if abs(ratio) < 0.15:
                self.get_logger().warn(
                    f'The {axis} axis barely responds to its joint (measured '
                    f'{ratio:+.2f}x of predicted). Either the camera is '
                    f'mounted rotated, so the two axes are swapped, or that '
                    f'joint cannot move from this pose. skip_probe:=false '
                    f'measures the mounting properly.',
                    throttle_duration_sec=20.0)
            elif not 0.6 < ratio < 1.6:
                # Name the per-axis parameter, because the two axes genuinely
                # differ -- the frame is wider than it is tall, so the vertical
                # edge is a smaller angle away -- and pointing both at
                # assumed_deg_per_error meant fixing one by breaking the other.
                current = self._assumed_v_deg if i else self._assumed_deg
                knob = ('assumed_v_deg_per_error' if i
                        else 'assumed_deg_per_error')
                suggested = current / max(ratio, 0.2)
                self.get_logger().info(
                    f'The {axis} axis responds {ratio:.2f}x as strongly as '
                    f'assumed -- tracking works but is '
                    f'{"under" if ratio < 1 else "over"}-damped. '
                    f'{knob}:={suggested:.0f} would match it.',
                    throttle_duration_sec=30.0)
        self._reset_sign_estimate()

    def _compensate(self, stamp: float, error: np.ndarray) -> np.ndarray:
        """Estimate the error NOW from a measurement of the error THEN.

        This measurement describes the scene at `stamp`. Every jog published
        since `stamp - command_lag` is motion the arm is making, or is about
        to make, that this frame cannot possibly show yet. Predict what those
        will do to the image and add it in; what comes out is where the hand
        will be by the time the next correction could act on it.

        Without this the loop re-commands corrections that are already on
        their way, which is overshoot by construction and shows up as the arm
        flicking back and forth past the target.
        """
        if not self._lag_comp or self._jfwd is None:
            return error
        in_flight = self._sent_between(stamp - self._command_lag,
                                       float('inf'))
        if not np.any(in_flight):
            return error
        adjust = self._jfwd @ in_flight
        # Bounded, because this is open loop. If a jog was clamped by the
        # driver, rejected while the link was down, or blocked by a joint
        # limit, the prediction is simply wrong, and a wrong prediction with
        # no ceiling drives the arm off on its own.
        norm = float(np.linalg.norm(adjust))
        if norm > self._max_comp:
            adjust = adjust * (self._max_comp / norm)
        return error + adjust

    def _update_velocity(self, stamp: float, measured: np.ndarray) -> np.ndarray:
        """Estimate how fast the TARGET is crossing the image.

        Differencing two sightings gives the image motion between them, but
        that is the sum of two quite different things: the hand moving, and
        the camera moving because we jogged it. Only the first is worth
        leading -- the second is already handled by the lag compensator, and
        feeding it back in here would have the loop chasing its own motion.
        So the jogs that landed between the two frames get subtracted, leaving
        the hand's own travel.

        Returns half-frames per second, smoothed. Zero until a second
        sighting arrives, which makes the lead term a no-op on the first
        detection of a lock rather than a guess.
        """
        prev = self._vel_prev
        self._vel_prev = (stamp, measured.copy())
        if prev is None or self._lead_time <= 0.0:
            return self._vel

        prev_stamp, prev_measured = prev
        dt = stamp - prev_stamp
        # Two readings from the same frame say nothing about speed and would
        # divide by ~0; a gap of a second means the target was lost and came
        # back, and the "motion" across that gap is not a velocity.
        if not (1e-3 < dt < 1.0):
            return self._vel

        own = np.zeros(2, dtype=float)
        if self._jfwd is not None:
            own = self._jfwd @ self._sent_between(
                prev_stamp - self._command_lag, stamp - self._command_lag)

        raw = (measured - prev_measured - own) / dt
        # One bad detection produces one enormous difference. Clamped, because
        # this is multiplied by lead_time and added straight to the error, so
        # an unbounded velocity is an unbounded lunge.
        speed = float(np.linalg.norm(raw))
        if speed > self._max_target_speed:
            raw = raw * (self._max_target_speed / speed)
            self._vel_saturated += 1
        else:
            # One clamped sample is a glitch and means nothing; it is only a
            # RUN of them that indicts the model. So the counter resets on the
            # first sane reading rather than decaying.
            self._vel_saturated = 0
            if self._vel_untrusted:
                self._vel_untrusted = False
                self.get_logger().info(
                    'Hand speed estimate is plausible again -- leading the '
                    'target once more.')

        if (not self._vel_untrusted
                and self._vel_saturated >= self._vel_sat_limit):
            # Nothing physical moves this fast this long. The own-motion
            # subtraction above must be crediting jogs that never executed,
            # which is what a blocked or unresponsive joint looks like from
            # here. Leading on that estimate amplifies the error and sustains
            # the saturation, so stop.
            self._vel_untrusted = True
            self.get_logger().warn(
                f'Hand speed has read at the {self._max_target_speed:.1f}/s '
                f'ceiling for {self._vel_saturated} detections running. A '
                'hand does not do that, so the jogs being subtracted here are '
                'probably not reaching the arm -- check the driver log for a '
                'joint whose commanded angle stops advancing. Dropping the '
                'lead term; tracking continues proportional-only.')

        if self._vel_untrusted:
            self._vel = np.zeros(2, dtype=float)
            return self._vel

        a = self._vel_smoothing
        self._vel = a * self._vel + (1.0 - a) * raw
        return self._vel

    def _reset_tracking(self) -> None:
        """Drop the diagnostics that only mean anything within one lock.

        The control law itself is stateless -- each correction comes purely
        from the error in front of it, which is the point -- so there is no
        accumulated term to unwind here. What does need clearing is the
        convergence history behind the sign/axis warnings: carrying it across
        a lost target would compare errors from two different attempts and
        report a runaway that never happened.

        The in-flight command list goes too. Between one lock and the next the
        arm may have swept, homed, or sat still for a minute; predicting the
        image effect of jogs from before all that is worse than not predicting
        at all.
        """
        self._growing = [0, 0]
        self._sent.clear()
        self._prev_meas = None
        # The hand that comes back after a dropout is not necessarily the one
        # that left, and it is certainly not still travelling at the speed it
        # was. Leading a stale velocity across a gap aims the arm at a place
        # nothing ever was.
        self._vel = np.zeros(2, dtype=float)
        self._vel_prev = None
        self._vel_saturated = 0
        self._vel_untrusted = False
        self._stuck = [0, 0]
        self._best_abs_err = [float('inf'), float('inf')]
        self._prev_abs_err = None
        # Let the next measurement through immediately rather than waiting for
        # one newer than whatever we last acted on before the reset.
        self._acted_point_time = 0.0

    def _search_sweep(self) -> None:
        """Tilt the wrist through its arc looking for a hand.

        Paced against the wall clock, not the loop counter: the step is
        whatever covers search_range_deg in search_sweep_seconds given the
        time that actually elapsed since the last tick. A fixed
        degrees-per-tick step silently stretches the sweep by however long
        the host stalled, which is how a sweep meant to take 15s ended up
        crawling a few degrees over a minute.

        Deliberately slow regardless: MediaPipe needs a few clean frames to
        lock on, and sweeping faster than it can detect means panning
        straight past a hand that was in view the whole time.
        """
        now = time.monotonic()
        if self._last_sweep_time is None:
            # First tick of this sweep -- no elapsed time to work from yet.
            self._last_sweep_time = now
            return
        dt = now - self._last_sweep_time
        self._last_sweep_time = now

        deg_per_sec = self._search_range / max(self._sweep_seconds, 0.1)
        step = deg_per_sec * dt * self._search_dir
        # After a stall dt is large; cap the catch-up so the arm eases back
        # into the sweep rather than lunging.
        step = max(-self._search_max_step,
                   min(self._search_max_step, step))

        self._send_jog(self._search_joint, step)
        self._search_travel += step

        # Reverse at the ends of the sweep. Range is measured from wherever the
        # search started, so it stays near the home pose instead of wandering.
        if abs(self._search_travel) >= self._search_range / 2.0:
            self._search_dir *= -1.0
            self.get_logger().info(
                f'Search sweep reversing at {self._search_travel:+.0f}deg')

        self.get_logger().info(
            f'Searching... {self._search_joint} at {self._search_travel:+.0f}deg '
            f'({deg_per_sec:.0f}deg/s)',
            throttle_duration_sec=3.0)

    def _check_sign(self, ex: float, ey: float) -> None:
        """Warn when the loop is driving the target away rather than in.

        Only meaningful when the orientation was assumed rather than
        measured: a wrong sign is precisely what the probe exists to rule
        out. Rather than leave it as "the arm twitches and never closes",
        name the axis and the parameter that fixes it.
        """
        cur = (abs(ex), abs(ey))
        if self._prev_abs_err is not None:
            for i in (0, 1):
                # Ignore changes too small to distinguish from hand tremor.
                if cur[i] > self._prev_abs_err[i] + 0.01:
                    self._growing[i] += 1
                elif cur[i] < self._prev_abs_err[i]:
                    self._growing[i] = 0
        self._prev_abs_err = cur

        for i in (0, 1):
            # Well outside the deadband and no better than it has ever been.
            if cur[i] > 0.25 and cur[i] >= self._best_abs_err[i] - 0.02:
                self._stuck[i] += 1
            else:
                self._stuck[i] = 0
            self._best_abs_err[i] = min(self._best_abs_err[i], cur[i])

        for i, (axis, joint_param) in enumerate(
                (('horizontal', 'horizontal_joint'),
                 ('vertical', 'vertical_joint'))):
            if self._stuck[i] >= 25:
                self.get_logger().warn(
                    f'The {axis} error has sat at {cur[i]:.2f} for '
                    f'{self._stuck[i]} updates without improving, while the '
                    f'joint kept moving -- that axis does not appear to steer '
                    f'the image at all. Either {joint_param} is the wrong '
                    f'joint for this camera mounting, or the two axes are '
                    f'swapped. skip_probe:=false measures it properly.',
                    throttle_duration_sec=10.0)
                self._stuck[i] = 0

        for i, (axis, param) in enumerate(
                (('horizontal', 'assumed_h_sign'),
                 ('vertical', 'assumed_v_sign'))):
            if self._growing[i] >= 6:
                self.get_logger().warn(
                    f'The {axis} error has grown {self._growing[i]} updates '
                    f'running -- the arm is pushing the target OUT of frame, '
                    f'not centring it. The {axis} sign is almost certainly '
                    f'inverted: relaunch with {param}:='
                    f'{-self._assumed_v_sign if i else -self._assumed_h_sign:+.0f}'
                    ', or skip_probe:=false to measure it.',
                    throttle_duration_sec=5.0)
                self._growing[i] = 0

    def _approach_step(self) -> float:
        """Degrees to move the approach joint to close in on the hand.

        Uses apparent palm size as the range proxy: bigger means nearer. Zero
        if approach is off, the probe could not determine a direction, or the
        hand already fills the target fraction of the frame.
        """
        if not self._approach_enabled:
            return 0.0
        if self._approach_sign == 0.0:
            self.get_logger().warn(
                'Not closing in: no approach direction known (the approach '
                'probe could not tell which way moves the camera nearer). '
                'Set assumed_approach_sign, or skip_probe:=false to measure.',
                throttle_duration_sec=10.0)
            return 0.0
        if self._last_size_px is None or self._width is None:
            self.get_logger().warn(
                'Not closing in: the tracker is not reporting palm size, so '
                'there is no range proxy to close against.',
                throttle_duration_sec=10.0)
            return 0.0

        current = self._last_size_px / float(self._width)
        error = self._target_size - current
        if abs(error) < self._approach_deadband:
            return 0.0

        step = self._approach_gain * error * self._approach_sign
        return max(-self._max_approach_step,
                   min(self._max_approach_step, step))

    def _target_age(self) -> float:
        if self._last_point is None:
            return float('inf')
        return time.monotonic() - self._last_point_time

    def _servo_step(self) -> None:
        if self._probing or not self._enabled:
            return

        # HOMING is driven by the service callback; nothing to do until it
        # completes and flips the state to IDLE.
        if self._state in (IDLE, HOMING):
            return

        visible = self._target_age() <= self._timeout

        if self._state == SEARCHING:
            if not visible:
                # Give up only if we had a target and lost it. A search that
                # has never seen anything keeps sweeping -- it was asked to
                # hunt, and homing after 15s of an empty room is not that.
                if (self._had_lock
                        and self._target_age() > self._lost_timeout):
                    self.get_logger().info(
                        f'No sighting for {self._lost_timeout:.0f}s -- '
                        'going home.')
                    self._set_state(HOMING)
                    self._go_home()
                    return
                self._search_sweep()
                return
            # Found one. Probe first if the camera orientation is still
            # unknown -- unless we have been told to take it on trust and
            # start moving straight away.
            if not self._probed and self._skip_probe:
                self._assume_orientation()
            if not self._probed:
                ok, detail = self._probe()
                if not ok:
                    self.get_logger().error(
                        f'Probe failed: {detail}. Returning home.')
                    self._set_state(HOMING)
                    self._go_home()
                    return
            self._had_lock = True
            self._set_state(TRACKING)
            return

        # --- TRACKING ---
        if not visible:
            lost_for = self._target_age()
            if lost_for > self._lost_timeout:
                self.get_logger().info(
                    f'No sighting for {self._lost_timeout:.0f}s -- going home.')
                self._set_state(HOMING)
                self._go_home()
                return
            if lost_for >= self._resume_search_after:
                # Long enough that it is not coming back on its own. Sweep
                # again rather than standing still until lost_timeout.
                self.get_logger().info(
                    f'Target lost {lost_for:.1f}s ago -- resuming search.')
                self._reset_tracking()
                self._set_state(SEARCHING)
                return
            # Hold position during a brief dropout rather than sweeping away
            # from a hand that is probably about to reappear. Nothing to wind
            # down: the control law carries no state between updates, so
            # holding is genuinely just doing nothing.
            self.get_logger().info(
                f'Target lost {lost_for:.1f}s ago; holding',
                throttle_duration_sec=2.0)
            return

        if self._jinv is None:
            self.get_logger().warn(
                'Tracking with no Jacobian -- probe did not run.',
                throttle_duration_sec=3.0)
            return

        err = self._current_error()
        if err is None:
            return
        ex, ey = err

        # Only act once per measurement. The timer runs at `rate` (15Hz) but
        # detections arrive slower than that, so without this the same reading
        # would be acted on several times over -- and since each correction is
        # sized as a fraction of the error it sees, repeating one reading means
        # commanding that fraction two or three times for a single observation.
        # That is precisely the over-correction the gain is chosen to avoid.
        if self._last_point_time <= self._acted_point_time:
            return
        self._acted_point_time = self._last_point_time

        # Deliberately after that gate. These counters ask "is the error
        # improving between one sighting and the next"; feeding them the same
        # reading three times running answers "no" three times over, which is
        # how the old build reported a stuck axis on a loop that was tracking
        # fine. One sighting, one verdict.
        #
        # Skipped entirely when auto_sign is doing the same job properly.
        # These heuristics infer a bad sign from the error failing to shrink,
        # which a hand moving away from the arm also produces; the correlation
        # check measures the response directly. Running both means two
        # verdicts that can disagree, and the weaker one shouting first.
        if not (self._auto_sign and self._skip_probe):
            self._check_sign(ex, ey)

        measured = np.array([ex, ey], dtype=float)
        stamp = self._last_point_stamp

        # Score the last jog against what the image actually did, and flip an
        # axis if it turns out to be wired backwards. Only meaningful when the
        # mounting was assumed: after a real probe the Jacobian is measured,
        # off-diagonal, and this per-axis pairing no longer holds.
        if self._auto_sign and self._skip_probe:
            self._update_sign_estimate(stamp, measured)

        # Correct for where the hand will be once the jogs already in flight
        # have landed, not where it was when this frame was captured.
        error = self._compensate(stamp, measured)

        # Then lead the hand's own motion. _compensate answers "where is it
        # now"; this answers "where will it be when this correction lands",
        # which is the one the arm should actually be aiming at. Without it a
        # proportional loop tracks a moving hand at a fixed distance behind
        # it -- in frame, never centred.
        velocity = self._update_velocity(stamp, measured)
        lead = velocity * self._lead_time
        # Bounded for the same reason the compensator is: it is a prediction,
        # and a prediction that can outweigh the measurement it is correcting
        # stops being a refinement and becomes the input.
        lead_mag = float(np.linalg.norm(lead))
        if lead_mag > self._max_lead:
            lead = lead * (self._max_lead / lead_mag)
        error = error + lead

        ce = float(np.linalg.norm(error))

        if ce < self._deadband:
            # Being centred is success, not a fault -- but it looks identical
            # to a dead loop from outside, so say so.
            self.get_logger().info(
                f'On target (error {ce:.3f} < deadband '
                f'{self._deadband}); holding.',
                throttle_duration_sec=5.0)
            # Centred is NOT done. Approach runs on its own axis and its own
            # error, so returning here meant the arm centred the hand and
            # then sat there forever -- the deadband on the centring error
            # silently gated the closing-in as well.
            da = self._approach_step()
            if da != 0.0:
                self._send_jog(self._approach_joint, da)
                self.get_logger().info(
                    f'Centred; closing in {self._approach_joint}{da:+.2f}deg',
                    throttle_duration_sec=2.0)
            return

        # Proportional, and only proportional. _jinv maps a unit image error
        # to the joint motion that would fully correct it -- either measured by
        # the probe or assumed from the camera geometry -- and gain takes a
        # fraction of that. `error` here is already lag-compensated, which is
        # what lets the fraction be large enough to feel responsive instead of
        # small enough to survive its own delay. Negated because we drive the
        # error toward zero.
        # progressive_gain (0 by default) makes this superlinear in the error;
        # see the parameter for why that is off and what it costs when on.
        # Keyed on the norm rather than per-axis so a hand far out diagonally
        # is chased as one target, not harder in x than in y.
        eff_gain = self._gain * (1.0 + self._progressive_gain * ce)
        delta = self._jinv @ (eff_gain * error) * -1.0
        dh, dv = float(delta[0]), float(delta[1])

        dh = max(-self._max_step, min(self._max_step, dh))
        dv = max(-self._max_step, min(self._max_step, dv))

        # Record the CLAMPED values, and record them before publishing. These
        # are what the arm will actually be asked to do, so these are what the
        # next compensation has to subtract; logging the unclamped intent here
        # would have the compensator predicting motion that never happens and
        # steadily under-driving the arm.
        self._record_sent(self._ros_now(), dh, dv)

        names = [self._h_joint, self._v_joint]
        deltas = [dh, dv]

        # Close in as well as centre. Sent in the same message so the arm
        # approaches and tracks together rather than alternating between them.
        da = self._approach_step()
        if da != 0.0:
            if self._approach_joint in names:
                deltas[names.index(self._approach_joint)] += da
            else:
                names.append(self._approach_joint)
                deltas.append(da)

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = names
        msg.displacements = deltas
        self._jog_pub.publish(msg)

        size_note = ''
        if self._last_size_px is not None and self._width:
            frac = self._last_size_px / float(self._width)
            size_note = f' size={frac:.2f}/{self._target_size:.2f}'

        # All three errors, because the differences between them are the whole
        # story when tracking misbehaves. `seen` is what the camera reported;
        # `aim` is that corrected for jogs still in flight and led by the
        # hand's own motion -- the point the arm is actually driving at. If
        # they are far apart the arm is chasing a lot of stale motion and
        # command_lag matters; if they are identical neither correction is
        # doing anything and something upstream (image timestamps, a dead
        # link) is why. `vel` says how fast the hand is judged to be moving,
        # which is what to check first if the arm leads or trails a wave.
        comp_note = ''
        if not np.allclose(error, (ex, ey), atol=1e-3):
            comp_note = f' aim=({error[0]:+.3f},{error[1]:+.3f})'
        vel_note = ''
        if self._lead_time > 0.0 and float(np.linalg.norm(velocity)) > 0.05:
            vel_note = f' vel=({velocity[0]:+.2f},{velocity[1]:+.2f})/s'

        # If this logs but the arm does not move, the jog is being rejected by
        # the driver -- check the driver's console, which now says why.
        self.get_logger().info(
            f'seen=({ex:+.3f},{ey:+.3f}){comp_note}{vel_note}{size_note} -> '
            + ' '.join(f'{n}{d:+.2f}' for n, d in zip(names, deltas)),
            throttle_duration_sec=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualServoNode()
    # Explicit thread count: the orientation probe blocks inside a service
    # callback (it sleeps while waiting to see where the target moved), and the
    # point subscription must keep running on another thread throughout or the
    # probe can never observe anything and always reports "target lost".
    # MultiThreadedExecutor() defaults to cpu_count(), which is 1 on some VMs.
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    node.get_logger().info('Executor: 4 threads')
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
