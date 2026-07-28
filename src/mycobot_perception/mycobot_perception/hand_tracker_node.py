"""
MediaPipe hand/finger tracking for the myCobot.

Subscribes to the camera stream and publishes where a fingertip is, in three
progressively more useful forms depending on what calibration is available:

  /hand/point_px    geometry_msgs/PointStamped
                    x, y = fingertip pixel position
                    z    = palm width IN PIXELS (not a depth in metres)
                    Always published when a hand is visible. Needs nothing.

                    Palm width rides in z so position and apparent size stay
                    in one atomic message. It is the servo's stand-in for
                    range: with no depth sensor, a hand that grows in the
                    frame is a hand getting closer, which is enough to drive
                    an approach. Using the palm specifically because it is
                    rigid -- a finger segment changes length as it curls.

  /hand/point_cam   geometry_msgs/PointStamped  metres in the camera frame
                    Requires real camera intrinsics on /camera/camera_info.
                    Until camera_node publishes a populated k matrix this
                    stays silent -- see scripts/calibrate_camera.py.

  /hand/annotated   sensor_msgs/Image           landmarks drawn on the frame

Getting to the ROBOT frame needs one more thing that does not exist yet: the
transform from camera to robot base. Where that comes from depends on how the
camera is mounted:

  - Mounted on the arm (eye-in-hand): the URDF already describes it, via
    camera_flange -> camera_link off link6_flange. Nothing to measure. This
    also enables image-based servoing, where you drive the arm to centre the
    target in view and never need a 3D point at all.

  - Mounted separately (eye-to-hand): the camera->base transform has to be
    measured and published as a static transform. This is the fiddly one.

DEPTH FROM ONE CAMERA IS AN ESTIMATE. A pixel is a ray, not a point. We
recover distance by exploiting the fact that a hand is a known-size object:
MediaPipe's world landmarks are metric (metres, hand-centred), so comparing a
real inter-landmark distance against its pixel distance gives range through
the pinhole relation Z = f * real_size / pixel_size. Expect a few centimetres
of error, worse at the frame edges and worse for unusual hand sizes. This is
precisely the step an OAK-D Pro removes, which is why it is isolated here.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        'mediapipe not installed:\n'
        '    pip install -r requirements.txt\n'
        f'(import error: {exc})'
    )

# MediaPipe 1.0 dropped the legacy solutions API this node is built on, and
# `pip install mediapipe` now gets you 1.x. Caught here rather than left to
# fail deeper in as "module 'mediapipe' has no attribute 'solutions'", which
# reads like a broken install rather than the wrong major version.
if not hasattr(mp, 'solutions'):
    raise SystemExit(
        f'mediapipe {getattr(mp, "__version__", "?")} has no .solutions -- '
        'that API was removed in 1.0, and this node uses it.\n'
        '    pip install "mediapipe<1.0"\n'
        'See requirements.txt; the version bounds there are load-bearing.'
    )


# MediaPipe hand landmark indices we care about.
WRIST = 0
INDEX_MCP = 5          # knuckle of the index finger
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17

# Landmark pair used as the "known size" ruler for the depth estimate.
# Index knuckle to pinky knuckle is a good choice: it spans the palm, which is
# rigid, so it does not change as fingers curl. Using a finger segment instead
# would make range jump around every time you bend a knuckle.
RULER_A = INDEX_MCP
RULER_B = PINKY_MCP


class HandTrackerNode(Node):

    def __init__(self) -> None:
        super().__init__('hand_tracker_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        # Which landmark to steer at. 9 (middle-finger knuckle) sits in the
        # middle of the palm and is the steadiest point on the hand: it is on
        # the rigid part, so it does not move when fingers flex. The index
        # fingertip (8) was the old default and made the servo chase finger
        # jitter rather than the hand.
        self.declare_parameter('target_landmark', MIDDLE_MCP)
        self.declare_parameter('min_detection_confidence', 0.6)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('max_num_hands', 1)
        self.declare_parameter('show_window', False)
        self.declare_parameter('window_name', 'Hand Tracking')
        # Exponential smoothing on the reported point. Raw MediaPipe output
        # jitters by a few pixels frame to frame; feeding that straight into a
        # servo loop makes the arm buzz. 0 = no smoothing, 0.9 = very heavy.
        self.declare_parameter('smoothing', 0.6)
        self.declare_parameter('publish_annotated', True)
        # MediaPipe landmark model: 0 is roughly twice as fast as 1 for a
        # modest accuracy cost. On a CPU-bound host the servo loop is starved
        # of fresh detections long before it is limited by landmark precision,
        # so 0 usually produces *smoother* tracking despite being the "worse"
        # model -- fresher data beats more accurate stale data in a control
        # loop. Raise to 1 if you have GPU inference or a fast host.
        self.declare_parameter('model_complexity', 1)
        # Skip frames that are already stale by the time we get to them.
        #
        # MediaPipe on a CPU is slower than the camera, so frames queue up
        # behind it. Processing a queued frame produces a detection that was
        # true a third of a second ago, and a servo loop acting on that is
        # correcting a position the hand has already left. Dropping it and
        # taking the next one costs a detection and buys back the whole delay:
        # in a control loop, fresher beats more.
        #
        # Set to 0 to disable, and watch the 'tracker:' line to see how many
        # frames this is actually discarding.
        self.declare_parameter('max_frame_age', 0.12)
        # ...but never starve the loop. If every frame is arriving stale --
        # a badly overloaded host, or clocks that disagree -- dropping them
        # all means publishing nothing at all, which is far worse than
        # publishing something late. Process one regardless after this long.
        self.declare_parameter('starvation_timeout', 0.4)

        self._image_topic = self.get_parameter('image_topic').value
        self._info_topic = self.get_parameter('camera_info_topic').value
        self._target_lm = int(self.get_parameter('target_landmark').value)
        self._smoothing = float(self.get_parameter('smoothing').value)
        self._show_window = bool(self.get_parameter('show_window').value)
        self._window_name = self.get_parameter('window_name').value
        self._publish_annotated = bool(self.get_parameter('publish_annotated').value)

        self._bridge = CvBridge()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=int(self.get_parameter('max_num_hands').value),
            model_complexity=int(self.get_parameter('model_complexity').value),
            min_detection_confidence=float(
                self.get_parameter('min_detection_confidence').value),
            min_tracking_confidence=float(
                self.get_parameter('min_tracking_confidence').value),
        )
        self._draw = mp.solutions.drawing_utils
        self._draw_styles = mp.solutions.drawing_styles

        # Focal length in pixels, from CameraInfo. None until we receive a
        # populated k matrix; without it we cannot convert pixels to metres.
        self._fx: float | None = None
        self._fy: float | None = None
        self._cx: float | None = None
        self._cy: float | None = None
        self._warned_no_intrinsics = False

        self._smoothed_px: tuple[float, float] | None = None

        self._max_frame_age = float(self.get_parameter('max_frame_age').value)
        self._starvation = float(self.get_parameter('starvation_timeout').value)
        self._last_processed = 0.0
        # Rolling counters for the periodic 'tracker:' report.
        self._n_seen = 0
        self._n_dropped = 0
        self._n_processed = 0
        self._age_sum = 0.0
        self._proc_sum = 0.0
        self._report_since = 0.0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub = self.create_subscription(
            Image, self._image_topic, self._image_cb, sensor_qos)
        self._info_sub = self.create_subscription(
            CameraInfo, self._info_topic, self._info_cb, 10)

        self._px_pub = self.create_publisher(PointStamped, 'hand/point_px', 10)
        self._cam_pub = self.create_publisher(PointStamped, 'hand/point_cam', 10)
        self._annotated_pub = self.create_publisher(Image, 'hand/annotated', 10)

        self.get_logger().info(
            f'Hand tracker ready. Watching {self._image_topic}, '
            f'reporting landmark {self._target_lm}.'
        )

    # ---- Camera intrinsics ----

    def _info_cb(self, msg: CameraInfo) -> None:
        """Latch intrinsics from CameraInfo, ignoring the uncalibrated stub.

        camera_node currently publishes width/height with an all-zero k
        matrix. Treat that as "no intrinsics" rather than believing a focal
        length of zero.
        """
        k = list(msg.k)
        if len(k) != 9 or k[0] <= 0.0 or k[4] <= 0.0:
            if not self._warned_no_intrinsics:
                self.get_logger().warn(
                    'CameraInfo has no usable intrinsics (k matrix is empty or '
                    'zero). Publishing pixel coordinates only; /hand/point_cam '
                    'stays silent. Run scripts/calibrate_camera.py and load the '
                    'result into camera_node to enable metric output.'
                )
                self._warned_no_intrinsics = True
            return

        if self._fx is None:
            self.get_logger().info(
                f'Got intrinsics: fx={k[0]:.1f} fy={k[4]:.1f} '
                f'cx={k[2]:.1f} cy={k[5]:.1f}'
            )
        self._fx, self._fy = k[0], k[4]
        self._cx, self._cy = k[2], k[5]

    # ---- Depth estimate ----

    def _estimate_depth_m(self, landmarks, world_landmarks, w: int, h: int):
        """Range to the hand in metres, or None if it cannot be estimated.

        Uses the palm width as a ruler of known metric size. See the module
        docstring for why this is approximate.
        """
        if self._fx is None or world_landmarks is None:
            return None

        a, b = landmarks[RULER_A], landmarks[RULER_B]
        pixel_dist = math.hypot((a.x - b.x) * w, (a.y - b.y) * h)
        if pixel_dist < 1.0:
            return None

        wa, wb = world_landmarks[RULER_A], world_landmarks[RULER_B]
        real_dist = math.sqrt(
            (wa.x - wb.x) ** 2 + (wa.y - wb.y) ** 2 + (wa.z - wb.z) ** 2)
        if real_dist <= 0.0:
            return None

        # Pinhole: an object of size S at range Z projects to S * f / Z pixels.
        return (real_dist * self._fx) / pixel_dist

    # ---- Main callback ----

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _report(self, now: float) -> None:
        """Publish the numbers that decide how fast the servo can track.

        Detection rate and frame age are the two limits on the whole system,
        and neither is visible from anywhere else. Without them, "it lags" is
        a feeling; with them it is either a slow camera, a slow model, or a
        slow arm, and those have different fixes.
        """
        if self._report_since == 0.0:
            self._report_since = now
            return
        span = now - self._report_since
        if span < 10.0:
            return
        rate = self._n_processed / span
        age = self._age_sum / max(self._n_processed, 1)
        proc = self._proc_sum / max(self._n_processed, 1)
        note = ''
        if rate < 6.0:
            note = (' -- too slow to track a moving hand; try '
                    'model_complexity:=0, or a smaller camera frame')
        self.get_logger().info(
            f'tracker: {rate:.1f} detections/s, {proc * 1000:.0f}ms per frame, '
            f'{age * 1000:.0f}ms old on arrival, dropped {self._n_dropped} '
            f'stale of {self._n_seen}{note}')
        self._report_since = now
        self._n_seen = self._n_dropped = self._n_processed = 0
        self._age_sum = self._proc_sum = 0.0

    def _image_cb(self, msg: Image) -> None:
        now = self._now()
        self._n_seen += 1
        self._report(now)

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = now - stamp
        # A negative or absurd age means the stamp is not usable -- treat the
        # frame as fresh rather than dropping every frame forever on the
        # strength of a bad clock.
        if not (0.0 <= age < 5.0):
            age = 0.0
        if (self._max_frame_age > 0.0
                and age > self._max_frame_age
                and now - self._last_processed < self._starvation):
            self._n_dropped += 1
            return
        self._last_processed = now
        self._age_sum += age

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge failed: {e}')
            return

        h, w = frame.shape[:2]
        # MediaPipe wants RGB; mark the buffer read-only so it can avoid a copy.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        t0 = self._now()
        results = self._hands.process(rgb)
        self._proc_sum += self._now() - t0
        self._n_processed += 1

        annotated = frame if self._publish_annotated or self._show_window else None

        if not results.multi_hand_landmarks:
            # Drop the smoothing state so a reappearing hand does not get
            # dragged in from wherever it was last seen.
            self._smoothed_px = None
            self._emit_annotated(annotated, msg)
            return

        landmarks = results.multi_hand_landmarks[0].landmark
        world = None
        if getattr(results, 'multi_hand_world_landmarks', None):
            world = results.multi_hand_world_landmarks[0].landmark

        tip = landmarks[self._target_lm]
        px, py = tip.x * w, tip.y * h

        # Smooth in pixel space before anything downstream sees it.
        if self._smoothed_px is None:
            self._smoothed_px = (px, py)
        else:
            a = self._smoothing
            self._smoothed_px = (
                a * self._smoothed_px[0] + (1.0 - a) * px,
                a * self._smoothed_px[1] + (1.0 - a) * py,
            )
        sx, sy = self._smoothed_px

        # Palm width in pixels: the servo's proxy for range. Same landmark pair
        # used for the metric depth estimate below, so the two agree.
        a, b = landmarks[RULER_A], landmarks[RULER_B]
        palm_px = math.hypot((a.x - b.x) * w, (a.y - b.y) * h)

        px_msg = PointStamped()
        px_msg.header = msg.header
        px_msg.point.x = float(sx)
        px_msg.point.y = float(sy)
        px_msg.point.z = float(palm_px)
        self._px_pub.publish(px_msg)

        depth_m = self._estimate_depth_m(landmarks, world, w, h)
        if depth_m is not None and self._cx is not None:
            # Back-project the pixel to a 3D point in the camera frame:
            # x right, y down, z forward, standard optical convention.
            x = (sx - self._cx) * depth_m / self._fx
            y = (sy - self._cy) * depth_m / self._fy
            cam_msg = PointStamped()
            cam_msg.header = msg.header
            cam_msg.point.x = float(x)
            cam_msg.point.y = float(y)
            cam_msg.point.z = float(depth_m)
            self._cam_pub.publish(cam_msg)

        if annotated is not None:
            self._draw.draw_landmarks(
                annotated,
                results.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS,
                self._draw_styles.get_default_hand_landmarks_style(),
                self._draw_styles.get_default_hand_connections_style(),
            )
            cv2.circle(annotated, (int(sx), int(sy)), 9, (0, 0, 255), 2)
            label = f'({sx:.0f}, {sy:.0f}) px'
            if depth_m is not None:
                label += f'   {depth_m * 100:.1f} cm'
            else:
                label += '   depth: no intrinsics'
            cv2.putText(annotated, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self._emit_annotated(annotated, msg)

    def _emit_annotated(self, annotated, src_msg: Image) -> None:
        if annotated is None:
            return
        if self._publish_annotated:
            try:
                out = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                out.header = src_msg.header
                self._annotated_pub.publish(out)
            except Exception as e:
                self.get_logger().warn(f'annotated publish failed: {e}')
        if self._show_window:
            cv2.imshow(self._window_name, annotated)
            cv2.waitKey(1)

    def destroy_node(self) -> bool:
        try:
            self._hands.close()
        except Exception:
            pass
        if self._show_window:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HandTrackerNode()
    try:
        rclpy.spin(node)
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
