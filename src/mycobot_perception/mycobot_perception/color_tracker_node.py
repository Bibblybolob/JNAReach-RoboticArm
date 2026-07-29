"""
Colour-blob tracking, publishing the same contract hand_tracker_node does.

    /color/point_px    geometry_msgs/PointStamped
                       x, y = blob centre in FULL-frame pixels
                       z    = blob diameter in pixels (not a depth)

That z convention is deliberate: it is exactly what hand_tracker_node puts
there (palm width), so visual_servo_node's approach axis works against a
colour target without knowing anything changed. Point the servo at this topic
and the whole existing loop -- lag compensation, lead, gain profile, approach
-- drives a red object instead of a hand:

    ./run.py track:=color

WHY A COLOUR TARGET TRACKS MORE SMOOTHLY THAN A HAND

Not because the control law is better. Because the measurement is.

  * A blob centroid on a rigid object is stable to well under a pixel. A
    MediaPipe hand landmark jitters by several, and every pixel of jitter is
    a correction the servo dutifully makes.
  * Thresholding a downscaled frame costs about a millisecond against
    MediaPipe's 8-19ms. That is dead time removed from a loop whose binding
    constraint is dead time, and it is the single biggest difference.
  * A colour blob either is or is not there. Hand detection drops out at
    frame edges and odd angles, and every dropout resets the servo's velocity
    estimate and lag window.

So expect this to look better than hand tracking on identical gains, and do
not read that as the gains having been wrong.

METHOD

LAB rather than HSV, following the same approach as the vendor code this was
modelled on. Red is awkward in HSV because its hue wraps around 0/180, so it
needs two ranges that then have to be OR'd; in LAB it is simply a high `a`,
one contiguous box. LAB also separates lightness from colour, which makes a
threshold hold up better as the lighting changes across the workspace.

Detection runs on a downscaled copy (`process_width`, 160px by default, as
the vendor code does) and the result is scaled back to full-frame pixels.
Nothing about a colour blob needs 640px to locate, and the cost is quadratic
in width.
"""

from __future__ import annotations

from . import _env
_env.check()   # before cv_bridge: see the module docstring

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped


# LAB thresholds, OpenCV's 8-bit convention (L 0-255, a/b 0-255 with 128
# neutral). Starting points only -- lighting varies enough that these are
# expected to be tuned, which is what scripts/tune_color.py is for.
PRESETS = {
    'red':    {'min': (0, 150, 120), 'max': (255, 255, 200)},
    'green':  {'min': (0, 0, 120),   'max': (255, 110, 200)},
    'blue':   {'min': (0, 100, 0),   'max': (255, 145, 110)},
    'yellow': {'min': (0, 100, 150), 'max': (255, 145, 255)},
}


class ColorTrackerNode(Node):

    def __init__(self) -> None:
        super().__init__('color_tracker_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('point_topic', '/color/point_px')
        self.declare_parameter('target_color', 'red')
        # Override the preset when the lighting does not match it. Six values:
        # L_min, a_min, b_min, L_max, a_max, b_max.
        self.declare_parameter('lab_bounds', [0.0] * 6)
        # Width the detection runs at. The vendor code uses 160; the cost is
        # quadratic in this and a blob does not need resolution to locate.
        self.declare_parameter('process_width', 160)
        # Ignore blobs smaller than this, in pixels of the DOWNSCALED image.
        # 10 at 160px wide is about a 4px-wide object in the full frame --
        # enough to reject sensor noise and coloured speckle without losing a
        # real target at range.
        self.declare_parameter('min_area_px', 10.0)
        # Exponential smoothing on the published centre, matching
        # hand_tracker_node's parameter of the same name. Lower here on
        # purpose: a blob centroid is already steady, and smoothing is dead
        # time, which is the one thing this loop cannot spare. See the module
        # docstring in visual_servo_node.
        self.declare_parameter('smoothing', 0.3)
        self.declare_parameter('publish_annotated', False)
        self.declare_parameter('show_window', False)

        self._image_topic = self.get_parameter('image_topic').value
        self._color = str(self.get_parameter('target_color').value).lower()
        self._proc_w = int(self.get_parameter('process_width').value)
        self._min_area = float(self.get_parameter('min_area_px').value)
        self._smoothing = float(self.get_parameter('smoothing').value)
        self._annotate = bool(self.get_parameter('publish_annotated').value)
        self._show = bool(self.get_parameter('show_window').value)

        bounds = list(self.get_parameter('lab_bounds').value)
        if len(bounds) == 6 and any(bounds):
            self._lo = np.array(bounds[:3], dtype=np.uint8)
            self._hi = np.array(bounds[3:], dtype=np.uint8)
            source = 'lab_bounds'
        else:
            preset = PRESETS.get(self._color)
            if preset is None:
                raise ValueError(
                    f'unknown target_color {self._color!r}; known: '
                    f'{sorted(PRESETS)} -- or pass lab_bounds directly')
            self._lo = np.array(preset['min'], dtype=np.uint8)
            self._hi = np.array(preset['max'], dtype=np.uint8)
            source = f'{self._color} preset'

        self._bridge = CvBridge()
        self._smoothed: tuple[float, float] | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(
            PointStamped, self.get_parameter('point_topic').value, 10)
        self._annotated_pub = (
            self.create_publisher(Image, 'color/annotated', sensor_qos)
            if self._annotate else None)
        self.create_subscription(
            Image, self._image_topic, self._image_cb, sensor_qos)

        self._n = 0
        self._hits = 0
        self._proc_sum = 0.0
        self._since = 0.0

        self.get_logger().info(
            f'Colour tracker ready: {self._color} from {source}, '
            f'LAB {tuple(int(v) for v in self._lo)}..'
            f'{tuple(int(v) for v in self._hi)}, detecting at '
            f'{self._proc_w}px wide. Publishing '
            f'{self.get_parameter("point_topic").value}.')

    # ---- detection ----

    def find_blob(self, bgr):
        """Largest blob of the target colour. Returns (cx, cy, diameter) in
        FULL-frame pixels, or None.

        Kept free of ROS so it can be tested against images directly; see
        test/test_color_tracker.py.
        """
        h, w = bgr.shape[:2]
        scale = self._proc_w / float(w)
        small = cv2.resize(bgr, (self._proc_w, max(1, int(round(h * scale)))))
        # Blur before thresholding, not after: it merges the speckle that
        # would otherwise survive as one-pixel contours and cost an erode.
        blurred = cv2.GaussianBlur(small, (3, 3), 3)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(lab, self._lo, self._hi)

        # Erode then dilate (an opening): drops speckle, then restores the
        # surviving blob to roughly its original size rather than leaving it
        # eroded, which would bias the diameter the approach axis uses.
        mask = cv2.erode(mask, self._kernel)
        mask = cv2.dilate(mask, self._kernel)

        contours = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self._min_area:
            return None

        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        inv = 1.0 / scale
        return cx * inv, cy * inv, radius * 2.0 * inv

    # ---- ROS ----

    def _image_cb(self, msg: Image) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._since == 0.0:
            self._since = now
        self._n += 1

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge failed: {e}')
            return

        t0 = self.get_clock().now().nanoseconds * 1e-9
        found = self.find_blob(frame)
        self._proc_sum += self.get_clock().now().nanoseconds * 1e-9 - t0

        if found is None:
            # Drop the filter state so a blob reappearing elsewhere is not
            # dragged in from where the last one was.
            self._smoothed = None
            self._report(now)
            self._emit(frame, None)
            return

        cx, cy, diameter = found
        if self._smoothed is None:
            self._smoothed = (cx, cy)
        else:
            a = self._smoothing
            self._smoothed = (a * self._smoothed[0] + (1.0 - a) * cx,
                              a * self._smoothed[1] + (1.0 - a) * cy)
        sx, sy = self._smoothed

        out = PointStamped()
        # The frame's own stamp, carried through untouched. The servo's lag
        # compensation reasons in it, and re-stamping here with "now" would
        # tell it the measurement is fresher than it is.
        out.header = msg.header
        out.point.x = float(sx)
        out.point.y = float(sy)
        out.point.z = float(diameter)
        self._pub.publish(out)

        self._hits += 1
        self._report(now)
        self._emit(frame, (sx, sy, diameter))

    def _report(self, now: float) -> None:
        span = now - self._since
        if span < 10.0:
            return
        self.get_logger().info(
            f'colour: {self._hits / span:.1f} detections/s, '
            f'{self._proc_sum / max(self._n, 1) * 1000:.1f}ms per frame, '
            f'{self._hits}/{self._n} frames had the target')
        self._since = now
        self._n = self._hits = 0
        self._proc_sum = 0.0

    def _emit(self, frame, found) -> None:
        if self._annotated_pub is None and not self._show:
            return
        if found is not None:
            sx, sy, diameter = found
            cv2.circle(frame, (int(sx), int(sy)), int(diameter / 2),
                       (0, 255, 0), 2)
            cv2.circle(frame, (int(sx), int(sy)), 3, (0, 0, 255), -1)
            cv2.putText(frame, f'{self._color} ({sx:.0f}, {sy:.0f})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        if self._annotated_pub is not None:
            try:
                self._annotated_pub.publish(
                    self._bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
            except Exception:
                pass
        if self._show:
            cv2.imshow('color_tracking', frame)
            cv2.waitKey(1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ColorTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.get_parameter('show_window').value:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
