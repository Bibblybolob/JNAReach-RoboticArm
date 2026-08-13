"""
Bridges vision_msgs/Detection2DArray + a depth image into the PointStamped
contract the visual servo expects.

    /button/point_px    geometry_msgs/PointStamped
                         x, y = detection bbox centre, in FULL-frame pixels
                         z    = depth at that point, in millimetres (0 if
                                unavailable)

Which detection gets published is controlled by the `target_label` parameter:

    ros2 param set /detection_bridge_node target_label button-5

Empty (the default) means "steer at the best button you can see, whatever it
is" -- the highest-confidence detection of any class. That is what bring-up
wants, and what button_servo.launch.py's own description has always claimed
this did; it used to publish nothing instead, which silently pinned the servo
in SEARCHING forever because /button/point_px never carried a single message.

Set the label once a specific floor is wanted. If more than one detection
matches, the highest-confidence one wins.

Depth is sampled as the median of a 5x5 patch around the bbox centre,
with zero (invalid) pixels excluded, so a single missing depth reading at the
exact centre pixel does not zero out the whole point.
"""

from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rcl_interfaces.msg import SetParametersResult

from cv_bridge import CvBridge

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class DetectionBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__('detection_bridge_node')

        self.declare_parameter('detection_topic', '/perception/button_detections')
        self.declare_parameter('depth_topic', '/camera/depth_raw')
        self.declare_parameter('output_topic', '/button/point_px')
        self.declare_parameter('target_label', '')

        # --- Depth sanity filter ---
        #
        # A button is a physical thing on a flat panel: it has a depth, that
        # depth is inside the sensor's valid range, and it does not move.
        # A false detection has none of those properties, and the detector
        # produces plenty of them -- measured on this panel, it locked onto
        # 'down' (a class this panel does not even have) at 0.21 and the
        # target teleported from (+0.94,-0.52) to (+0.40,-0.96) to
        # (+0.96,-0.60) between consecutive sightings, saturating every jog.
        #
        # Depth rejects that for free, without touching the model: noise
        # lands wherever it lands, and mostly not on the panel.
        self.declare_parameter('require_depth', True)
        # D405 range. Outside 70-500mm its depth is not trustworthy, which is
        # a property of the sensor rather than a tuning choice.
        self.declare_parameter('min_depth_mm', 70.0)
        self.declare_parameter('max_depth_mm', 500.0)
        # How far from the recently-agreed panel distance a detection may sit
        # before it is treated as something else in the scene. 0 disables.
        self.declare_parameter('depth_consistency_mm', 80.0)

        self.detection_topic = self.get_parameter('detection_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_label = self.get_parameter('target_label').value
        self.require_depth = self.get_parameter('require_depth').value
        self.min_depth_mm = self.get_parameter('min_depth_mm').value
        self.max_depth_mm = self.get_parameter('max_depth_mm').value
        self.depth_consistency_mm = self.get_parameter('depth_consistency_mm').value

        self._bridge = CvBridge()
        self._depth_image: np.ndarray | None = None
        self._published_any = False
        self._warned_no_match = False
        # Recent accepted depths, for the consistency test. A median over a
        # short window rather than a running mean: one bad reading should not
        # drag the reference, and the panel does not move.
        self._recent_depths: list[float] = []
        self._rejected = {'no_depth': 0, 'out_of_range': 0, 'inconsistent': 0}
        self._accepted = 0
        # Depths actually observed at rejected detections. Without these the
        # filter is a black box: "rejected 150" cannot distinguish "the panel
        # is out of range" from "these are all background noise", and those
        # need opposite fixes.
        self._seen_depths: list[float] = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._detection_sub = self.create_subscription(
            Detection2DArray, self.detection_topic, self._detection_cb, qos)
        self._depth_sub = self.create_subscription(
            Image, self.depth_topic, self._depth_cb, qos)

        self._point_pub = self.create_publisher(PointStamped, self.output_topic, 10)

        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f'detection_bridge_node up: {self.detection_topic} + '
            f'{self.depth_topic} -> {self.output_topic}, '
            f"target_label={self.target_label!r}")

    # Tunable while the stack runs. The depth filter is exactly the kind of
    # thing you want to adjust against a live scene rather than by relaunching
    # and losing the state you were looking at.
    _LIVE = ('require_depth', 'min_depth_mm', 'max_depth_mm',
             'depth_consistency_mm')

    def _on_set_parameters(self, params) -> SetParametersResult:
        for p in params:
            if p.name == 'target_label':
                self.target_label = p.value
                self._warned_no_match = False
                self.get_logger().info(
                    f'target_label -> {self.target_label!r}'
                    + ('' if self.target_label else ' (best detection of any class)'))
            elif p.name in self._LIVE:
                setattr(self, p.name, p.value)
                # Forget the agreed panel distance: it was learned under the
                # old rules, and keeping it would let a stale reference veto
                # detections the new settings are meant to admit.
                self._recent_depths.clear()
                self.get_logger().info(f'{p.name} -> {p.value}')
        return SetParametersResult(successful=True)

    def _depth_cb(self, msg: Image) -> None:
        try:
            self._depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'depth conversion failed: {exc}')

    def _panel_depth(self) -> float | None:
        """The distance the panel has recently been agreed to be at."""
        if len(self._recent_depths) < 3:
            return None
        return float(np.median(self._recent_depths))

    # uint16 saturation. A D405 reports this where it resolved nothing, so it
    # is "no measurement", not "65 metres away" -- and reporting it as a
    # distance makes the diagnostics nonsense ("depths seen: 2021-65535mm").
    DEPTH_INVALID = 65535

    def _depth_ok(self, depth_mm: float) -> tuple[bool, str]:
        """Could a real button be here? (accepted, reason-if-not)"""
        if depth_mm <= 0.0 or depth_mm >= self.DEPTH_INVALID:
            # No depth at all. On a D405 that means nothing solid was
            # resolved there -- which is what empty space returns, and what
            # most false positives sit on.
            return (not self.require_depth), 'no_depth'
        if not (self.min_depth_mm <= depth_mm <= self.max_depth_mm):
            return False, 'out_of_range'
        ref = self._panel_depth()
        if (self.depth_consistency_mm > 0 and ref is not None
                and abs(depth_mm - ref) > self.depth_consistency_mm):
            # Something solid, but not on the surface the panel is on.
            return False, 'inconsistent'
        return True, ''

    def _detection_cb(self, msg: Detection2DArray) -> None:
        # vision_msgs in Humble uses string class_id (see food_detector_node).
        #
        # Pick the best candidate that PASSES the depth test, not the best
        # candidate overall. Filtering after the choice would let one
        # high-scoring phantom suppress a real button behind it.
        target = None
        any_label_match = False
        for det in msg.detections:
            for result in det.results:
                class_id = result.hypothesis.class_id
                score = result.hypothesis.score
                # No label set: any class will do, best score wins.
                if self.target_label and class_id.lower() != self.target_label.lower():
                    continue
                any_label_match = True
                d = self._sample_depth(det.bbox.center.position.x,
                                       det.bbox.center.position.y)
                ok, why = self._depth_ok(d)
                if not ok:
                    self._rejected[why] = self._rejected.get(why, 0) + 1
                    if 0 < d < self.DEPTH_INVALID:
                        self._seen_depths.append(d)
                        del self._seen_depths[:-200]
                    continue
                if target is None or score > target[1]:
                    target = (det, score, class_id, d)

        if target is None and any_label_match:
            total = sum(self._rejected.values())
            if total and total % 50 == 0:
                ref = self._panel_depth()
                obs = ''
                if self._seen_depths:
                    arr = np.array(self._seen_depths)
                    obs = (f'Depths seen at rejected detections: '
                           f'{arr.min():.0f}-{arr.max():.0f}mm '
                           f'(median {np.median(arr):.0f}). ')
                self.get_logger().info(
                    f'depth filter has rejected {total} detection(s): '
                    f'{self._rejected} (accepted {self._accepted}). '
                    + obs
                    + (f'Panel taken to be at {ref:.0f}mm. '
                       if ref is not None else '')
                    + f'Accepting {self.min_depth_mm:.0f}-'
                      f'{self.max_depth_mm:.0f}mm. If those depths look like '
                      'your panel, widen the range; if they are far away, the '
                      'detector is firing on background and the panel is not '
                      'in view.',
                    throttle_duration_sec=10.0)

        if target is None:
            if self.target_label and msg.detections and not self._warned_no_match:
                self._warned_no_match = True
                seen = sorted({r.hypothesis.class_id
                               for d in msg.detections for r in d.results})
                self.get_logger().warn(
                    f'target_label={self.target_label!r} matches nothing; '
                    f'detector is reporting {seen}. Nothing will be published '
                    f'until it matches, so the servo will keep searching.')
            return

        det, score, class_id, depth_mm = target

        # Only accepted depths shape the panel reference, so a rejected
        # outlier can never drag it toward itself and start admitting more
        # like it.
        if depth_mm > 0.0:
            self._recent_depths.append(depth_mm)
            del self._recent_depths[:-25]
        self._accepted += 1

        if not self._published_any:
            self._published_any = True
            self.get_logger().info(
                f'first target: {class_id!r} score {score:.2f} at '
                f'{depth_mm:.0f}mm -- publishing to {self.output_topic}')

        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y

        point = PointStamped()
        point.header = msg.header
        point.point.x = float(cx)
        point.point.y = float(cy)
        point.point.z = float(depth_mm)
        self._point_pub.publish(point)

    def _sample_depth(self, cx: float, cy: float) -> float:
        if self._depth_image is None:
            return 0.0

        h, w = self._depth_image.shape[:2]
        ix, iy = int(round(cx)), int(round(cy))

        x0, x1 = max(0, ix - 2), min(w, ix + 3)
        y0, y1 = max(0, iy - 2), min(h, iy + 3)
        if x0 >= x1 or y0 >= y1:
            return 0.0

        patch = self._depth_image[y0:y1, x0:x1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return 0.0

        return float(np.median(valid))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
