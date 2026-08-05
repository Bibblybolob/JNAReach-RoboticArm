#!/usr/bin/env python3
"""Bridge between a YOLO-style Detection2DArray and the PointStamped the
visual servo expects.

Subscribes to `vision_msgs/Detection2DArray` button detections and an
optional 16UC1 depth image (RealSense, mm), and republishes the selected
button's bbox centre as `geometry_msgs/PointStamped` on `/button/point_px`.

Target selection is a runtime ROS parameter (`target_label`), following the
same live-tunable pattern as `visual_servo_node.py` — no custom service type
is needed since ROS 2 already exposes parameter get/set as services. When
`target_label` is empty, the highest-confidence detection of any label is
forwarded; when set, only detections whose `class_id` matches
(case-insensitive) are considered, and the highest-confidence match wins.

Depth lookup takes a 5x5 median around the detection centre for noise
rejection and clamps to positive values; if depth is zero or the depth topic
has not produced a frame yet, `point.z` falls back to the bbox diagonal in
pixels so the servo still has a distance proxy to reason about.
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class DetectionBridgeNode(Node):
    def __init__(self):
        super().__init__('detection_bridge_node')

        self.declare_parameter(
            'detections_topic', '/perception/button_detections',
            ParameterDescriptor(description='Detection2DArray topic from the YOLO detector'))
        self.declare_parameter(
            'depth_topic', '/camera/depth_raw',
            ParameterDescriptor(description='16UC1 depth image topic, mm'))
        self.declare_parameter(
            'output_topic', '/button/point_px',
            ParameterDescriptor(description='PointStamped output topic for the servo'))
        self.declare_parameter(
            'target_label', '',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "Floor label to target, e.g. '3'. Empty = highest-confidence "
                    "detection of any label. Set live with "
                    "`ros2 param set /detection_bridge_node target_label 3`.")))
        self.declare_parameter(
            'use_depth', True,
            ParameterDescriptor(description='Look up depth at the detection centre when true'))
        self.declare_parameter(
            'min_confidence', 0.2,
            ParameterDescriptor(description='Minimum hypothesis score to consider a detection'))

        self._detections_topic = self.get_parameter('detections_topic').value
        self._depth_topic = self.get_parameter('depth_topic').value
        self._output_topic = self.get_parameter('output_topic').value
        self._target_label = self.get_parameter('target_label').value
        self._use_depth = self.get_parameter('use_depth').value
        self._min_confidence = self.get_parameter('min_confidence').value

        self.add_on_set_parameters_callback(self._on_set_parameters)

        self._bridge = CvBridge()
        self._latest_depth = None  # numpy uint16 array, mm

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10)
        depth_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        self._det_sub = self.create_subscription(
            Detection2DArray, self._detections_topic, self._on_detections, reliable_qos)
        self._depth_sub = self.create_subscription(
            Image, self._depth_topic, self._on_depth, depth_qos)
        self._point_pub = self.create_publisher(PointStamped, self._output_topic, reliable_qos)

        self.get_logger().info(
            f"detection_bridge_node up: detections='{self._detections_topic}' "
            f"depth='{self._depth_topic}' output='{self._output_topic}' "
            f"target_label='{self._target_label}' use_depth={self._use_depth}")

    # -- parameter handling -------------------------------------------------

    def _on_set_parameters(self, params):
        for p in params:
            if p.name == 'target_label':
                new_label = p.value
                if new_label != self._target_label:
                    self.get_logger().info(
                        f"target_label changed: '{self._target_label}' -> '{new_label}'")
                self._target_label = new_label
            elif p.name == 'use_depth':
                self._use_depth = p.value
            elif p.name == 'min_confidence':
                self._min_confidence = p.value
            elif p.name == 'detections_topic':
                return SetParametersResult(
                    successful=False,
                    reason='detections_topic is structural; restart the node to change it')
            elif p.name == 'depth_topic':
                return SetParametersResult(
                    successful=False,
                    reason='depth_topic is structural; restart the node to change it')
            elif p.name == 'output_topic':
                return SetParametersResult(
                    successful=False,
                    reason='output_topic is structural; restart the node to change it')
        return SetParametersResult(successful=True)

    # -- subscriptions --------------------------------------------------

    def _on_depth(self, msg: Image):
        try:
            self._latest_depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().warn(f'depth conversion failed: {exc}', throttle_duration_sec=1.0)

    def _on_detections(self, msg: Detection2DArray):
        best = self._select_detection(msg)
        if best is None:
            return

        cx = best.bbox.center.position.x
        cy = best.bbox.center.position.y
        size_x = best.bbox.size_x
        size_y = best.bbox.size_y

        depth_mm = None
        if self._use_depth:
            depth_mm = self._lookup_depth(cx, cy)

        if depth_mm is None or depth_mm <= 0:
            z = float(np.hypot(size_x, size_y))
            self.get_logger().warn(
                'depth unavailable at detection centre, falling back to bbox diagonal',
                throttle_duration_sec=1.0)
        else:
            z = float(depth_mm)

        point = PointStamped()
        point.header = best.header if best.header.stamp.sec or best.header.stamp.nanosec else msg.header
        point.point.x = float(cx)
        point.point.y = float(cy)
        point.point.z = z
        self._point_pub.publish(point)

        label = self._best_label(best)
        self.get_logger().info(
            f"selected button '{label}' at ({cx:.1f}, {cy:.1f}) z={z:.1f}",
            throttle_duration_sec=1.0)

    # -- selection logic --------------------------------------------------

    def _select_detection(self, msg: Detection2DArray):
        target = self._target_label.strip().lower()
        best = None
        best_score = -1.0

        for det in msg.detections:
            if not det.results:
                continue
            hyp_result, score = self._best_result(det)
            if hyp_result is None or score < self._min_confidence:
                continue

            if target:
                if hyp_result.hypothesis.class_id.strip().lower() != target:
                    continue

            if score > best_score:
                best_score = score
                best = det

        return best

    @staticmethod
    def _best_result(det):
        best_result = None
        best_score = -1.0
        for r in det.results:
            score = r.hypothesis.score
            if score > best_score:
                best_score = score
                best_result = r
        return best_result, best_score

    def _best_label(self, det):
        result, _ = self._best_result(det)
        return result.hypothesis.class_id if result is not None else '?'

    # -- depth ------------------------------------------------------------

    def _lookup_depth(self, cx: float, cy: float):
        if self._latest_depth is None:
            return None

        depth = self._latest_depth
        h, w = depth.shape[:2]
        ix, iy = int(round(cx)), int(round(cy))

        x0, x1 = max(0, ix - 2), min(w, ix + 3)
        y0, y1 = max(0, iy - 2), min(h, iy + 3)
        if x1 <= x0 or y1 <= y0:
            return None

        patch = depth[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        return float(np.median(valid))


def main(args=None):
    rclpy.init(args=args)
    node = DetectionBridgeNode()
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
