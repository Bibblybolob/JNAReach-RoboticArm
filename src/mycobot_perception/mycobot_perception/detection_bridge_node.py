"""
Bridges vision_msgs/Detection2DArray + a depth image into the PointStamped
contract the visual servo expects.

    /button/point_px    geometry_msgs/PointStamped
                         x, y = detection bbox centre, in FULL-frame pixels
                         z    = depth at that point, in millimetres (0 if
                                unavailable)

Which detection gets published is controlled by the `target_label` parameter
(empty by default, which means "publish nothing"). Set it at runtime:

    ros2 param set /detection_bridge_node target_label button_5

If more than one detection matches the label, the highest-confidence one
wins. Depth is sampled as the median of a 5x5 patch around the bbox centre,
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

        self.detection_topic = self.get_parameter('detection_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_label = self.get_parameter('target_label').value

        self._bridge = CvBridge()
        self._depth_image: np.ndarray | None = None

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

    def _on_set_parameters(self, params) -> SetParametersResult:
        for p in params:
            if p.name == 'target_label':
                self.target_label = p.value
                self.get_logger().info(f'target_label -> {self.target_label!r}')
        return SetParametersResult(successful=True)

    def _depth_cb(self, msg: Image) -> None:
        try:
            self._depth_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'depth conversion failed: {exc}')

    def _detection_cb(self, msg: Detection2DArray) -> None:
        if not self.target_label:
            return

        # vision_msgs in Humble uses string class_id (see food_detector_node).
        target = None
        for det in msg.detections:
            for result in det.results:
                class_id = result.hypothesis.class_id
                score = result.hypothesis.score
                if class_id.lower() == self.target_label.lower():
                    if target is None or score > target[1]:
                        target = (det, score)

        if target is None:
            return

        det = target[0]
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y

        depth_mm = self._sample_depth(cx, cy)

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
