"""
ROS 2 elevator-button-detection node for myCobot 280.

Subscribes to a sensor_msgs/Image stream (default: /camera/image_raw), runs
inference with a trained YOLOv11n model, and publishes:

  1. A CV2 window with bboxes + labels (for humans).
  2. The annotated frame on /perception/button_detections/image
     (so RViz2 / other consumers can subscribe).
  3. Structured detections on /perception/button_detections as
     vision_msgs/Detection2DArray (for downstream targeting/pressing logic).

No class filtering is applied -- every detection the model produces is
published. The model's own class names (e.g. "3", "lobby", "open") go into
`hypothesis.hypothesis.class_id`.

Design notes
------------
- QoS is BEST_EFFORT, depth=1 -- camera frames are "latest wins".
- `cv2.waitKey(1)` is mandatory after `imshow` to pump the GUI loop.
- All tunables are ROS 2 parameters; same binary works headless
  (`show_window:=false`) or with a custom-trained .pt
  (`model_path:=/path/to/elevator_buttons.pt`).
"""

from __future__ import annotations

import torch

# cuDNN OFF, deliberately, and this must run before any CUDA work.
#
# NVIDIA's torch 2.5.0a0 for JetPack 6.1 links against cuDNN 9 and will not
# even import without libcudnn.so.9 present, but the cuDNN 9 pip wheels
# (cu12 and cu13 both) fail at runtime on this Jetson's CUDA 12.2 with
# CUDNN_STATUS_NOT_INITIALIZED the moment a convolution runs. So the symlinks
# into site-packages/nvidia/cudnn/lib have to STAY -- removing them breaks the
# import -- while cuDNN itself has to be switched off.
#
# Disabling it falls back to native CUDA convolutions, which cost nothing that
# matters here: measured 24 fps against roughly 2 fps on CPU. Deleting this
# line does not "re-enable acceleration", it crashes the node on the first
# frame.
torch.backends.cudnn.enabled = False

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from ultralytics import YOLO


class ButtonDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__('button_detector_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_path', 'elevator_buttons.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'Button Detection')
        self.declare_parameter('device', 'cuda:0')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self._conf = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self._show_window = self.get_parameter('show_window').get_parameter_value().bool_value
        self._window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self._device = self.get_parameter('device').get_parameter_value().string_value

        self._bridge = CvBridge()

        self.get_logger().info(f'Loading model: {model_path} (device={self._device})')
        self._model = YOLO(model_path)
        self._class_names = self._model.names

        # BEST_EFFORT + depth 1 = latest-frame-wins, drop stale frames.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._sub = self.create_subscription(
            Image, image_topic, self._image_cb, sensor_qos,
        )
        self._annotated_pub = self.create_publisher(
            Image, '/perception/button_detections/image', 10,
        )
        self._detections_pub = self.create_publisher(
            Detection2DArray, '/perception/button_detections', 10,
        )

        self.get_logger().info(f'Subscribed to {image_topic}, ready for inference')

    def _image_cb(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'cv_bridge failed: {e}')
            return

        # `verbose=False` silences ultralytics per-frame stdout spam.
        results = self._model.predict(
            frame,
            conf=self._conf,
            device=self._device,
            verbose=False,
        )
        result = results[0]

        annotated, detection_array = self._build_outputs(frame, result, msg.header)

        self._annotated_pub.publish(
            self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8'),
        )
        self._detections_pub.publish(detection_array)

        if self._show_window:
            cv2.imshow(self._window_name, annotated)
            # Required to pump the GUI event loop. 1ms is enough.
            cv2.waitKey(1)

    def _build_outputs(
        self,
        frame: np.ndarray,
        result,
        header,
    ) -> tuple[np.ndarray, Detection2DArray]:
        annotated = frame.copy()
        det_array = Detection2DArray()
        det_array.header = header

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return annotated, det_array

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clses):
            label = self._class_names.get(int(cls_id), str(cls_id))
            self._draw_box(annotated, x1, y1, x2, y2, label, conf)
            det_array.detections.append(
                self._make_detection(x1, y1, x2, y2, label, conf, header),
            )

        return annotated, det_array

    @staticmethod
    def _draw_box(
        img: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        label: str, conf: float,
    ) -> None:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        text = f'{label} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), (0, 255, 0), -1)
        cv2.putText(
            img, text, (p1[0] + 2, p1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _make_detection(
        x1: float, y1: float, x2: float, y2: float,
        label: str, conf: float, header,
    ) -> Detection2D:
        det = Detection2D()
        det.header = header
        # vision_msgs 4.x: BoundingBox2D.center is a vision_msgs/Pose2D where
        # x/y live inside a nested Point2D `position` field (NOT directly on
        # the Pose2D like the older geometry_msgs/Pose2D layout).
        det.bbox.center.position.x = float((x1 + x2) / 2.0)
        det.bbox.center.position.y = float((y1 + y2) / 2.0)
        det.bbox.center.theta = 0.0
        det.bbox.size_x = float(x2 - x1)
        det.bbox.size_y = float(y2 - y1)

        hypothesis = ObjectHypothesisWithPose()
        # vision_msgs in Humble uses string class_id.
        hypothesis.hypothesis.class_id = label
        hypothesis.hypothesis.score = float(conf)
        det.results.append(hypothesis)
        return det

    def destroy_node(self) -> bool:
        if self._show_window:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ButtonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == '__main__':
    main()
