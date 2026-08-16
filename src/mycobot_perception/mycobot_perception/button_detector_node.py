"""
ROS 2 elevator-button-detection node for myCobot 280.

Subscribes to a sensor_msgs/Image stream (default: /camera/image_raw), runs
TWO-STAGE inference -- find the buttons, then read their legends -- and
publishes:

  1. A CV2 window with bboxes + labels (for humans).
  2. The annotated frame on /perception/button_detections/image
     (so RViz2 / other consumers can subscribe).
  3. Structured detections on /perception/button_detections as
     vision_msgs/Detection2DArray (for downstream targeting/pressing logic).

No class filtering is applied -- every detection the model produces is
published, including the ones the arm must never press, because "there is a
key switch here" is information the caller needs rather than noise.

Two stages, and why
-------------------
  Stage A  a 9-class detector: up, down, floor, open, close, help, stop,
           keyhole, other. Every class has hundreds to thousands of training
           examples.
  Stage B  a classifier over the crop of each `floor` box, which says WHICH
           floor: 0-36, B, B1, B2, B3, G, L, LG, M, CH, -1, or `unreadable`.

The single-stage version this replaces gave the detector one class per floor
and stopped at button-10, which fails against a real elevator in both
directions -- a building with a 14th floor had it called background, and the
top of the range never had the examples to be learned. It also had no `up` and
no `down` class at all, which are the two highest-priority buttons in the
project. See button_classes.py for the measured counts behind all of that.

What lands in `hypothesis.hypothesis.class_id` is what a caller would
naturally ask for: `up`, `down`, `help`, or a bare legend like `7` / `B1` /
`G`. detection_bridge_node matches `target_label` against exactly that string,
so `target_label:=up` and `target_label:=7` both work.

**A floor whose legend was not read confidently publishes as `floor`.** That
is a deliberate refusal, not a fallback: the button stays visible, but
`target_label:=7` will not match it, so the arm cannot press it believing it
is 7. Pressing the wrong floor is the failure this pipeline is built to avoid,
and an unread button costs a retry while a misread one costs a trip to the
wrong storey with nothing in the logs to say it happened.

The reader is optional. If it will not load, floor buttons are still detected
and published as `floor` -- the arm loses the number, not the panel.

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

from mycobot_perception.button_classes import (
    FORBIDDEN, PRESSABLE, UNREADABLE)


class ButtonDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__('button_detector_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        # TensorRT, because it is the only GPU path that runs on this board.
        # Measured back to back 2026-08-14 on one frame, median of 6:
        #
        #   elevator_buttons.pt  on cpu    535.9ms   works
        #   elevator_buttons.pt  on cuda        -    cuDNN error:
        #                                            CUDNN_STATUS_EXECUTION_FAILED_CUDART
        #   elevator_buttons.engine (TRT)   18.7ms   works
        #
        # So the PyTorch CUDA path is genuinely broken and the earlier note
        # about it was right -- but TensorRT does not go through cuDNN, and it
        # is 28x faster than the CPU fallback. The engine had been sitting in
        # the repo unused since 2026-08-11.
        #
        # The trap that note recorded still stands: every check short of a real
        # forward pass says the GPU is fine. torch.cuda.is_available() returns
        # True, the device reports as Orin, torch.version.cuda is 12.6, cuDNN
        # reports 9.2.4, and a bare torch conv on CUDA now succeeds. It is
        # Ultralytics' .pt path that fails, at kernel execution. So do NOT
        # switch model_path back to .pt on device=0 without RUNNING it.
        #
        # Falls back to the .pt on CPU if the engine is missing, since the
        # engine is built for this specific board and does not travel.
        self.declare_parameter('model_path', 'button_detect.engine')
        self.declare_parameter('fallback_model_path', 'button_detect.pt')
        self.declare_parameter('confidence_threshold', 0.5)

        # --- stage B: reading the floor legend off the crop ----------------
        #
        # Stage A says `floor`; this says WHICH floor. Split in two because
        # the detector's per-class support collapses down the floor range
        # (1200 examples of floor 2, 55 of floor 33) while a classifier on an
        # already-centred 128px crop sees the numeral an order of magnitude
        # larger. See scripts/train_button_reader.py.
        #
        # Optional by construction: if the reader will not load, every floor
        # button still DETECTS, it just publishes as the generic `floor`
        # instead of `7`. Losing the number degrades the arm to "I can see a
        # button", which is recoverable; a node that refuses to start is not.
        self.declare_parameter('reader_model_path', 'button_read.engine')
        self.declare_parameter('reader_fallback_path', 'button_read.pt')
        self.declare_parameter('read_legends', True)
        # Below this the legend is NOT published, and the button reports as
        # `floor` rather than a guess. Set high on purpose: the failure this
        # project cares about is pressing the WRONG floor, and an unread
        # button costs a retry while a misread one costs a trip to the wrong
        # storey with nothing in the logs to say so. 6/9 is the confusion
        # that actually happens -- they are a 180-degree rotation apart.
        self.declare_parameter('reader_min_confidence', 0.75)
        # Fraction of box size added around each crop before reading, so the
        # reader sees the button rim rather than a tight numeral. MUST match
        # --crop-pad in scripts/build_button_dataset.py (0.12): a classifier
        # fed a tighter or looser crop than it trained on loses accuracy for
        # no visible reason.
        self.declare_parameter('reader_crop_pad', 0.12)
        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'Button Detection')
        # A STRING, because it is read with .string_value below and because
        # ultralytics accepts '0' and 'cpu' alike. Declaring it as an int here
        # makes that read return '' and the device silently wrong.
        self.declare_parameter('device', '0')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self._conf = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self._show_window = self.get_parameter('show_window').get_parameter_value().bool_value
        self._window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self._device = self.get_parameter('device').get_parameter_value().string_value

        self._bridge = CvBridge()

        # A TensorRT engine is built for the board it was built on, so a
        # missing or unloadable one is an expected case rather than a crash --
        # fall back to the portable .pt on CPU and SAY which is running, since
        # a 29x speed difference that is not announced gets diagnosed as
        # something else entirely.
        fallback = self.get_parameter(
            'fallback_model_path').get_parameter_value().string_value
        try:
            self.get_logger().info(
                f'Loading model: {model_path} (device={self._device})')
            self._model = YOLO(model_path, task='detect')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warning(
                f'{model_path} would not load ({e}); falling back to '
                f'{fallback} on CPU. Expect ~536ms/frame against ~19ms -- '
                'rebuild the engine for this board to get it back.')
            self._model = YOLO(fallback, task='detect')
            self._device = 'cpu'
        self._class_names = self._model.names

        self._reader = None
        self._reader_names: dict[int, str] = {}
        self._read_min = self.get_parameter(
            'reader_min_confidence').get_parameter_value().double_value
        self._crop_pad = self.get_parameter(
            'reader_crop_pad').get_parameter_value().double_value
        if self.get_parameter('read_legends').get_parameter_value().bool_value:
            self._load_reader()

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

    def _load_reader(self) -> None:
        """Load the legend classifier. Never fatal -- see the parameter note."""
        path = self.get_parameter(
            'reader_model_path').get_parameter_value().string_value
        fallback = self.get_parameter(
            'reader_fallback_path').get_parameter_value().string_value
        for cand in (path, fallback):
            if not cand:
                continue
            try:
                self._reader = YOLO(cand, task='classify')
                self._reader_names = self._reader.names
                self.get_logger().info(
                    f'Legend reader: {cand} ({len(self._reader_names)} '
                    f'legends, min confidence {self._read_min:.2f})')
                return
            except Exception as e:  # noqa: BLE001
                self.get_logger().warning(f'reader {cand} would not load ({e})')
        self.get_logger().warning(
            'No legend reader loaded. Floor buttons will still be DETECTED '
            'and published as the generic class `floor`, but not numbered -- '
            'so target_label:=7 will match nothing while target_label:=up '
            'still works. Train one with scripts/train_button_reader.py.')

    def _read_legends(self, frame: np.ndarray, boxes: list) -> list:
        """Classify each floor crop. Returns a legend (or None) per box.

        Batched in one call: a panel has a dozen floor buttons and running
        them one at a time would pay the per-inference overhead a dozen
        times over for crops that are 128px each.
        """
        out: list = [None] * len(boxes)
        if self._reader is None or not boxes:
            return out

        H, W = frame.shape[:2]
        crops, idx = [], []
        for i, (x1, y1, x2, y2, kind) in enumerate(boxes):
            if kind != 'floor':
                continue
            pw = (x2 - x1) * self._crop_pad
            ph = (y2 - y1) * self._crop_pad
            cx1 = int(max(0, x1 - pw))
            cy1 = int(max(0, y1 - ph))
            cx2 = int(min(W, x2 + pw))
            cy2 = int(min(H, y2 + ph))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size:
                crops.append(crop)
                idx.append(i)

        if not crops:
            return out

        try:
            results = self._reader.predict(crops, device=self._device,
                                           verbose=False)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'legend reader failed: {e}')
            return out

        for i, res in zip(idx, results):
            probs = getattr(res, 'probs', None)
            if probs is None:
                continue
            conf = float(probs.top1conf)
            name = self._reader_names.get(int(probs.top1), '')
            # Two separate refusals, and they mean different things. Below
            # threshold: the model is not sure enough to be trusted with a
            # floor. UNREADABLE: it IS sure, and what it is sure of is that
            # the legend cannot be made out -- a blank, blurred or unknown
            # button. Both publish as `floor`; only the second is a
            # confident answer.
            if conf < self._read_min or name == UNREADABLE:
                continue
            out[i] = (name, conf)
        return out

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

        kinds = [self._class_names.get(int(c), str(c)) for c in clses]
        packed = [(x1, y1, x2, y2, k)
                  for (x1, y1, x2, y2), k in zip(xyxy, kinds)]
        legends = self._read_legends(frame, packed)

        for (x1, y1, x2, y2), conf, kind, legend in zip(
                xyxy, confs, kinds, legends):
            # What goes out as class_id is what `target_label` is matched
            # against in detection_bridge_node, so it is chosen to be the
            # thing a caller would naturally ask for: `up`, `down`, `help`,
            # or a bare floor legend like `7` / `B1` / `G`.
            #
            # A floor whose legend was not read confidently publishes as
            # `floor`. That is deliberate rather than a fallback: it stays
            # visible and pressable-in-principle, but `target_label:=7` will
            # not match it, so the arm cannot press it BELIEVING it is 7.
            if kind == 'floor' and legend is not None:
                label = legend[0]
                score = float(conf) * float(legend[1])
            else:
                label = kind
                score = float(conf)
            self._draw_box(annotated, x1, y1, x2, y2, label, score, kind)
            det_array.detections.append(
                self._make_detection(x1, y1, x2, y2, label, score, header),
            )

        return annotated, det_array

    @staticmethod
    def _draw_box(
        img: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        label: str, conf: float, kind: str = '',
    ) -> None:
        # Colour by whether the arm may press it, because the overlay is what
        # a human checks before letting it move. Red is not "low confidence",
        # it is "never press this": an emergency stop, a key switch, or a
        # button whose legend could not be read.
        if kind in FORBIDDEN:
            colour = (0, 0, 255)
        elif kind in ('up', 'down'):
            colour = (0, 200, 255)      # top priority, called out on sight
        elif kind in PRESSABLE:
            colour = (0, 255, 0)
        else:
            colour = (200, 200, 200)

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, colour, 2)
        text = f'{label} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), colour, -1)
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
