"""
ROS 2 camera node: publishes /camera/image_raw and /camera/camera_info.

TWO SOURCES, BECAUSE THE CAMERA IS MOVING MACHINES

`source:=mjpeg` (default) reads the Pi's camera_stream.py over HTTP. The
multipart stream is parsed by hand rather than through cv2.VideoCapture,
which is unreliable across OpenCV/GStreamer builds for this kind of stream.

`source:=device` opens a local V4L2/CSI camera directly. That is the Jetson
configuration, and it is not merely a convenience: on the networked path a
frame is captured, JPEG-encoded on the Pi, pushed over WiFi, and decoded here
before anything looks at it, and none of that is measured anywhere. Opening
the camera locally deletes the whole leg rather than optimising it.

Worth being clear about what that does and does not buy, because it is easy
to over-invest here. Perception is already the cheap half of this system --
frames arrive ~13ms old by the node's own reckoning and MediaPipe takes
8-19ms -- while the round trip the servo compensates for is nearer 250ms, and
most of that is the arm command path. Moving the CAMERA local removes a leg
that is real but not the dominant one, and once the Pi and the Jetson are on
Ethernet rather than WiFi that leg gets cheaper still. Removing the ARM's
network hop is what would change the servo's constraints, and that means the
Jetson driving the arm, not just watching it.

Note also that the ~13ms figure is measured from this node's publish stamp,
which is written AFTER the frame has been encoded on the Pi, crossed the
network and been decoded here. The Pi-to-host leg is therefore not measured
anywhere, and nothing here should be taken as evidence about its size.
"""
import os
import threading

import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        # Built from MYCOBOT_IP like everything else, so `ros2 run` on this
        # node alone reaches the same robot the launch files do rather than a
        # stale literal. A hostname works here too -- see the README note on
        # mDNS, since the Pi's DHCP address moves.
        _default_host = os.environ.get('MYCOBOT_IP', '192.168.0.15')
        self.declare_parameter(
            'camera_url', f'http://{_default_host}:8080/?action=stream')
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('stream_read_timeout', 30.0)
        # 'mjpeg' reads the Pi over HTTP; 'device' opens a local camera. The
        # second is the Jetson layout, where the camera is on the same board.
        self.declare_parameter('source', 'mjpeg')
        # V4L2 index or a GStreamer pipeline string. An integer opens
        # /dev/videoN; anything else is handed to OpenCV as-is, which is how
        # a CSI camera on the Jetson gets in (nvarguscamerasrc ... ! appsink).
        self.declare_parameter('device', '0')
        self.declare_parameter('device_width', 640)
        self.declare_parameter('device_height', 480)
        # Ask the camera for MJPG rather than raw YUYV. Same reasoning as on
        # the Pi: YUYV at 640x480 saturates USB 2.0 and the camera answers by
        # dropping frames.
        self.declare_parameter('device_mjpg', True)

        self._source = self.get_parameter('source').get_parameter_value().string_value
        self._url = self.get_parameter('camera_url').get_parameter_value().string_value
        rate = self.get_parameter('frame_rate').get_parameter_value().double_value
        self._frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self._read_timeout = self.get_parameter(
            'stream_read_timeout').get_parameter_value().double_value

        self._bridge = CvBridge()
        # Sensor-data QoS for the image stream: latest frame wins. The default
        # RELIABLE/depth-10 profile is wrong for video -- it makes DDS queue
        # and retransmit frames that are already obsolete by the time they
        # arrive, which on a loaded host adds latency and memory churn for
        # data nobody wants. Every consumer (hand_tracker, food_detector)
        # already requests BEST_EFFORT, so this also stops the publisher doing
        # reliability bookkeeping no subscriber asked for.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._image_pub = self.create_publisher(Image, 'camera/image_raw', sensor_qos)
        # CameraInfo stays RELIABLE: it is tiny, and consumers need to receive
        # it once rather than catch it in flight.
        self._info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        self._width = None
        self._height = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()

        if self._source == 'device':
            target = self._device_reader
            self.get_logger().info(
                f'Opening LOCAL camera: {self.get_parameter("device").value} '
                f'({self.get_parameter("device_width").value}x'
                f'{self.get_parameter("device_height").value}). No network '
                'leg on this path.')
        elif self._source == 'mjpeg':
            target = self._stream_reader
            self.get_logger().info(f'Opening camera stream: {self._url}')
        else:
            raise ValueError(
                f"source must be 'mjpeg' or 'device', got {self._source!r}")
        self._reader_thread = threading.Thread(target=target, daemon=True)
        self._reader_thread.start()

        self._timer = self.create_timer(1.0 / rate, self._publish_latest_frame)

    def _open_device(self):
        """Open a local camera. An integer is a /dev/videoN index; anything
        else is a GStreamer pipeline, which is how a CSI camera on the Jetson
        gets in (nvarguscamerasrc ... ! appsink)."""
        spec = str(self.get_parameter('device').value)
        if spec.isdigit():
            cap = cv2.VideoCapture(int(spec))
            if self.get_parameter('device_mjpg').value:
                # Before the size, as on the Pi: some V4L2 drivers only offer
                # the higher rates for a resolution once the format is MJPG.
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,
                    int(self.get_parameter('device_width').value))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                    int(self.get_parameter('device_height').value))
        else:
            cap = cv2.VideoCapture(spec, cv2.CAP_GSTREAMER)
        try:
            # Freshest frame beats a queued one; a servo loop acting on a
            # buffered frame is correcting where the hand already was.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _device_reader(self):
        """Runs in a background thread: reads a local camera directly.

        The Jetson path. Deliberately mirrors _stream_reader's contract --
        fill _latest_frame, let the timer publish it -- so everything
        downstream is identical and only the source changes.
        """
        cap = None
        fails = 0
        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = self._open_device()
                if not cap.isOpened():
                    self.get_logger().error(
                        f'Cannot open camera '
                        f'{self.get_parameter("device").value}; retrying in 2s')
                    self._stop_event.wait(2.0)
                    continue
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.get_logger().info(f'Local camera open at {w}x{h}')

            ok, frame = cap.read()
            if not ok:
                # Back off rather than spin. The same tight retry loop on the
                # Pi was enough to starve the board it was running on.
                fails += 1
                if fails in (1, 10) or fails % 200 == 0:
                    self.get_logger().warn(
                        f'Camera read failed ({fails} in a row)')
                self._stop_event.wait(min(0.5, 0.01 * fails))
                if fails > 50:
                    cap.release()
                    cap = None
                    fails = 0
                continue
            fails = 0
            with self._frame_lock:
                self._latest_frame = frame

        if cap is not None:
            cap.release()

    def _stream_reader(self):
        """Runs in a background thread: continuously reads the MJPEG
        multipart HTTP stream and decodes frames as they arrive."""
        while not self._stop_event.is_set():
            try:
                # (connect timeout, read timeout). A single number here would
                # apply to both, and requests treats it as a per-read stall
                # detector on a streaming response, not a total-request
                # deadline: a plain `nc -zv` to the port succeeds instantly
                # because it only checks the TCP handshake, but a brief gap
                # between MJPEG frames (Wi-Fi jitter, the Pi's stream loop
                # getting descheduled) is enough to trip a short read timeout
                # and force a reconnect even though the stream is healthy.
                resp = requests.get(
                    self._url, stream=True, timeout=(5, self._read_timeout))
                self.get_logger().info('Camera stream connected')
                buf = b''
                for chunk in resp.iter_content(chunk_size=4096):
                    if self._stop_event.is_set():
                        return
                    if not chunk:
                        continue
                    buf += chunk
                    start = buf.find(b'\xff\xd8')  # JPEG SOI marker
                    end = buf.find(b'\xff\xd9')    # JPEG EOI marker
                    if start != -1 and end != -1 and end > start:
                        jpg = buf[start:end + 2]
                        buf = buf[end + 2:]
                        frame = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if frame is not None:
                            with self._frame_lock:
                                self._latest_frame = frame
            except Exception as e:
                self.get_logger().error(f'Camera stream error: {e}; retrying in 2s')
                self._stop_event.wait(2.0)

    def _publish_latest_frame(self):
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return

        if self._width is None:
            self._height, self._width = frame.shape[:2]
            self.get_logger().info(f'Streaming {self._width}x{self._height}')

        stamp = self.get_clock().now().to_msg()
        img_msg = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self._frame_id
        self._image_pub.publish(img_msg)

        info_msg = CameraInfo()
        info_msg.header.stamp = stamp
        info_msg.header.frame_id = self._frame_id
        info_msg.width = self._width
        info_msg.height = self._height
        self._info_pub.publish(info_msg)

    def destroy_node(self):
        self._stop_event.set()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
