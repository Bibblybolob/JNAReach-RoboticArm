"""
ROS 2 camera node for myCobot 280 Pi.
Consumes an MJPEG HTTP stream from the Pi's camera_stream.py server
and republishes it as sensor_msgs/Image on /camera/image_raw plus
sensor_msgs/CameraInfo on /camera/camera_info.
The Pi runs a lightweight Python MJPEG HTTP server (pi/camera_stream.py)
on port 8080. This node manually parses the multipart MJPEG HTTP stream
rather than relying on cv2.VideoCapture, which is unreliable across
different OpenCV/GStreamer builds for this kind of stream.
"""
import threading

import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.declare_parameter('camera_url', 'http://192.168.1.46:8080/?action=stream')
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('frame_id', 'camera_link')

        self._url = self.get_parameter('camera_url').get_parameter_value().string_value
        rate = self.get_parameter('frame_rate').get_parameter_value().double_value
        self._frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self._bridge = CvBridge()
        self._image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self._info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        self._width = None
        self._height = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()

        self.get_logger().info(f'Opening camera stream: {self._url}')
        self._reader_thread = threading.Thread(target=self._stream_reader, daemon=True)
        self._reader_thread.start()

        self._timer = self.create_timer(1.0 / rate, self._publish_latest_frame)

    def _stream_reader(self):
        """Runs in a background thread: continuously reads the MJPEG
        multipart HTTP stream and decodes frames as they arrive."""
        while not self._stop_event.is_set():
            try:
                resp = requests.get(self._url, stream=True, timeout=5)
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
