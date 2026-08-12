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
import time

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
        self.declare_parameter('device_fps', 30.0)
        # V4L2 capture queue depth. 0 means "do not touch it", which is the
        # default because setting it to 1 HALVES the frame rate on this
        # driver: 30.0 fps untouched against 18.5 with BUFFERSIZE=1, measured
        # back to back on the same camera.
        #
        # A depth of 1 is the obvious choice for a servo loop -- a queued
        # frame is a stale frame, and this loop would rather have a new
        # picture than every picture. But it only pays off if the reader is
        # SLOWER than the camera, and _device_reader below is a tight loop
        # with no pacing, so it drains as fast as frames arrive and the queue
        # never builds. Paying half the frame rate to prevent a queue that
        # does not form is the wrong trade twice over, since detection rate is
        # the ceiling on the whole servo loop.
        #
        # Set it to 1 if the transit or staleness figures ever suggest frames
        # really are queueing.
        self.declare_parameter('device_buffersize', 0)
        # See _open_device: auto-exposure silently caps the frame rate in dim
        # light. Turn it off and the rate triples; the image gets darker.
        self.declare_parameter('device_auto_exposure', True)
        # Only used when auto exposure is off. 0 leaves whatever the driver
        # defaults to. Units are driver-specific -- smaller is shorter, and
        # what matters is that exposure time bounds the frame period.
        #
        # USUALLY NEEDED ALONGSIDE device_auto_exposure:=false, because the
        # driver's manual default is often nearly as long as the auto value it
        # replaced. Measured on this webcam at 640x480 MJPG:
        #
        #     auto                                    10.2 fps
        #     manual, driver default exposure         17.2 fps
        #     manual, exposure 50                     30.2 fps
        #
        # So turning auto off is most of the setup and none of the win. Pick
        # the largest value that still hits the rate you want; a shorter
        # exposure than necessary only costs brightness, and detection needs
        # the light.
        self.declare_parameter('device_exposure', 0.0)

        # --- source:=realsense ------------------------------------------
        # An Intel RealSense via librealsense rather than V4L2. A RealSense
        # does enumerate UVC video nodes, so source:=device can sometimes grab
        # SOMETHING off one of them -- but which /dev/videoN carries colour is
        # not stable across replugs, the formats are Y8/Y16/Z16 rather than the
        # MJPG this path asks for, and it throws away depth and the factory
        # intrinsics, which are the entire reason to own the camera.
        #
        # 640x480 by default to match the rest of the project: the servo's
        # tuning is expressed in fractions of a 640-wide frame.
        self.declare_parameter('rs_width', 640)
        self.declare_parameter('rs_height', 480)
        self.declare_parameter('rs_fps', 30)
        # Serial number, for when more than one is plugged in. Empty = first.
        self.declare_parameter('rs_serial', '')
        # Depth costs USB bandwidth and hand detection is purely 2D, so it is
        # off by default. Turn it on for the approach axis and for
        # /hand/point_cam, which needs the intrinsics this path supplies.
        self.declare_parameter('rs_depth', False)
        # D405 SPECIFIC: its colour comes from the same stereo imagers that
        # produce depth, so the two are natively registered and aligning is a
        # no-op that costs CPU. On a D435/D455, whose RGB module is a separate
        # sensor on a different baseline, alignment is required for a pixel in
        # one image to mean anything in the other.
        self.declare_parameter('rs_align_depth_to_color', False)
        # AUTO-EXPOSURE IS A FRAME RATE CONTROL. The same trap as the V4L2
        # path, and it was missed here: a sensor in dim light lengthens its
        # exposure to brighten the image, and frame time cannot be shorter
        # than exposure time, so the camera quietly delivers a fraction of the
        # rate it was asked for while still reporting the profile it agreed
        # to. Measured on a D405 indoors: 14-18 fps of a requested 30, with a
        # third of frames arriving too stale for the tracker to use.
        #
        # Two ways out, and the first is usually enough. auto_exposure_priority
        # 0 tells the sensor to hold the frame rate and accept a darker image
        # rather than the other way round -- auto exposure otherwise, so the
        # picture still adapts. Setting rs_auto_exposure:=false with an
        # explicit rs_exposure pins it hard.
        self.declare_parameter('rs_auto_exposure', True)
        # Hold the requested frame rate even on auto. Only meaningful while
        # rs_auto_exposure is true.
        self.declare_parameter('rs_constant_fps', True)
        # Microseconds. 0 leaves whatever the sensor chose. Only used when
        # rs_auto_exposure is false. Too short breaks detection outright, so
        # measure rather than assume: camera_node reports the rate it really
        # captures.
        self.declare_parameter('rs_exposure', 0.0)

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
        # Only created when a depth stream was actually asked for, so nothing
        # advertises a topic it will never publish on.
        self._depth_pub = None
        if (self.get_parameter('source').value == 'realsense'
                and bool(self.get_parameter('rs_depth').value)):
            self._depth_pub = self.create_publisher(
                Image, 'camera/depth_raw', sensor_qos)

        self._width = None
        self._height = None
        self._latest_frame = None
        self._latest_depth = None
        # Millimetres per raw depth unit; set once the device is open.
        self._depth_scale_mm = 1.0
        self._frame_lock = threading.Lock()
        # Camera intrinsics. None until a source supplies them, which today
        # only source:=realsense does -- the other two paths have no way to
        # know them and calibrate_camera.py has never been run. CameraInfo
        # goes out with empty k/p/d in that case, exactly as it always has.
        self._k = None
        self._p = None
        self._d = None
        # Transit measurement: the smallest (arrival - capture) seen acts as
        # the unknown clock offset, and everything is reported relative to it.
        self._transit_floor = None
        self._transit_n = 0
        self._transit_sum = 0.0
        self._transit_max = 0.0
        self._transit_report = 0.0
        self._stop_event = threading.Event()

        if self._source == 'device':
            target = self._device_reader
            self.get_logger().info(
                f'Opening LOCAL camera: {self.get_parameter("device").value} '
                f'({self.get_parameter("device_width").value}x'
                f'{self.get_parameter("device_height").value}). No network '
                'leg on this path.')
        elif self._source == 'realsense':
            target = self._realsense_reader
            self.get_logger().info(
                f'Opening RealSense via librealsense at '
                f'{self.get_parameter("rs_width").value}x'
                f'{self.get_parameter("rs_height").value}@'
                f'{self.get_parameter("rs_fps").value} '
                f'(depth {"on" if self.get_parameter("rs_depth").value else "off"})')
        elif self._source == 'mjpeg':
            target = self._stream_reader
            self.get_logger().info(f'Opening camera stream: {self._url}')
        else:
            raise ValueError(
                f"source must be 'mjpeg', 'device' or 'realsense', got "
                f'{self._source!r}')
        self._reader_thread = threading.Thread(target=target, daemon=True)
        self._reader_thread.start()

        self._timer = self.create_timer(1.0 / rate, self._publish_latest_frame)

    def _note_transit(self, headers: bytes) -> None:
        """Measure how long this frame took to get here from the sensor.

        The Pi stamps each part with its own capture time. The two clocks are
        not synchronised, so the absolute difference is meaningless -- but the
        offset between them is CONSTANT, so the smallest difference seen in a
        run is the best case (near-zero queueing) and everything above it is
        real, measurable delay. Reporting excess-over-best sidesteps the clock
        problem entirely and still answers the only question being asked:
        is it lagging right now, and by how much.

        This is the one leg of the pipeline nothing else measures. The servo's
        `frames Nms old` counts from this node's PUBLISH stamp, which is
        written after capture, encode, network and decode have already
        happened, so a stall out here is invisible to it.
        """
        idx = headers.find(b'X-Capture-Us:')
        if idx == -1:
            return                      # older Pi build; nothing to measure
        try:
            captured_us = int(headers[idx + 13:].split(b'\r\n', 1)[0])
        except (ValueError, IndexError):
            return

        diff = time.time() - captured_us / 1e6
        if self._transit_floor is None or diff < self._transit_floor:
            # Best case seen so far; treat it as the clock offset. Frames
            # cannot arrive before they were taken, so the minimum is the
            # closest thing to a zero available without synchronised clocks.
            self._transit_floor = diff
        excess = diff - self._transit_floor

        self._transit_n += 1
        self._transit_sum += excess
        self._transit_max = max(self._transit_max, excess)

        now = time.monotonic()
        if now - self._transit_report < 10.0 or self._transit_n < 10:
            return
        mean = self._transit_sum / self._transit_n
        note = ''
        if self._transit_max > 0.25:
            note = (' -- frames are arriving in bursts; the Pi or the link is '
                    'stalling, and the servo cannot see this from its own '
                    'staleness figure')
        self.get_logger().info(
            f'camera transit: +{mean * 1000:.0f}ms mean, '
            f'+{self._transit_max * 1000:.0f}ms worst, over best case'
            f'{note}')
        self._transit_report = now
        self._transit_n = 0
        self._transit_sum = 0.0
        self._transit_max = 0.0

    def _open_device(self):
        """Open a local camera. An integer is a /dev/videoN index; anything
        else is a GStreamer pipeline, which is how a CSI camera on the Jetson
        gets in (nvarguscamerasrc ... ! appsink)."""
        spec = str(self.get_parameter('device').value)
        if spec.isdigit():
            # CAP_V4L2 explicitly. Left to choose, OpenCV picked a backend on
            # this machine whose read() blocked forever -- the node opened the
            # camera, reported success, and then never produced a frame.
            cap = cv2.VideoCapture(int(spec), cv2.CAP_V4L2)
            if self.get_parameter('device_mjpg').value:
                # Before the size, as on the Pi: some V4L2 drivers only offer
                # the higher rates for a resolution once the format is MJPG.
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,
                    int(self.get_parameter('device_width').value))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                    int(self.get_parameter('device_height').value))
            cap.set(cv2.CAP_PROP_FPS,
                    float(self.get_parameter('device_fps').value))

            # AUTO-EXPOSURE IS A FRAME RATE CONTROL, which is not obvious and
            # cost a lot of pipeline work to discover. A UVC camera in dim
            # light lengthens its exposure to brighten the image, and frame
            # time cannot be shorter than exposure time -- so the camera
            # quietly drops to whatever rate the room allows while still
            # REPORTING 30fps when asked. Measured on this webcam: 10.2 fps on
            # auto, 30.2 fps with a short manual exposure. Same camera, same
            # resolution, same everything else.
            #
            # Detection rate is the ceiling on the whole servo loop, so this
            # can matter more than anything tuned in the servo. Left on auto
            # by default because a too-short exposure makes the image dark
            # enough to break detection outright, which is worse than a slow
            # one; the achieved rate is measured below and says so when it
            # looks like this.
            if not bool(self.get_parameter('device_auto_exposure').value):
                # 1 is "manual" in V4L2's UVC mapping, 3 is "aperture
                # priority" (auto). OpenCV passes the value straight through.
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                exposure = float(self.get_parameter('device_exposure').value)
                if exposure > 0:
                    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        else:
            cap = cv2.VideoCapture(spec, cv2.CAP_GSTREAMER)
        depth = int(self.get_parameter('device_buffersize').value)
        if depth > 0:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, depth)
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
                self.get_logger().info(
                    f'Local camera open at {w}x{h}, camera reports '
                    f'{cap.get(cv2.CAP_PROP_FPS):.0f} fps '
                    f'(auto exposure '
                    f'{"on" if self.get_parameter("device_auto_exposure").value else "OFF"})')
                grabbed = 0
                rate_since = time.monotonic()

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

            # What the camera ACTUALLY delivers, which is regularly not what
            # it claims when asked. Reported rather than assumed because the
            # gap is the single cheapest thing to fix in this pipeline and is
            # invisible from anywhere else.
            grabbed += 1
            elapsed = time.monotonic() - rate_since
            if elapsed >= 10.0:
                actual = grabbed / elapsed
                want = float(self.get_parameter('device_fps').value)
                note = ''
                if (actual < 0.6 * want
                        and bool(self.get_parameter(
                            'device_auto_exposure').value)):
                    note = (' -- well under what was asked for. Auto-exposure '
                            'caps frame rate in dim light: this camera '
                            'measured 10fps on auto and 30fps with a short '
                            'manual exposure. Try '
                            'device_auto_exposure:=false, or add light.')
                self.get_logger().info(
                    f'local camera: {actual:.1f} fps captured '
                    f'(asked for {want:.0f}){note}')
                grabbed = 0
                rate_since = time.monotonic()

        if cap is not None:
            cap.release()

    def _realsense_reader(self):
        """Runs in a background thread: reads an Intel RealSense.

        Same contract as the other two readers -- fill _latest_frame, let the
        timer publish it -- so nothing downstream knows or cares which camera
        is attached. What this path adds over source:=device is the factory
        intrinsics, which arrive with the stream and cost nothing: CameraInfo
        has been going out with width and height and empty k/p/d, which is why
        /hand/point_cam has never had anything to say.

        D405 NOTES, since that is the one this project is buying:

        - Its colour comes from the same stereo pair as depth, so there is no
          separate RGB module and the two images are natively registered.
          rs_align_depth_to_color is therefore off by default; on a D435/D455
          it would be required.
        - Depth is only valid from about 7cm to 50cm. That is the point of the
          camera for pressing buttons, and a real limitation for following a
          hand across a room -- but HAND DETECTION IS UNAFFECTED, because
          MediaPipe works on the colour image and never looks at depth. A hand
          at 2m tracks exactly as well; only the range reading goes away.
        - It needs USB 3. On USB 2 librealsense will either refuse the profile
          or quietly hand back a slower one.
        """
        try:
            import pyrealsense2 as rs
        except ImportError:
            self.get_logger().error(
                'source:=realsense needs pyrealsense2, which is not '
                'installed:\n'
                '    pip install pyrealsense2\n'
                'PyPI serves an aarch64 wheel, so this works on a Jetson '
                'without building librealsense. This node publishes nothing '
                'until it is installed; the mjpeg and device sources are '
                'unaffected.')
            return

        width = int(self.get_parameter('rs_width').value)
        height = int(self.get_parameter('rs_height').value)
        fps = int(self.get_parameter('rs_fps').value)
        want_depth = bool(self.get_parameter('rs_depth').value)
        serial = str(self.get_parameter('rs_serial').value)

        while not self._stop_event.is_set():
            pipeline = None
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                if serial:
                    config.enable_device(serial)
                config.enable_stream(rs.stream.color, width, height,
                                     rs.format.bgr8, fps)
                if want_depth:
                    config.enable_stream(rs.stream.depth, width, height,
                                         rs.format.z16, fps)
                profile = pipeline.start(config)

                dev = profile.get_device()
                name = dev.get_info(rs.camera_info.name)
                usb = 'unknown'
                try:
                    usb = dev.get_info(rs.camera_info.usb_type_descriptor)
                except Exception:
                    pass
                self.get_logger().info(
                    f'RealSense open: {name}, USB {usb}, '
                    f'{width}x{height}@{fps}')
                if usb.startswith('2'):
                    self.get_logger().warn(
                        f'This camera is on USB {usb}. A RealSense needs USB '
                        f'3 for full rate -- expect a silently reduced frame '
                        f'rate, which caps the whole servo loop.')

                # Depth scale: metres per raw unit. NOT the same across
                # models, and assuming it is silently scales every distance.
                # A D435 uses 0.001 (1mm per unit) which matches the 16UC1
                # convention of millimetres, so publishing raw is correct
                # there. A D405 uses 0.0001 -- tenths of a millimetre -- so
                # raw values are 10x the millimetre figure.
                #
                # That was live for a whole session: the depth filter saw
                # "3423-35561mm, median 5451" and rejected everything as far
                # beyond the sensor's 50cm range, when those were really
                # 342-3556mm with the panel sitting at 545mm. The panel was
                # in view the entire time.
                try:
                    ds = dev.first_depth_sensor().get_depth_scale()
                except Exception:
                    ds = 0.001
                self._depth_scale_mm = ds * 1000.0
                self.get_logger().info(
                    f'depth scale: {ds:.6f} m per unit '
                    f'({self._depth_scale_mm:.4f} mm) -- /camera/depth_raw is '
                    'published in millimetres regardless of model.')

                self._apply_rs_exposure(rs, dev)

                # Factory intrinsics. Nothing else in this project has ever
                # had them, so this is what unblocks depth downstream.
                intr = (profile.get_stream(rs.stream.color)
                        .as_video_stream_profile().get_intrinsics())
                self._set_intrinsics(intr.fx, intr.fy, intr.ppx, intr.ppy,
                                     list(intr.coeffs))
                self.get_logger().info(
                    f'intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} '
                    f'cx={intr.ppx:.1f} cy={intr.ppy:.1f} -- CameraInfo now '
                    f'carries a real projection matrix.')
                self._warn_field_of_view(intr.fx, intr.fy, width, height)

                align = None
                if want_depth and bool(self.get_parameter(
                        'rs_align_depth_to_color').value):
                    align = rs.align(rs.stream.color)

                grabbed = 0
                rate_since = time.monotonic()
                while not self._stop_event.is_set():
                    frames = pipeline.wait_for_frames(5000)
                    if align is not None:
                        frames = align.process(frames)
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    frame = np.asanyarray(color.get_data())
                    with self._frame_lock:
                        self._latest_frame = frame
                    if want_depth:
                        depth = frames.get_depth_frame()
                        if depth:
                            raw = np.asanyarray(depth.get_data())
                            # Convert to millimetres HERE, so every consumer
                            # gets the same units whatever camera is fitted.
                            # Scaling downstream instead means each consumer
                            # has to know the model, and one that does not
                            # silently reads distances 10x too large.
                            scale = getattr(self, '_depth_scale_mm', 1.0)
                            if abs(scale - 1.0) > 1e-6:
                                mm = raw.astype(np.float32) * scale
                                # Preserve 0 as "no measurement" rather than
                                # letting it round into a real distance.
                                out = np.zeros_like(raw)
                                np.clip(mm, 0, 65535, out=mm)
                                out[raw > 0] = mm[raw > 0].astype(np.uint16)
                                raw = out
                            with self._frame_lock:
                                self._latest_depth = raw

                    grabbed += 1
                    elapsed = time.monotonic() - rate_since
                    if elapsed >= 10.0:
                        self.get_logger().info(
                            f'realsense: {grabbed / elapsed:.1f} fps captured '
                            f'(asked for {fps})')
                        grabbed = 0
                        rate_since = time.monotonic()
            except Exception as e:
                self.get_logger().error(
                    f'RealSense error: {e.__class__.__name__}: {e}; '
                    f'retrying in 2s')
                self._stop_event.wait(2.0)
            finally:
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass

    def _apply_rs_exposure(self, rs, dev) -> None:
        """Stop the sensor trading frame rate for brightness.

        Applied to every sensor that advertises the option rather than to a
        named one: on a D405 colour and depth come from the same stereo pair,
        so there is no separate RGB sensor to reach for, and the option lives
        wherever librealsense decided to put it for that model.
        """
        auto = bool(self.get_parameter('rs_auto_exposure').value)
        constant = bool(self.get_parameter('rs_constant_fps').value)
        exposure = float(self.get_parameter('rs_exposure').value)
        applied = []
        for sensor in dev.query_sensors():
            name = 'sensor'
            try:
                name = sensor.get_info(rs.camera_info.name)
            except Exception:
                pass

            def opt(option, value, label):
                try:
                    if sensor.supports(option):
                        sensor.set_option(option, value)
                        applied.append(f'{name}: {label}={value:g}')
                except Exception as e:
                    self.get_logger().warn(
                        f'{name}: could not set {label} -- '
                        f'{e.__class__.__name__}: {e}')

            opt(rs.option.enable_auto_exposure, 1.0 if auto else 0.0,
                'auto_exposure')
            if auto and constant:
                # 0 = hold the frame rate, darkening the image if it must.
                # 1 = the default, which lets exposure grow past the frame
                # period and silently halves or thirds the rate.
                opt(rs.option.auto_exposure_priority, 0.0,
                    'auto_exposure_priority')
            if not auto and exposure > 0.0:
                opt(rs.option.exposure, exposure, 'exposure_us')

        if applied:
            self.get_logger().info('exposure: ' + '; '.join(applied))
        else:
            self.get_logger().warn(
                'No sensor accepted an exposure option. If the captured rate '
                'comes in well under what was asked for, that is why.')

    def _set_intrinsics(self, fx, fy, cx, cy, coeffs):
        self._k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self._p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._d = [float(c) for c in coeffs]

    def _warn_field_of_view(self, fx, fy, width, height):
        """Report the real field of view, which the servo needs as geometry.

        The servo's error is normalised per axis, so "1.0" means the frame edge
        on any camera -- but how many DEGREES the edge is away is a property of
        the lens. Its `assumed_deg_per_error` was a hand-set 25 because no
        intrinsics existed. On this path they do, and the servo takes them off
        CameraInfo automatically (auto_deg_per_error).

        The direction matters and is easy to get backwards: a WIDER lens means
        the edge is FURTHER away in degrees, so an assumed value that is too
        SMALL scales every command down. It under-corrects, the error never
        closes, and the arm parks far enough out that the progressive gain
        holds it against the max_step_deg clamp -- which then presents as jitter
        rather than as sluggishness.
        """
        import math
        h_deg = math.degrees(math.atan2(width / 2.0, fx))
        v_deg = math.degrees(math.atan2(height / 2.0, fy))
        self.get_logger().info(
            f'field of view: frame edge is {h_deg:.0f} deg horizontally, '
            f'{v_deg:.0f} deg vertically')
        if h_deg > 32.0:
            self.get_logger().warn(
                f'This lens is wider than the ~25/19 deg the servo gains were '
                f'fitted against, so a hand-set assumed_deg_per_error would '
                f'command only {25.0 / h_deg * 100:.0f}% of what centring '
                f'needs. visual_servo_node should be taking {h_deg:.0f}/'
                f'{v_deg:.0f} off CameraInfo -- check its log says so, and if '
                f'auto_deg_per_error is off, pass '
                f'assumed_deg_per_error:={h_deg:.0f} '
                f'assumed_v_deg_per_error:={v_deg:.0f}.')

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
                        # Everything before the SOI is this part's headers,
                        # which is where the Pi puts its capture time.
                        self._note_transit(buf[:start])
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
        if self._k is not None:
            info_msg.distortion_model = 'plumb_bob'
            info_msg.k = self._k
            info_msg.p = self._p
            # plumb_bob wants exactly 5 coefficients; librealsense hands back
            # 5 for Brown-Conrady and zeros for the rest.
            info_msg.d = (self._d + [0.0] * 5)[:5]
        self._info_pub.publish(info_msg)

        if self._depth_pub is not None:
            with self._frame_lock:
                depth = (None if self._latest_depth is None
                         else self._latest_depth.copy())
            if depth is not None:
                depth_msg = self._bridge.cv2_to_imgmsg(
                    depth, encoding='16UC1')
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = self._frame_id
                self._depth_pub.publish(depth_msg)

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
