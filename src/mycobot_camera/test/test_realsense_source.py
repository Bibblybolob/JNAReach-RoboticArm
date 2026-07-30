#!/usr/bin/env python3
"""source:=realsense, exercised without a RealSense attached.

    python3 src/mycobot_camera/test/test_realsense_source.py

pyrealsense2 is stubbed, so this runs on a machine with no camera and no
librealsense -- which is the situation the code was written in, and the reason
it is worth testing at all. What it pins:

  - the reader fills _latest_frame with the colour image, so everything
    downstream is unchanged from source:=device
  - factory intrinsics reach CameraInfo as a real k/p, since publishing them
    empty is why /hand/point_cam has never had anything to say
  - a missing pyrealsense2 is reported and does not take the node down
  - the field-of-view check fires on a D405-class lens, because the servo
    gains were fitted against a much narrower one
"""
import math
import os
import sys
import types

import numpy as np

# The documented way to run this is `python3 src/mycobot_camera/test/...` from
# the repo root, which does not put the package on the path. Add it here so
# the command in CLAUDE.md works as written rather than only under a
# PYTHONPATH the reader has to know about.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_fake_rs(width=640, height=480, fx=380.0, fy=380.0, usb='3.2',
                 name='Intel RealSense D405'):
    """A pyrealsense2 stub covering exactly the surface camera_node uses."""
    rs = types.ModuleType('pyrealsense2')

    class Stream:
        color, depth = 'color', 'depth'

    class Format:
        bgr8, z16 = 'bgr8', 'z16'

    class CameraInfoEnum:
        name = 'name'
        usb_type_descriptor = 'usb'

    class Intrinsics:
        def __init__(self):
            self.fx, self.fy = fx, fy
            self.ppx, self.ppy = width / 2.0, height / 2.0
            self.coeffs = [0.1, -0.2, 0.0, 0.0, 0.0]

    class VideoStreamProfile:
        def get_intrinsics(self):
            return Intrinsics()

    class StreamProfile:
        def as_video_stream_profile(self):
            return VideoStreamProfile()

    class Device:
        def get_info(self, which):
            return {'name': name, 'usb': usb}[which]

    class PipelineProfile:
        def get_device(self):
            return Device()

        def get_stream(self, which):
            return StreamProfile()

    class Frame:
        def __init__(self, arr):
            self.arr = arr

        def __bool__(self):
            return True

        def get_data(self):
            return self.arr

    class Frames:
        def __init__(self, colour, depth):
            self._c, self._d = colour, depth

        def get_color_frame(self):
            return Frame(self._c)

        def get_depth_frame(self):
            return Frame(self._d)

    class Config:
        def __init__(self):
            self.streams = []
            self.serial = None

        def enable_device(self, s):
            self.serial = s

        def enable_stream(self, *a):
            self.streams.append(a)

    class Pipeline:
        started = 0

        def start(self, config):
            Pipeline.started += 1
            self.config = config
            return PipelineProfile()

        def wait_for_frames(self, timeout_ms=None):
            colour = np.full((height, width, 3), 7, dtype=np.uint8)
            depth = np.full((height, width), 1234, dtype=np.uint16)
            return Frames(colour, depth)

        def stop(self):
            pass

    rs.stream, rs.format, rs.camera_info = Stream, Format, CameraInfoEnum
    rs.pipeline, rs.config = Pipeline, Config
    rs.align = lambda which: types.SimpleNamespace(process=lambda f: f)
    return rs


class Recorder:
    """Collects log lines so assertions can be made about what was said."""

    def __init__(self):
        self.lines = []

    def _add(self, level, msg):
        self.lines.append(f'{level}: {msg}')

    def info(self, m):
        self._add('INFO', m)

    def warn(self, m):
        self._add('WARN', m)

    def error(self, m):
        self._add('ERROR', m)

    def has(self, needle):
        return any(needle.lower() in ln.lower() for ln in self.lines)


def build_node(params, logger):
    """A CameraNode with rclpy's machinery replaced, so no ROS graph is
    needed. Only the pieces _realsense_reader touches are provided."""
    try:
        from mycobot_camera.camera_node import CameraNode
    except ImportError as e:
        # camera_node imports rclpy and cv_bridge at module level, so this
        # needs ROS on the path even though no ROS graph is started.
        sys.exit(f'Cannot import camera_node: {e}\n\n'
                 'Source ROS first:\n'
                 '    source /opt/ros/humble/setup.bash')
    import threading

    node = CameraNode.__new__(CameraNode)
    node.get_parameter = lambda n: types.SimpleNamespace(value=params[n])
    node.get_logger = lambda: logger
    node._frame_lock = threading.Lock()
    node._stop_event = threading.Event()
    node._latest_frame = None
    node._latest_depth = None
    node._k = node._p = node._d = None
    return node


def run_reader_once(node):
    """Let the reader loop turn over a few frames, then stop it."""
    import threading
    t = threading.Thread(target=node._realsense_reader, daemon=True)
    t.start()
    for _ in range(200):
        if node._latest_frame is not None:
            break
        __import__('time').sleep(0.01)
    node._stop_event.set()
    t.join(timeout=3.0)


PARAMS = {
    'rs_width': 640, 'rs_height': 480, 'rs_fps': 30, 'rs_serial': '',
    'rs_depth': True, 'rs_align_depth_to_color': False,
}


def test_colour_reaches_latest_frame():
    sys.modules['pyrealsense2'] = make_fake_rs()
    log = Recorder()
    node = build_node(PARAMS, log)
    run_reader_once(node)
    assert node._latest_frame is not None, 'reader never produced a frame'
    assert node._latest_frame.shape == (480, 640, 3), node._latest_frame.shape
    assert node._latest_depth is not None, 'rs_depth was on but no depth'
    assert node._latest_depth.dtype == np.uint16
    assert log.has('RealSense open'), log.lines
    print('  colour and depth reach the publish path')


def test_intrinsics_become_a_projection_matrix():
    sys.modules['pyrealsense2'] = make_fake_rs(fx=380.0, fy=380.0)
    log = Recorder()
    node = build_node(PARAMS, log)
    run_reader_once(node)
    assert node._k is not None, 'intrinsics never captured'
    assert node._k[0] == 380.0 and node._k[4] == 380.0, node._k
    assert node._k[2] == 320.0 and node._k[5] == 240.0, node._k
    assert node._k[8] == 1.0, node._k
    assert len(node._p) == 12 and node._p[0] == 380.0, node._p
    assert node._d[:2] == [0.1, -0.2], node._d
    print(f'  k = fx {node._k[0]:.0f}, fy {node._k[4]:.0f}, '
          f'cx {node._k[2]:.0f}, cy {node._k[5]:.0f}')


def test_wide_lens_is_flagged():
    """A D405 is ~87 deg horizontal; the servo was tuned at ~25."""
    # fx for an 87-degree horizontal FOV at 640 wide.
    fx = (640 / 2.0) / math.tan(math.radians(87 / 2.0))
    sys.modules['pyrealsense2'] = make_fake_rs(fx=fx, fy=fx)
    log = Recorder()
    node = build_node(PARAMS, log)
    run_reader_once(node)
    assert log.has('wider than the one the servo was tuned'), log.lines
    assert log.has('skip_probe'), 'should name the fix'
    print(f'  fx={fx:.0f} (87 deg lens) -> warned, and named skip_probe')

    # And a narrow lens must NOT warn, or the warning is noise.
    narrow = (640 / 2.0) / math.tan(math.radians(50 / 2.0))
    sys.modules['pyrealsense2'] = make_fake_rs(fx=narrow, fy=narrow)
    log2 = Recorder()
    run_reader_once(build_node(PARAMS, log2))
    assert not log2.has('wider than the one the servo'), log2.lines
    print(f'  fx={narrow:.0f} (50 deg lens) -> silent, as it should be')


def test_usb2_is_flagged():
    sys.modules['pyrealsense2'] = make_fake_rs(usb='2.1')
    log = Recorder()
    run_reader_once(build_node(PARAMS, log))
    assert log.has('needs USB 3'), log.lines
    print('  USB 2 attachment warned about')


def test_missing_pyrealsense2_is_survivable():
    sys.modules.pop('pyrealsense2', None)
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == 'pyrealsense2':
            raise ImportError('No module named pyrealsense2')
        return real_import(name, *a, **k)

    builtins.__import__ = blocked
    try:
        log = Recorder()
        node = build_node(PARAMS, log)
        node._realsense_reader()          # must return, not raise
    finally:
        builtins.__import__ = real_import
    assert log.has('pip install pyrealsense2'), log.lines
    assert log.has('jetson'), 'should warn about ARM64 wheels'
    assert node._latest_frame is None
    print('  missing pyrealsense2 reported without taking the node down')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    print(f'{len(tests)} tests, no RealSense required\n')
    for fn in tests:
        print(f'{fn.__name__}:')
        fn()
    print(f'\nAll {len(tests)} passed.')
