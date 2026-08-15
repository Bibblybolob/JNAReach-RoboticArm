"""
Bridges vision_msgs/Detection2DArray + a depth image into the PointStamped
contract the visual servo expects.

    /button/point_px    geometry_msgs/PointStamped
                         x, y = detection bbox centre, in FULL-frame pixels
                         z    = depth at that point, in millimetres (0 if
                                unavailable)

    /button/point_base  geometry_msgs/PointStamped
                         the same detection in the ARM's base frame, METRES.
                         Silent unless an eye-to-hand calibration exists.

Pixels steer a servo; they cannot be reached for. The base-frame topic is what
lets the arm be commanded AT the button rather than merely turned toward it,
and it needs three things the pixel topic does not: the camera intrinsics
(source:=realsense supplies them), the eye-to-hand calibration, and the
MEASURED joint1 angle -- which on this mount is the one joint that moves the
camera, so the transform depends on it.

Every one of those degrades quietly and says why once. A missing calibration
is the normal state until someone runs it, and it must never take
/button/point_px down with it: the servo depends on that topic and needs none
of this.

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

import json
import math
import os

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from rcl_interfaces.msg import SetParametersResult

from cv_bridge import CvBridge

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from vision_msgs.msg import Detection2DArray

from mycobot_perception.target_in_base import TargetError, target_in_base


class DetectionBridgeNode(Node):

    def __init__(self) -> None:
        super().__init__('detection_bridge_node')

        self.declare_parameter('detection_topic', '/perception/button_detections')
        self.declare_parameter('depth_topic', '/camera/depth_raw')
        self.declare_parameter('output_topic', '/button/point_px')
        self.declare_parameter('target_label', '')

        # --- Base-frame output ---
        #
        # The same detection expressed where the ARM can act on it. Off until
        # a calibration exists, which is the honest default: without one there
        # is no transform from the camera to the base, and a guess would be a
        # confident wrong point that something downstream would drive to.
        self.declare_parameter('base_output_topic', '/button/point_base')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        # Where the calibration scripts actually WRITE. This used to default to
        # ~/mycobot_project/calibration/, a directory nothing creates and
        # nothing writes to, so a perfectly good calibration would still leave
        # this topic silent -- and the message says "no calibration", which
        # reads as "it was never run" rather than "it is not here".
        # calibrate_hand_eye.py and verify_calibration.py both use ~/hand_eye.
        self.declare_parameter('calibration_path',
                               '~/hand_eye/eye_to_hand.json')
        self.declare_parameter('base_frame', 'base_link')
        # Which joint pans the camera. joint1 for the first-arm-piece mount;
        # a parameter rather than a constant so remounting is a config change.
        self.declare_parameter('pan_joint', 'joint1')

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
        self.base_output_topic = self.get_parameter('base_output_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.calibration_path = self.get_parameter('calibration_path').value
        self.base_frame = self.get_parameter('base_frame').value
        self.pan_joint = self.get_parameter('pan_joint').value

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
        self._base_pub = self.create_publisher(
            PointStamped, self.base_output_topic, 10)

        # joint1 is the only joint that moves this camera, so it is the only
        # one the base-frame transform needs -- but it must be the MEASURED
        # angle, not the commanded one, or the point is wrong by however far
        # the arm is trailing.
        self._joint1_deg = None
        self._joint_sub = self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10)
        self._cam_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, 10)
        self._intrinsics = None
        self._published_base_any = False
        self._warned = set()
        self._cam_to_base = self._load_calibration()

        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f'detection_bridge_node up: {self.detection_topic} + '
            f'{self.depth_topic} -> {self.output_topic}, '
            f"target_label={self.target_label!r}")

    def _load_calibration(self):
        """The eye-to-hand transform, or None with a reason said once.

        Absent is the normal state until someone runs the calibration, and it
        must degrade quietly: /button/point_px needs none of this and the
        servo depends on it.
        """
        path = os.path.expanduser(self.calibration_path)
        if not os.path.isfile(path):
            self.get_logger().info(
                f'no eye-to-hand calibration at {path}, so '
                f'{self.base_output_topic} stays silent. Pixels still publish. '
                'Run scripts/calibrate_hand_eye.py --eye-to-hand to enable it.')
            return None
        try:
            with open(path) as f:
                calib = json.load(f)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'could not read {path}: {e}')
            return None
        if 'camera_to_base' not in calib:
            # The likeliest wrong file: an eye-IN-hand result, which describes
            # a camera that is no longer on the flange.
            self.get_logger().error(
                f'{path} has no camera_to_base. That looks like an eye-in-hand '
                'calibration, which describes a camera mounted on the flange '
                '-- this one is on the first arm piece. Re-run with '
                '--eye-to-hand.')
            return None
        self.get_logger().info(
            f'eye-to-hand calibration loaded from {path}, solved at joint1='
            f'{calib.get("joint1_deg", 0.0):.1f}deg')
        return calib

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self._intrinsics is not None:
            return
        k = list(msg.k)
        if len(k) < 9 or k[0] == 0.0:
            # CameraInfo went out with an empty k for most of this project's
            # life. Naming the fix beats a stream of transform failures.
            self._warn_once(
                'intrinsics',
                'CameraInfo carries no intrinsics (k is empty or zero), so no '
                'base-frame point can be computed. Run with source:=realsense, '
                'which supplies the factory calibration.')
            return
        self._intrinsics = {'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]}
        self.get_logger().info(
            f'intrinsics: fx={k[0]:.1f} fy={k[4]:.1f} '
            f'cx={k[2]:.1f} cy={k[5]:.1f}')

    def _on_joint_states(self, msg: JointState) -> None:
        try:
            i = list(msg.name).index(self.pan_joint)
        except ValueError:
            self._warn_once(
                'panjoint',
                f'{self.pan_joint!r} is not in /joint_states (saw '
                f'{list(msg.name)}), so the camera pan angle is unknown.')
            return
        if i < len(msg.position):
            self._joint1_deg = math.degrees(msg.position[i])

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

        self._publish_base_point(msg.header, cx, cy, depth_mm)

    def _publish_base_point(self, header, cx, cy, depth_mm) -> None:
        """The same detection as a point in the ARM's frame, when we can.

        Pixels steer a servo; they cannot be reached for. This is the output
        that lets the arm be commanded AT the button rather than merely turned
        toward it, and it needs three things the pixel topic does not: the
        camera's intrinsics, the eye-to-hand calibration, and joint1 -- which
        on this mount is the one joint that moves the camera.

        Silent when any of them is missing, and says why ONCE. A missing
        calibration must not stop /button/point_px, which the servo depends
        on and which needs none of this.
        """
        if self._cam_to_base is None or self._intrinsics is None:
            return
        if self._joint1_deg is None:
            self._warn_once('joint1',
                            'No /joint_states yet, so the camera\'s pan angle '
                            'is unknown and the base-frame point cannot be '
                            'computed. Is the driver running?')
            return

        try:
            x, y, z = target_in_base(
                cx, cy, depth_mm, self._intrinsics,
                self._cam_to_base, self._joint1_deg)
        except TargetError as e:
            # Not a warning worth repeating every frame: out-of-range depth is
            # the normal state of affairs while the panel is far away.
            self.get_logger().debug(f'no base-frame point: {e}')
            return

        out = PointStamped()
        out.header = header
        out.header.frame_id = self.base_frame
        out.point.x, out.point.y, out.point.z = float(x), float(y), float(z)
        self._base_pub.publish(out)

        if not self._published_base_any:
            self._published_base_any = True
            self.get_logger().info(
                f'first base-frame target: ({x*1000:.0f}, {y*1000:.0f}, '
                f'{z*1000:.0f})mm in {self.base_frame}, with joint1 at '
                f'{self._joint1_deg:.1f}deg -- publishing to '
                f'{self.base_output_topic}')

    def _warn_once(self, key: str, text: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        self.get_logger().warn(text)

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
