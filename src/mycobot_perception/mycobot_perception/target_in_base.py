"""Turn a detected pixel plus a depth reading into a point the arm can reach.

This is the step the whole button-pressing plan rests on: the servo steers in
IMAGE space -- "the button is left of centre" -- which centres a target and
nothing more. Pressing needs "the button is 180mm in front of the base, 40mm
left", and that is this module.

    pixel (u,v) + depth  --K-->  camera frame  --calibration-->  base frame

Deliberately free of ROS and of pyrealsense2, so it can be tested without a
camera, a workspace or an arm: see test/test_target_in_base.py.

The mount this is written for
-----------------------------
The camera is on the FIRST ARM PIECE, so joint1 -- and only joint1 -- moves
it. That has one consequence the maths must carry: the calibration is solved
at one joint1 angle, and at any other pan the camera has rotated about the
base's own z axis. So

    T_cam2base(q1) = Rz(q1 - q1_solved) @ T_cam2base(q1_solved)

Get that composition wrong and every reading is correct at exactly one pan
angle and quietly wrong everywhere else -- which is the sort of error that
looks like a calibration that "drifts".

Units, because this project has paid for them twice
---------------------------------------------------
**Depth in, millimetres.** `camera_node` already converts using the sensor's
real depth scale, so `/camera/depth_raw` is in mm even though a D405 counts in
tenths of one. Metres out, because that is what the arm's FK and IK use.
Passing metres in by mistake does not silently scale the answer by 1000: it
lands outside the D405's valid range and is refused.
"""
from __future__ import annotations

import math

# The D405 sees nothing useful outside this band. Its stereo baseline puts the
# near limit around 7cm and the far usable limit around 50cm -- exactly right
# for pressing a button and wrong for following something across a room. A
# reading outside it is not a distant target, it is noise, and returning a
# point for it would put the arm somewhere arbitrary.
MIN_DEPTH_MM = 70.0
MAX_DEPTH_MM = 500.0


class TargetError(ValueError):
    """A reading that must not be turned into a point."""


def pixel_to_camera(u, v, depth_mm, fx, fy, cx, cy):
    """(u, v, depth_mm) -> (x, y, z) metres in the camera frame.

    Plain pinhole. The D405's colour comes from its stereo imagers and is
    delivered rectified, so there is no distortion term here -- if that ever
    stops being true the intrinsics will carry non-zero coefficients and this
    needs cv2.undistortPoints in front of it.

    Camera frame is OpenCV's: +x right across the image, +y down it, +z out
    along the optical axis.
    """
    if not (MIN_DEPTH_MM <= depth_mm <= MAX_DEPTH_MM):
        raise TargetError(
            f'depth {depth_mm:.1f}mm is outside the D405\'s usable '
            f'{MIN_DEPTH_MM:.0f}-{MAX_DEPTH_MM:.0f}mm range. A reading out '
            'here is noise, not a distant target -- and if this looks like a '
            'factor of 1000, the caller passed metres where mm were wanted.')
    z = depth_mm / 1000.0
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


def _rz(theta_rad):
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _apply(T, p):
    return tuple(
        sum(T[r][k] * p[k] for k in range(3)) + T[r][3] for r in range(3))


def camera_to_base(p_cam, cam_to_base, joint1_deg, solved_joint1_deg=0.0):
    """A point in camera coords -> the arm's base frame.

    `cam_to_base` is the 4x4 from `calibrate_hand_eye.py --eye-to-hand`, valid
    at `solved_joint1_deg`. joint1 rotates the camera about the base z axis,
    so any other pan composes as Rz(delta) in front of it.
    """
    p = _apply(cam_to_base, p_cam)
    d = math.radians(joint1_deg - solved_joint1_deg)
    if d == 0.0:
        return p
    R = _rz(d)
    return tuple(sum(R[r][k] * p[k] for k in range(3)) for r in range(3))


def target_in_base(u, v, depth_mm, intrinsics, calibration, joint1_deg):
    """The whole chain: pixel + depth -> (x, y, z) metres in the base frame.

    `intrinsics` is a mapping with fx, fy, cx (or ppx), cy (or ppy).
    `calibration` is the dict written by `calibrate_hand_eye.py --eye-to-hand`,
    carrying `camera_to_base` and the `joint1_deg` it was solved at.

    Raises TargetError rather than returning a plausible-looking wrong point.
    That is the deliberate choice here: everything downstream drives the arm,
    and a bad point is a collision, while a refusal is a retry.
    """
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics.get('cx', intrinsics.get('ppx'))
    cy = intrinsics.get('cy', intrinsics.get('ppy'))
    if None in (fx, fy, cx, cy) or fx == 0 or fy == 0:
        raise TargetError(
            'intrinsics are missing or zero. CameraInfo went out with empty '
            'k for most of this project\'s life -- run with source:=realsense, '
            'which supplies the factory calibration.')

    T = calibration.get('camera_to_base')
    if T is None:
        raise TargetError(
            'no camera_to_base in the calibration. This needs the EYE-TO-HAND '
            'solve (calibrate_hand_eye.py --eye-to-hand); a camera_to_flange '
            'transform describes a camera that is no longer on the flange.')

    p_cam = pixel_to_camera(u, v, depth_mm, fx, fy, cx, cy)
    return camera_to_base(p_cam, T, joint1_deg,
                          calibration.get('joint1_deg', 0.0))
