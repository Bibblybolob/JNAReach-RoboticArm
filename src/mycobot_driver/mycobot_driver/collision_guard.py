"""Refuse poses that would drive the arm into the desk, its base, or itself.

Why this is needed
------------------
The visual servo jogs joints straight from image error, bypassing MoveIt --
so nothing between "the detector saw something low in the frame" and the arm
moving there checks whether the destination is occupied. The joint limits in
the driver only bound each joint separately, and every self-collision on this
arm is reachable well inside them: joint2 and joint3 are individually legal at
angles that fold the forearm through the base.

The realistic collisions on a desk-mounted 280, in the order they will happen:

  1. **Into the desk.** The arm is bolted to a surface at z=0 and the search
     sweep tilts joint5 through +/-90deg. An elbow reaches the desk before the
     tool does, which is why every joint origin is checked and not just the
     flange.
  2. **Into its own base.** Folding joint3 hard while joint2 is low swings the
     forearm back through the column.
  3. **Into the panel it is approaching.** Handled by target_depth_mm in the
     servo, not here -- this module knows nothing about what the camera sees.

Approach
--------
Forward kinematics for a CANDIDATE pose, then test every joint origin and the
tool tip against a small set of keep-out volumes. Cheap: six 4x4 multiplies,
run before a jog is sent rather than as a background monitor, so a bad pose is
never commanded rather than being noticed afterwards.

Deliberately geometric and conservative. It models the arm as points at the
joints plus samples along each link, not as meshes -- a real mesh check needs
MoveIt and the whole point here is that the servo path does not use MoveIt.
Points-plus-samples with a margin catches the collisions that actually happen
and costs microseconds.

No ROS import, so it is testable without a workspace: see
test/test_collision_guard.py.
"""
from __future__ import annotations

import math

# (xyz, rpy) per joint, straight out of the URDF -- same chain ik_demo.py uses.
CHAIN = [
    ((0.0, 0.0, 0.13956), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, -0.001), (0.0, 1.5708, -1.5708)),
    ((-0.1104, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.096, 0.0, 0.06462), (0.0, 0.0, -1.5708)),
    ((0.0, -0.07318, -0.001), (1.5708, -1.5708, 0.0)),
    ((0.0, 0.0456, 0.0), (-1.5708, 0.0, 0.0)),
]

# The D405 and its mount stick out past the flange. Guarding the flange alone
# would let the camera hit things first -- it is the part furthest out and the
# part worth least to replace.
TOOL_OFFSET_M = 0.055


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def _rpy(r, p, y):
    """URDF rpy is fixed-axis: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]]


def joint_points(angles_deg, tool_offset_m: float = TOOL_OFFSET_M):
    """(x, y, z) of every joint origin, plus the tool tip, in metres.

    Base frame: origin at the mounting face, +z up. So z<0 is inside whatever
    the arm is bolted to.
    """
    q = [math.radians(a) for a in angles_deg]
    T = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    pts = [(0.0, 0.0, 0.0)]
    for i in range(6):
        xyz, rpy_ = CHAIN[i]
        R = _rpy(*rpy_)
        c, s = math.cos(q[i]), math.sin(q[i])
        Rz = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        M = [[sum(R[r][k] * Rz[k][cc] for k in range(3)) for cc in range(3)]
             + [xyz[r]] for r in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
        T = _matmul(T, M)
        pts.append((T[0][3], T[1][3], T[2][3]))

    # Tool tip: straight out along the flange's own z.
    if tool_offset_m:
        pts.append((T[0][3] + T[0][2] * tool_offset_m,
                    T[1][3] + T[1][2] * tool_offset_m,
                    T[2][3] + T[2][2] * tool_offset_m))
    return pts


def _sampled(pts, per_link: int = 4, skip_first_links: int = 1):
    """Joint points plus samples along each link.

    A link can pass through the desk with both of its ENDS above it -- the
    forearm swinging down and back up is exactly that. Checking only the
    joints misses it, and it is the case that actually breaks a camera.

    skip_first_links drops the base column itself. The mount sits AT the
    origin and the first link runs straight up through the base, so both are
    permanently inside every keep-out volume by construction -- checking them
    rejects every pose including home, which is exactly what the first version
    of this did.
    """
    out = list(pts[skip_first_links:])
    for a, b in zip(pts[skip_first_links:], pts[skip_first_links + 1:]):
        for k in range(1, per_link):
            f = k / per_link
            out.append((a[0] + (b[0] - a[0]) * f,
                        a[1] + (b[1] - a[1]) * f,
                        a[2] + (b[2] - a[2]) * f))
    return out


class CollisionGuard:
    """Geometric keep-out test for a candidate joint pose.

    Every distance is metres in the arm's base frame.
    """

    def __init__(self,
                 min_z: float = 0.02,
                 base_radius: float = 0.06,
                 base_height: float = 0.12,
                 max_reach: float = 0.32,
                 tool_offset_m: float = TOOL_OFFSET_M,
                 enabled: bool = True):
        # Clearance above the mounting surface. Not 0: the arm is bolted to a
        # desk that is not perfectly flat, the FK ignores link thickness, and
        # servo error measured 0.8deg -- which at full reach is several
        # millimetres. 2cm buys all of that back.
        self.min_z = min_z
        # The base column. Nothing may enter this cylinder below base_height.
        self.base_radius = base_radius
        self.base_height = base_height
        # Beyond this the arm is stretched out further than it can hold
        # steadily -- the pose that browned out the controller on 2026-08-12
        # was exactly this shape. Geometry cannot see torque, but reach is a
        # decent proxy for it.
        self.max_reach = max_reach
        self.tool_offset_m = tool_offset_m
        self.enabled = enabled

    def check(self, angles_deg):
        """(ok, reason). reason is '' when the pose is fine."""
        if not self.enabled:
            return True, ''
        try:
            pts = joint_points(angles_deg, self.tool_offset_m)
        except Exception as e:  # noqa: BLE001
            # Never let a maths failure become a silent permit.
            return False, f'kinematics failed: {e}'

        for i, (x, y, z) in enumerate(_sampled(pts)):
            where = f'point {i}'
            if z < self.min_z:
                return False, (f'{where} would be {z * 1000:.0f}mm above the '
                               f'mounting surface, below the {self.min_z * 1000:.0f}mm '
                               'floor -- that is into the desk')
            r = math.hypot(x, y)
            if r < self.base_radius and z < self.base_height:
                return False, (f'{where} would be {r * 1000:.0f}mm from the '
                               f'column at {z * 1000:.0f}mm high -- that is '
                               'inside the base')
            if r > self.max_reach:
                return False, (f'{where} would be {r * 1000:.0f}mm out, past '
                               f'the {self.max_reach * 1000:.0f}mm reach limit '
                               '-- the arm does not hold that pose steadily')
        return True, ''

    def worst_margin(self, angles_deg) -> float:
        """Smallest clearance to any keep-out, in metres. Negative = inside.

        For logging how close a pose came, which is what tells you whether a
        limit is set sensibly or is about to start refusing useful poses.
        """
        pts = _sampled(joint_points(angles_deg, self.tool_offset_m))
        margins = []
        for x, y, z in pts:
            margins.append(z - self.min_z)
            r = math.hypot(x, y)
            if z < self.base_height:
                margins.append(r - self.base_radius)
            margins.append(self.max_reach - r)
        return min(margins) if margins else 0.0
