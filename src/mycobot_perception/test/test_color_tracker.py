"""Checks on the colour blob detector, no camera and no ROS required.

    python3 src/mycobot_perception/test/test_color_tracker.py

find_blob feeds the servo directly, so what matters is that the coordinates
it returns are in FULL-frame pixels despite detection running downscaled --
an off-by-a-scale-factor here would look exactly like a badly tuned gain, and
would be tuned around for a session before anyone checked.
"""

import ast
import os
import sys

import cv2
import numpy as np

SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'mycobot_perception', 'color_tracker_node.py')

W, H = 640, 480


def _load():
    """Pull find_blob out by source, as the other test files here do, so this
    runs without rclpy."""
    tree = ast.parse(open(SRC).read())
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == 'ColorTrackerNode')
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == 'find_blob')
    presets = next(n for n in tree.body
                   if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], 'id', '') == 'PRESETS')
    ns = {'cv2': cv2, 'np': np}
    exec(compile(ast.Module(body=[presets, fn], type_ignores=[]), SRC, 'exec'),
         ns)
    return ns['find_blob'], ns['PRESETS']


find_blob, PRESETS = _load()


class Tracker:
    """Minimal stand-in carrying only what find_blob touches."""

    def __init__(self, color='red', proc_w=160, min_area=10.0):
        self._proc_w = proc_w
        self._min_area = min_area
        self._lo = np.array(PRESETS[color]['min'], dtype=np.uint8)
        self._hi = np.array(PRESETS[color]['max'], dtype=np.uint8)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


Tracker.find_blob = find_blob


def scene(blobs, bg=(40, 40, 40)):
    """BGR frame with filled circles: blobs = [(x, y, r, (b,g,r)), ...]"""
    img = np.full((H, W, 3), bg, dtype=np.uint8)
    for x, y, r, colour in blobs:
        cv2.circle(img, (x, y), r, colour, -1)
    return img


RED = (0, 0, 255)
GREEN = (0, 200, 0)
BLUE = (255, 0, 0)


def test_finds_a_red_blob():
    got = Tracker().find_blob(scene([(400, 300, 40, RED)]))
    assert got is not None, 'missed an obvious red blob'
    cx, cy, dia = got
    assert abs(cx - 400) < 12, f'x off: {cx}'
    assert abs(cy - 300) < 12, f'y off: {cy}'


def test_coordinates_are_full_frame_not_downscaled():
    """The whole point of the scale factor. A blob on the right of the frame
    must report near 640, not near the 160px the detector actually ran at."""
    cx, cy, _ = Tracker().find_blob(scene([(560, 240, 35, RED)]))
    assert cx > W / 2, f'{cx} looks like downscaled coordinates'
    assert abs(cx - 560) < 15


def test_diameter_is_full_frame_too():
    """z carries diameter and the approach axis divides it by frame width, so
    a scale error here silently changes what 'close enough' means."""
    _, _, dia = Tracker().find_blob(scene([(320, 240, 50, RED)]))
    assert abs(dia - 100) < 20, f'diameter {dia}, expected ~100'


def test_ignores_other_colours():
    for colour in (GREEN, BLUE):
        assert Tracker().find_blob(scene([(320, 240, 45, colour)])) is None


def test_picks_the_largest_when_several_are_present():
    got = Tracker().find_blob(scene([
        (120, 120, 12, RED),
        (500, 350, 45, RED),
        (300, 100, 15, RED),
    ]))
    cx, cy, _ = got
    assert abs(cx - 500) < 20 and abs(cy - 350) < 20, \
        f'followed a smaller blob: {cx},{cy}'


def test_nothing_there_is_none_not_a_guess():
    assert Tracker().find_blob(scene([])) is None


def test_speckle_is_rejected():
    """Single stray pixels must not become a target; the arm would chase
    sensor noise across the frame."""
    img = scene([])
    rng = np.random.default_rng(0)
    for _ in range(60):
        x, y = rng.integers(0, W), rng.integers(0, H)
        img[y, x] = RED
    assert Tracker().find_blob(img) is None


def test_a_distant_small_blob_is_still_found():
    """min_area must reject noise without rejecting a real target at range."""
    assert Tracker().find_blob(scene([(320, 240, 14, RED)])) is not None


def test_detection_is_cheap():
    """It has to be: the point of a colour target over MediaPipe is dead time
    removed, and dead time is what limits this loop."""
    import time
    t = Tracker()
    img = scene([(400, 300, 40, RED)])
    t.find_blob(img)                      # warm up
    t0 = time.perf_counter()
    for _ in range(50):
        t.find_blob(img)
    per_ms = (time.perf_counter() - t0) / 50 * 1000
    print(f'      ({per_ms:.2f} ms/frame)')
    assert per_ms < 5.0, f'{per_ms:.1f}ms is too slow to be worth it'


if __name__ == '__main__':
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS  {name}')
            passed += 1
    print(f'\n{passed} passed')
