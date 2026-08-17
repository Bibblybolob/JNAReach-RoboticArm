#!/usr/bin/env python3
"""What the model detector makes ASKABLE. Runs with no camera, arm or GPU.

    python3 src/mycobot_perception/test/test_press_targets.py

The models are stubbed. What is under test is the labelling policy in
press_button.find_with_model -- which detections become press targets, which
are deliberately withheld, and which are present but unaskable. That policy is
the safety boundary between "the arm can see it" and "the arm may press it",
and it is small enough to get wrong silently.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mycobot_perception'))
sys.path.insert(0, os.path.join(_ROOT, 'src', 'mycobot_driver'))

from mycobot_perception.button_classes import DETECT_CLASSES  # noqa: E402
import press_button as pb  # noqa: E402

PASS = FAIL = 0


def check(msg, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'PASS  {msg}')
    else:
        FAIL += 1
        print(f'FAIL  {msg}')


def _wrap(arr):
    return types.SimpleNamespace(
        cpu=lambda: types.SimpleNamespace(numpy=lambda: np.array(arr)))


class _Boxes:
    def __init__(self, xy, cl):
        self.xyxy = _wrap(xy)
        self.cls = _wrap(cl)

    def __len__(self):
        return len(self.cls.cpu().numpy())


class _Det:
    """Stage A: one of each interesting kind."""

    KINDS = ('up', 'down', 'floor', 'floor', 'keyhole', 'stop', 'help')

    def predict(self, img, **kw):
        xy = [[20 * i, 10, 20 * i + 18, 30] for i in range(len(self.KINDS))]
        cl = [DETECT_CLASSES.index(k) for k in self.KINDS]
        return [types.SimpleNamespace(boxes=_Boxes(xy, cl))]


class _Reader:
    """Stage B: first floor reads as `5`, second is unreadable."""

    names = {0: '5', 1: 'unreadable'}

    def predict(self, crops, **kw):
        out = []
        for i in range(len(crops)):
            out.append(types.SimpleNamespace(
                probs=types.SimpleNamespace(
                    top1=0 if i == 0 else 1, top1conf=0.99)))
        return out


def run(read_min=0.95, reader=None):
    pb._MODELS.clear()
    pb._MODELS.update({'det': _Det(),
                       'rdr': _Reader() if reader is None else reader})
    args = types.SimpleNamespace(conf=0.5, device='cpu', crop_pad=0.12,
                                 read_min=read_min, detect_weights='',
                                 read_weights='')
    return pb.find_with_model(np.zeros((240, 320, 3), np.uint8), args)


found = run()

# --- priority 1: the hall call is askable at all -------------------------
# It was not, before this: `button` took an int, so `up` could not be named.
check('up is a press target', 'up' in found)
check('down is a press target', 'down' in found)

# --- priority 2: floors are asked for by their LEGEND ---------------------
check('a confidently read floor is askable by its legend', '5' in found)
check('the raw class `floor` is NOT askable when a legend was read',
      'floor' not in found)

# --- the refusal that matters --------------------------------------------
# A button whose legend was not read is real, and located, and must not be
# askable -- otherwise the arm presses it believing it is a floor it is not.
unread = [k for k in found if k.startswith('floor?')]
check('an unread floor is present but unaskable', len(unread) == 1)
check('an unread floor keeps a position for the panel plane fit',
      all(len(found[k]) == 4 for k in unread))

# --- never-press classes are withheld, not merely deprioritised -----------
check('keyhole is never a target', 'keyhole' not in found)
check('stop is never a target', 'stop' not in found)
check('help IS a target -- it is priority 3, not forbidden', 'help' in found)

# --- shape contract with keypad_finder ------------------------------------
# The rest of the chain reads (x, y, half_w, half_h) and does not care which
# detector produced it. A shape change here surfaces as a wrong press.
x, y, a, b = found['up']
check('labels carry (x, y, half_w, half_h) like keypad_finder',
      abs(a - 9.0) < 1e-6 and abs(b - 10.0) < 1e-6)
check('the centre is the box centre', abs(y - 20.0) < 1e-6)

# --- a stricter threshold declines more, never guesses more ---------------
strict = run(read_min=1.01)          # nothing can clear this
check('an unreachable threshold makes every floor unaskable',
      '5' not in strict and len([k for k in strict if k.startswith('floor?')]) == 2)
check('a stricter threshold does not affect up/down',
      'up' in strict and 'down' in strict)

# --- no reader at all ------------------------------------------------------
# The reader is optional by design: losing it costs the numbers, not the panel.
pb._MODELS.clear()
pb._MODELS.update({'det': _Det(), 'rdr': None})
args = types.SimpleNamespace(conf=0.5, device='cpu', crop_pad=0.12,
                             read_min=0.95, detect_weights='', read_weights='')
blind = pb.find_with_model(np.zeros((240, 320, 3), np.uint8), args)
check('with no reader, up and down still work',
      'up' in blind and 'down' in blind)
check('with no reader, floors are present but unnumbered',
      len([k for k in blind if k.startswith('floor?')]) == 2)

print(f'\n{PASS} passed' + (f', {FAIL} FAILED' if FAIL else ''))
sys.exit(1 if FAIL else 0)
