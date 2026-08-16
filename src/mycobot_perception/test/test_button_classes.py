#!/usr/bin/env python3
"""Pin the button vocabulary. Runs without ROS.

    python3 src/mycobot_perception/test/test_button_classes.py

Class order is a contract (CLAUDE.md, failure #11): ids are stored as
integers, so reordering DETECT_CLASSES silently relabels every box in every
dataset and every trained checkpoint, with no error raised anywhere. A
detector trained before the change and a data.yaml written after it disagree
completely and both look healthy.

So the expected order is written out longhand below. If this test fails
because you meant to change the list, the dataset must be rebuilt and both
models retrained -- that is the test doing its job, not an obstacle.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mycobot_perception.button_classes import (  # noqa: E402
    DETECT_CLASSES, FORBIDDEN, PRESSABLE, UNREADABLE,
    to_detect_class, to_reader_label)

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


# --- the contract ----------------------------------------------------------
check(DETECT_CLASSES == ['up', 'down', 'floor', 'open', 'close', 'help',
                         'stop', 'keyhole', 'other'],
      f'DETECT_CLASSES changed: {DETECT_CLASSES}')

# The two top-priority classes are ids 0 and 1, which the builder relies on
# by NUMBER when it decides what to oversample.
check(DETECT_CLASSES.index('up') == 0, 'up is no longer id 0')
check(DETECT_CLASSES.index('down') == 1, 'down is no longer id 1')

# --- priority 1: the hall call --------------------------------------------
# These are the variants that were being silently dropped before 2026-08-16,
# which is 44 instances of the highest-priority classes.
for name in ('up', 'UP', 'U', 'Up'):
    check(to_detect_class(name) == 'up', f'{name!r} should map to up')
for name in ('down', 'D', 'DN', 'Down'):
    check(to_detect_class(name) == 'down', f'{name!r} should map to down')

# `U` and `D` must NOT fall through to the floor branch -- they match the
# floor-legend regex, so only the alias table's precedence keeps a hall call
# from being read as a storey.
check(to_reader_label('U') is None, 'U leaked into the reader as a floor')
check(to_reader_label('D') is None, 'D leaked into the reader as a floor')

# --- priority 2: floors ----------------------------------------------------
for name in ('1', '12', 'button-7', 'floor-3', 'B', 'B1', 'G', 'LG', '-1'):
    check(to_detect_class(name) == 'floor', f'{name!r} should map to floor')
check(to_reader_label('button-7') == '7', 'the button- prefix must be stripped')
check(to_reader_label('b1') == 'B1', 'reader labels are upper-cased')

# Spelled-out storeys canonicalise onto the abbreviation. ENTC writes
# `floor-ground` where Sun Moon writes `G`; if these produced a separate
# `GROUND` legend, one storey's examples would be split across two labels and
# the reader would be permanently unsure between them.
check(to_detect_class('floor-ground') == 'floor', 'floor-ground is a floor')
check(to_reader_label('floor-ground') == 'G', 'ground must canonicalise to G')
check(to_reader_label('lobby') == 'L', 'lobby must canonicalise to L')
check(to_reader_label('button-g') == 'G', 'button-g is the same storey as G')

# ENTC spells the hall call both ways; both must reach the same class.
check(to_detect_class('button-up') == 'up', 'button-up should map to up')
check(to_detect_class('button-down') == 'down', 'button-down should map to down')
# The door STATE, which is not a button at all.
check(to_detect_class('closed-door') is None, 'closed-door is not a button')

# `B` is a basement, never a bell -- the alias table lists `bell` explicitly
# so that this stays true.
check(to_detect_class('B') == 'floor', 'B must be the basement, not a bell')
check(to_detect_class('bell') == 'help', 'bell is a help button')

# --- priority 3: help ------------------------------------------------------
for name in ('alarm', 'call', 'emergency', 'intercom', 'bell'):
    check(to_detect_class(name) == 'help', f'{name!r} should map to help')

# --- never press -----------------------------------------------------------
check(to_detect_class('stop') == 'stop', 'stop must stay its own class')
check('stop' in FORBIDDEN, 'stop must never be pressable')
check('keyhole' in FORBIDDEN and 'other' in FORBIDDEN,
      'keyhole and other must never be pressable')
check(not set(FORBIDDEN) & set(PRESSABLE),
      'a class cannot be both pressable and forbidden')
check(set(PRESSABLE) | set(FORBIDDEN) == set(DETECT_CLASSES),
      'every detect class must be either pressable or forbidden')

# --- unreadable ------------------------------------------------------------
for name in ('empty', 'blur', 'unknown'):
    check(to_detect_class(name) == 'other', f'{name!r} should map to other')
check(to_reader_label('empty') is None,
      'other-class names are not floor legends')

# --- things that are not buttons ------------------------------------------
# Printed legends beside a button, indicator lamps and speakers. If one of
# these starts mapping to a class, the detector learns to fire on the label
# next to the button instead of the button.
for name in ('text', 'text_OPEN', 'text_DOWN', 'led', 'indicator', 'speaker',
             'light', 'fan', 'updown'):
    check(to_detect_class(name) is None, f'{name!r} should be dropped')

# `text_DOWN` is the trap: it contains the name of a top-priority class and
# is a printed word, not a button.
check(to_detect_class('text_DOWN') is None,
      'text_DOWN is a printed legend, not the down button')

check(UNREADABLE == 'unreadable', 'the node matches this name literally')

if failures:
    print(f'FAIL ({len(failures)})')
    for f in failures:
        print('  -', f)
    raise SystemExit(1)
print(f'ok -- {len(DETECT_CLASSES)} detect classes, order pinned')
