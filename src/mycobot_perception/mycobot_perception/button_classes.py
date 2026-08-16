"""The button vocabulary, in one place, because class order is a contract.

Every `data.yaml`, every training run and the detector node itself read their
class list from here. It used to be written out by hand in three files
(`~/panel_dataset/data.yaml`, `~/datasets/public_panel/data.yaml` and TARGET
in `scripts/remap_dataset.py`) which have to agree exactly -- a reordering is
failure #11 in CLAUDE.md: labels are stored as integers, so a different order
silently relabels every box and nothing anywhere raises.


Why two stages
--------------
The old design gave the detector one class per floor and stopped at
`button-10`. Against a real elevator that is unworkable in both directions: a
building with a 14th floor gets it called background, while the rare classes
at the top of the range never had enough examples to learn. Measured over the
Sun Moon set (22,843 instances, 368 classes): the counts fall off a cliff
after floor ~12 and there is a long tail of `B3`, `LG2`, `P4A` and the like
with single-digit support.

So the work is split the way the CUHK paper splits it:

  Stage A (detect)  WHERE is a button, and WHAT KIND -- 9 classes, every one
                    with hundreds to thousands of examples.
  Stage B (read)    Given the crop, WHICH FLOOR -- a classifier, which is far
                    more sample-efficient than detection because the crop is
                    already centred, scaled and 128px of pure signal.

That also puts the priority order the right way up. `up` and `down` are
top-priority and are the two classes a per-floor detector was spending none of
its capacity on; here they are 2 of 9.


Why these nine
--------------
Read off the measured per-class counts, not chosen in advance:

  up       414   the hall call. Includes the `U` legend variant -- inspected
                 2026-08-16 and they are unambiguously up-call buttons with a
                 letter instead of an arrow, so dropping them was throwing
                 away 21 examples of the highest-priority class.
  down     359   likewise, with `D` and `DN`.
  floor  ~12000  every numbered or lettered floor button, merged. Stage B
                 recovers which one.
  open    1072   door open.
  close    971   door close.
  help    1081   alarm + call + bell + emergency + intercom. One class: they
                 are one button to a robot, and separately none of them has
                 enough support to be worth splitting.
  stop      88   thin, and kept separate anyway -- deliberately. It is the one
                 button on the panel that must never be pressed by accident,
                 and folding it into `help` would make "press for help" reach
                 for it. A class the robot AVOIDS can be thin in a way a class
                 it targets cannot.
  keyhole   826   not a target. It is here as a class rather than ignored
                 because it is round, metallic and button-sized -- exactly
                 what gets detected as a floor button otherwise. Naming it is
                 what stops it being guessed at.
  other    2391   a button whose legend cannot be read: blank, blurred or
                 unrecognised. Also not a target, also better named than
                 dropped, and the reader's `unreadable` verdict routes here.

`empty`/`blur`/`unknown` becoming a real class rather than being discarded is
the safety-relevant part: the failure mode this project cares about is
pressing the WRONG floor, so "there is a button here and I cannot read it" has
to be expressible. A dropped class is not -- the detector would simply find
nothing there, which reads identically to no button at all.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Stage A: detection
# ---------------------------------------------------------------------------

# ORDER IS THE CONTRACT. Append only; never reorder, never delete in place.
# The two priorities the arm is built around come first so that a truncated
# read of this list still starts with them.
DETECT_CLASSES = [
    'up',        # 0  hall call, up
    'down',      # 1  hall call, down
    'floor',     # 2  a numbered/lettered floor button; stage B reads which
    'open',      # 3  door open
    'close',     # 4  door close
    'help',      # 5  alarm / call / bell / emergency / intercom
    'stop',      # 6  emergency stop -- never press
    'keyhole',   # 7  key switch -- never press
    'other',     # 8  a button whose legend cannot be read
]

# What the arm is allowed to aim at, in the priority order Jonathan set on
# 2026-08-16: the up/down call first, then a numbered floor, then help.
PRESSABLE = ['up', 'down', 'floor', 'help', 'open', 'close']

# Never a press target, whatever the confidence.
FORBIDDEN = ['stop', 'keyhole', 'other']


# Source-dataset class names that mean one of the above. Deliberately
# conservative: anything not matched here is DROPPED rather than guessed,
# because a wrong mapping is worse than a missing one -- it teaches the model
# that a thing is a button it is not.
#
# Keys are compared lowercased, after PREFIX below is stripped.
_DETECT_ALIASES = {
    # --- priority 1: the hall call --------------------------------------
    'up': 'up', 'u': 'up', 'uparrow': 'up', 'call-up': 'up', 'hall-up': 'up',
    'down': 'down', 'd': 'down', 'dn': 'down', 'downarrow': 'down',
    'call-down': 'down', 'hall-down': 'down',
    # `updown` is a single box drawn around BOTH arrows (3 instances). It is
    # not either class and splitting a box in half is a guess, so it goes.

    # --- priority 3: help ------------------------------------------------
    'alarm': 'help', 'bell': 'help', 'emergency': 'help', 'call': 'help',
    'intercom': 'help', 'phone': 'help', 'help': 'help', 'sos': 'help',

    # --- doors ------------------------------------------------------------
    'open': 'open', 'door-open': 'open', 'openbutton': 'open', 'do': 'open',
    'close': 'close', 'door-close': 'close', 'closebutton': 'close',
    'dc': 'close',

    # --- never press -------------------------------------------------------
    'stop': 'stop',
    'keyhole': 'keyhole', 'bt_keyhole': 'keyhole', 'key': 'keyhole',
    'keyswitch': 'keyhole', 'bt_switch': 'keyhole', 'switch': 'keyhole',

    # --- present but unreadable --------------------------------------------
    'empty': 'other', 'blur': 'other', 'unknown': 'other',
}

# Everything that is not a button at all: annotations for the printed legend
# beside a button, indicator lamps, speakers. Listed explicitly so that they
# are dropped ON PURPOSE and a genuinely new name still shows up in the
# builder's "dropped" report instead of hiding in the noise.
_NOT_A_BUTTON = re.compile(
    r'^(text|indicator|led|light|speaker|fan|hat|fire|-|s\d*|)$', re.I)

# Different sets put different prefixes in front of the same thing: ENTC uses
# `button-1` and `floor-1`, Sun Moon uses a bare `1`.
_PREFIX = re.compile(r'^(button[-_ ]?|floor[-_ ]?|btn[-_ ]?)', re.I)

# A floor legend: a number, or a letter code like B2, LG, G, PH1, -1, 12A.
# Matched only AFTER the alias table, so `D` is the down-call button and never
# a floor, and `B` is a basement and never anything else.
_FLOOR_LEGEND = re.compile(r'^-?\d+[a-z]?$|^[a-z]{1,3}-?\d{0,2}$', re.I)


def normalise(name: str) -> str:
    """Source class name -> a comparable key."""
    return _PREFIX.sub('', str(name).strip()).strip().lower()


def to_detect_class(name: str) -> str | None:
    """Source class name -> one of DETECT_CLASSES, or None to drop it."""
    n = normalise(name)
    if n in _DETECT_ALIASES:
        return _DETECT_ALIASES[n]
    if _NOT_A_BUTTON.match(n):
        return None
    if _FLOOR_LEGEND.match(n):
        return 'floor'
    return None


# ---------------------------------------------------------------------------
# Stage B: reading the floor legend off a crop
# ---------------------------------------------------------------------------

# The reader's vocabulary is NOT fixed here the way the detector's is -- it is
# built from whatever the source data actually supports, by
# scripts/build_button_dataset.py, and written to a labels file next to the
# weights. Pinning it in source would mean this file has to change every time
# a dataset is added, and the reader is a classifier, so its class list is
# read back from the checkpoint at load time rather than assumed.
#
# What IS fixed is the reject class, because the node reasons about it by name.
UNREADABLE = 'unreadable'

# A floor legend has to survive being read wrong. These are the pairs that
# actually get confused on a 128px crop -- worth knowing when the node decides
# whether a reading is trustworthy enough to press.
CONFUSABLE = [('6', '9'), ('8', 'B'), ('0', 'D'), ('1', '7'), ('5', 'S'),
              ('G', '6'), ('2', 'Z')]


def to_reader_label(name: str) -> str | None:
    """Source class name -> the floor legend it shows, or None if not a floor.

    Only called on boxes that stage A maps to `floor`, so the alias table has
    already claimed `U`, `D`, `B`(ell) and friends.
    """
    if to_detect_class(name) != 'floor':
        return None
    return normalise(name).upper()
