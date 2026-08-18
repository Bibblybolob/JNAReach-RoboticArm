"""Find numbered keypad buttons geometrically, without a trained model.

Why this exists
---------------
`elevator_buttons.pt` cannot detect the buttons on this panel, and neither a
lower threshold nor the GPU changes that. Its class list is

    alarm, button-1, button-2, button-3, button-down, button-g, button-up,
    close, closed-door, down, floor-1, floor-2, floor-3, floor-ground, key,
    open, up

There is **no class for any numeral above 3**, so a 12-floor keypad is
undetectable by construction. Measured 2026-08-14: zero detections at the 0.5
production threshold, nothing above 0.12 even at 0.05, and the few boxes that
did appear were on the arm and the wall socket.

The buttons are, however, strongly geometric -- equal-sized rounded shapes on
a regular lattice. That prior is worth more here than a learned detector, runs
in ~50ms of CPU instead of 536ms, and does not care what digit is printed.

Ellipses, not circles
---------------------
A panel viewed off-axis projects round buttons to ELLIPSES. `HoughCircles`
models only circles, and on the first attempt here it found the near column
and missed the foreshortened far one entirely. Contours plus `fitEllipse` has
no such bias, which is why that is what this uses.

Four filters, each removing a different kind of false positive
--------------------------------------------------------------
1. **shape** -- the contour's area must match its fitted ellipse's area
2. **equal size** -- keypad buttons are all one size; drop outliers
3. **packing** -- keep the largest group of mutually-close candidates, which
   discards isolated blobs elsewhere in the scene
4. **global lattice** -- every row uses the SAME columns. Clustering each row
   independently let a stray blob invent a third column in one row only.

Labels are a table, not a formula
---------------------------------
`PANEL_12` is the layout of the panel in this lab, read off the photograph.
It is deliberately explicit: the rows ascend in pairs from the bottom, EXCEPT
the 9/10 row which is reversed, so any clever formula would be wrong. Getting
this wrong means pressing the wrong button, so `label()` REFUSES when the row
count does not match the layout rather than shifting every label by one.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# Row-major from the TOP, each row left-to-right. See the module docstring:
# this is read off the panel, not derived.
PANEL_12 = [
    [11, 12],
    [10, 9],
    [7, 8],
    [5, 6],
    [3, 4],
    [1, 2],
]


# The LAB's printed test panel, which is NOT the same layout as PANEL_12 and
# must not be conflated with it. Read off the panel 2026-08-17:
#
#   - it has EIGHT rows of two, not six. Below the numbers sit door-open /
#     door-close, then a fan and an alarm bell. `_rows_of`'s note about the
#     symbol row being excluded because it has THREE buttons on its own
#     spacing does not hold here -- these are two-wide on the same spacing as
#     the numbers, so they come through as rows 7 and 8 and a six-row layout
#     refuses the whole panel.
#   - every row runs odd-left, even-right, INCLUDING 9/10. PANEL_12 reverses
#     that row. Both cannot describe the same panel; PANEL_12 is kept as-is
#     for the real lift, and this describes the print on the board.
#
# Getting these two the wrong way round presses the wrong floor, which is the
# specific error `label()` refuses on, so they are separate constants rather
# than one with a flag.
PANEL_12_PRINTED = [
    [11, 12],
    [9, 10],
    [7, 8],
    [5, 6],
    [3, 4],
    [1, 2],
    ['open', 'close'],
    ['fan', 'alarm'],
]


# The numbered block of the printed panel ALONE, for when the symbol rows run
# off the bottom of the frame -- which is the normal case once the panel is
# close enough to press, since it then fills the view.
#
# This is the RISKY layout of the three and is offered rather than defaulted:
# with six rows expected and eight present, a row block that is not
# top-anchored labels every button one row out. Nothing here can detect that,
# so confirm against `--save` before commanding a press. Prefer
# PANEL_12_PRINTED whenever all eight rows are actually in view.
PANEL_12_PRINTED_NUMBERS = PANEL_12_PRINTED[:6]


# The separate call panel: up above down, one column. Names not numbers,
# because that is what is printed on them.
PANEL_CALL = [
    ['up'],
    ['down'],
]


class KeypadError(ValueError):
    """No usable keypad in this image."""


def _candidates(bgr):
    """Ellipse candidates, swept over several thresholds and deduplicated."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)
    g = cv2.createCLAHE(3.0, (8, 8)).apply(g)

    # Scale with the image. The block sizes and area limits below were tuned
    # at 640x480; at 1280x720 the same numbers find nothing at all, because a
    # block that spans a whole button at one resolution spans a quarter of it
    # at twice the scale. Measured 2026-08-14: 3/4 frames at 640, 0/4 at 1280
    # before this. Everything downstream is already scale-free (tolerances are
    # in button radii), so this is the only place resolution leaks in.
    sc = bgr.shape[1] / 640.0
    blocks = [int(round(b * sc)) | 1 for b in (21, 31, 41)]   # must be odd
    a_lo, a_hi = 150 * sc * sc, 3000 * sc * sc

    out = []
    for blk in blocks:
        for c_ in (2, 5, 8):
            th = cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blk, c_)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
            cs, _ = cv2.findContours(th, cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_NONE)
            for c in cs:
                if len(c) < 12:
                    continue
                a = cv2.contourArea(c)
                if not (a_lo < a < a_hi):
                    continue
                (cx, cy), (MA, ma), _ = cv2.fitEllipse(c)
                major, minor = max(MA, ma), min(MA, ma)
                if minor < 1 or minor / major < 0.45:
                    continue
                # Does the ellipse actually explain the contour?
                if not (0.65 < a / (np.pi * major * minor / 4) < 1.35):
                    continue
                out.append((cx, cy, major / 2, minor / 2))

    keep = []
    for c in out:
        if not any(np.hypot(c[0] - k[0], c[1] - k[1]) < 10 * sc for k in keep):
            keep.append(c)
    if not keep:
        raise KeypadError('no button-shaped contours at any threshold')
    return np.array(keep)


def _cluster(v, tol):
    order = np.argsort(v)
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if abs(v[i] - v[cur[-1]]) <= tol:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    return groups


def _depth_ok(K, depth_mm, lo, hi):
    """Drop candidates that depth shows are at the WRONG distance.

    The decisive filter against the arm itself: its white casing has round
    mouldings that pass every shape test, and they were matched as the up/down
    pair once proximity pulled the search near the keypad. They sit ~200mm
    from the camera while the panel is ~350mm.

    Rejects only on CONFIDENT evidence. A candidate whose depth is missing or
    too sparse to trust is KEPT, not dropped -- absence of a reading is not
    evidence of being in the wrong place. Getting this backwards cost a
    working detector: the keypad spans 305-369mm against a 300mm floor, its
    depth drops out on a few buttons from frame to frame, and requiring a
    valid reading silently deleted real buttons at random. That presented as
    "detection worked a minute ago and now finds four rows".
    """
    out = []
    h, w = depth_mm.shape
    for c in K:
        x, y = int(round(c[0])), int(round(c[1]))
        if not (2 <= x < w - 2 and 2 <= y < h - 2):
            out.append(c)
            continue
        p = depth_mm[y-2:y+3, x-2:x+3]
        v = p[p > 0]
        if v.size < 4:            # no usable reading -> no evidence -> keep
            out.append(c)
            continue
        if lo <= float(np.median(v)) <= hi:
            out.append(c)
    return np.array(out) if out else np.empty((0, 4))


def _groups(bgr, link=6.0, depth_mm=None, depth_range=None):
    """All densely-packed groups of equal-sized buttons, biggest first.

    A lift panel is usually more than one cluster: the numbered keypad, and a
    separate call panel with just up and down. Returning every group lets the
    caller take the keypad AND the call buttons, instead of the largest group
    silently discarding the smaller one -- which is what happened when the
    up/down buttons went unrecognised.

    `link` must sit between the widest spacing INSIDE a panel and the gap
    BETWEEN panels, both measured in button radii. On this panel: columns are
    ~57px apart at a radius of ~12 (4.8 radii), and the call panel is ~110px
    away (9 radii). It was 4.0, which is below the column spacing -- so the
    two columns of the keypad were sometimes separate groups and the largest
    group was a single column of six. That failed intermittently, because
    whether it linked depended on the median radius the frame happened to
    produce. 6.0 clears the columns with margin and stays well inside the gap
    to the call panel.
    """
    K = _candidates(bgr)
    if depth_mm is not None and depth_range is not None:
        K = _depth_ok(K, depth_mm, *depth_range)
        if len(K) < 2:
            raise KeypadError(
                f'{len(K)} candidates survive the depth gate '
                f'{depth_range[0]:.0f}-{depth_range[1]:.0f}mm. The rest were '
                'confidently at another distance -- usually the arm itself.')
    rad = float(np.median(K[:, 2]))
    K = K[np.abs(K[:, 2] - rad) <= 0.35 * rad]
    if len(K) < 2:
        raise KeypadError(
            f'{len(K)} candidates survived the equal-size filter. A keypad\'s '
            'buttons are all one size; a scene that fails this is not one.')

    xy = K[:, :2]
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    adj = (D > 0) & (D < rad * link)
    seen, comps = set(), []
    for i in range(len(xy)):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k); comp.append(k)
            stack.extend(np.nonzero(adj[k])[0].tolist())
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return [K[c] for c in comps], rad


def _rows_along(K, rad, n_cols, down, across, align_tol):
    """_rows_of's body, measured along a given pair of axes."""
    u = K[:, :2] @ across          # across a row: separates COLUMNS
    t = K[:, :2] @ down            # down a column: separates ROWS

    colg = _cluster(u, rad * 1.3)
    colg.sort(key=len, reverse=True)
    colg = colg[:n_cols]
    if len(colg) < n_cols:
        return []
    centres = sorted(float(np.mean(u[gr])) for gr in colg)
    keep = sorted(i for gr in colg for i in gr)
    Kk, uk, tk = K[keep], u[keep], t[keep]

    rowg = _cluster(tk, rad * 1.3)
    # Cluster along the lattice axis, but ORDER by image y. The projection's
    # sign comes from an eigenvector, and an eigenvector's direction is
    # arbitrary -- `_lattice_axes` pins it, but only for the assignment it
    # returns, and `_rows_of` also tries the SWAPPED one. Ordering by `t`
    # therefore inverts whenever the swapped axes win, which is a silent
    # top-to-bottom flip of the whole layout: photographed 2026-08-18 with
    # the fan and alarm symbols labelled 11 and 12, and a requested button 1
    # aimed at the button printed 7. Image y has no such ambiguity.
    rowg.sort(key=lambda gr: float(np.mean(K[keep][gr, 1])))
    rows = []
    for gr in rowg:
        idx = sorted(gr, key=lambda i: uk[i])
        if len(idx) != n_cols:
            continue
        if any(abs(uk[i] - c) > rad * align_tol
               for i, c in zip(idx, centres)):
            continue
        rows.append([tuple(float(v) for v in Kk[i]) for i in idx])
    return rows


def _rows_of(K, rad, n_cols, align_tol=0.7):
    """Arrange one group's buttons into rows, filtered to n_cols columns.

    Rows whose buttons do not line up with the dominant columns are dropped.
    That is what separates the NUMBERED rows from the door-open/alarm/close
    row underneath them: the symbol row has three buttons on its own spacing,
    so once the arm stopped occluding it, it appeared as a seventh row and the
    layout check refused everything. Alignment is the real distinction between
    "part of this grid" and "some other buttons that happen to be nearby".

    Measured along the LATTICE's axes, not the image's. Clustering raw x to
    find columns assumes the panel is square-on, and a rotated one then fails
    completely rather than gracefully: measured 2026-08-17 at 19.9deg of
    rotation, each column's x drifted 127px top to bottom while adjacent
    buttons differed by only ~30px, so `_cluster` -- which is single-linkage --
    chained straight down the panel and returned ONE cluster of 14 spanning
    256px instead of two columns of six. Every row then failed the alignment
    check and a fully visible panel yielded nothing at all, 0/30 frames.

    The angle is FOUND BY SWEEP rather than estimated, and that matters. The
    first version took the principal axis of the button centres, which is only
    as good as the points fed to it: on 2026-08-18 three stray candidates off
    the panel -- chained into the group by the link distance -- pulled the
    estimate to 34deg where the true column tilt was 17deg, and the panel then
    yielded 0 rows with 19 candidates sitting plainly on it.

    A sweep has no such failure. A wrong angle simply produces no rows, and an
    outlier that belongs to no row cannot drag the answer, so the score is
    robust to exactly the contamination that breaks a least-squares fit. It
    costs ~90 clusterings of a few dozen points, which is nothing next to
    finding the candidates in the first place.
    """
    best = []
    for deg in range(-90, 90, 2):
        th = math.radians(deg)
        # across[0] >= 0 over this range and down[1] > 0, so "left to right"
        # and "top to bottom" keep their ordinary meanings.
        across = np.array([math.cos(th), math.sin(th)])
        down = np.array([-math.sin(th), math.cos(th)])
        rows = _rows_along(K, rad, n_cols, down, across, align_tol)
        if len(rows) > len(best):
            best = rows
    return best


def find_grid(bgr, n_cols=2, depth_mm=None, depth_range=None):
    """Locate the keypad. Returns (rows, semi_major_px).

    `rows` is a list, top row first, of lists of (x, y, a, b) left to right.
    The largest packed group is taken as the keypad; `_rows_of` then keeps
    only rows aligned with its dominant columns, which is what excludes the
    door-open/alarm/close row sitting underneath the numbers.
    """
    groups, rad = _groups(bgr, depth_mm=depth_mm, depth_range=depth_range)
    if not groups:
        raise KeypadError('no button groups found')
    rows = _rows_of(groups[0], rad, n_cols=n_cols)
    if len(rows) < 2:
        raise KeypadError(
            f'the largest group yielded {len(rows)} aligned rows, too few to '
            'be a keypad')
    return rows, rad


def complete_lattice(bgr, rows, rad, n_rows, min_score=0.12):
    """Fill in rows the detector missed, using the grid it did find.

    Low-contrast buttons do not produce a closed contour -- buttons 1 and 2 on
    this panel are visibly there and were found zero times at any threshold.
    Loosening the filters to catch them readmits the false positives those
    filters exist to remove, so instead: a lattice of equally-spaced rows
    PREDICTS where a missing row must be, and the buttons already found give a
    template to confirm it is really there.

    Which end is missing is not assumed. Every alignment of the found rows
    within `n_rows` is scored by how well the predicted cells match the
    average found button, and the best alignment wins -- so a missing TOP row
    is handled as naturally as a missing bottom one, and that matters because
    guessing wrong shifts every label.
    """
    if len(rows) >= n_rows or len(rows) < 2:
        return rows

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    half = int(round(rad))

    def patch(x, y):
        xi, yi = int(round(x)), int(round(y))
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            return None
        p = gray[yi-half:yi+half+1, xi-half:xi+half+1]
        s = p.std()
        return (p - p.mean()) / s if s > 1e-6 else None

    ref = [patch(c[0], c[1]) for r in rows for c in r]
    ref = [p for p in ref if p is not None]
    if not ref:
        return rows
    template = np.mean(ref, axis=0)

    ys = [np.mean([c[1] for c in r]) for r in rows]

    # The pitch below assumes the rows found are CONSECUTIVE. When they are
    # not, (last-first)/(n-1) is a blend of the real pitch and whatever gap
    # was skipped, and every predicted row lands somewhere arbitrary.
    # Measured 2026-08-17: rows at y=337,388,435,687 -- a 252px hole, five
    # pitches wide -- gave dy=117 against a true pitch of ~49, and completion
    # duly invented two rows at y=104 and y=221, ABOVE THE PANEL, then
    # labelled all twelve buttons with confidence. That is the wrong-floor
    # failure this module exists to refuse, so refuse it: uneven spacing means
    # the row set is not a contiguous run and cannot be extrapolated from.
    gaps = np.diff(ys)
    if len(gaps) and float(np.max(gaps)) > 1.5 * float(np.min(gaps)):
        return rows

    dy = (ys[-1] - ys[0]) / (len(ys) - 1)
    # Column x per row drifts if the panel is tilted; take the per-column
    # slope from the rows we have rather than assuming vertical columns.
    ncol = len(rows[0])
    slopes, x0s = [], []
    for ci in range(ncol):
        xs_c = [r[ci][0] for r in rows if len(r) > ci]
        if len(xs_c) < 2:
            slopes.append(0.0); x0s.append(xs_c[0] if xs_c else 0.0)
        else:
            A = np.polyfit(ys[:len(xs_c)], xs_c, 1)
            slopes.append(float(A[0])); x0s.append(float(A[1]))

    ranked = []
    for off in range(n_rows - len(rows) + 1):
        filled, scores = [], []
        unverified = False
        for ri in range(n_rows):
            if off <= ri < off + len(rows):
                filled.append(rows[ri - off])
                continue
            y = ys[0] + (ri - off) * dy
            cells = []
            for ci in range(ncol):
                x = slopes[ci] * y + x0s[ci]
                p = patch(x, y)
                # Both patches are zero-mean unit-variance, so the mean of
                # their product is the normalised correlation.
                if p is None:
                    # OFF THE IMAGE. Not evidence of absence -- there is
                    # simply nothing to look at, and scoring it -1 makes it
                    # evidence AGAINST, which inverts the answer whenever the
                    # panel runs to the frame edge. Measured 2026-08-17: the
                    # bottom symbol row sat past y=720, so the correct
                    # alignment (both missing rows at the bottom) scored -1
                    # and LOST to one that invented a row above the panel --
                    # shifting every label by a row, so a commanded press of
                    # 5 went to the button printed 7. Flag it and refuse.
                    unverified = True
                else:
                    scores.append(float(np.mean(p * template)))
                cells.append((float(x), float(y), rad, rad))
            filled.append(cells)
        ranked.append((np.mean(scores) if scores else -1.0, off, filled,
                       unverified))

    ranked.sort(key=lambda t: -t[0])
    best_score, _, best, best_unverified = ranked[0]

    # An alignment that puts a predicted row off the edge of the image cannot
    # be confirmed OR denied, so completing on it is a guess -- and a guess
    # here renumbers the whole panel. Refuse and let `label()` say what to do
    # about it, which is to re-frame so every row is in view.
    if best_unverified:
        return rows

    # A RELATIVE test, not an absolute one. Measured on this panel 2026-08-14:
    # buttons the detector DID find self-correlate at 0.43-0.62, while the
    # genuinely-present but low-contrast bottom row scores 0.21 -- so any
    # absolute bar high enough to mean "this is a button" also rejects the
    # real ones. What separates the right answer from the wrong one is the
    # MARGIN: 0.213 for the missing bottom row against 0.079 for a missing
    # top row, a factor of 2.7. So require the winner to beat the runner-up
    # clearly, and refuse when the alignment is genuinely ambiguous, because
    # choosing wrong shifts every label by a row.
    if best_score < min_score:
        return rows
    if len(ranked) > 1:
        second = ranked[1][0]
        if second > 0 and best_score < 1.5 * second:
            return rows
    return best


def _row_above(bgr, rows, rad, min_score=0.12):
    """Is there another button row ABOVE the ones found? Score, or None.

    The check a short layout cannot do for itself. `PANEL_12_PRINTED_NUMBERS`
    describes six rows of a panel that physically has eight, so it is correct
    only if the six found are the TOP six -- and nothing about six rows in
    isolation says whether the block starts at the top or one row down.

    Measured 2026-08-18: the faint 11/12 row went undetected, the six rows
    found were rows 2-7, and every button was labelled one row out -- a
    requested 12 aimed at the printed 10 and a requested 1 at the door-open
    symbol. Looking one pitch above the top row costs one correlation and
    catches exactly that.
    """
    if len(rows) < 2:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    half = int(round(rad))

    def patch(x, y):
        xi, yi = int(round(x)), int(round(y))
        if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
            return None
        p = gray[yi - half:yi + half + 1, xi - half:xi + half + 1]
        s = p.std()
        return (p - p.mean()) / s if s > 1e-6 else None

    ref = [patch(c[0], c[1]) for r in rows for c in r]
    ref = [p for p in ref if p is not None]
    if not ref:
        return None
    template = np.mean(ref, axis=0)

    # Step each COLUMN back by its own spacing vector, rather than predicting
    # one y for the whole row. On a rotated panel a row's buttons differ in y
    # -- 14px apart at 10deg here -- so a single row y puts each patch off its
    # button by half that, and the correlation collapses to noise. The step
    # vector carries the rotation for free and is exact on a uniform lattice.
    scores = []
    for ci in range(len(rows[0])):
        col = [r[ci] for r in rows if len(r) > ci]
        if len(col) < 2:
            continue
        first = np.array(col[0][:2], dtype=float)
        step = (np.array(col[-1][:2], dtype=float) - first) / (len(col) - 1)
        above = first - step
        p = patch(above[0], above[1])
        if p is not None:
            scores.append(float(np.mean(p * template)))
    if not scores:
        return None
    score = float(np.mean(scores))
    return score if score >= min_score else None


def label(rows, layout=PANEL_12):
    """Map the found grid onto printed numbers. Returns {number: (x,y,a,b)}.

    REFUSES on a row-count mismatch. A missing top row would otherwise shift
    every label by one and press the wrong floor -- a silent, confident error
    of exactly the kind that is expensive here.
    """
    if len(rows) != len(layout):
        raise KeypadError(
            f'found {len(rows)} rows but the layout has {len(layout)}. '
            'Refusing to label: a missed row shifts every number and presses '
            'the wrong button. Re-frame the panel so all rows are visible, or '
            'pass a layout matching what is actually in view.')
    out = {}
    for found, names in zip(rows, layout):
        if len(found) != len(names):
            raise KeypadError(
                f'row has {len(found)} buttons, layout expects {len(names)}')
        for cell, n in zip(found, names):
            out[n] = cell
    return out


def find_labelled(bgr, layout=PANEL_12, complete=True, depth_mm=None,
                  depth_range=None, top_anchored=False):
    """image -> {number: (x, y, semi_major, semi_minor)}.

    `top_anchored` is for layouts that describe only the TOP of a longer
    panel, such as PANEL_12_PRINTED_NUMBERS against the eight-row print. Such
    a layout is right only if the rows found start at the panel's top row, and
    the row count cannot tell you that. With it set, a row detected above the
    block is a refusal rather than a silent renumbering.
    """
    rows, rad = find_grid(bgr, n_cols=len(layout[0]), depth_mm=depth_mm,
                          depth_range=depth_range)
    if complete:
        rows = complete_lattice(bgr, rows, rad, len(layout))
    if top_anchored:
        score = _row_above(bgr, rows, rad)
        if score is not None:
            raise KeypadError(
                f'there is another button row above the ones found '
                f'(correlation {score:.2f}), so these are not the top rows of '
                'the panel and this layout would number every button one row '
                'out. Re-frame so the whole panel is visible and use the full '
                'layout.')
        # Extra rows BELOW are fine once the top is pinned. How many of the
        # symbol rows fall inside the frame changes with every nudge of the
        # panel -- 8 rows, then 6, then 7 across three placements on
        # 2026-08-18 -- and an exact-count match refuses all but one of those
        # for no good reason. The numbered block is at the top, so with
        # nothing above it the first len(layout) rows ARE the layout.
        if len(rows) > len(layout):
            rows = rows[:len(layout)]
    return label(rows, layout)


def _fine_candidates(bgr):
    """Candidates at a finer scale, for the small faint call buttons.

    Separate from `_candidates` on purpose. The call buttons are r~9 against
    the keypad's r~12 and much lower contrast (std 18.8 vs 25.2), and they
    form no closed contour until the adaptive-threshold block drops to ~9-15.
    Feeding those settings into the SHARED sweep was tried and it broke the
    keypad detector outright -- far more candidates, and the grouping stopped
    resolving the two columns. So the fine sweep lives here and only the call
    search pays for it.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 50, 50)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    out = []
    for blk in (9, 11, 15, 21):
        for c_ in (1, 2, 3, 5):
            th = cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blk, c_)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
            cs, _ = cv2.findContours(th, cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_NONE)
            for c in cs:
                if len(c) < 10:
                    continue
                a = cv2.contourArea(c)
                if not (80 < a < 1200):
                    continue
                (cx, cy), (MA, ma), _ = cv2.fitEllipse(c)
                mj, mn = max(MA, ma), min(MA, ma)
                if mn < 1 or mn / mj < 0.5:
                    continue
                if not (0.6 < a / (np.pi * mj * mn / 4) < 1.4):
                    continue
                out.append((cx, cy, mj / 2, mn / 2))
    keep = []
    for c in sorted(out, key=lambda t: t[1]):
        if not any(np.hypot(c[0] - k[0], c[1] - k[1]) < 7 for k in keep):
            keep.append(c)
    return np.array(keep) if keep else np.empty((0, 4))


def _strict_depth(K, depth_mm, lo, hi):
    """Keep only candidates with a CONFIRMED in-range depth.

    The opposite policy to `_depth_ok`, and deliberately so. The keypad search
    must tolerate missing depth or it deletes real buttons at random. The call
    search must not: a pair of blobs with no depth at the frame edge -- the
    cut-off sliver of the keypad -- matched the up/down shape perfectly and
    won. Requiring depth costs nothing here, because a call button that cannot
    be ranged cannot be pressed either.
    """
    out = []
    h, w = depth_mm.shape
    for c in K:
        x, y = int(round(c[0])), int(round(c[1]))
        if not (2 <= x < w - 2 and 2 <= y < h - 2):
            continue
        p = depth_mm[y-2:y+3, x-2:x+3]
        v = p[p > 0]
        if v.size >= 4 and lo <= float(np.median(v)) <= hi:
            out.append(c)
    return np.array(out) if out else np.empty((0, 4))


def find_call_buttons(bgr, depth_mm=None, depth_range=None, near=None,
                      gap_radii=(2.0, 5.5), x_tol_radii=0.8,
                      size_tol=0.35, isolation_radii=4.5,
                      edge_margin_radii=2.5):
    """The up/down call panel, as {'up': cell, 'down': cell}.

    Found as a PAIR, not as a grid. Two buttons are not a lattice -- there is
    no row or column structure to exploit and the keypad machinery kept either
    discarding them (largest-group-wins) or matching two round mouldings on
    the arm's own casing instead. What actually identifies them is simple and
    specific: two blobs of the SAME size, stacked VERTICALLY, a couple of radii
    apart. Up is the higher one, because that is what the arrows mean.
    """
    K = _fine_candidates(bgr)
    if depth_mm is not None and depth_range is not None:
        K = _strict_depth(K, depth_mm, *depth_range)
    if len(K) < 2:
        raise KeypadError(
            f'{len(K)} fine candidates with a confirmed in-range depth -- '
            'need at least two')

    best = None
    for i in range(len(K)):
        for j in range(len(K)):
            if i == j:
                continue
            a, b = K[i], K[j]
            if a[1] >= b[1]:                     # a must be the upper one
                continue
            r = (a[2] + b[2]) / 2.0
            if abs(a[2] - b[2]) > size_tol * r:  # same size
                continue
            if abs(a[0] - b[0]) > x_tol_radii * r:   # vertically stacked
                continue
            gap = b[1] - a[1]
            if not (gap_radii[0] * r <= gap <= gap_radii[1] * r):
                continue
            # A pair touching the frame EDGE cannot be judged isolated: its
            # neighbours may simply be outside the image. That is exactly how
            # a half-visible keypad at the border kept winning this search --
            # its cut-off column looks like a lonely pair.
            if min(a[0], b[0]) < edge_margin_radii * r or \
                    min(a[1], b[1]) < edge_margin_radii * r:
                continue
            H, W = bgr.shape[:2]
            if max(a[0], b[0]) > W - edge_margin_radii * r or \
                    max(a[1], b[1]) > H - edge_margin_radii * r:
                continue
            # ISOLATION -- the discriminator that shape cannot provide.
            # Keypad buttons also come in vertically-stacked, equally-sized,
            # correctly-spaced pairs at the right depth, and two of them won
            # this search twice before this check existed. What is unique
            # about the call panel is that it stands ALONE: exactly two
            # buttons, no third of the same size anywhere near. Every keypad
            # button has neighbours.
            others = 0
            for k in range(len(K)):
                if k in (i, j):
                    continue
                c = K[k]
                if abs(c[2] - r) > size_tol * r:
                    continue
                if (np.hypot(c[0] - a[0], c[1] - a[1]) < isolation_radii * r or
                        np.hypot(c[0] - b[0], c[1] - b[1]) < isolation_radii * r):
                    others += 1
            if others:
                continue
            # Prefer the pair nearest a supplied anchor, else the tidiest one.
            score = (abs(a[0] - b[0]) / r + abs(a[2] - b[2]) / r)
            if near is not None:
                score += np.hypot((a[0] + b[0]) / 2 - near[0],
                                  (a[1] + b[1]) / 2 - near[1]) / (10.0 * r)
            if best is None or score < best[0]:
                best = (score, a, b)

    if best is None:
        raise KeypadError(
            'no vertically-stacked, equally-sized pair found. The call panel '
            'may be occluded -- park the arm clear of it.')
    _, up, down = best
    return {'up': tuple(float(v) for v in up),
            'down': tuple(float(v) for v in down)}


def find_all(bgr, depth_mm=None, depth_range=None):
    """Everything on the panel: numbered buttons plus up/down, in one pass.

    Returns {'floors': {n: cell}, 'call': {'up': cell, 'down': cell}}. Either
    may be missing with its reason attached rather than raising, because one
    group being occluded should not hide the other.
    """
    out = {}
    try:
        out['floors'] = find_labelled(bgr, depth_mm=depth_mm,
                                      depth_range=depth_range)
    except KeypadError as e:
        out['floors'], out['floors_error'] = None, str(e)

    # Anchor the call-panel search on the keypad when we have it, so a
    # similarly-shaped cluster elsewhere in the scene cannot win.
    near = None
    if out.get('floors'):
        near = (float(np.mean([c[0] for c in out['floors'].values()])),
                float(np.mean([c[1] for c in out['floors'].values()])))
    try:
        out['call'] = find_call_buttons(bgr, near=near, depth_mm=depth_mm,
                                        depth_range=depth_range)
    except KeypadError as e:
        out['call'], out['call_error'] = None, str(e)
    return out
