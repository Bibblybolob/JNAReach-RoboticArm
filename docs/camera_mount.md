# Camera on the first arm piece — measured, 2026-08-13

The D405 moved from the flange to the first arm piece (link 1), which puts
only **joint1** upstream of it. Everything below is measured on hardware with
`scripts/measure_jacobian.py`, not derived from the URDF.

## The image Jacobian

Global phase correlation between a frame before each jog and after it, ±6° per
joint, 4 samples each, from the all-zeros pose.

| joint | px/deg x | px/deg y | normalised/deg x | response |
|---|---|---|---|---|
| **1** | **+10.49** | −1.77 | **+0.0328** | 0.467 |
| 2 | +0.08 | +0.00 | +0.0002 | 1.000 |
| 3 | −0.02 | +0.01 | −0.0001 | 1.000 |
| 4 | −0.00 | −0.00 | −0.0000 | 0.999 |
| 5 | +0.04 | −0.01 | +0.0001 | 1.000 |
| 6 | −0.00 | +0.00 | −0.0000 | 1.002 |

The response column is the confirmation, not a footnote. A correlation of
**1.000 means the two frames are identical** — the scene did not move at all.
Joint1 scores 0.467 precisely because the scene *did* shift, so the overlap is
smaller. Joints 2–6 are not "small"; they are nothing, to within a twentieth
of a pixel per degree.

**So the servo's 2x2 image Jacobian is singular by construction.** No choice of
`vertical_joint` fixes it, because no second joint moves the view from any
pose — this is the kinematic chain, not a tuning problem. `skip_probe:=false`
refuses for exactly this reason and should be left on.

Why it is dangerous by default: `skip_probe` defaults to `True`, and the
servo's usual guard (`Jog target pinned to the measured pose`) detects a joint
that will not MOVE. Joint5 moves perfectly well here — it just has no effect
on the image. The lag compensator then books the missing image motion as the
target fleeing, leads harder, and tilts further, with no warning attached.

## Intrinsics, from the factory calibration

```
640x480   fx=393.8  fy=393.4  ppx=318.1  ppy=236.5   inverse_brown_conrady
FOV       78.2 deg horizontal (39.1 half)   62.8 deg vertical (31.4 half)
```

**CLAUDE.md's "~43°" for a D405 is the right order but not the number** — at
640x480 the half-angle is 39.1° horizontally and 31.4° vertically. The ratio
between the axes (1.25) is what `assumed_v_deg_per_error` exists to carry.

## The gain depends on how far away the scene is

Pure rotation of a camera predicts `fx * pi/180` = **6.87 px/deg**. Joint1
measured **10.49**, a ratio of **1.53**.

The excess is translation. The camera is mounted off the joint1 axis, so
rotating joint1 both turns the camera and swings it sideways by `r * theta`.
At scene distance `Z` that adds `f * r * theta / Z` of image motion, giving

    measured / predicted  =  1 + r/Z

so `r/Z ≈ 0.53` at the distance this was measured at. **The joint1 gain is
therefore not a constant** — it approaches 6.87 px/deg for distant scenes and
grows as the target gets closer, by half again at the range measured here.

A flange camera has this effect too, but it is worth flagging now because a
servo tuned at arm's length will be over-gained by ~50% on a panel at 10cm,
which is exactly the approach this project ends in.

## Mounting roll

Joint1's image motion is **9.6° off the image horizontal** (−1.77 against
+10.49). The camera is rolled about its optical axis by that much. Small
enough to ignore for a pan-only loop; it is not ignorable if a second axis is
ever added, since roll cross-couples the axes and no per-axis scalar corrects
for it.

## The board is printed at 39mm, not 30mm — and depth is what found it

Measured 2026-08-13, and it invalidates the existing `~/hand_eye/hand_eye.json`.

Two independent ways of locating the same 22 ChArUco corners disagreed by
**30%**: colour geometry put the board at 215mm, depth at 280mm. Depth was
right. Measuring adjacent corner spacings directly gives **38.99mm against the
30mm the script assumed** — 34 spacings, 0.57mm spread. A printer "fit to
page", almost certainly.

Depth is believable here for reasons that do not depend on the board:

- `depth_scale` reads 0.1mm/unit, exactly as documented for a D405
- it reconstructs a known-flat board flat to **1.31mm rms** at that range
- 34 independent spacings agree to 0.57mm

**Nothing in the colour-only path can catch this**, and that is the lesson
worth keeping. The board's pose is solved *from* the assumed square size, so a
wrong size produces a wrong range that is perfectly self-consistent. All four
solvers agreed to 1mm — because all four were handed the same wrong number.
Solver agreement measures agreement, not accuracy, and the previous
calibration's "±8mm between runs" was measuring run-to-run noise sitting on
top of a 30% scale error nobody was looking for.

The colour geometry's own flatness figure, 0.00mm rms, is tautological: those
corners are coplanar by construction. It is not evidence of anything.

`--measure-square` now does this measurement in one command. Run it once per
printed board and pass the result to `--square-mm`.

## Consequences

- **Hand-eye calibration is degenerate, not merely stale.** `AX=XB` needs
  rotation about two non-parallel axes; there is one. The 25° rotational-spread
  check in `calibrate_hand_eye.py` measures magnitude, not axis rank, so
  single-axis data can clear that gate while remaining unsolvable.
  The method that does work here is eye-to-hand: marker on the flange, joint1
  held still, move the ARM through many poses, solve camera→base against the
  arm's own FK.
- **Buttons become pan-to-centre plus IK reach.** Joint1 turns the camera and
  the arm's workspace together, so a target centred in the image lies in the
  arm's sagittal plane. That removes the dead-time problem entirely — correcting
  no longer moves the camera — at the cost of closed-loop feedback on the final
  approach.
