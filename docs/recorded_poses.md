# Recorded arm poses

Measured on hardware, not chosen. Each is the mean of 30 consecutive reads
with the spread noted — a pose worth hard-coding should read identically
every time, and all of these do.

Servo zeros were recalibrated 2026-08-10 (`scripts/calibrate_zero.py`), so
these angles are only meaningful against that calibration. If the Atom is
reflashed the zeros are lost and every pose here must be re-measured.

## camera-forward, 2026-08-11

Chosen so the flange camera looks outward rather than at the ceiling. The
search sweep only tilts joint5 from wherever the arm starts, so hunting for
buttons from a folded pose scans empty air.

```
degrees  [0.0, -30.0, -30.0, 0.0, -30.0, 0.0]
radians  [0.0, -0.5236, -0.5236, 0.0, -0.5236, 0.0]
```

Reached to within 1 degree on hardware.

## operator-set pose, 2026-08-11

Set by hand and recorded on request. 30/30 clean reads, 0.00 degrees spread
on every joint.

```
degrees   [-1.1, 90.6, -148.6, 56.1, 6.3, 0.6]
radians   [-0.0183, 1.5814, -2.5939, 0.9786, 0.1103, 0.0106]
encoders  [2060, 1017, 357, 1410, 1976, 2041]
```

## mechanical home

The stock myCobot home. Reaches to 1.1 degrees and holds with zero drift over
8 seconds — there is no torque problem with it, contrary to a claim made
during the 2026-08-10 session. It does fold the arm over and point the camera
up, which is why it is poor for button hunting.

```
degrees  [0.0, 90.0, -90.0, 0.0, 0.0, 0.0]
radians  [0.0, 1.5708, -1.5708, 0.0, 0.0, 0.0]
```

## The FK chain checked against a ruler, 2026-08-13

At all-zeros the chain in `collision_guard.py` predicts the flange frame
**418mm above the mounting face**, 78mm out from the base axis. Measured with
a ruler: **418mm**.

Small check, load-bearing conclusion. Inverse kinematics, the collision guard
and the touch calibration all reason from that chain, and nothing had ever
compared it to the physical arm — a systematic error in it would have been
invisible in every test, because the tests check the chain against itself.

It also settles `--tool-offset-mm` for `calibrate_touch.py`: the flange FRAME
sits on the flange FACE, so touching with a bare flange is an offset of
**0**, and that is the default.

## all-zeros

Every joint on its notch. This is the calibration reference pose — use it to
check visually whether the zeros are still true.

```
degrees  [0, 0, 0, 0, 0, 0]
encoders [2048, 2048, 2048, 2048, 2048, 2048]
```
