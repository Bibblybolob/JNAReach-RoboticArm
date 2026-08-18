#!/usr/bin/env python3
"""Fit camera->base from tag POSITION only, discarding marker orientation.

    ./scripts/solve_hand_eye_position.py --tag-id 20 --tag-mm 62.75

Why this exists
---------------
`calibrate_hand_eye.py --solve` runs four closed-form solvers (TSAI, PARK,
HORAUD, DANIILIDIS) and refuses when they disagree by more than 15mm. On this
rig they always do -- 397.6mm on 2026-08-17, 38.4mm on the captures of
2026-08-14 -- because every one of them consumes the marker's ORIENTATION, and
the orientation of a near-frontal planar marker cannot be reconciled with FK
here below ~26deg. The cause was never isolated between the FK rotation model
and the weak observability of out-of-plane rotation for a flat target; what
was established is that POSITION is unaffected.

So this fits position only. It is not a fallback -- it is the method that
produced the only calibration this project has ever verified well (3.2mm
out-of-sample on 2026-08-14, against 38.4mm for the closed-form fit on the
same captures). It was written ad-hoc that day and never committed, which is
why it had to be written twice.

The model
---------
The tag is bolted to the flange, so its position in the FLANGE frame is one
unknown constant. For every capture:

    p_flange = inv(FK(joint angles)) @ camera_to_base @ p_camera

and that must come out the same every time. Nine unknowns -- six for the
camera pose, three for the tag offset -- against three equations per capture.

That objective is deliberately the same quantity `verify_calibration.py`
reports as "spread", so a good fit here and a good verification are the same
statement measured on different poses.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'mycobot_driver',
                                'mycobot_driver'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from collision_guard import flange_transform  # noqa: E402

OUT_DIR = os.path.expanduser('~/hand_eye')
# Factory intrinsics for the D405 colour stream at 640x480, which is the
# resolution --collect captures at. These MUST match the images: the same
# camera at 1280x720 reports fx=654.6, and using one set on the other's images
# is a silent ~40% scale error in every pose estimate.
DEFAULT_K = np.array([[393.8, 0, 318.1], [0, 393.4, 236.5], [0, 0, 1.0]])


def observations(records, K, tag_id, tag_mm):
    """Tag position in the camera frame, plus FK, for each usable capture."""
    dist = np.zeros(5)
    adict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    obs = []
    for r in records:
        img = cv2.imread(r['image'])
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, adict)
        if ids is None:
            continue
        hit = [k for k, m in enumerate(ids.ravel()) if int(m) == tag_id]
        if len(hit) != 1:
            # Zero means not seen. More than one means the id is duplicated
            # somewhere else in the scene and there is no way to tell which is
            # the flange -- refusing beats guessing, same as the main script.
            continue
        _, tv, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[hit[0]]], tag_mm / 1000.0, K, dist)
        obs.append({
            'name': os.path.basename(r['image']),
            'p_cam': np.asarray(tv).reshape(3),
            'T_bf': np.array(flange_transform(r['angles'])),
        })
    return obs


def residuals(x, obs):
    """Per-capture deviation of the tag from its own mean, in the flange frame."""
    R = cv2.Rodrigues(x[:3].reshape(3, 1))[0]
    t = x[3:6]
    p_f = x[6:9]
    out = []
    for o in obs:
        p_base = R @ o['p_cam'] + t
        T_fb = np.linalg.inv(o['T_bf'])
        p_flange = T_fb[:3, :3] @ p_base + T_fb[:3, 3]
        out.append(p_flange - p_f)
    return np.concatenate(out)


def fit(obs, seed_C, seed_pf, loss='soft_l1'):
    x0 = np.zeros(9)
    x0[:3] = cv2.Rodrigues(seed_C[:3, :3])[0].reshape(3)
    x0[3:6] = seed_C[:3, 3]
    x0[6:9] = seed_pf
    # soft_l1 rather than plain least squares: a capture where the marker was
    # caught mid-move, or detected at a grazing angle, is a large outlier that
    # would otherwise drag the whole transform toward it.
    r = least_squares(residuals, x0, args=(obs,), loss=loss, f_scale=0.005,
                      max_nfev=2000)
    C = np.eye(4)
    C[:3, :3] = cv2.Rodrigues(r.x[:3].reshape(3, 1))[0]
    C[:3, 3] = r.x[3:6]
    return C, r.x[6:9], r


def per_pose_error(x, obs):
    res = residuals(x, obs).reshape(-1, 3)
    return np.linalg.norm(res, axis=1) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag-id', type=int, default=20)
    ap.add_argument('--tag-mm', type=float, required=True,
                    help='printed side of the marker, BLACK SQUARE only')
    ap.add_argument('--drop-worse-than', type=float, default=15.0,
                    help='refit without captures whose residual exceeds this '
                         '(mm); 0 disables')
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'eye_to_hand.json'))
    ap.add_argument('--K', type=float, nargs=9, default=None)
    args = ap.parse_args()

    rec_path = os.path.join(OUT_DIR, 'poses.json')
    if not os.path.isfile(rec_path):
        print(f'no captures at {rec_path} -- run --collect first')
        return 1
    records = json.load(open(rec_path))
    K = (np.array(args.K, dtype=float).reshape(3, 3)
         if args.K else DEFAULT_K)
    print(f'intrinsics fx={K[0,0]:.1f} fy={K[1,1]:.1f} '
          f'cx={K[0,2]:.1f} cy={K[1,2]:.1f}')

    obs = observations(records, K, args.tag_id, args.tag_mm)
    print(f'{len(obs)} of {len(records)} captures had a usable marker '
          f'{args.tag_id} view')
    if len(obs) < 8:
        print('Not enough usable views to fit 9 parameters with any margin.')
        return 1

    # Seed from the existing calibration when there is one -- it is in the
    # right neighbourhood even when stale, and this objective is non-convex in
    # the rotation.
    seed_C = np.eye(4)
    if os.path.isfile(args.out):
        try:
            seed_C = np.array(json.load(open(args.out))['camera_to_base'],
                              dtype=float)
            print('seeded from the existing calibration')
        except Exception:  # noqa: BLE001
            pass
    seed_pf = np.array([0.025, 0.022, -0.086])

    C, p_f, r = fit(obs, seed_C, seed_pf)
    err = per_pose_error(r.x, obs)
    print(f'\nfit over {len(obs)} captures: '
          f'mean {err.mean():.1f}mm, worst {err.max():.1f}mm')

    kept = obs
    if args.drop_worse_than > 0 and err.max() > args.drop_worse_than:
        bad = [o['name'] for o, e in zip(obs, err) if e > args.drop_worse_than]
        kept = [o for o, e in zip(obs, err) if e <= args.drop_worse_than]
        print(f'dropping {len(bad)} capture(s) worse than '
              f'{args.drop_worse_than}mm: {", ".join(bad)}')
        if len(kept) < 8:
            print('too few left to refit; keeping the full-set fit')
            kept = obs
        else:
            C, p_f, r = fit(kept, C, p_f)
            err = per_pose_error(r.x, kept)
            print(f'refit over {len(kept)} captures: '
                  f'mean {err.mean():.1f}mm, worst {err.max():.1f}mm')

    # Leave-one-out: how much does the answer move when any single capture is
    # removed? A fit that swings here is being carried by one observation.
    jack = []
    for i in range(len(kept)):
        sub = kept[:i] + kept[i + 1:]
        Ci, _, _ = fit(sub, C, p_f, loss='linear')
        jack.append(Ci[:3, 3])
    jack_mm = float(np.max(np.ptp(np.array(jack), axis=0)) * 1000)
    print(f'jackknife spread of the camera position: {jack_mm:.2f}mm')

    print(f'\ncamera in the BASE frame (mm): '
          f'x={C[0,3]*1000:+.1f}  y={C[1,3]*1000:+.1f}  z={C[2,3]*1000:+.1f}')
    print(f'tag in the FLANGE frame (mm):  '
          f'x={p_f[0]*1000:+.1f}  y={p_f[1]*1000:+.1f}  z={p_f[2]*1000:+.1f}')

    if os.path.isfile(args.out):
        bak = args.out.replace('.json', f'.{date.today().isoformat()}.bak.json')
        shutil.copy2(args.out, bak)
        print(f'\nprevious calibration backed up to {os.path.basename(bak)}')

    doc = {
        'camera_to_base': C.tolist(),
        'mount': 'fixed stand; the camera does not move with the arm',
        'compose_note': 'camera_to_base is absolute -- do NOT compose it with '
                        'joint1. See docs/camera_mount.md.',
        'solver': f'position-only least squares over {len(kept)} captures',
        'captures_used': len(kept),
        'tag_id': args.tag_id,
        'tag_mm': args.tag_mm,
        'fit_residual_mm': {'mean': round(float(err.mean()), 2),
                            'worst': round(float(err.max()), 2)},
        'jackknife_spread_mm': round(jack_mm, 2),
        'accuracy_note': 'Fitted from tag POSITION only, orientation '
                         'discarded. Confirm with verify_calibration.py on '
                         'fresh poses -- the in-sample residual above is not '
                         'a substitute for that.',
        'known_limitation': 'Marker ORIENTATION could not be reconciled with '
                            'FK below ~26deg, cause not isolated between the '
                            'FK orientation model and the weak observability '
                            'of out-of-plane rotation for a near-frontal '
                            'planar marker. Do not derive tool ORIENTATION '
                            'from this calibration without re-checking.',
    }
    with open(args.out, 'w') as f:
        json.dump(doc, f, indent=2)
    print(f'wrote {args.out}')
    print('\nNow verify on FRESH poses, which is the only number that counts:')
    print(f'  ./scripts/verify_calibration.py --tag-id {args.tag_id} '
          f'--tag-mm {args.tag_mm}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
