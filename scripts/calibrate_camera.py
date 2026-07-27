#!/usr/bin/env python3
"""
Measure the webcam's intrinsics with a checkerboard.

camera_node currently publishes a CameraInfo stub -- width and height and
nothing else. Without a real k matrix a pixel cannot be turned into a ray, so
no depth estimate and no 3D targeting is possible. This produces that matrix.

WHAT YOU NEED
  A checkerboard. Print one (search "OpenCV checkerboard 9x6 pdf"), tape it to
  something rigid and FLAT -- a clipboard or a hardback book. A sagging sheet
  of paper will quietly ruin the calibration.

  Count INNER corners, not squares. A board with 10x7 squares has 9x6 inner
  corners, and 9x6 is what you pass here.

USAGE
  Live from the Pi stream (the same source camera_node reads):
      python3 scripts/calibrate_camera.py --stream http://192.168.0.15:8080/?action=stream

  From a local USB webcam:
      python3 scripts/calibrate_camera.py --device 0

  From a folder of stills you already took:
      python3 scripts/calibrate_camera.py --images ./calib/*.jpg

CAPTURING GOOD FRAMES
  Press SPACE to capture when the board is detected (corners drawn), q to
  finish. Aim for 15-20 captures and vary them:
    - board near and far
    - board in each corner of the frame, not just the middle
    - tilted maybe 30-45 degrees in different directions
  All frames taken head-on from the same distance will produce a confident,
  wrong answer. The tilted views are what actually constrain the focal length.

OUTPUT
  Writes camera_intrinsics.yaml and prints the values to paste into
  camera_node.py. A reprojection error under ~0.5 px is good; over ~1.0 px
  means recapture with more variety.
"""

import argparse
import glob
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit('opencv not installed: pip install opencv-python')


def collect_from_images(paths, pattern):
    """Find checkerboard corners in a set of image files."""
    frames = []
    for p in sorted(paths):
        img = cv2.imread(p)
        if img is None:
            print(f'  skip (unreadable): {p}')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        print(f'  {"OK  " if found else "miss"} {p}')
        if found:
            frames.append((gray.shape[::-1], corners, gray))
    return frames


def collect_interactive(cap, pattern):
    """Live capture loop: SPACE grabs a frame when the board is visible."""
    frames = []
    print('\nSPACE = capture (only works when corners are drawn), q = done\n')
    while True:
        ok, img = cap.read()
        if not ok:
            print('camera read failed')
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        display = img.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern, corners, found)
        cv2.putText(
            display, f'captured: {len(frames)}   {"BOARD OK" if found else "no board"}',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow('calibration', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and found:
            frames.append((gray.shape[::-1], corners, gray))
            print(f'  captured {len(frames)}')

    cv2.destroyAllWindows()
    return frames


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--stream', help='MJPEG URL, e.g. http://IP:8080/?action=stream')
    src.add_argument('--device', type=int, help='local camera index, e.g. 0')
    src.add_argument('--images', nargs='+', help='image files or a glob')
    ap.add_argument('--cols', type=int, default=9, help='INNER corners across')
    ap.add_argument('--rows', type=int, default=6, help='INNER corners down')
    ap.add_argument('--square', type=float, default=0.025,
                    help='square size in metres (default 25mm)')
    ap.add_argument('--out', default='camera_intrinsics.yaml')
    args = ap.parse_args()

    pattern = (args.cols, args.rows)

    # Object points: the board's corners in its own frame, z=0, scaled to the
    # real square size so translation comes out in metres.
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square

    if args.images:
        paths = []
        for pat in args.images:
            paths.extend(glob.glob(pat))
        if not paths:
            sys.exit('no images matched')
        print(f'Scanning {len(paths)} images for a {args.cols}x{args.rows} board:')
        frames = collect_from_images(paths, pattern)
    else:
        source = args.stream if args.stream else args.device
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f'could not open {source}')
        frames = collect_interactive(cap, pattern)
        cap.release()

    if len(frames) < 5:
        sys.exit(f'\nOnly {len(frames)} usable frames -- need at least 5, '
                 f'and 15+ for a trustworthy result.')

    print(f'\nCalibrating from {len(frames)} frames...')
    size = frames[0][0]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints, imgpoints = [], []
    for _, corners, gray in frames:
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(refined)

    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, size, None, None)

    # Mean reprojection error is the honest quality number; RMS can look fine
    # while individual frames are bad.
    total = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], k, dist)
        total += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    mean_err = total / len(objpoints)

    print(f'\nRMS: {rms:.4f}')
    print(f'Mean reprojection error: {mean_err:.4f} px', end='  ')
    if mean_err < 0.5:
        print('(good)')
    elif mean_err < 1.0:
        print('(acceptable)')
    else:
        print('(POOR -- recapture with more angles and distances)')

    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    print(f'\nfx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}')
    print(f'resolution: {size[0]}x{size[1]}')

    with open(args.out, 'w') as f:
        f.write('# Generated by scripts/calibrate_camera.py\n')
        f.write('# These are valid ONLY for the resolution below. If you change\n')
        f.write('# the camera resolution, recalibrate or scale fx/fy/cx/cy.\n')
        f.write(f'image_width: {size[0]}\n')
        f.write(f'image_height: {size[1]}\n')
        f.write('camera_matrix:\n')
        f.write(f'  data: [{", ".join(f"{v:.6f}" for v in k.flatten())}]\n')
        f.write('distortion_coefficients:\n')
        f.write(f'  data: [{", ".join(f"{v:.6f}" for v in dist.flatten())}]\n')
        f.write(f'reprojection_error: {mean_err:.4f}\n')

    print(f'\nWrote {args.out}')
    print('\nLoad it in camera_node.py so /camera/camera_info carries a real k')
    print('matrix -- hand_tracker_node stays in pixel-only mode until it does.')


if __name__ == '__main__':
    main()
