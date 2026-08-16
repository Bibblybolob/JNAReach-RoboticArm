#!/usr/bin/env python3
"""Export the trained detector and reader to TensorRT engines for this board.

    ./scripts/export_button_engine.py                    # both stages
    ./scripts/export_button_engine.py --stage detect

TensorRT is not an optimisation here, it is the only working GPU path.
Measured 2026-08-14 and recorded in button_detector_node.py:

    elevator_buttons.pt      on cpu      535.9ms   works
    elevator_buttons.pt      on cuda          -    cuDNN
                                               CUDNN_STATUS_EXECUTION_FAILED_CUDART
    elevator_buttons.engine  (TRT)        18.7ms   works

Ultralytics' .pt CUDA path fails at kernel execution on this JetPack build
while TensorRT, which does not go through cuDNN, is fine. So a .pt is a
28x-slower CPU fallback, not an alternative.

**Export through Ultralytics, not trtexec.** Recorded 2026-08-11 in the
TensorRT note: a trtexec-built engine loses the Ultralytics metadata (class
names, imgsz, task) and the node then loads an engine that detects correctly
and cannot name anything.

**An engine is built for one board and does not travel.** It is tied to the
GPU architecture and the TensorRT version. That is why the node treats a
failed engine load as an expected case and falls back to the .pt on CPU
rather than crashing -- and why the engines are not committed.
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch

torch.backends.cudnn.enabled = False

from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export(weights: str, imgsz: int, half: bool, dest: str) -> bool:
    if not os.path.isfile(weights):
        print(f'  no weights at {weights} -- skipping')
        return False
    print(f'  exporting {weights} @ {imgsz} (half={half})')
    model = YOLO(weights)
    out = model.export(format='engine', imgsz=imgsz, half=half, device=0,
                       workspace=4)
    if dest and out and os.path.exists(str(out)):
        shutil.copy2(str(out), dest)
        print(f'  -> {dest}')
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('detect', 'read', 'both'),
                    default='both')
    ap.add_argument('--detect-weights', default=os.path.join(
        REPO, 'runs_buttons', 'detect_v1', 'weights', 'best.pt'))
    ap.add_argument('--read-weights', default=os.path.join(
        REPO, 'runs_buttons', 'read_v1', 'weights', 'best.pt'))
    ap.add_argument('--detect-imgsz', type=int, default=640)
    ap.add_argument('--read-imgsz', type=int, default=128)
    # FP16 by default: measured 1.71x on this board (33.3 -> 19.5ms) for no
    # accuracy change worth measuring on a detection task at this scale.
    ap.add_argument('--no-half', action='store_true')
    args = ap.parse_args()

    half = not args.no_half
    ok = True
    if args.stage in ('detect', 'both'):
        print('stage A (detect):')
        ok &= export(args.detect_weights, args.detect_imgsz, half,
                     os.path.join(REPO, 'button_detect.engine'))
    if args.stage in ('read', 'both'):
        print('stage B (read):')
        ok &= export(args.read_weights, args.read_imgsz, half,
                     os.path.join(REPO, 'button_read.engine'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
