#!/usr/bin/env python3
"""Train a YOLOv11n model on elevator button images from Roboflow.

Usage:
    pip install roboflow ultralytics
    python3 scripts/train_button_detector.py

The script downloads a dataset from Roboflow, trains YOLOv11n, and exports
the result as elevator_buttons.pt — the file button_detector_node expects.

Set ROBOFLOW_API_KEY in your environment, or the script will prompt for it.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default='',
                        help='Roboflow workspace name (prompted if empty)')
    parser.add_argument('--project', default='',
                        help='Roboflow project name (prompted if empty)')
    parser.add_argument('--version', type=int, default=0,
                        help='Dataset version number (prompted if 0)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='0',
                        help='CUDA device index or "cpu"')
    parser.add_argument('--output', default='elevator_buttons.pt')
    parser.add_argument('--data-yaml', default='',
                        help='Skip the Roboflow download and train directly '
                             'on an existing dataset/data.yaml')
    args = parser.parse_args()

    if args.data_yaml:
        data_yaml = args.data_yaml
        print(f'Using existing dataset: {data_yaml}')
    else:
        api_key = os.environ.get('ROBOFLOW_API_KEY', '')
        if not api_key:
            api_key = input('Roboflow API key: ').strip()
            if not api_key:
                print('No API key provided.', file=sys.stderr)
                sys.exit(1)

        workspace = args.workspace or input('Roboflow workspace: ').strip()
        project = args.project or input('Roboflow project: ').strip()
        version = args.version or int(input('Dataset version: ').strip())

        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download('yolov11')
        data_yaml = os.path.join(dataset.location, 'data.yaml')

        print(f'\nDataset downloaded to {dataset.location}')
        print(f'data.yaml: {data_yaml}')

    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project='runs/button_detect',
        name='train',
    )

    # Read the save dir back from the trainer rather than assuming
    # 'runs/button_detect/train' -- Ultralytics nests it under an extra
    # 'detect/' segment even with an explicit `project`, so a hardcoded path
    # missed the real weights on the first run this was used.
    best = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
    if os.path.exists(best):
        import shutil
        shutil.copy2(best, args.output)
        print(f'\nTrained model saved to {args.output}')
        print(f'Copy it where button_detector_node can find it, or pass')
        print(f'  model_path:={os.path.abspath(args.output)}')
    else:
        print(f'Training finished but {best} not found — check runs/',
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
