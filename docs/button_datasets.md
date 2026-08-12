# Button training data: what we use, and why the shipped model failed

## The problem, measured

`elevator_buttons.pt` is trained on the ENTC Roboflow set. Our panel has
buttons 1-10 plus P; ENTC's 17 classes contain **no button-4..button-10 and no
P**. Running `scripts/remap_dataset.py` against it onto our 14 classes shows
exactly what that means:

| class | examples in ENTC |
|---|---|
| button-1 / 2 / 3 | 440 / 445 / 419 |
| **button-4 … button-10** | **0** |
| **button-P** | **0** |
| open / close / alarm | 351 / 348 / 350 |

Eight of fourteen classes have zero training examples. That is the whole of
the detection failure: watched live on 2026-08-12 the model called button 10
`floor-ground`, button 6 `close`, and locked onto `down` -- a class this panel
does not have -- at 0.21 confidence while the target teleported around the
frame. No threshold, gain or lead-time setting fixes a class that was never
trained.

## What we use instead

[Sun Moon University, Elevator Button Recognition][sm] -- 2.02k images, 368
classes including floor numbers 0-50+. CC BY 4.0.

Remapped onto our 14 classes it gives **1650 usable images** (1165 train / 324
valid / 161 test):

| class | examples | |
|---|---|---|
| button-1 / 2 / 3 | 1140 / 1200 / 1090 | |
| button-4 / 5 / 6 / 7 | 894 / 798 / 665 / 593 | |
| button-8 / 9 / 10 | 564 / 501 / 395 | |
| **button-P** | **21** | thin -- see below |
| open / close / alarm | 1072 / 971 / 1081 | |

`button-P` is the weak class. 21 examples will not be learned reliably, and
our panel has a P, so that is the one our own labelled images most need to
cover.

## Reproducing it

Roboflow needs an API key -- the Universe page returns 403 to a script and the
API says so plainly. Get one from roboflow.com -> Settings -> API Keys.

```bash
export ROBOFLOW_API_KEY=<key>
python3 -c "
from roboflow import Roboflow
import os
Roboflow(api_key=os.environ['ROBOFLOW_API_KEY']) \
  .workspace('sun-moon-university').project('elevator-button-recognition') \
  .version(1).download('yolov11', location='~/datasets/sunmoon-buttons')"

./scripts/remap_dataset.py --src ~/datasets/sunmoon-buttons --dry-run
./scripts/remap_dataset.py --src ~/datasets/sunmoon-buttons \
    --out ~/datasets/public_panel
```

Always `--dry-run` first: the per-class counts are what decide whether a
source is worth training on, and a class with a handful of examples is worse
than an absent one -- the model fires on it rarely and wrongly.

The datasets live under `~/datasets/`, deliberately outside the repo. They are
gigabytes of other people's images and are reproducible from the commands
above.

## Sources considered

| source | verdict |
|---|---|
| [Sun Moon University][sm] (2.02k imgs, 368 classes, CC BY 4.0) | **used** -- the only one with numbered floors 1-10 |
| [ENTC][entc] (393 imgs, 17 classes) | what the shipped model was trained on; no buttons 4-10 |
| [CUHK Button Dataset][cuhk] (3,718 imgs, 35,100 labels) | larger, but the SharePoint download 404s -- link rot on a 2021 paper |

No Kaggle or HuggingFace mirror of any of these exists; they are Roboflow-only,
which is why an API key is unavoidable.

## Two stages, not one

Public data fixes **class coverage**. It does not fix **domain gap** -- these
are still other people's panels, lighting and cameras.

1. Train on the remapped public set for vocabulary and volume:
   `./scripts/finetune_panel.py --data ~/datasets/public_panel/data.yaml --name public_pretrain`
2. Fine-tune that on our own images (`~/panel_dataset`, captured with
   `scripts/capture_panel_dataset.py`) for domain.

Stage 2 is why our own labelling still matters -- but 60 images is enough on
top of stage 1, where it would not have been on its own.

**Class order is a contract.** `~/panel_dataset/data.yaml`,
`~/datasets/public_panel/data.yaml` and `TARGET` in `scripts/remap_dataset.py`
must list the same 14 names in the same order, or a model trained on one and
fine-tuned on the other learns shifted ids and every label is wrong.

## Prior negative result, so it is not retried

A `yolo11s @ imgsz=960` run on the ENTC set (93 epochs) came in *worse* than
the `yolo11n @ 640` baseline: mAP50 0.9252 vs 0.9292, mAP50-95 0.5366 vs
0.5446. At 393 images the model is data-limited, not capacity-limited, so a
larger backbone overfits sooner and the extra resolution has nothing to
resolve. More images beat a bigger model here.

[sm]: https://universe.roboflow.com/sun-moon-university/elevator-button-recognition
[entc]: https://universe.roboflow.com/murge-data/entc-elevator-button-detection
[cuhk]: https://github.com/zhudelong/elevator_button_recognition
