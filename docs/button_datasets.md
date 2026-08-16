# Button recognition: the data, the classes, and why it is two stages

Rewritten 2026-08-16. The earlier version of this document described a flat
14-class detector; the section at the bottom keeps its measurements, because
they are what ruled that design out.

## The task, in priority order

Set by Jonathan on 2026-08-16, and the class list is built around it:

1. **the hall call -- up and down**
2. **floor numbers**
3. **help buttons**

The target is a **real elevator in the building**, not the printed panel on the
lab bench. That is what makes generalisation the requirement and public data
the primary training set rather than a warm-up for our own 60 images.

Timing: the arm reads the panel while **stationary**, and the whole press must
take under 5 seconds. So inference has a budget of hundreds of milliseconds,
not the 33ms a servo loop would demand.

## What was wrong before, measured

The 14 classes in the old `~/panel_dataset/data.yaml` were
`button-1..button-10`, `button-P`, `open`, `close`, `alarm`.

**There was no `up` and no `down` class.** `map_name()` in
`scripts/remap_dataset.py` dropped both on the floor. Counted over the Sun
Moon set on 2026-08-16, that discarded:

| dropped source class | instances | images |
|---|---|---|
| `up` | 393 | 349 |
| `down` | 335 | 308 |
| `U` (up rendered as a letter) | 21 | 21 |
| `D` / `DN` | 24 | 24 |

773 instances of the two highest-priority classes, thrown away at remap time.
This is why the shipped model was seen "locking onto `down` at 0.21
confidence" -- ENTC had that class, and our retrain deleted it.

The `U`/`D` variants were checked visually before being reclaimed, not
assumed: cropped and inspected 2026-08-16, they are unambiguously hall-call
buttons with a letter where another panel puts an arrow.

**And the floor range stopped at 10.** Against a real elevator a numbered
class list is unworkable in both directions at once: floor 14 gets called
background, while the top of whatever range you pick never has the examples to
be learned. Measured over 22,843 instances in 368 classes, support collapses
down the range -- floor 2 has 1200 examples, floor 33 has 55, and there is a
long tail of `B4`, `LG2`, `P4A` in single digits.

## The baseline: the old model is narrow, not bad

Measured 2026-08-16 with `scripts/eval_buttons.py`, which maps the old
17-class model's predictions through the same taxonomy as the ground truth so
the two are scored on identical terms. Both are held-out test splits.

| | `up` recall | `down` recall | `floor` recall |
|---|---|---|---|
| on **ENTC test** (its own training domain, 19 imgs) | **1.000** | 0.800 | 0.985 |
| on **Sun Moon test** (202 imgs) | **0.065** | **0.000** | **0.006** |

Precision on ENTC is 0.92-1.00 across `up`, `down`, `floor`, `open` and
`close`. On Sun Moon it predicts most classes not at all.

**So `elevator_buttons.pt` was never broken.** It is an excellent detector of
the panels in its own 393-image training set and effectively blind to any
other, which is a far more precise diagnosis than "the detector does not
work" and matches the 2026-08-14 finding that it scored zero on the lab's
printed panel. The same failure, twice, from the same cause.

Two consequences for how the retrain is judged:

- **The gain to look for is breadth and vocabulary, not accuracy.** On ENTC
  there is no headroom left on `up` -- 1.000 recall cannot be beaten. A new
  model that comes in slightly lower there while transforming the Sun Moon
  column is a better model, and reporting only the split that flatters it
  would be the mistake `compare_buttons.py` refuses to make silently.
- **Beware the small split.** ENTC test is 19 images with n=12 for `up`, so
  those figures are noisy, and `stop` and `other` do not occur in it at all.

## The design: find, then read

    Stage A  detect   WHERE is a button, and WHAT KIND   9 classes
    Stage B  classify given the crop, WHICH FLOOR        47 legends + reject

Stage B is the part that makes the floor range tractable. A detector spends
its capacity on localisation and classification jointly, over a 640px frame in
which a button is maybe 30px across. The classifier starts from a crop that is
already found, centred and scaled, and resamples it to 128px -- the same
photons, an order of magnitude more of the network looking at them, and a
class list that can cover 0-36 without the detector paying for it.

It is also what puts the priorities the right way up. `up` and `down` are 2 of
9 classes here, where a per-floor detector spent essentially none of its
capacity on them.

### Stage A -- the nine classes

Chosen from the measured counts, not decided in advance. Owned by
`src/mycobot_perception/mycobot_perception/button_classes.py`, which is the
single source of truth -- every `data.yaml` is **generated** from it.

| class | instances | note |
|---|---|---|
| `up` | 417 | priority 1. Includes `U`, `UP` |
| `down` | 359 | priority 1. Includes `D`, `DN` |
| `floor` | 13871 | every numbered/lettered storey; stage B reads which |
| `open` | 1091 | |
| `close` | 983 | |
| `help` | 1081 | priority 3: alarm + call + bell + emergency + intercom |
| `stop` | 88 | thin, and kept separate deliberately -- see below |
| `keyhole` | 931 | never press. Named rather than ignored |
| `other` | 2391 | a button whose legend cannot be read |

Three of those need justifying.

**`help` is one class, not four.** `alarm`, `call`, `bell` and `emergency` are
one button to a robot, and separately none of them has the support to be worth
splitting.

**`stop` stays separate at 88 examples**, below the threshold this document
otherwise applies. A thin class normally fires rarely and wrongly, which is
worse than an absent one. The asymmetry is that this is a class the arm
AVOIDS, not one it targets: a false `stop` costs a skipped button, where
folding it into `help` would make "press for help" reach for the emergency
stop.

**`other` exists so the arm can say "I cannot read this".** `empty`, `blur`
and `unknown` become a real class rather than being discarded, and this is the
safety-relevant choice in the whole pipeline. The failure that matters is
pressing the WRONG floor. A dropped class cannot express uncertainty -- it
produces no detection, which reads identically to no button being there.

### Stage B -- 47 legends plus a reject

Floors `0`-`36`, plus `B`, `B1`, `B2`, `B3`, `G`, `L`, `LG`, `M`, `CH`, `-1`,
plus `unreadable`. 14,768 crops.

The cut is `--min-reader-support 40`. 197 rarer legends fall below it (`R`(38),
`SG`(30), `P1`(26), `P`(21), `E`(23) ...). A button carrying one of those still
**detects** as a floor button; the reader simply returns low confidence and the
arm declines to press it. That is the intended behaviour, not a gap.

Note that `P` at 21 examples is one of the casualties, and the lab's printed
panel has a P. Against a real elevator that is the right trade; if the printed
panel matters again, P is the class our own labelling most needs to cover.

## Two augmentations that must stay off

**`fliplr=0.0` in stage A, and the reason is not the digits.** Stage A never
reads a number, so mirrored numerals are not the issue. The issue is that
`open` and `close` are mirror images OF EACH OTHER -- `|<->|` against
`->||<-`. A horizontal flip does not corrupt those classes, it SWAPS them: a
consistent wrong label on 2,074 instances.

**`flipud=0.0`**, one level up: a vertical flip turns `up` into `down`.

Rotation is capped at 12 degrees in stage A and 15 in stage B for the related
reason -- `6` and `9` are a 180-degree rotation apart, and pressing 9 for 6 is
a real error with no cue that it happened.

## The imbalance, and the oversampling

`floor` has 13,871 instances against `up`'s 417 -- 33:1 on the class that
matters most. YOLO's loss does not reweight and Ultralytics has no per-class
sampler, so the lever that works is showing the rare images more often:
`--oversample 3` writes three hardlinked copies of every **train** image
containing an up or a down. Train only -- oversampling the val split would
make the headline mAP a report on the duplication.

Duplicates are not augmentations; they differ only through what the trainer
applies on top. Worth it against being outvoted 33:1 in every batch, but the
reason `--oversample 8` is not better: past a point it overfits the same 349
images harder.

## Running it

```bash
./scripts/build_button_dataset.py --src ~/datasets/sunmoon-buttons --dry-run
./scripts/build_button_dataset.py --src ~/datasets/sunmoon-buttons \
    --out ~/datasets/buttons
./scripts/train_buttons_all.sh          # both stages, then TensorRT export
```

Always `--dry-run` first: the per-class counts decide whether a source is
worth training on, and you want that before spending a night on it.

The datasets live under `~/datasets/`, outside the repo -- gigabytes of other
people's images, reproducible from the commands here. Getting the source needs
a Roboflow API key (the Universe page returns 403 to a script):

```bash
export ROBOFLOW_API_KEY=<key>          # roboflow.com -> Settings -> API Keys
python3 -c "
from roboflow import Roboflow
import os
Roboflow(api_key=os.environ['ROBOFLOW_API_KEY']) \
  .workspace('sun-moon-university').project('elevator-button-recognition') \
  .version(1).download('yolov11', location='~/datasets/sunmoon-buttons')"
```

`--src` is repeatable, and adding sources is the highest-value work left --
see below.

## Model size: the data binds before the budget does

Under a 5-second press budget, inference is nearly free -- a yolo11m or l
would fit easily. Two other things bind first.

**The images are 416x416 on disk.** Every one of the 2,019 Sun Moon images,
checked 2026-08-16. Training at 960 feeds the network a 2.3x upscale of an
image with no detail above 416 to recover, at 2.2x the cost per epoch. 640 is
a mild upsample, which YOLO does benefit from for small objects, and it
matches the D405's 640x480 delivery so the engine is built at the resolution
it will actually see.

**Training time is not free, measured on this Orin.** yolo11n@640 batch 8 runs
1.8 it/s over 245 iterations -- 140s an epoch. yolo11s is ~223s. yolo11m is
roughly 5x the n: 58 hours for 300 epochs, to overfit a 2,000-image set
harder.

So: **yolo11s @ 640, 150 epochs, ~11 hours.** The step up from n is justified
by this set being 5x the 393 images the old ENTC runs used; the step to m is
not justified by anything yet.

**GPU training works on this board**, verified 2026-08-16 with a full epoch
before the long run was committed. That is worth stating because it contradicts
what the *inference* path does: `.pt` on CUDA fails at kernel execution with
`CUDNN_STATUS_EXECUTION_FAILED_CUDART`, which is why the node runs TensorRT.
Training with `torch.backends.cudnn.enabled = False` is fine. Do not
generalise the inference failure into "the GPU is broken".

## What to do next, in order of value

1. **More sources.** This is the highest-value work and the repeat of this
   document's own earlier conclusion: more images beat a bigger model here.
   `--src` is repeatable and the builder namespaces filenames per source so
   two Roboflow exports cannot collide. ENTC is already known to carry up/down.
2. **Our own images of a real panel**, mixed IN rather than fine-tuned on.
   A sequential fine-tune on 60 images of the printed keypad -- which contains
   no up, no down and no help button -- would teach the model that all three
   are background, destroying priority 1 to serve priority 2.
3. Only then, a bigger backbone.

## Sources

| source | verdict |
|---|---|
| [Sun Moon University][sm] (2.02k imgs, 368 classes, CC BY 4.0) | **used** -- the only one with numbered floors past 3, and it carries up/down |
| [ENTC][entc] (393 imgs, 17 classes) | what the original shipped model trained on. Has up/down; worth adding as a second `--src` |
| [CUHK Button Dataset][cuhk] (3,718 imgs, 35,100 labels) | larger, but the SharePoint download 404s -- link rot on a 2021 paper |

No Kaggle or HuggingFace mirror of any of these exists; they are Roboflow-only,
which is why an API key is unavoidable.

## Prior negative results, kept so they are not retried

**A bigger model on a small set lost.** `yolo11s @ imgsz=960` on the 393-image
ENTC set (93 epochs) came in *worse* than the `yolo11n @ 640` baseline: mAP50
0.9252 vs 0.9292, mAP50-95 0.5366 vs 0.5446. Read at the time as
capacity-limited overfitting; the 416px source resolution is at least as
likely a cause, and both point the same way.

**The old 14-class remap, for the record.** Remapping Sun Moon onto
`button-1..10 + P + open/close/alarm` gave 1,650 images with `button-P` at 21
examples and nothing above `button-10` at all. `scripts/remap_dataset.py` is
kept only to reproduce those numbers; `build_button_dataset.py` replaces it.

**The domain gap on the printed panel is real and is a separate problem.**
`elevator_buttons.pt` scored zero on the lab's inkjet panel at every scale and
threshold, and YOLO-World zero-shot found nothing across three prompt sets
(2026-08-14). `keypad_finder.py` solves that geometrically and still does.
Nothing here supersedes it -- this pipeline targets real elevators, which is a
different problem from an inkjet print taped to a board.

[sm]: https://universe.roboflow.com/sun-moon-university/elevator-button-recognition
[entc]: https://universe.roboflow.com/murge-data/entc-elevator-button-detection
[cuhk]: https://github.com/zhudelong/elevator_button_recognition
