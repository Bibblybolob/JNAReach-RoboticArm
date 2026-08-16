#!/usr/bin/env bash
# Everything that happens AFTER the detector finishes training:
# train the reader, export both engines, score the new pipeline on the same
# held-out splits the baseline used, and print the before/after.
#
#   ./scripts/finish_and_compare.sh
#
# Split out of train_buttons_all.sh so it can be re-run on its own when the
# detector is already trained -- retraining an 11-hour model to regenerate a
# comparison table is not an acceptable way to fix a typo in the table.
set -u
cd "$(dirname "$0")/.." || exit 1

EV=eval_results
mkdir -p "$EV"

echo "=== stage B: legend reader    $(date)"
./scripts/train_button_reader.py > /tmp/read_train.log 2>&1
echo "    exit $?"

echo "=== export TensorRT           $(date)"
./scripts/export_button_engine.py > /tmp/export.log 2>&1
echo "    exit $?"

# Score on BOTH splits, for opposite reasons.
#
#   sunmoon/test  the diverse set. Out of domain for the old model, which is
#                 the honest picture of what the arm faced before -- but it
#                 flatters the new one, whose training data came from the same
#                 collection.
#   entc/test     the old model's OWN training domain, i.e. its best case, and
#                 held out for both. This is the conservative comparison and
#                 the one to believe if the two disagree.
#
# Reporting only the first would be the flattering mistake compare_buttons.py
# warns about, so both are always produced.
echo "=== eval: sunmoon/test        $(date)"
./scripts/eval_buttons.py --src ~/datasets/sunmoon-buttons --split test \
    --label 'AFTER: two-stage yolo11s + reader' \
    --json "$EV/after_sunmoon.json" > "$EV/after_sunmoon.txt" 2>&1
echo "    exit $?"

echo "=== eval: entc/test           $(date)"
./scripts/eval_buttons.py --src ~/datasets/entc --split test \
    --label 'AFTER: two-stage yolo11s + reader' \
    --json "$EV/after_entc.json" > "$EV/after_entc.txt" 2>&1
echo "    exit $?"

for s in sunmoon entc; do
    if [ -f "$EV/before_$s.json" ] && [ -f "$EV/after_$s.json" ]; then
        echo
        echo "##### $s #####"
        ./scripts/compare_buttons.py "$EV/before_$s.json" "$EV/after_$s.json" \
            | tee "$EV/comparison_$s.txt"
    fi
done

echo
echo "=== done                      $(date)"
echo "tables: $EV/comparison_*.txt"
