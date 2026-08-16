#!/usr/bin/env bash
# Train both stages and export the engines, in sequence.
#
#   ./scripts/train_buttons_all.sh
#
# Sequence rather than parallel ON PURPOSE. The Orin Nano has one GPU and a
# 7GB pool shared with the CPU; the detector run alone sits at 2.2GB and two
# trainings that each fit alone will OOM together -- which surfaces as a
# CUDA error partway into whichever run happened to allocate second, hours
# in, with the other run's log looking perfectly healthy.
#
# Safe to run with nothing attached: it commands no motion and touches no
# serial port.
set -u

cd "$(dirname "$0")/.." || exit 1
LOG_DIR=${LOG_DIR:-/tmp}

echo "=== stage A: detector      $(date)"
./scripts/train_button_detector.py "$@" > "$LOG_DIR/detect_train.log" 2>&1
A=$?
echo "    exit $A"

echo "=== stage B: legend reader $(date)"
./scripts/train_button_reader.py > "$LOG_DIR/read_train.log" 2>&1
B=$?
echo "    exit $B"

# Export whatever trained. A failed stage leaves the other one usable: the
# node runs detect-only without a reader by design, and a reader with no
# detector is simply unused.
echo "=== export TensorRT        $(date)"
./scripts/export_button_engine.py > "$LOG_DIR/export.log" 2>&1
echo "    exit $?"

echo "=== done                   $(date)"
echo "detector: $LOG_DIR/detect_train.log"
echo "reader:   $LOG_DIR/read_train.log"
echo "export:   $LOG_DIR/export.log"
exit $(( A || B ))
