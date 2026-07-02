#!/usr/bin/env bash
set -euo pipefail

# One-condition ACDC downstream pipeline:
# 1. Dehaze ACDC images for one condition/split.
# 2. Evaluate YOLO11 models from smallest to largest on original and dehazed images.
#
# Example:
#   bash experiments/acdc_yolo11/run_one_condition_all_yolo.sh \
#     fog val \
#     ./config/Dataset_NHHAZE_8flow_2block_paper_loss.yaml \
#     ./output/<run>/model_save/<checkpoint>.pth.tar \
#     ./data/Dataset_NHHAZE/trainB/01_GT.png

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONDITION="${1:-fog}"
SPLIT="${2:-val}"
CONFIG="${3:-./config/Dataset_NHHAZE_8flow_2block_paper_loss.yaml}"
CHECKPOINT="${4:-}"
REFERENCE="${5:-./data/Dataset_NHHAZE/trainB/01_GT.png}"

ACDC_ROOT="${ACDC_ROOT:-./data/acdc}"
DEHAZED_ROOT="${DEHAZED_ROOT:-./output/acdc_dehazed}"
RESULT_ROOT="${RESULT_ROOT:-./output/acdc_yolo11_comparison}"
IMG_SIZE="${IMG_SIZE:-1280}"
LIMIT="${LIMIT:-}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "Missing checkpoint path."
  echo "Usage: bash $0 <condition> <split> <config> <checkpoint> <reference>"
  exit 1
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

echo "=== ACDC downstream experiment ==="
echo "condition : $CONDITION"
echo "split     : $SPLIT"
echo "config    : $CONFIG"
echo "checkpoint: $CHECKPOINT"
echo "reference : $REFERENCE"
echo "dehazed   : $DEHAZED_ROOT"
echo "results   : $RESULT_ROOT"
echo

echo "=== Step 1: dehaze ACDC ${CONDITION}/${SPLIT} ==="
python experiments/acdc_yolo11/dehaze_acdc.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --acdc-root "$ACDC_ROOT" \
  --condition "$CONDITION" \
  --split "$SPLIT" \
  --reference "$REFERENCE" \
  --output-root "$DEHAZED_ROOT" \
  "${LIMIT_ARGS[@]}"

echo
echo "=== Step 2: YOLO11 original vs dehazed ==="
for MODEL in yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt; do
  echo
  echo "--- $MODEL ---"
  python experiments/acdc_yolo11/run_downstream_comparison.py \
    --condition "$CONDITION" \
    --split "$SPLIT" \
    --acdc-root "$ACDC_ROOT" \
    --yolo-model "$MODEL" \
    --dehazed-root "$DEHAZED_ROOT" \
    --output-root "$RESULT_ROOT/$MODEL" \
    --imgsz "$IMG_SIZE" \
    "${LIMIT_ARGS[@]}"
done

echo
echo "Done. Results are under:"
echo "$RESULT_ROOT"
