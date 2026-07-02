#!/usr/bin/env bash
set -euo pipefail

# XWOD weather downstream pipeline:
# 1. Dehaze prefixed XWOD test images through the dehazing model.
# 2. Build prefix-only YOLO validation datasets for original and dehazed images.
# 3. Validate YOLO11 models from smallest to largest on both sets.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PREFIX="${1:-fog_}"
CONFIG="${2:-./config/Dataset_NHHAZE_8flow_2block_paper_loss.yaml}"
CHECKPOINT="${3:-}"
REFERENCE="${4:-./data/Dataset_NHHAZE/trainB/01_GT.png}"

XWOD_ROOT="${XWOD_ROOT:-./data/XWOD}"
DEHAZED_ROOT="${DEHAZED_ROOT:-./output/xwod_dehazed}"
DATASET_ROOT="${DATASET_ROOT:-./output/xwod_fog_yolo}"
RESULT_ROOT="${RESULT_ROOT:-./output/xwod_yolo11_comparison}"
IMG_SIZE="${IMG_SIZE:-256}"
LIMIT="${LIMIT:-}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "Missing checkpoint path."
  echo "Usage: bash $0 <prefix> <config> <checkpoint> <reference>"
  exit 1
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

SAFE_PREFIX="${PREFIX%_}"
SAFE_PREFIX="${SAFE_PREFIX//[^A-Za-z0-9_-]/_}"

echo "=== XWOD downstream experiment ==="
echo "config    : $CONFIG"
echo "checkpoint: $CHECKPOINT"
echo "reference : $REFERENCE"
echo "prefix    : $PREFIX"
echo "xwod      : $XWOD_ROOT"
echo "dehazed   : $DEHAZED_ROOT"
echo "datasets  : $DATASET_ROOT"
echo "results   : $RESULT_ROOT"
echo

echo "=== Step 1: dehaze XWOD ${PREFIX}test images ==="
python experiments/xwod_yolo11/dehaze_xwod.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --xwod-root "$XWOD_ROOT" \
  --split test \
  --prefix "$PREFIX" \
  --reference "$REFERENCE" \
  --output-root "$DEHAZED_ROOT" \
  "${LIMIT_ARGS[@]}"

echo
echo "=== Step 2: build prefix-only YOLO datasets ==="
python experiments/xwod_yolo11/prepare_fog_subset.py \
  --xwod-root "$XWOD_ROOT" \
  --image-root "$XWOD_ROOT/test/images" \
  --split test \
  --prefix "$PREFIX" \
  --output-root "$DATASET_ROOT/$SAFE_PREFIX/original" \
  "${LIMIT_ARGS[@]}"

python experiments/xwod_yolo11/prepare_fog_subset.py \
  --xwod-root "$XWOD_ROOT" \
  --image-root "$DEHAZED_ROOT/test/images" \
  --split test \
  --prefix "$PREFIX" \
  --output-root "$DATASET_ROOT/$SAFE_PREFIX/dehazed" \
  "${LIMIT_ARGS[@]}"

echo
echo "=== Step 3: YOLO11 original vs dehazed validation ==="
for MODEL in yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt; do
  MODEL_NAME="${MODEL%.pt}"
  echo
  echo "--- $MODEL original ---"
  python experiments/xwod_yolo11/eval_yolo11_xwod.py \
    --model "$MODEL" \
    --data "$DATASET_ROOT/$SAFE_PREFIX/original/data.yaml" \
    --imgsz "$IMG_SIZE" \
    --output-root "$RESULT_ROOT/$SAFE_PREFIX/$MODEL_NAME" \
    --name original

  echo
  echo "--- $MODEL dehazed ---"
  python experiments/xwod_yolo11/eval_yolo11_xwod.py \
    --model "$MODEL" \
    --data "$DATASET_ROOT/$SAFE_PREFIX/dehazed/data.yaml" \
    --imgsz "$IMG_SIZE" \
    --output-root "$RESULT_ROOT/$SAFE_PREFIX/$MODEL_NAME" \
    --name dehazed
done

echo
echo "=== Step 4: summarize results ==="
python experiments/xwod_yolo11/summarize_yolo_results.py \
  --results-root "$RESULT_ROOT" \
  --output-dir "$RESULT_ROOT/$SAFE_PREFIX/summary_plots"

echo
echo "Done. Results are under:"
echo "$RESULT_ROOT"
echo "Summary plots:"
echo "$RESULT_ROOT/$SAFE_PREFIX/summary_plots"
