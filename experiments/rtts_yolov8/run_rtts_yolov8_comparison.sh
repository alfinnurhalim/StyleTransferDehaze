#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"

CONFIG="${1:-./config/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001.yaml}"
CHECKPOINT="${2:-./output/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001_120_100_8_3/model_save/epoch_110_val.pth.tar}"
REFERENCE="${3:-./data/Dataset_NHHAZE/trainB/01_GT.png}"

RTTS_ROOT="${RTTS_ROOT:-./data/RTTS}"
DEHAZED_ROOT="${DEHAZED_ROOT:-./output/rtts_dehazed}"
RESULT_ROOT="${RESULT_ROOT:-./output/rtts_yolov8_comparison}"
SPLIT="${SPLIT:-test}"
IMG_SIZE="${IMG_SIZE:-250}"
DEVICE="${DEVICE:-0}"
LIMIT="${LIMIT:-}"
YOLO_MODELS="${YOLO_MODELS:-yolov8n.pt yolov8s.pt yolov8m.pt}"
FORCE_DEHAZE="${FORCE_DEHAZE:-0}"

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

FORCE_ARGS=()
if [[ "$FORCE_DEHAZE" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

echo "=== RTTS YOLOv8 downstream experiment ==="
echo "split     : $SPLIT"
echo "config    : $CONFIG"
echo "checkpoint: $CHECKPOINT"
echo "reference : $REFERENCE"
echo "rtts      : $RTTS_ROOT"
echo "dehazed   : $DEHAZED_ROOT"
echo "results   : $RESULT_ROOT"
echo "models    : $YOLO_MODELS"
echo "device    : $DEVICE"
echo

echo "=== Step 1: dehaze RTTS ==="
python experiments/rtts_yolov8/dehaze_rtts.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --rtts-root "$RTTS_ROOT" \
  --split "$SPLIT" \
  --reference "$REFERENCE" \
  --output-root "$DEHAZED_ROOT" \
  "${LIMIT_ARGS[@]}" \
  "${FORCE_ARGS[@]}"

echo
echo "=== Step 2: YOLOv8 original vs dehazed ==="
for MODEL in $YOLO_MODELS; do
  MODEL_NAME="${MODEL%.pt}"
  echo
  echo "--- $MODEL original ---"
  python experiments/rtts_yolov8/eval_yolov8_rtts.py \
    --rtts-root "$RTTS_ROOT" \
    --images-root "$RTTS_ROOT/JPEGImages" \
    --split "$SPLIT" \
    --model "$MODEL" \
    --imgsz "$IMG_SIZE" \
    --device "$DEVICE" \
    --output-dir "$RESULT_ROOT/$MODEL_NAME/original" \
    "${LIMIT_ARGS[@]}"

  echo
  echo "--- $MODEL dehazed ---"
  python experiments/rtts_yolov8/eval_yolov8_rtts.py \
    --rtts-root "$RTTS_ROOT" \
    --images-root "$DEHAZED_ROOT/JPEGImages" \
    --split "$SPLIT" \
    --model "$MODEL" \
    --imgsz "$IMG_SIZE" \
    --device "$DEVICE" \
    --output-dir "$RESULT_ROOT/$MODEL_NAME/dehazed" \
    "${LIMIT_ARGS[@]}"
done

echo
echo "Done. Results are under:"
echo "$RESULT_ROOT"
