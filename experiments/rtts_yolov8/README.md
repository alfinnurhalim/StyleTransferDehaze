# RTTS YOLOv8 Downstream Evaluation

RTTS uses VOC XML annotations. The evaluator converts RTTS annotations to COCO format, maps COCO YOLOv8 predictions to RTTS classes, and reports COCO mAP metrics.

Run a quick smoke test:

```bash
LIMIT=5 YOLO_MODELS="yolov8n.pt" bash experiments/rtts_yolov8/run_rtts_yolov8_comparison.sh
```

Run the default comparison with YOLOv8 n/s/m:

```bash
bash experiments/rtts_yolov8/run_rtts_yolov8_comparison.sh \
  ./config/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001.yaml \
  ./output/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001_120_100_8_3/model_save/epoch_110_val.pth.tar \
  ./data/Dataset_NHHAZE/trainB/01_GT.png
```

Outputs are written to:

```text
output/rtts_dehazed
output/rtts_yolov8_comparison
```
