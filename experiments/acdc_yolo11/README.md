# ACDC YOLO11 Downstream Experiment

ACDC is organized as:

```text
data/acdc/
  images/{fog,night,rain,snow}/{train,val,test}/...
  labels/{fog,night,rain,snow}/instancesonly_*_{train,val,test}_*.json
```

Use `val` for downstream detection metrics because `test` has image info only.

## 1. Evaluate YOLO11 on original ACDC

```bash
python experiments/acdc_yolo11/eval_yolo11_acdc.py \
  --condition fog \
  --split val \
  --model yolo11n.pt \
  --output-dir ./output/acdc_yolo11/fog_val_original
```

## 2. Dehaze ACDC images

Use a trained dehazing checkpoint and one clear reference image for style conditioning:

```bash
python experiments/acdc_yolo11/dehaze_acdc.py \
  --config ./config/Dataset_NHHAZE_8flow_2block_paper_loss.yaml \
  --checkpoint ./output/<run>/model_save/<checkpoint>.pth.tar \
  --condition fog \
  --split val \
  --reference ./data/Dataset_NHHAZE/trainB/01_GT.png \
  --output-root ./output/acdc_dehazed
```

The script preserves relative paths under `output/acdc_dehazed`, for example:

```text
output/acdc_dehazed/fog/val/<sequence>/<image>.png
```

## 3. Compare YOLO11 original vs dehazed

```bash
python experiments/acdc_yolo11/run_downstream_comparison.py \
  --condition fog \
  --split val \
  --yolo-model yolo11n.pt \
  --dehazed-root ./output/acdc_dehazed \
  --output-root ./output/acdc_yolo11_comparison
```

Outputs include:

```text
detections_coco.json
summary.json
summary.csv
coco_metrics.json
```

COCO classes from YOLO11 are mapped to ACDC categories:

```text
person, bicycle, car, motorcycle, bus, train, truck
```

ACDC `rider` has no direct COCO class mapping and is not predicted by the default COCO YOLO11 model.
