# XWOD Fog YOLO11 Downstream Test

XWOD is already in YOLO format:

- `data/XWOD/test/images/fog_test_*.jpg`
- `data/XWOD/test/labels/fog_test_*.txt`
- `data/XWOD/data.yaml`

Run the fog-only downstream comparison:

```bash
bash experiments/xwod_yolo11/run_fog_all_yolo.sh \
  fog_ \
  ./config/Dataset_NHHAZE_8flow_2block_paper_loss.yaml \
  ./output/Dataset_NHHAZE_8flow_2block_paper_loss_100_8_2/model_save/chpt.tar \
  ./data/Dataset_NHHAZE/trainB/01_GT.png
```

The dehazer internally resizes each XWOD image to the configured model size, for example `256x256`, then resizes the dehazed output back to the original resolution before YOLO validation. YOLO validation defaults to `imgsz=256` for this experiment.
The pipeline calls Ultralytics through `python`, not the standalone `yolo` executable, so it uses the same environment as the dehazing step.

For a quick smoke test:

```bash
LIMIT=5 bash experiments/xwod_yolo11/run_fog_all_yolo.sh \
  fog_ \
  ./config/Dataset_NHHAZE_8flow_2block_paper_loss.yaml \
  ./output/Dataset_NHHAZE_8flow_2block_paper_loss_100_8_2/model_save/chpt.tar \
  ./data/Dataset_NHHAZE/trainB/01_GT.png
```

For heavy-rain XWOD test images, use:

```bash
bash experiments/xwod_yolo11/run_fog_all_yolo.sh \
  heavy_rain_ \
  ./config/Dataset_NHHAZE_8flow_2block_paper_loss.yaml \
  ./output/Dataset_NHHAZE_8flow_2block_paper_loss_100_8_2/model_save/chpt.tar \
  ./data/Dataset_NHHAZE/trainB/01_GT.png
```

Each full run also writes summary CSVs, plots, and `report.md` to:

```text
output/xwod_yolo11_comparison/<prefix>/summary_plots
```
