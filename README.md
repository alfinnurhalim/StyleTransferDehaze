# StyleTransferDehaze

Glow-based image dehazing experiments with paired hazy/clear supervision, VGG perceptual loss, optional TV regularization, and downstream YOLO11 evaluation scripts.

This repository is currently organized for NH-HAZE, Dense-Haze, ACDC, and XWOD experiments. Large datasets, outputs, and pretrained weights are intentionally ignored by git.

## Setup

```bash
conda create -n dehazeflow python=3.9
conda activate dehazeflow
pip install -r requirements.txt
```

Put the VGG weights at:

```text
./vgg_normalised.pth
```

The training code expects this path unless you edit the config.

## Dataset Layout

Paired dehazing datasets should use this folder structure:

```text
data/Dataset_NHHAZE/
├── trainA/   # hazy inputs
├── trainB/   # clear GT/reference images
├── valA/     # optional validation hazy inputs
├── valB/     # optional validation clear GT/reference images
├── testA/    # test hazy inputs
└── testB/    # test clear GT/reference images
```

Dense-Haze can use the same structure. If `valA/valB` are missing, training will still run test evaluation, but validation checkpoints are only saved when a validation split exists.

## Main Configs

Current recommended 8-flow/3-block configs with paper-style loss and a small TV term:

```text
config/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001.yaml
config/Dataset_DENSEHAZE_8flow_3block_paper_loss_cosine_tv001.yaml
```

Both use:

```yaml
n_flow: 8
n_block: 3
batch_size: 1
lr_scheduler: cosine
lr: 0.0002
lr_min: 0.00002
recon_weight: 1.0
content_weight: 0.1
smooth_weight: 0.001
style_weight: 0
total_epoch: 70
```

## Training

NH-HAZE:

```bash
python train.py --config ./config/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001.yaml
```

Dense-Haze:

```bash
python train.py --config ./config/Dataset_DENSEHAZE_8flow_3block_paper_loss_cosine_tv001.yaml
```

Outputs are written under:

```text
output/<job_name>_<keep_ratio_percent>_<n_flow>_<n_block>/
```

During training, validation metrics are saved to `validation_metrics.csv`, test metrics to `test_metrics.csv`, and validation checkpoints are saved as:

```text
model_save/epoch_000_val.pth.tar
model_save/epoch_010_val.pth.tar
...
```

The trainer also saves:

```text
model_save/last.pth.tar
model_save/best_val_psnr.pth.tar
model_save/best_val_ssim.pth.tar
training_metrics.csv
best_metrics.json
plots/train_losses.png
plots/learning_rate.png
plots/psnr_curves.png
plots/ssim_curves.png
```

## Evaluate A Pretrained Model

Download the pretrained checkpoint from the shared Google Drive link, then place it somewhere local, for example:

```text
pretrained/nhhaze_8flow_3block_tv001.pth.tar
```

Run paired validation/test:

```bash
python validate.py \
  --config ./config/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001.yaml \
  --phase test \
  --checkpoint ./pretrained/nhhaze_8flow_3block_tv001.pth.tar
```

For Dense-Haze:

```bash
python validate.py \
  --config ./config/Dataset_DENSEHAZE_8flow_3block_paper_loss_cosine_tv001.yaml \
  --phase test \
  --checkpoint ./pretrained/densehaze_8flow_3block_tv001.pth.tar
```

The script reports PSNR/SSIM and writes manual metrics under the corresponding output folder.

## Resume Or Fine-Tune

Set `resume` in the config to a checkpoint path:

```yaml
resume: ./output/<run_name>/model_save/epoch_040_val.pth.tar
```

Then run training normally:

```bash
python train.py --config ./config/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001.yaml
```

Current resume behavior loads model and optimizer state. New validation checkpoints also store epoch, scheduler state, and metrics for cleaner future continuation.

## XWOD Downstream YOLO11 Evaluation

The XWOD scripts compare YOLO11 detection on original weather images versus dehazed outputs. They also generate summary CSVs, plots, and a Markdown report.

Example for fog:

```bash
bash experiments/xwod_yolo11/run_fog_all_yolo.sh \
  fog_ \
  ./config/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001.yaml \
  ./pretrained/nhhaze_8flow_3block_tv001.pth.tar \
  ./data/Dataset_NHHAZE/trainB/01_GT.png
```

Example for heavy rain:

```bash
bash experiments/xwod_yolo11/run_fog_all_yolo.sh \
  heavy_rain_ \
  ./config/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001.yaml \
  ./pretrained/nhhaze_8flow_3block_tv001.pth.tar \
  ./data/Dataset_NHHAZE/trainB/01_GT.png
```

Quick smoke test:

```bash
LIMIT=5 bash experiments/xwod_yolo11/run_fog_all_yolo.sh \
  fog_ \
  ./config/Dataset_NHHAZE_8flow_3block_paper_loss_cosine_tv001.yaml \
  ./pretrained/nhhaze_8flow_3block_tv001.pth.tar \
  ./data/Dataset_NHHAZE/trainB/01_GT.png
```

Summary outputs:

```text
output/xwod_yolo11_comparison/<prefix>/summary_plots/report.md
```

## Notes

- `data/`, `output/`, pretrained weights, checkpoints, archives, and generated metric files are ignored by git.
- Keep only configs, source code, and experiment scripts in the repository.
- Use paired mode for paired dehazing experiments: `train_pairing: paired`, `test_pairing: paired`.
