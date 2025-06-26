
# 🌫️ DehazeFlow: Style-Aware Dehazing with Attention

This repository implements a **style transfer-based image dehazing** model using **Glow-style invertible networks**, **adaptive instance normalization (AdaIN)**, and **attention mechanisms** (spatial, channel, or hybrid). It supports both reference-based and reference-free dehazing with flexible encoder and loss designs.

---

## 🔧 Features
- ✨ Style-aware image restoration using style transfer
- 🌈 Glow-based generative architecture
- 🎯 Attention-enhanced style injection (spatial, channel, CBAM)
- 🎨 Supports VGG-based and learned style encoders
- 🧠 Perceptual + reconstruction losses
- 📈 Logging, checkpointing, and attention map visualization

---

## 📁 Project Structure

```
.
├── train.py                     # Entry point to train the model
├── config
│   ├── config.yaml              # Editable config file for job settings
├── model/
│   ├── network/
│   │   ├── net.py               # Style and content encoder
│   │   ├── glow.py              # Main Glow model
│   │   ├── attention.py         # Optional attention layers
│   ├── trainers/
│   │   └── Trainer.py           # Training logic
│   ├── losses/
│   │   └── tv_loss.py           # TV and perceptual loss
│   └── utils/
│       └── utils.py             # Scheduler, seed setup, etc.
├── data/                        # Input image directory
│   ├── trainA/                  # Hazy images
│   └── trainB/                  # Clear style/reference images
├── output/
│   ├── model_save/              # Saved checkpoints
│   ├── img_save/                # Training visual logs
│   └── test_save/               # Inference results
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda create -n dehazeflow python=3.9
conda activate dehazeflow
pip install -r requirements.txt
```

### 2. Prepare Dataset
Ensure your dataset folder follows this structure:
```
dataset/
├── trainA/   # hazy input images
├── trainB/   # clear reference images
```

### 3. Edit Config
Edit the file `config.yaml` to match your training preferences:
```yaml
job_name: dehazeflow_v1
lr: 1e-4
n_flow: 8
n_block: 3
z_dim: 1920
content_weight: 1.0
style_weight: 10.0
recon_weight: 1.0
use_attention: true
keep_ratio: 1.0
```

### 4. Train the Model
```bash
python train.py
```

---

## 🧪 Inference
During inference, you can provide a hazy image and a reference clear image (if enabled), or use a fixed style code (for reference-free mode).

---

## 📝 Citation
If you use this work in your research or publication, please cite your thesis and include:

> *"Style-Aware Dehazing via Glow-based Style Transfer with Attention Modulation" – [Alfin Nurhalim], [Institut Teknologi Bandung], 2025*
