# BTNet: Depth Reliability-Aware Dual-Stage RGB-D Salient Object Segmentation

Paper implementation in PyTorch

---

## File Structure

```
btnet/
├── model.py        # Full network architecture (BTNet + all sub-modules)
├── loss.py         # Multi-stage deep supervision loss (BCE + Dice)
├── train.py        # Dataset, trainer, inference function, CLI entry point
└── test_btnet.py   # Quick architecture sanity check (no real data required)
```

---

## Requirements

```bash
pip install torch torchvision timm Pillow numpy
```

> `timm` is used to load Swin-T pretrained weights. If unavailable, the model automatically falls back to a simple CNN encoder (for debugging only).

---

## Quick Sanity Check

```bash
python test_btnet.py
```

Validates the forward pass, output shapes, loss computation, and backward pass using random tensors — no real data needed.

---

## Data Preparation

Organize your dataset in the following directory structure (using NJU2K as an example):

```
data/
  NJU2K/
    train/
      RGB/      *.jpg
      depth/    *.png   # single-channel depth map
      GT/       *.png   # binary annotation (foreground=255, background=0)
    test/
      RGB/      *.jpg
      depth/    *.png
      GT/       *.png
```

Supported datasets: DUT-RGBD, LFSD, NJU2K, NLPR, SIP, STERE (same directory structure for all).

---

## Training

```bash
python train.py \
  --mode train \
  --data_root data/NJU2K \
  --save_dir  checkpoints \
  --epochs    100 \
  --batch_size 8 \
  --lr        1e-4 \
  --img_size  352 \
  --unified_dim 64 \
  --eval_freq 5
```

Evaluation runs every 5 epochs. The best model is saved automatically as `btnet_best.pth`.

---

## Inference

```bash
python train.py \
  --mode       infer \
  --model_path checkpoints/btnet_best.pth \
  --rgb_path   test.jpg \
  --depth_path test_depth.png \
  --save_path  result.png
```

---

## Network Architecture Overview

```
BTNet
├── DualBranchEncoder
│   ├── RGB Branch:   Swin-T (pretrained on ImageNet-1K)
│   └── Depth Branch: Lightweight 4-stage CNN (trained from scratch)
│
├── ChannelAlign x 8   -> project all features to unified_dim (64) channels
│
├── MFFM x 4 scales
│   ├── CJE: Cross-modal Joint Embedding (concatenation + compression)
│   ├── MAR: Modality-Aware Reweighting (depth reliability gate)
│   ├── RFE: Residual Fusion Enhancement (RGB semantic anchor)
│   └── CrossScale: top-down cross-scale consistency modeling
│
├── LightDecoder -> coarse saliency mask Mcoarse (H/8)
│
└── MGR (Mask-Guided Refinement)
    ├── Stage 3 (H/16): mask modulation -> prediction
    ├── Stage 2 (H/8):  mask modulation + upsample fusion -> prediction
    └── Stage 1 (H/4):  mask modulation + upsample fusion -> final prediction
```

---

## Loss Function

```
L_total = 0.3 x L_coarse
        + 0.4 x L_stage3
        + 0.6 x L_stage2
        + 1.0 x L_stage1

where each stage loss: L = L_BCE + lambda x L_Dice
```

---

## Evaluation Metrics

The codebase includes a lightweight built-in evaluator that computes:
- **MAE**: Mean Absolute Error
- **maxF**: Maximum F-measure (harmonic mean of precision and recall)

For full S-measure and E-measure evaluation, use the official MATLAB toolbox or the `py-sod-metrics` Python library.
