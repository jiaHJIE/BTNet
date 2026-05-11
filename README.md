# BTNet：深度可靠性感知双阶段 RGB-D 显著目标分割

论文复现实现 | PyTorch

---

## 文件结构

```
btnet/
├── model.py        # 完整网络结构（BTNet + 所有子模块）
├── loss.py         # 多级深度监督损失（BCE + Dice）
├── train.py        # 数据集、训练器、推理函数、CLI 入口
└── test_btnet.py   # 结构快速验证（无需真实数据）
```

---

## 环境依赖

```bash
pip install torch torchvision timm Pillow numpy
```

> `timm` 用于加载 Swin-T 预训练权重。若不可用，模型会自动回退到简单 CNN 占位编码器（仅用于调试）。

---

## 快速验证

```bash
python test_btnet.py
```

用随机张量验证前向传播、输出形状、损失计算、反向传播均正确。

---

## 数据准备

按以下目录结构组织数据集（以 NJU2K 为例）：

```
data/
  NJU2K/
    train/
      RGB/      *.jpg
      depth/    *.png   # 单通道深度图
      GT/       *.png   # 二值标注（前景=255，背景=0）
    test/
      RGB/      *.jpg
      depth/    *.png
      GT/       *.png
```

支持数据集：DUT-RGBD、LFSD、NJU2K、NLPR、SIP、STERE（目录结构相同）。

---

## 训练

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

训练时每 5 个 Epoch 评估一次，自动保存最优模型（`btnet_best.pth`）。

---

## 推理

```bash
python train.py \
  --mode      infer \
  --model_path checkpoints/btnet_best.pth \
  --rgb_path   test.jpg \
  --depth_path test_depth.png \
  --save_path  result.png
```

---

## 网络结构概览

```
BTNet
├── DualBranchEncoder
│   ├── RGB Branch:   Swin-T（ImageNet-1K 预训练）
│   └── Depth Branch: 轻量 4 级 CNN（从头训练）
│
├── ChannelAlign × 8   → 统一到 unified_dim (64) 通道
│
├── MFFM × 4 级
│   ├── CJE: 跨模态联合嵌入（拼接 + 压缩）
│   ├── MAR: 模态感知重加权（深度可靠性门控）
│   ├── RFE: 残差融合增强（RGB 语义锚定）
│   └── CrossScale: 自顶向下跨尺度一致性
│
├── LightDecoder → 粗略显著性掩码 Mcoarse (H/8)
│
└── MGR (掩码引导精细化)
    ├── 阶段3 (H/16): 掩码调制 → 预测
    ├── 阶段2 (H/8):  掩码调制 + 上采样融合 → 预测
    └── 阶段1 (H/4):  掩码调制 + 上采样融合 → 最终预测
```

---

## 损失函数

```
L_total = 0.3 × L_coarse
        + 0.4 × L_stage3
        + 0.6 × L_stage2
        + 1.0 × L_stage1

其中每级 L = L_BCE + λ × L_Dice
```

---

## 评估指标

代码内置轻量评估器，计算：
- **MAE**：平均绝对误差
- **maxF**：最大 F-measure（精确率/召回率综合）

完整的 S-measure / E-measure 建议使用官方 MATLAB 工具箱或 `py-sod-metrics` 库。
