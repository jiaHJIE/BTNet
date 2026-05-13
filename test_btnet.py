"""
BTNet 快速结构验证脚本
运行：python test_btnet.py
无需真实数据，用随机张量验证前向传播 & 损失计算是否正确
"""

import torch
from model import BTNet
from loss  import BTNetLoss


def test_forward():
    print("=" * 55)
    print("BTNet 结构验证")
    print("=" * 55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    B, H, W = 2, 352, 352

    # 构建模型（pretrained=False 避免下载）
    model = BTNet(unified_dim=64, pretrained_swin=False).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  总参数量: {total_params:.1f} M")

    rgb   = torch.randn(B, 3, H, W).to(device)
    depth = torch.randn(B, 1, H, W).to(device)
    gt    = torch.randint(0, 2, (B, 1, H, W)).float().to(device)

    # ── 训练模式 ──────────────────────────────
    model.train()
    outputs = model(rgb, depth)
    print(f"\n  [训练输出]")
    print(f"    m_coarse : {outputs['m_coarse'].shape}")
    for i, p in enumerate(outputs['preds']):
        print(f"    preds[{i}] : {p.shape}")

    criterion = BTNetLoss()
    losses = criterion(outputs, gt)
    print(f"\n  [损失]")
    for k, v in losses.items():
        print(f"    {k:8s}: {v.item():.4f}")

    losses['total'].backward()
    print("  反向传播: ✓")

    # ── 推理模式 ──────────────────────────────
    model.eval()
    with torch.no_grad():
        pred = model(rgb, depth)
    print(f"\n  [推理输出] {pred.shape}  (应为 {(B,1,H,W)})")
    assert pred.shape == (B, 1, H, W), "输出形状不匹配！"
    assert pred.min() >= 0 and pred.max() <= 1, "输出不在 [0,1]！"

    print("\n  所有检查通过 ✓")
    print("=" * 55)


if __name__ == '__main__':
    test_forward()
