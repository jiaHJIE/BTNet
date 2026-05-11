"""
BTNet 损失函数
多阶段深度监督：BCE + Dice 组合损失，不同阶段赋予不同权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    单阶段组合损失：BCE（像素级）+ λ × Dice（结构级）
    对应论文公式 (19)(20)(21)
    """
    def __init__(self, lam: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B,1,H,W) sigmoid 输出，已在 [0,1]
            gt:   (B,1,H,W) 二值标注
        """
        # ── BCE ──────────────────────────────
        gt_r = F.interpolate(gt, size=pred.shape[2:], mode='nearest')
        bce = F.binary_cross_entropy(pred, gt_r)

        # ── Dice ─────────────────────────────
        inter = (pred * gt_r).sum(dim=(1, 2, 3))
        union = pred.pow(2).sum(dim=(1, 2, 3)) + gt_r.pow(2).sum(dim=(1, 2, 3))
        dice  = 1 - (2 * inter + self.eps) / (union + self.eps)
        dice  = dice.mean()

        return bce + self.lam * dice


class BTNetLoss(nn.Module):
    """
    多级深度监督总损失，对应论文公式 (22)

    权重分配（由低到高分辨率）：
      coarse mask          → alpha_0 = 0.3（全局先验，权重较低）
      MGR stage3 (H/16)    → alpha_1 = 0.4
      MGR stage2 (H/8)     → alpha_2 = 0.6
      MGR stage1 (H/4)     → alpha_3 = 1.0（最终输出，权重最高）
    """
    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.seg_loss = BCEDiceLoss(lam=lam)
        # alpha：[coarse, mgr_deep, mgr_mid, mgr_shallow]
        self.alphas = [0.3, 0.4, 0.6, 1.0]

    def forward(self, outputs: dict, gt: torch.Tensor) -> dict:
        """
        Args:
            outputs: {'m_coarse': Tensor, 'preds': [p3, p2, p1]}
            gt:      (B, 1, H, W) 二值标注
        Returns:
            {'total': Tensor, 'coarse': Tensor, 'stage1..3': Tensor}
        """
        m_coarse = outputs['m_coarse']
        preds    = outputs['preds']          # [p3(H/16), p2(H/8), p1(H/4)]

        loss_coarse = self.seg_loss(m_coarse, gt)
        loss_stages = [self.seg_loss(p, gt) for p in preds]

        total = (self.alphas[0] * loss_coarse
                 + self.alphas[1] * loss_stages[0]   # stage3
                 + self.alphas[2] * loss_stages[1]   # stage2
                 + self.alphas[3] * loss_stages[2])  # stage1 (最终)

        return {
            'total':   total,
            'coarse':  loss_coarse,
            'stage3':  loss_stages[0],
            'stage2':  loss_stages[1],
            'stage1':  loss_stages[2],
        }
