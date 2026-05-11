"""
BTNet: Depth Reliability-Aware Dual-Stage Network for RGB-D Salient Object Segmentation
论文复现实现 - PyTorch

结构：
  DualBranchEncoder (Swin-T + 轻量深度编码器)
    └─ MFFM × 4级 (CJE → MAR → RFE → 跨尺度一致性)
         ├─ LightDecoder → 粗略显著性掩码
         └─ MGR (掩码引导精细化) → 最终分割结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. 深度结构编码器（Depth Structural Encoder）
# ─────────────────────────────────────────────

class StructuralConvBlock(nn.Module):
    """SCB: 带残差的深度结构卷积块，可选空洞卷积增大感受野"""
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


class DepthStructuralEncoder(nn.Module):
    """
    轻量四阶段深度编码器
    输出四个尺度特征：H/4, H/8, H/16, H/32，通道 96/192/384/768
    从头训练，不使用预训练权重
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        channels = [96, 192, 384, 768]

        # Stem: 单通道深度图 → 初始浅层特征
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.stages = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            stage = nn.Sequential(
                # 步长卷积：降分辨率 + 通道扩展（第0级不降分辨率，stem已处理）
                nn.Conv2d(in_ch, out_ch, 3, stride=2 if i > 0 else 1,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                # 第一个 SCB 用空洞卷积扩大感受野
                StructuralConvBlock(out_ch, dilation=2),
                StructuralConvBlock(out_ch, dilation=1),
            )
            self.stages.append(stage)
            in_ch = out_ch

    def forward(self, d: torch.Tensor):
        """
        Args:
            d: (B, 1, H, W) 深度图
        Returns:
            List[(B, C_i, H/s_i, W/s_i)] × 4 级
        """
        feats = []
        x = self.stem(d)          # (B, 96, H, W)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        # feats: [H/4, H/8, H/16, H/32]，通道 [96,192,384,768]
        return feats


# ─────────────────────────────────────────────
# 2. 通道对齐（统一到 unified_dim）
# ─────────────────────────────────────────────

class ChannelAlign(nn.Module):
    """将任意通道数对齐到 unified_dim（论文中为 64）"""
    def __init__(self, in_channels: int, unified_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, unified_dim, 1, bias=False),
            nn.BatchNorm2d(unified_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────
# 3. MFFM 子模块
# ─────────────────────────────────────────────

class CJE(nn.Module):
    """
    Cross-modal Joint Embedding
    拼接 RGB + 深度特征，压缩编码，捕捉跨模态一致性与差异
    """
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_rgb: torch.Tensor, f_d: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([f_rgb, f_d], dim=1)   # (B, 2C, H, W)
        return self.fuse(cat)                   # (B, C, H, W)


class MAR(nn.Module):
    """
    Modality-Aware Reweighting
    基于联合特征，通过全局平均池化 + FC 生成通道级深度可靠性门控权重
    wi  → RGB 置信度；1-wi → 深度可靠性
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, f_joint: torch.Tensor, f_rgb: torch.Tensor, f_d: torch.Tensor):
        """
        Returns:
            f_rgb_rw: 可靠性重加权后的 RGB 特征
            f_d_rw:   可靠性重加权后的深度特征
            w:        RGB 置信度向量 (B, C)
        """
        w = self.gate(f_joint)              # (B, C)
        w = w.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        f_rgb_rw = w * f_rgb
        f_d_rw   = (1 - w) * f_d
        return f_rgb_rw, f_d_rw


class RFE(nn.Module):
    """
    Residual Fusion Enhancement
    将重加权后的 RGB + 深度特征融合，加入 RGB 残差项稳定语义
    F_fuse = φ(F_rgb + F_d) + F_rgb
    """
    def __init__(self, channels: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_rgb: torch.Tensor, f_d: torch.Tensor) -> torch.Tensor:
        fused = self.relu(self.phi(f_rgb + f_d) + f_rgb)
        return fused


# ─────────────────────────────────────────────
# 4. 多级特征融合模块（MFFM）
# ─────────────────────────────────────────────

class MFFM(nn.Module):
    """
    Multi-Level Feature Fusion Module
    单个尺度的融合单元：CJE → MAR → RFE
    """
    def __init__(self, channels: int):
        super().__init__()
        self.cje = CJE(channels)
        self.mar = MAR(channels)
        self.rfe = RFE(channels)

    def forward(self, f_rgb: torch.Tensor, f_d: torch.Tensor) -> torch.Tensor:
        f_joint = self.cje(f_rgb, f_d)
        f_rgb_rw, f_d_rw = self.mar(f_joint, f_rgb, f_d)
        f_fuse = self.rfe(f_rgb_rw, f_d_rw)
        return f_fuse


class CrossScaleFusion(nn.Module):
    """
    跨尺度一致性建模
    将高层特征上采样后与当前层特征融合（拼接 + 卷积）
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_cur: torch.Tensor, f_upper: torch.Tensor) -> torch.Tensor:
        f_up = F.interpolate(f_upper, size=f_cur.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([f_cur, f_up], dim=1))


# ─────────────────────────────────────────────
# 5. 轻量解码器（Light Decoder） → 粗略掩码
# ─────────────────────────────────────────────

class LightDecoder(nn.Module):
    """
    从高层融合特征渐进上采样生成全局粗略显著性掩码
    输入：F3, F2（对应 H/16, H/8 尺度，统一 unified_dim 通道）
    输出：Mcoarse ∈ [0,1]^{H×W}
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.pred = nn.Conv2d(channels, 1, 1)

    def forward(self, f4: torch.Tensor, f3: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f4: (B, C, H/32, W/32) 最深层
            f3: (B, C, H/16, W/16)
            f2: (B, C, H/8, W/8)
        Returns:
            Mcoarse: (B, 1, H/8, W/8)，Sigmoid 激活后的粗略掩码
        """
        # D3 = Up(F4) ⊕ F3
        d3 = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.conv3(d3 + f3)

        # D2 = Up(D3) ⊕ F2
        d3_up = F.interpolate(d3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.conv2(torch.cat([d3_up, f2], dim=1))

        m_coarse = torch.sigmoid(self.pred(d2))
        return m_coarse


# ─────────────────────────────────────────────
# 6. 掩码引导精细化模块（MGR）
# ─────────────────────────────────────────────

class MaskGuidedModulation(nn.Module):
    """
    单尺度掩码引导调制
    F_mg = φ(M ⊙ F_fuse) + F_fuse
    """
    def __init__(self, channels: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_fuse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_fuse: (B, C, H, W)
            mask:   (B, 1, H, W) 粗略掩码（已缩放到对应分辨率）
        """
        f_guided = mask * f_fuse                # 空间加权
        f_mg = self.phi(f_guided) + f_fuse      # 残差保护
        return f_mg


class MGR(nn.Module):
    """
    Mask-Guided Refinement
    逐尺度注入粗略掩码约束，渐进上采样生成多阶段输出
    输入尺度：F1(H/4), F2(H/8), F3(H/16)，均为 unified_dim 通道
    """
    def __init__(self, channels: int):
        super().__init__()
        # 各尺度的掩码引导调制
        self.modulate = nn.ModuleList([
            MaskGuidedModulation(channels) for _ in range(3)
        ])
        # 渐进解码卷积
        self.decode_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ) for _ in range(2)
        ])
        # 各阶段预测头（深度监督）
        self.pred_heads = nn.ModuleList([
            nn.Conv2d(channels, 1, 1) for _ in range(3)
        ])

    def forward(self, feats: list, m_coarse: torch.Tensor):
        """
        Args:
            feats:    [f1, f2, f3]，尺度 H/4, H/8, H/16
            m_coarse: (B, 1, H/8, W/8) 粗略显著性掩码
        Returns:
            preds: List[Tensor(B,1,H_i,W_i)] 各阶段预测（由浅到深）
        """
        f1, f2, f3 = feats      # H/4, H/8, H/16
        preds = []

        # ── 阶段3（最深，H/16）──────────────────
        m3 = F.adaptive_avg_pool2d(m_coarse, f3.shape[2:])
        f3_mg = self.modulate[2](f3, m3)
        preds.append(torch.sigmoid(self.pred_heads[2](f3_mg)))

        # ── 阶段2（H/8）────────────────────────
        f3_up = F.interpolate(f3_mg, size=f2.shape[2:], mode='bilinear', align_corners=False)
        m2 = F.adaptive_avg_pool2d(m_coarse, f2.shape[2:])
        f2_mg = self.modulate[1](f2, m2)
        f2_cat = self.decode_convs[0](torch.cat([f3_up, f2_mg], dim=1))
        preds.append(torch.sigmoid(self.pred_heads[1](f2_cat)))

        # ── 阶段1（最浅，H/4）──────────────────
        f2_up = F.interpolate(f2_cat, size=f1.shape[2:], mode='bilinear', align_corners=False)
        m1 = F.adaptive_avg_pool2d(m_coarse, f1.shape[2:])
        f1_mg = self.modulate[0](f1, m1)
        f1_cat = self.decode_convs[1](torch.cat([f2_up, f1_mg], dim=1))
        preds.append(torch.sigmoid(self.pred_heads[0](f1_cat)))

        return preds  # [pred_stage3, pred_stage2, pred_stage1]


# ─────────────────────────────────────────────
# 7. BTNet 整体网络
# ─────────────────────────────────────────────

class BTNet(nn.Module):
    """
    BTNet: 深度可靠性感知双阶段 RGB-D 显著目标分割网络

    输入:
        rgb:   (B, 3, H, W)
        depth: (B, 1, H, W)
    输出（训练模式）:
        {
          'm_coarse': (B,1,H/8,W/8),   粗略掩码（用于深度监督）
          'preds':    [(B,1,H_i,W_i)]   MGR 各阶段输出，最后一个分辨率最高
        }
    输出（推理模式）:
        (B,1,H,W) 上采样到原始分辨率的最终分割图
    """

    # Swin-T 各阶段输出通道
    SWIN_CHANNELS = [96, 192, 384, 768]

    def __init__(self, unified_dim: int = 64, pretrained_swin: bool = True):
        super().__init__()
        self.unified_dim = unified_dim

        # ── RGB 编码器：Swin-T ─────────────────
        self.rgb_encoder = self._build_swin_encoder(pretrained_swin)

        # ── 深度编码器 ─────────────────────────
        self.depth_encoder = DepthStructuralEncoder(in_channels=1)

        # ── 通道对齐（RGB & Depth → unified_dim）
        self.rgb_aligns = nn.ModuleList([
            ChannelAlign(c, unified_dim) for c in self.SWIN_CHANNELS
        ])
        self.d_aligns = nn.ModuleList([
            ChannelAlign(c, unified_dim) for c in self.SWIN_CHANNELS
        ])

        # ── 各尺度 MFFM ───────────────────────
        self.mffms = nn.ModuleList([
            MFFM(unified_dim) for _ in range(4)
        ])

        # ── 跨尺度一致性（从 i+1 → i）──────────
        self.cross_scales = nn.ModuleList([
            CrossScaleFusion(unified_dim) for _ in range(3)   # 3→2, 2→1, 1→0
        ])

        # ── 轻量解码器 → 粗略掩码 ─────────────
        self.light_decoder = LightDecoder(unified_dim)

        # ── 掩码引导精细化 ─────────────────────
        self.mgr = MGR(unified_dim)

        # 权重初始化
        self._init_weights()

    # ------------------------------------------
    def _build_swin_encoder(self, pretrained: bool):
        """
        尝试加载 timm 库中的 Swin-T；若不可用则回退到占位编码器
        """
        try:
            import timm
            model = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            return model
        except Exception:
            print("[BTNet] timm 不可用，使用简单 CNN 占位编码器（仅用于调试）")
            return _SimpleCNNEncoder(3, self.SWIN_CHANNELS)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        # ── 1. 提取多尺度特征 ──────────────────
        rgb_feats   = self.rgb_encoder(rgb)        # 4×(B,C_i,H/s,W/s)
        depth_feats = self.depth_encoder(depth)    # 4×(B,C_i,H/s,W/s)

        # ── 2. 通道对齐 ───────────────────────
        rgb_feats   = [self.rgb_aligns[i](rgb_feats[i])   for i in range(4)]
        depth_feats = [self.d_aligns[i](depth_feats[i])   for i in range(4)]

        # ── 3. 各尺度 MFFM 融合 ───────────────
        fused = [self.mffms[i](rgb_feats[i], depth_feats[i]) for i in range(4)]
        # fused: [F0(H/4), F1(H/8), F2(H/16), F3(H/32)]

        # ── 4. 跨尺度一致性（自顶向下）─────────
        # F3 → F2 → F1 → F0
        cs_feats = [None, None, None, fused[3]]
        for i in range(2, -1, -1):                         # i = 2,1,0
            cs_feats[i] = self.cross_scales[i](fused[i], cs_feats[i + 1])

        # ── 5. 粗略掩码预测 ───────────────────
        # Light Decoder 使用 F3(H/32), F2(H/16), F1(H/8)
        m_coarse = self.light_decoder(cs_feats[3], cs_feats[2], cs_feats[1])
        # m_coarse: (B, 1, H/8, W/8)

        # ── 6. 掩码引导精细化 ─────────────────
        mgr_preds = self.mgr([cs_feats[0], cs_feats[1], cs_feats[2]], m_coarse)
        # mgr_preds[-1] 是最高分辨率的精细结果 (H/4)

        if self.training:
            return {'m_coarse': m_coarse, 'preds': mgr_preds}
        else:
            # 推理：上采样到原始分辨率
            final = mgr_preds[-1]
            final = F.interpolate(final, size=rgb.shape[2:], mode='bilinear', align_corners=False)
            return final


# ─────────────────────────────────────────────
# 辅助：简单 CNN 占位编码器（timm 不可用时）
# ─────────────────────────────────────────────

class _SimpleCNNEncoder(nn.Module):
    def __init__(self, in_ch: int, channels: list):
        super().__init__()
        self.stages = nn.ModuleList()
        c = in_ch
        for out_c in channels:
            self.stages.append(nn.Sequential(
                nn.Conv2d(c, out_c, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ))
            c = out_c

    def forward(self, x):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats
