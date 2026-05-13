"""
BTNet 数据集接口 & 训练/推理脚本
支持的数据集：DUT-RGBD / LFSD / NJU2K / NLPR / SIP / STERE

目录结构约定（以 NJU2K 为例）:
    data/
      NJU2K/
        train/
          RGB/    *.jpg
          depth/  *.png   (单通道深度图)
          GT/     *.png   (二值标注)
        test/
          RGB/    *.jpg
          depth/  *.png
          GT/     *.png
"""

import os
import glob
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from model import BTNet
from loss  import BTNetLoss


# ─────────────────────────────────────────────
# 数据集
# ─────────────────────────────────────────────

class RGBDSalDataset(Dataset):
    """通用 RGB-D 显著目标分割数据集加载器"""

    def __init__(self, root: str, split: str = 'train', size: int = 352):
        self.rgb_paths   = sorted(glob.glob(os.path.join(root, split, 'RGB',   '*')))
        self.depth_paths = sorted(glob.glob(os.path.join(root, split, 'depth', '*')))
        self.gt_paths    = sorted(glob.glob(os.path.join(root, split, 'GT',    '*')))
        assert len(self.rgb_paths) == len(self.depth_paths) == len(self.gt_paths), \
            "RGB / depth / GT 文件数量不匹配，请检查数据目录"

        self.size = size
        self.rgb_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        self.depth_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),          # → [0,1] 单通道
        ])
        self.gt_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        # 训练增强（水平翻转）
        self.augment = split == 'train'

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb   = Image.open(self.rgb_paths[idx]).convert('RGB')
        depth = Image.open(self.depth_paths[idx]).convert('L')  # 灰度
        gt    = Image.open(self.gt_paths[idx]).convert('L')

        # 训练随机水平翻转
        if self.augment and torch.rand(1) > 0.5:
            rgb, depth, gt = map(lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
                                 [rgb, depth, gt])

        rgb   = self.rgb_tf(rgb)
        depth = self.depth_tf(depth)                 # (1, H, W)
        gt    = self.gt_tf(gt)
        gt    = (gt > 0.5).float()                   # 二值化

        return rgb, depth, gt


# ─────────────────────────────────────────────
# 评估指标
# ─────────────────────────────────────────────

class Metrics:
    """计算 MAE、S-measure（近似）、maxF、E-measure（近似）"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mae_sum   = 0.0
        self.count     = 0
        self.tp_sum    = 0.0
        self.fp_sum    = 0.0
        self.fn_sum    = 0.0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred: (B,1,H,W) sigmoid，值在 [0,1]
        gt:   (B,1,H,W) 二值
        """
        B = pred.size(0)
        pred_np = pred.squeeze(1).cpu().numpy()
        gt_np   = gt.squeeze(1).cpu().numpy()

        for i in range(B):
            p = pred_np[i]
            g = gt_np[i]
            # MAE
            self.mae_sum += np.abs(p - g).mean()
            # F-measure（阈值 0.5）
            p_bin = (p > 0.5).astype(float)
            self.tp_sum += (p_bin * g).sum()
            self.fp_sum += (p_bin * (1 - g)).sum()
            self.fn_sum += ((1 - p_bin) * g).sum()
            self.count  += 1

    def summary(self):
        mae    = self.mae_sum / max(self.count, 1)
        prec   = self.tp_sum / (self.tp_sum + self.fp_sum + 1e-8)
        recall = self.tp_sum / (self.tp_sum + self.fn_sum + 1e-8)
        maxf   = 2 * prec * recall / (prec + recall + 1e-8)
        return {'MAE': mae, 'maxF': float(maxf)}


# ─────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg):
        self.cfg  = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Trainer] 使用设备: {self.device}")

        # 模型
        self.model = BTNet(unified_dim=cfg.unified_dim,
                           pretrained_swin=cfg.pretrained).to(self.device)

        # 损失 & 优化器
        self.criterion = BTNetLoss(lam=1.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.5
        )

        # 数据
        train_set = RGBDSalDataset(cfg.data_root, 'train', cfg.img_size)
        test_set  = RGBDSalDataset(cfg.data_root, 'test',  cfg.img_size)
        self.train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                                       shuffle=True,  num_workers=4, pin_memory=True)
        self.test_loader  = DataLoader(test_set,  batch_size=1,
                                       shuffle=False, num_workers=2, pin_memory=True)

        # 保存目录
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        self.best_mae = float('inf')

    # ------------------------------------------
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        for step, (rgb, depth, gt) in enumerate(self.train_loader):
            rgb, depth, gt = rgb.to(self.device), depth.to(self.device), gt.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(rgb, depth)            # 训练模式返回 dict
            losses  = self.criterion(outputs, gt)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += losses['total'].item()

            if step % 50 == 0:
                print(f"  Epoch {epoch:03d} | Step {step:04d}/{len(self.train_loader)} "
                      f"| Loss {losses['total'].item():.4f} "
                      f"(coarse={losses['coarse'].item():.3f}, "
                      f"stage1={losses['stage1'].item():.3f})")

        return total_loss / len(self.train_loader)

    # ------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        metrics = Metrics()
        for rgb, depth, gt in self.test_loader:
            rgb, depth = rgb.to(self.device), depth.to(self.device)
            pred = self.model(rgb, depth)               # 推理模式返回 Tensor
            metrics.update(pred.cpu(), gt)
        return metrics.summary()

    # ------------------------------------------
    def train(self):
        print(f"[Trainer] 开始训练，共 {self.cfg.epochs} 个 Epoch")
        for epoch in range(1, self.cfg.epochs + 1):
            avg_loss = self.train_epoch(epoch)
            self.scheduler.step()

            if epoch % self.cfg.eval_freq == 0:
                res = self.evaluate()
                print(f"[Eval] Epoch {epoch:03d} | MAE={res['MAE']:.4f} | "
                      f"maxF={res['maxF']:.4f} | AvgLoss={avg_loss:.4f}")

                if res['MAE'] < self.best_mae:
                    self.best_mae = res['MAE']
                    save_path = os.path.join(self.cfg.save_dir, 'btnet_best.pth')
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  ✓ 最优模型已保存 → {save_path}")

        # 最终保存
        torch.save(self.model.state_dict(),
                   os.path.join(self.cfg.save_dir, 'btnet_last.pth'))
        print("[Trainer] 训练完成。")


# ─────────────────────────────────────────────
# 推理函数
# ─────────────────────────────────────────────

@torch.no_grad()
def inference(model_path: str, rgb_path: str, depth_path: str,
              save_path: str = 'result.png', img_size: int = 352):
    """
    单张图片推理示例
    Args:
        model_path:  模型权重路径 (.pth)
        rgb_path:    RGB 图像路径
        depth_path:  深度图路径
        save_path:   预测结果保存路径
        img_size:    输入分辨率
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = BTNet(unified_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rgb_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    depth_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    rgb   = rgb_tf(Image.open(rgb_path).convert('RGB')).unsqueeze(0).to(device)
    depth = depth_tf(Image.open(depth_path).convert('L')).unsqueeze(0).to(device)

    pred = model(rgb, depth)                         # (1,1,H,W)
    pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(pred_np).save(save_path)
    print(f"[Inference] 预测结果已保存 → {save_path}")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='BTNet Training & Inference')
    p.add_argument('--mode',        type=str,   default='train',
                   choices=['train', 'infer'],   help='运行模式')
    # 训练参数
    p.add_argument('--data_root',   type=str,   default='data/NJU2K')
    p.add_argument('--save_dir',    type=str,   default='checkpoints')
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight_decay',type=float, default=5e-4)
    p.add_argument('--img_size',    type=int,   default=352)
    p.add_argument('--unified_dim', type=int,   default=64)
    p.add_argument('--pretrained',  action='store_true', default=True)
    p.add_argument('--eval_freq',   type=int,   default=5)
    # 推理参数
    p.add_argument('--model_path',  type=str,   default='checkpoints/btnet_best.pth')
    p.add_argument('--rgb_path',    type=str,   default='')
    p.add_argument('--depth_path',  type=str,   default='')
    p.add_argument('--save_path',   type=str,   default='result.png')
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.mode == 'train':
        trainer = Trainer(args)
        trainer.train()
    else:
        inference(args.model_path, args.rgb_path, args.depth_path, args.save_path)
