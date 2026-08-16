import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed: int = 42):
    """Set random seed for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-10) -> float:
    """
    Calculate PSNR for float tensors strictly in [0, 1].
    pred, target: (B, C, H, W) or (C, H, W) or (H, W)
    """
    pred = torch.clamp(pred, 0.0, max_val)
    target = torch.clamp(target, 0.0, max_val)
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 100.0
    return float(10.0 * np.log10((max_val ** 2) / mse))

def calculate_batch_psnr(preds: torch.Tensor, targets: torch.Tensor, max_val: float = 1.0) -> list[float]:
    """Calculate PSNR for each sample in a batch."""
    preds = torch.clamp(preds, 0.0, max_val)
    targets = torch.clamp(targets, 0.0, max_val)
    batch_size = preds.shape[0]
    psnrs = []
    for i in range(batch_size):
        mse = torch.mean((preds[i] - targets[i]) ** 2).item()
        if mse < 1e-10:
            psnrs.append(100.0)
        else:
            psnrs.append(float(10.0 * np.log10((max_val ** 2) / mse)))
    return psnrs

def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate SSIM for 4D/3D tensors in [0, 1] using pytorch_msssim.
    """
    pred = torch.clamp(pred, 0.0, max_val)
    target = torch.clamp(target, 0.0, max_val)
    
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)
        
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(0)
        
    try:
        from pytorch_msssim import ssim
        val = ssim(pred, target, data_range=max_val, size_average=True)
        return float(val.item())
    except Exception:
        # Fallback simple structural calculation
        mu_x = pred.mean()
        mu_y = target.mean()
        sigma_x = pred.std()
        sigma_y = target.std()
        sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        ssim_val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
        return float(ssim_val.item())

class LPIPSCalculator:
    def __init__(self, device: torch.device):
        self.device = device
        self.loss_fn = None
        try:
            import lpips
            self.loss_fn = lpips.LPIPS(net='alex').to(device).eval()
            for param in self.loss_fn.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"[Warning] Failed to initialize LPIPS: {e}")

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if self.loss_fn is None:
            return 0.0
        with torch.no_grad():
            p = torch.clamp(pred, 0.0, 1.0) * 2.0 - 1.0
            t = torch.clamp(target, 0.0, 1.0) * 2.0 - 1.0
            if p.ndim == 2:
                p = p.unsqueeze(0).unsqueeze(0)
            elif p.ndim == 3:
                p = p.unsqueeze(0)
            if t.ndim == 2:
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.ndim == 3:
                t = t.unsqueeze(0)
            if p.shape[1] == 1:
                p = p.repeat(1, 3, 1, 1)
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)
            val = self.loss_fn(p.to(self.device), t.to(self.device))
            return float(val.mean().item())

def bicubic_upsample(x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
    """Bicubic upsampling for baseline or residual connections."""
    return F.interpolate(x, scale_factor=scale_factor, mode='bicubic', align_corners=False)
