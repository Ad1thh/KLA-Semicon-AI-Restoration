import random
import numpy as np
import torch
import torch.nn.functional as F

def add_gaussian_noise(img: torch.Tensor, mean: float = 0.0, std_range: tuple[float, float] = (0.01, 0.08)) -> torch.Tensor:
    """
    Additive Gaussian noise: I_noisy = I + N(mean, std^2).
    img: torch.Tensor of shape (..., H, W)
    """
    std = random.uniform(std_range[0], std_range[1])
    noise = torch.randn_like(img) * std + mean
    return img + noise

def add_speckle_noise(img: torch.Tensor, mean: float = 0.0, std_range: tuple[float, float] = (0.01, 0.08)) -> torch.Tensor:
    """
    Multiplicative speckle noise: I_noisy = I + I * N(mean, std^2).
    img: torch.Tensor of shape (..., H, W)
    """
    std = random.uniform(std_range[0], std_range[1])
    noise = torch.randn_like(img) * std + mean
    return img + img * noise

def apply_downsampling(img: torch.Tensor, scale_factor: float = 0.5, mode: str = "bicubic") -> torch.Tensor:
    """
    Downsample image by scale_factor.
    img: torch.Tensor of shape (B, C, H, W) or (C, H, W)
    """
    squeeze = False
    if img.ndim == 3:
        img = img.unsqueeze(0)
        squeeze = True
    
    if mode in ["bicubic", "bilinear"]:
        out = F.interpolate(img, scale_factor=scale_factor, mode=mode, align_corners=False)
    else:
        out = F.interpolate(img, scale_factor=scale_factor, mode="area")
        
    if squeeze:
        out = out.squeeze(0)
    return out

class SyntheticDegradationPipeline:
    """
    Simulates the 3 degradation mechanisms from KLA Semiconductor inspection:
    1. Speckle Noise
    2. Additive Gaussian Noise
    3. Downsampling (2x)
    Applied in arbitrary/random permutation orders.
    """
    def __init__(self, 
                 scale_factor: float = 0.5,
                 gaussian_std_range: tuple[float, float] = (0.01, 0.08),
                 speckle_std_range: tuple[float, float] = (0.01, 0.08)):
        self.scale_factor = scale_factor
        self.gaussian_std_range = gaussian_std_range
        self.speckle_std_range = speckle_std_range

    def __call__(self, clean_gt: torch.Tensor) -> torch.Tensor:
        """
        Takes clean GT tensor [0, 1] of shape (C, H, W) or (B, C, H, W).
        Returns degraded tensor without clipping (values may exceed [0, 1]).
        """
        ops = ["gaussian", "speckle", "downsample"]
        random.shuffle(ops)
        
        degraded = clean_gt.clone()
        for op in ops:
            if op == "gaussian":
                degraded = add_gaussian_noise(degraded, std_range=self.gaussian_std_range)
            elif op == "speckle":
                degraded = add_speckle_noise(degraded, std_range=self.speckle_std_range)
            elif op == "downsample":
                degraded = apply_downsampling(degraded, scale_factor=self.scale_factor)
                
        return degraded
