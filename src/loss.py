import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (differentiable L1 approximation).
    L = sqrt(||y_hat - y||^2 + eps^2) with eps = 1e-3.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.float()
        t = target.float()
        diff = p - t
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return torch.mean(loss)

class FFTLoss(nn.Module):
    """
    Frequency-domain Focal Frequency Loss computed via 2D Fast Fourier Transform (torch.fft.rfft2)
    to enforce periodic layout grating fidelity and sharp line edge transitions in semiconductor inspection.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.float()
        t = target.float()
        pred_fft = torch.fft.rfft2(p, norm='ortho')
        target_fft = torch.fft.rfft2(t, norm='ortho')
        
        diff_real = pred_fft.real - target_fft.real
        diff_imag = pred_fft.imag - target_fft.imag
        
        loss = torch.sqrt(diff_real * diff_real + diff_imag * diff_imag + self.eps)
        return torch.mean(loss)

class MSSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity Loss: 1 - MS-SSIM(y_hat, y).
    Resolves multi-scale semiconductor geometries and contact holes.
    """
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range
        self.ms_ssim_module = None
        try:
            from pytorch_msssim import MS_SSIM
            self.ms_ssim_module = MS_SSIM(data_range=data_range, size_average=True, channel=1)
        except Exception as e:
            print(f"[Warning] pytorch_msssim.MS_SSIM initialization warning: {e}", flush=True)
            self.ms_ssim_module = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(pred.float(), 0.0, self.data_range)
        t = torch.clamp(target.float(), 0.0, self.data_range)
        if self.ms_ssim_module is not None:
            val = self.ms_ssim_module(p, t)
            val = torch.clamp(val, 0.0, 1.0)
            return 1.0 - val
        else:
            from src.utils import calculate_ssim
            ssim_val = calculate_ssim(p, t)
            return 1.0 - torch.tensor(ssim_val, device=pred.device, dtype=torch.float32, requires_grad=True)

class CompositeRestorationLoss(nn.Module):
    """
    Multi-Scale Composite Structural Loss for Semiconductor Inspection:
    L_total = 1.0 * L_Charbonnier + 1.0 * L_MS_SSIM + 0.2 * L_FFT
    """
    def __init__(self,
                 w_charbonnier: float = 1.0,
                 w_ms_ssim: float = 1.0,
                 w_fft: float = 0.2,
                 w_ssim: float = None,
                 w_lpips: float = 0.0,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.w_charbonnier = w_charbonnier
        self.w_ms_ssim = w_ms_ssim if w_ms_ssim is not None else (w_ssim if w_ssim is not None else 1.0)
        self.w_fft = w_fft
        self.w_lpips = w_lpips

        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.fft = FFTLoss(eps=1e-8)
        self.ms_ssim = MSSSIMLoss(data_range=1.0)

    def to(self, device):
        self.ms_ssim = self.ms_ssim.to(device)
        return super().to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l_char = self.charbonnier(pred, target)
        l_ms_ssim = self.ms_ssim(pred, target)
        l_fft = self.fft(pred, target)

        total_loss = (
            self.w_charbonnier * l_char +
            self.w_ms_ssim * l_ms_ssim +
            self.w_fft * l_fft
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_charbonnier": l_char.item(),
            "loss_ms_ssim": l_ms_ssim.item(),
            "loss_ssim": l_ms_ssim.item(),
            "loss_fft": l_fft.item(),
        }

        return total_loss, loss_dict
