import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (differentiable L1 approximation)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.float()
        t = target.float()
        diff = p - t
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return torch.mean(loss)

class FFTLoss(nn.Module):
    """Frequency-domain Charbonnier loss for semiconductor line edge and pitch fidelity."""
    def __init__(self, eps: float = 1e-6):
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

class SSIMLoss(nn.Module):
    """Structural Similarity Loss."""
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range
        self.ssim_module = None
        try:
            from pytorch_msssim import SSIM
            self.ssim_module = SSIM(data_range=data_range, size_average=True, channel=1)
        except Exception:
            self.ssim_module = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(pred.float(), 0.0, self.data_range)
        t = torch.clamp(target.float(), 0.0, self.data_range)
        if self.ssim_module is not None:
            val = self.ssim_module(p, t)
            val = torch.clamp(val, 0.0, 1.0)
            return 1.0 - val
        else:
            from src.utils import calculate_ssim
            ssim_val = calculate_ssim(p, t)
            return 1.0 - torch.tensor(ssim_val, device=pred.device, dtype=torch.float32, requires_grad=True)

class LPIPSLoss(nn.Module):
    """Perceptual Loss using LPIPS (AlexNet)."""
    def __init__(self, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device
        self.lpips_fn = None
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"[Warning] LPIPS initialization deferred: {e}", flush=True)

    def to(self, device):
        self.device = device
        if self.lpips_fn is not None:
            self.lpips_fn = self.lpips_fn.to(device)
        return super().to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lpips_fn is None:
            return torch.tensor(0.0, device=pred.device, dtype=torch.float32, requires_grad=True)

        if next(self.lpips_fn.parameters()).device != pred.device:
            self.lpips_fn = self.lpips_fn.to(pred.device)

        p = torch.clamp(pred.float(), 0.0, 1.0) * 2.0 - 1.0
        t = torch.clamp(target.float(), 0.0, 1.0) * 2.0 - 1.0
        if p.shape[1] == 1:
            p = p.repeat(1, 3, 1, 1)
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        val = self.lpips_fn(p, t)
        return val.mean()

class CompositeRestorationLoss(nn.Module):
    """
    Multi-Domain Composite Loss:
    L_total = 1.0 * L_Charbonnier + 0.5 * L_SSIM + 0.1 * L_FFT + 0.02 * L_LPIPS
    """
    def __init__(self,
                 w_charbonnier: float = 1.0,
                 w_ssim: float = 0.5,
                 w_fft: float = 0.1,
                 w_lpips: float = 0.02,
                 w_ms_ssim: float = None,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.w_charbonnier = w_charbonnier
        self.w_ssim = w_ms_ssim if w_ms_ssim is not None else w_ssim
        self.w_fft = w_fft
        self.w_lpips = w_lpips

        self.charbonnier = CharbonnierLoss()
        self.fft = FFTLoss()
        self.ssim = SSIMLoss(data_range=1.0)
        self.lpips = LPIPSLoss(device=device) if w_lpips > 0.0 else None

    def to(self, device):
        if self.lpips is not None:
            self.lpips = self.lpips.to(device)
        self.ssim = self.ssim.to(device)
        return super().to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l_char = self.charbonnier(pred, target)
        l_fft = self.fft(pred, target)
        l_ssim = self.ssim(pred, target)
        l_lpips = self.lpips(pred, target) if self.lpips is not None else torch.tensor(0.0, device=pred.device)

        total_loss = (
            self.w_charbonnier * l_char +
            self.w_ssim * l_ssim +
            self.w_fft * l_fft +
            self.w_lpips * l_lpips
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "loss_charbonnier": l_char.item(),
            "loss_ssim": l_ssim.item(),
            "loss_ms_ssim": l_ssim.item(),
            "loss_fft": l_fft.item(),
            "loss_lpips": l_lpips.item(),
        }

        return total_loss, loss_dict
