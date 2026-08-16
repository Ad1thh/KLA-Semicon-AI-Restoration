import os
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.utils import set_seed, calculate_psnr, calculate_ssim, LPIPSCalculator, bicubic_upsample
from src.dataset import SemiconductorDataset, get_dataloaders
from src.model import NAFNetSR
from src.loss import CompositeRestorationLoss

def log_to_claude_mem(key: str, data: dict):
    """Autonomous state logger recording experiment tracking and metrics."""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "claude_mem_state.yaml")
    
    current_data = {}
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                current_data = yaml.safe_load(f) or {}
        except Exception:
            current_data = {}
            
    current_data[key] = data
    with open(log_file, "w") as f:
        yaml.dump(current_data, f, default_flow_style=False)
    print(f"[claude-mem] Successfully recorded '{key}' to {log_file}", flush=True)

def run_bicubic_baseline(val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Compute standard Bicubic baseline metrics on validation set."""
    print("\n" + "=" * 60, flush=True)
    print("--- Computing Bicubic Baseline Metrics ---", flush=True)
    print("=" * 60, flush=True)
    
    lpips_calc = LPIPSCalculator(device)
    psnr_list, ssim_list, lpips_list = [], [], []

    start_time = time.time()
    for batch in val_loader:
        degraded = batch["degraded"].to(device)
        gt = batch["gt"].to(device)

        with torch.no_grad():
            bicubic = bicubic_upsample(degraded, scale_factor=2)
            bicubic = torch.clamp(bicubic, 0.0, 1.0)

            for i in range(bicubic.shape[0]):
                p = calculate_psnr(bicubic[i:i+1], gt[i:i+1])
                s = calculate_ssim(bicubic[i:i+1], gt[i:i+1])
                l = lpips_calc(bicubic[i:i+1], gt[i:i+1])
                psnr_list.append(p)
                ssim_list.append(s)
                lpips_list.append(l)

    elapsed = time.time() - start_time
    results = {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)),
        "num_samples": len(psnr_list),
        "time_seconds": float(elapsed)
    }

    print(f"Bicubic Baseline Results (N={len(psnr_list)}):", flush=True)
    print(f"  PSNR:  {results['psnr']:.4f} dB", flush=True)
    print(f"  SSIM:  {results['ssim']:.4f}", flush=True)
    print(f"  LPIPS: {results['lpips']:.4f}", flush=True)
    print(f"  Total evaluation time: {results['time_seconds']:.2f}s", flush=True)
    print("=" * 60 + "\n", flush=True)

    log_to_claude_mem("bicubic_baseline", results)
    return results

def run_overfit_test(config: dict, device: torch.device):
    """Sanity Check: Overfit on exactly 2 pairs first; confirm loss drops and PSNR > 40 dB."""
    print("\n" + "=" * 60, flush=True)
    print("--- Phase 4: Running 2-Pair Karpathy Overfit Test ---", flush=True)
    print("=" * 60, flush=True)
    
    set_seed(config["seed"])
    train_dataset = SemiconductorDataset(
        degraded_dir=config["dataset"]["train_degraded_dir"],
        gt_dir=config["dataset"]["train_gt_dir"],
        augment=False,
        preload=True
    )
    
    two_pair_ds = Subset(train_dataset, [0, 1])
    two_pair_loader = DataLoader(two_pair_ds, batch_size=2, shuffle=False)

    model = NAFNetSR(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        width=config["model"]["width"],
        enc_blk_nums=config["model"]["enc_blk_nums"],
        middle_blk_num=config["model"]["middle_blk_num"],
        dec_blk_nums=config["model"]["dec_blk_nums"],
        scale_factor=config["dataset"]["scale_factor"]
    ).to(device)

    criterion = CompositeRestorationLoss(
        w_charbonnier=config["loss_weights"]["charbonnier"],
        w_ms_ssim=config["loss_weights"]["ms_ssim"],
        w_fft=config["loss_weights"]["fft"],
        w_lpips=config["loss_weights"]["lpips"],
        device=device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    batch = next(iter(two_pair_loader))
    deg = batch["degraded"].to(device)
    gt = batch["gt"].to(device)

    best_psnr = 0.0
    for step in range(1, 401):
        model.train()
        optimizer.zero_grad()
        pred = model(deg)
        loss, loss_dict = criterion(pred, gt)
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                model.eval()
                eval_pred = model(deg)
                p0 = calculate_psnr(eval_pred[0:1], gt[0:1])
                p1 = calculate_psnr(eval_pred[1:2], gt[1:2])
                avg_psnr = (p0 + p1) / 2.0
                best_psnr = max(best_psnr, avg_psnr)
                print(f"Step {step:03d}/400 | Total Loss: {loss.item():.5f} | Charb: {loss_dict['loss_charbonnier']:.5f} | PSNR: {avg_psnr:.2f} dB (Sample 0: {p0:.2f} dB, Sample 1: {p1:.2f} dB)", flush=True)
                
                if avg_psnr >= 40.0:
                    print(f"\n>>> Overfit Target Met! PSNR = {avg_psnr:.2f} dB (> 40 dB) at step {step}! <<<", flush=True)
                    break

    log_to_claude_mem("karpathy_overfit_test", {
        "status": "PASSED" if best_psnr >= 40.0 else "FAILED",
        "final_psnr": float(best_psnr),
        "target_psnr": 40.0,
        "steps_run": step
    })
    
    assert best_psnr >= 40.0, f"Overfit test failed: achieved {best_psnr:.2f} dB, expected > 40 dB"
    print("=" * 60, flush=True)
    print(f"--- Karpathy Overfit Test PASSED with Peak PSNR: {best_psnr:.2f} dB ---\n", flush=True)

def train_full(config: dict, device: torch.device, target_epochs: int = None):
    """
    Launch full training with Mixed Precision (AMP) and Cosine Annealing.
    """
    print("\n" + "=" * 60, flush=True)
    print("--- Phase 5: Launching Full NAFNet-SR Training ---", flush=True)
    print("=" * 60, flush=True)
    
    set_seed(config["seed"])
    torch.set_float32_matmul_precision('high')
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    weights_dir = config["train"]["checkpoint_dir"]
    os.makedirs(weights_dir, exist_ok=True)
    best_weight_path = config["train"]["best_weight_path"]

    train_gt_dir = config["dataset"]["train_gt_dir"]
    train_deg_dir = config["dataset"]["train_degraded_dir"]
    val_gt_dir = "./data/val/gt"
    val_deg_dir = "./data/val/degraded"

    print("[DataLoader] Preloading Train and Val datasets into high-speed memory...", flush=True)
    t0 = time.time()
    train_loader, val_loader = get_dataloaders(
        train_degraded_dir=train_deg_dir,
        train_gt_dir=train_gt_dir,
        val_degraded_dir=val_deg_dir,
        val_gt_dir=val_gt_dir,
        batch_size=config["train"]["batch_size"],
        preload=True,
        num_workers=0
    )
    print(f"[DataLoader] Ready in {time.time()-t0:.2f}s! ({len(train_loader.dataset)} Train, {len(val_loader.dataset)} Val).", flush=True)

    # Initialize Model
    model = NAFNetSR(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        width=config["model"]["width"],
        enc_blk_nums=config["model"]["enc_blk_nums"],
        middle_blk_num=config["model"]["middle_blk_num"],
        dec_blk_nums=config["model"]["dec_blk_nums"],
        scale_factor=config["dataset"]["scale_factor"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: NAFNet-SR | Total Trainable Parameters: {total_params:,}", flush=True)

    # Loss & Optimizer
    criterion = CompositeRestorationLoss(
        w_charbonnier=config["loss_weights"]["charbonnier"],
        w_ms_ssim=config["loss_weights"]["ms_ssim"],
        w_fft=config["loss_weights"]["fft"],
        w_lpips=config["loss_weights"]["lpips"],
        device=device
    )

    base_lr = config["train"].get("learning_rate", 3e-4)
    if base_lr > 3e-4:
        base_lr = 3e-4  # Ensure safe learning rate

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    epochs = target_epochs or config["train"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config["train"]["min_lr"]
    )

    amp_enabled = config["train"].get("amp_enabled", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_val_psnr = -1.0
    best_metrics = {}

    # Resume from checkpoint if available
    if os.path.exists(best_weight_path):
        try:
            ckpt = torch.load(best_weight_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_psnr = ckpt.get("val_psnr", 0.0)
                best_metrics = {
                    "epoch": ckpt.get("epoch", 0),
                    "val_psnr": best_val_psnr,
                    "val_ssim": ckpt.get("val_ssim", 0.0),
                    "val_lpips": ckpt.get("val_lpips", 0.0)
                }
                print(f"[Resume] Loaded checkpoint from {best_weight_path} (Starting Epoch: {start_epoch}, Best Val PSNR: {best_val_psnr:.4f} dB)", flush=True)
        except Exception as e:
            print(f"[Warning] Could not resume checkpoint: {e}", flush=True)

    lpips_calc = LPIPSCalculator(device)
    accum_steps = config["train"].get("gradient_accumulation_steps", 1)

    print(f"Starting Training: Epoch {start_epoch} to {epochs} (AMP: {amp_enabled}, Base LR: {base_lr})...\n", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        valid_steps = 0
        
        optimizer.zero_grad()
        start_epoch_time = time.time()
        total_batches = len(train_loader)

        for step, batch in enumerate(train_loader, 1):
            deg = batch["degraded"].to(device)
            gt = batch["gt"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(deg)
                loss, loss_dict = criterion(pred, gt)
                loss_scaled = loss / accum_steps

            # NaN Guard: Skip batch if numerical instability detected
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [Warning] Skipping NaN/Inf at Batch {step}/{total_batches}", flush=True)
                optimizer.zero_grad()
                continue

            scaler.scale(loss_scaled).backward()

            if step % accum_steps == 0 or step == total_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss_dict["total_loss"]
            valid_steps += 1

            # High-frequency 10-batch live logging
            if step % 10 == 0 or step == total_batches:
                avg_step_loss = train_loss / max(1, valid_steps)
                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Batch {step:03d}/{total_batches:03d} | Batch Loss: {loss_dict['total_loss']:.5f} (Charb: {loss_dict['loss_charbonnier']:.5f}, FFT: {loss_dict['loss_fft']:.5f}) | Running: {avg_step_loss:.5f}", flush=True)

        scheduler.step()
        epoch_avg_loss = train_loss / max(1, valid_steps)
        epoch_time = time.time() - start_epoch_time

        # Fast GPU Validation phase
        model.eval()
        val_psnr_list, val_ssim_list, val_lpips_list = [], [], []
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                deg = batch["degraded"].to(device)
                gt = batch["gt"].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    pred = model(deg)
                for i in range(pred.shape[0]):
                    val_psnr_list.append(calculate_psnr(pred[i:i+1], gt[i:i+1]))
                    val_ssim_list.append(calculate_ssim(pred[i:i+1], gt[i:i+1]))
                    if idx < 5:
                        val_lpips_list.append(lpips_calc(pred[i:i+1], gt[i:i+1]))

        val_psnr = float(np.mean(val_psnr_list))
        val_ssim = float(np.mean(val_ssim_list))
        val_lpips = float(np.mean(val_lpips_list)) if val_lpips_list else 0.0

        lr_curr = optimizer.param_groups[0]["lr"]
        print(f"\n>> Epoch [{epoch:02d}/{epochs:02d}] Completed in {epoch_time:.1f}s | LR: {lr_curr:.6f} | Train Loss: {epoch_avg_loss:.5f} | Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f} | Val LPIPS: {val_lpips:.4f} <<", flush=True)

        # Checkpoint if best
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_metrics = {
                "epoch": epoch,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_lpips": val_lpips,
                "train_loss": epoch_avg_loss
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_lpips": val_lpips
            }, best_weight_path)
            print(f"  [*] New Best Checkpoint Saved! (Val PSNR: {val_psnr:.4f} dB -> {best_weight_path})\n", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"--- Full Training Complete! Best Val PSNR: {best_val_psnr:.4f} dB at Epoch {best_metrics.get('epoch', 0)} ---", flush=True)
    print(f"--- Best Val SSIM: {best_metrics.get('val_ssim', 0.0):.4f} | Best Val LPIPS: {best_metrics.get('val_lpips', 0.0):.4f} ---", flush=True)
    print("=" * 60 + "\n", flush=True)

    log_to_claude_mem("full_training_results", {
        "best_epoch": best_metrics.get("epoch", 0),
        "best_val_psnr": float(best_val_psnr),
        "best_val_ssim": float(best_metrics.get("val_ssim", 0.0)),
        "best_val_lpips": float(best_metrics.get("val_lpips", 0.0)),
        "total_epochs": epochs,
        "checkpoint": best_weight_path
    })

def main():
    parser = argparse.ArgumentParser(description="Train NAFNet-SR for Semiconductor Inspection")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--baseline_only", action="store_true", help="Only run Bicubic baseline calculation")
    parser.add_argument("--overfit_test", action="store_true", help="Run 2-pair Karpathy overfit test")
    parser.add_argument("--full_train", action="store_true", help="Run full training pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    if args.baseline_only:
        _, val_loader = get_dataloaders(
            train_degraded_dir=config["dataset"]["train_degraded_dir"],
            train_gt_dir=config["dataset"]["train_gt_dir"],
            val_degraded_dir="./data/val/degraded",
            val_gt_dir="./data/val/gt",
            batch_size=config["train"]["batch_size"],
            num_workers=0
        )
        run_bicubic_baseline(val_loader, device)
    elif args.overfit_test:
        run_overfit_test(config, device)
    elif args.full_train:
        train_full(config, device, target_epochs=args.epochs)
    else:
        _, val_loader = get_dataloaders(
            train_degraded_dir=config["dataset"]["train_degraded_dir"],
            train_gt_dir=config["dataset"]["train_gt_dir"],
            val_degraded_dir="./data/val/degraded",
            val_gt_dir="./data/val/gt",
            batch_size=config["train"]["batch_size"],
            num_workers=0
        )
        run_bicubic_baseline(val_loader, device)
        run_overfit_test(config, device)
        train_full(config, device, target_epochs=args.epochs)

if __name__ == "__main__":
    main()
