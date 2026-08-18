import os
import time
import argparse
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.utils import set_seed, calculate_psnr, calculate_ssim, LPIPSCalculator, bicubic_upsample
from src.dataset import SemiconductorDataset, get_dataloaders
from src.model import NAFNetSR
from src.loss import CompositeRestorationLoss

def log_experiment_metrics(key: str, data: dict):
    """Experiment state logger recording tracking and evaluation metrics."""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # YAML log
    yaml_file = os.path.join(log_dir, "experiment_metrics.yaml")
    current_data = {}
    if os.path.exists(yaml_file):
        try:
            with open(yaml_file, "r") as f:
                current_data = yaml.safe_load(f) or {}
        except Exception:
            current_data = {}
    current_data[key] = data
    with open(yaml_file, "w") as f:
        yaml.dump(current_data, f, default_flow_style=False)

    # Claude-mem json log
    json_file = os.path.join(log_dir, "claude_mem.json")
    json_data = {}
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                json_data = json.load(f)
        except Exception:
            json_data = {}
    json_data[key] = data
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"[Logger] Recorded '{key}' to {yaml_file} and {json_file}", flush=True)

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

    log_experiment_metrics("bicubic_baseline", results)
    return results

def run_overfit_test(config: dict, device: torch.device):
    """
    STEP 3: KARPATHY SANITY CHECK (OVERFIT 2 PAIRS)
    1. Isolate exactly 2 image pairs from the training set.
    2. Train the NAFNet-SR model on these 2 pairs for 200 iterations using AdamW (lr=1e-3).
    3. Assertion: Verify that loss decreases and validation PSNR on these 2 pairs reaches >= 40.0 dB.
    4. If this check fails, halt execution.
    """
    print("\n" + "=" * 60, flush=True)
    print("--- STEP 3: Running 2-Pair Karpathy Sanity Check ---", flush=True)
    print("=" * 60, flush=True)
    
    set_seed(config["seed"])
    train_deg_dir = config["dataset"]["train_degraded_dir"]
    train_gt_dir = config["dataset"]["train_gt_dir"]

    deg_files = sorted(os.listdir(train_deg_dir))[:2]
    deg_list, gt_list = [], []
    for f in deg_files:
        d = np.load(os.path.join(train_deg_dir, f)).astype(np.float32)
        g = np.load(os.path.join(train_gt_dir, f)).astype(np.float32)
        if d.ndim == 2: d = d[None, ...]
        if g.ndim == 2: g = g[None, ...]
        g = np.clip(g, 0.0, 1.0)
        deg_list.append(torch.from_numpy(d))
        gt_list.append(torch.from_numpy(g))

    deg = torch.stack(deg_list).to(device)
    gt = torch.stack(gt_list).to(device)

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
    print(f"Model Parameters: {total_params:,} ({total_params/1e6:.2f}M) - Budget Target < 4.0M", flush=True)
    assert total_params < 4_000_000, f"Parameter budget exceeded: {total_params} >= 4,000,000"

    criterion = CompositeRestorationLoss(
        w_charbonnier=config["loss_weights"]["charbonnier"],
        w_ms_ssim=config["loss_weights"]["ms_ssim"],
        w_fft=config["loss_weights"]["fft"],
        device=device
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)

    best_psnr = 0.0
    initial_loss = None
    passed_step = None

    for step in range(1, 201):
        model.train()
        optimizer.zero_grad()
        pred = model(deg)
        loss, loss_dict = criterion(pred.float(), gt.float())
        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                model.eval()
                eval_pred = model(deg)
                p0 = calculate_psnr(eval_pred[0:1], gt[0:1])
                p1 = calculate_psnr(eval_pred[1:2], gt[1:2])
                avg_psnr = (p0 + p1) / 2.0
                best_psnr = max(best_psnr, avg_psnr)
                print(f"Step {step:03d}/200 | Total Loss: {loss.item():.5f} (Charb: {loss_dict['loss_charbonnier']:.5f}, MS-SSIM: {loss_dict['loss_ms_ssim']:.5f}, FFT: {loss_dict['loss_fft']:.5f}) | PSNR: {avg_psnr:.2f} dB", flush=True)
                
                if avg_psnr >= 40.0 and passed_step is None:
                    passed_step = step
                    print(f"\n>>> Karpathy Overfit Target Met! PSNR = {avg_psnr:.2f} dB (>= 40.0 dB) at step {step}! <<<", flush=True)

    log_experiment_metrics("karpathy_overfit_test", {
        "status": "PASSED" if best_psnr >= 40.0 else "FAILED",
        "initial_loss": float(initial_loss),
        "final_loss": float(loss.item()),
        "peak_psnr": float(best_psnr),
        "target_psnr": 40.0,
        "passed_step": passed_step or 200,
        "total_steps": 200,
        "trainable_parameters": total_params
    })
    
    assert best_psnr >= 40.0, f"Overfit sanity check failed: peak PSNR was {best_psnr:.2f} dB (< 40.0 dB). Halting training."
    print("=" * 60, flush=True)
    print(f"--- Karpathy Sanity Check PASSED with Peak PSNR: {best_psnr:.2f} dB ---\n", flush=True)

def train_full(config: dict, device: torch.device, target_epochs: int = None):
    """
    STEP 4: FULL TRAINING EXECUTION
    - Train on 80/20 train/val split (seed=42)
    - AdamW (beta1=0.9, beta2=0.999, weight_decay=1e-4)
    - CosineAnnealingLR (T_max=100, eta_min=1e-6) with 5-epoch linear warmup
    - Precision: torch.amp.autocast('cuda', dtype=torch.float16) with GradScaler
    - Batch size: 8 with gradient accumulation steps = 2
    - Checkpoint strictly to weights/nafnet_sr_best.pt
    """
    print("\n" + "=" * 60, flush=True)
    print("--- STEP 4: Launching Full NAFNet-SR Training ---", flush=True)
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

    batch_size = config["train"].get("batch_size", 8)
    grad_accum_steps = config["train"].get("gradient_accumulation_steps", 2)
    epochs = target_epochs or config["train"].get("epochs", 100)

    print(f"[DataLoader] Initializing DataLoaders (batch_size={batch_size}, grad_accum={grad_accum_steps})...", flush=True)
    t0 = time.time()
    train_loader, val_loader = get_dataloaders(
        train_degraded_dir=train_deg_dir,
        train_gt_dir=train_gt_dir,
        val_degraded_dir=val_deg_dir,
        val_gt_dir=val_gt_dir,
        batch_size=batch_size,
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
    print(f"Model Architecture: NAFNet-SR | Total Trainable Parameters: {total_params:,} ({total_params/1e6:.2f}M)", flush=True)
    assert total_params < 4_000_000, f"Parameter budget exceeded: {total_params} >= 4,000,000"

    # Composite Loss
    criterion = CompositeRestorationLoss(
        w_charbonnier=config["loss_weights"]["charbonnier"],
        w_ms_ssim=config["loss_weights"]["ms_ssim"],
        w_fft=config["loss_weights"]["fft"],
        device=device
    ).to(device)

    # Optimizer
    base_lr = config["train"].get("learning_rate", 1e-3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # 5-epoch linear warmup + CosineAnnealingLR (T_max=epochs - 5, eta_min=1e-6)
    warmup_epochs = 5
    if epochs > warmup_epochs:
        warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=config["train"].get("min_lr", 1e-6))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config["train"].get("min_lr", 1e-6))

    # Mixed Precision Scaler
    amp_enabled = config["train"].get("amp_enabled", True) and (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    print(f"Mixed Precision FP16 Autocast: {'ENABLED' if amp_enabled else 'DISABLED'}", flush=True)

    best_val_psnr = -1.0
    best_metrics = {}
    total_batches = len(train_loader)

    print(f"Starting Full Training: 1 to {epochs} Epochs (Base LR: {base_lr}, Effective Batch Size: {batch_size * grad_accum_steps})...\n", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        start_epoch_time = time.time()

        for step, batch in enumerate(train_loader, 1):
            deg = batch["degraded"].to(device)
            gt = batch["gt"].to(device)

            with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.float16):
                pred = model(deg)

            # Compute composite loss in FP32 for numerical stability
            loss, loss_dict = criterion(pred.float(), gt.float())
            scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()

            if (step % grad_accum_steps == 0) or (step == total_batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss_dict["total_loss"]

            if step % 20 == 0 or step == total_batches:
                avg_step_loss = train_loss / step
                print(f"  [Epoch {epoch:02d}/{epochs:02d}] Batch {step:03d}/{total_batches:03d} | Loss: {loss_dict['total_loss']:.5f} (Charb: {loss_dict['loss_charbonnier']:.5f}, MS-SSIM: {loss_dict['loss_ms_ssim']:.5f}, FFT: {loss_dict['loss_fft']:.5f}) | Running Avg: {avg_step_loss:.5f}", flush=True)

        scheduler.step()
        epoch_avg_loss = train_loss / total_batches
        epoch_time = time.time() - start_epoch_time

        # Validation Phase
        model.eval()
        val_psnr_list, val_ssim_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                deg = batch["degraded"].to(device)
                gt = batch["gt"].to(device)
                with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=torch.float16):
                    pred = model(deg)
                for i in range(pred.shape[0]):
                    val_psnr_list.append(calculate_psnr(pred[i:i+1], gt[i:i+1]))
                    val_ssim_list.append(calculate_ssim(pred[i:i+1], gt[i:i+1]))

        val_psnr = float(np.mean(val_psnr_list))
        val_ssim = float(np.mean(val_ssim_list))
        lr_curr = optimizer.param_groups[0]["lr"]

        print(f"\n>> Epoch [{epoch:02d}/{epochs:02d}] Finished in {epoch_time:.1f}s | LR: {lr_curr:.6f} | Train Loss: {epoch_avg_loss:.5f} | Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f} <<", flush=True)

        # Checkpoint if best
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_metrics = {
                "epoch": epoch,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "train_loss": epoch_avg_loss
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim
            }, best_weight_path)
            print(f"  [*] New Best Checkpoint Saved! (Val PSNR: {val_psnr:.4f} dB -> {best_weight_path})\n", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"--- Full Training Complete! Best Val PSNR: {best_val_psnr:.4f} dB at Epoch {best_metrics.get('epoch', 0)} ---", flush=True)
    print(f"--- Best Val SSIM: {best_metrics.get('val_ssim', 0.0):.4f} ---", flush=True)
    print("=" * 60 + "\n", flush=True)

    log_experiment_metrics("full_training_results", {
        "best_epoch": best_metrics.get("epoch", 0),
        "best_val_psnr": float(best_val_psnr),
        "best_val_ssim": float(best_metrics.get("val_ssim", 0.0)),
        "total_epochs": epochs,
        "checkpoint": best_weight_path,
        "trainable_parameters": total_params
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
