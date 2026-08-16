import os
import sys
import time
import re
import glob
import subprocess
from datetime import datetime

# Enable ANSI escape sequences in Windows Command Prompt / PowerShell
if os.name == "nt":
    os.system("")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TOTAL_EPOCHS = 10
TOTAL_BATCHES_PER_EPOCH = 320

def get_gpu_info():
    """Query nvidia-smi for GPU metrics without using PyTorch CUDA context."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 4:
                return {
                    "util": parts[0] + "%",
                    "mem_used": f"{parts[1]} MiB",
                    "mem_total": f"{parts[2]} MiB",
                    "temp": f"{parts[3]}C"
                }
    except Exception:
        pass
    return None

def parse_active_log():
    """Parse exact real-time training progress from the active task log file."""
    info = {
        "epoch": 1,
        "total_epochs": TOTAL_EPOCHS,
        "batch": 0,
        "total_batches": TOTAL_BATCHES_PER_EPOCH,
        "batch_loss": "N/A",
        "running_loss": "N/A",
        "val_psnr": None,
        "val_ssim": None,
        "val_lpips": None,
        "latest_line": "Training in progress..."
    }
    try:
        tasks_dir = r"C:\Users\hp\.gemini\antigravity-ide\brain\b6b10320-59ba-47f6-a0f9-a79bfb2a474c\.system_generated\tasks"
        log_files = glob.glob(os.path.join(tasks_dir, "task-*.log"))
        if not log_files:
            return info
        
        latest_file = max(log_files, key=os.path.getmtime)
        with open(latest_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for l in reversed(lines):
            m = re.search(r"\[Epoch\s+(\d+)/(\d+)\]\s+Batch\s+(\d+)/(\d+)\s+\|\s+Batch Loss:\s+([0-9\.]+)\s+.*?\|\s+Running:\s+([0-9\.]+)", l)
            if m:
                info["epoch"] = int(m.group(1))
                info["total_epochs"] = int(m.group(2))
                info["batch"] = int(m.group(3))
                info["total_batches"] = int(m.group(4))
                info["batch_loss"] = m.group(5)
                info["running_loss"] = m.group(6)
                info["latest_line"] = l
                break
            
            m_val = re.search(r">> Epoch\s+\[(\d+)/(\d+)\].*?Val PSNR:\s+([0-9\.]+)\s+dB\s+\|\s+Val SSIM:\s+([0-9\.]+)", l)
            if m_val:
                info["epoch"] = int(m_val.group(1))
                info["total_epochs"] = int(m_val.group(2))
                info["batch"] = TOTAL_BATCHES_PER_EPOCH
                info["val_psnr"] = float(m_val.group(3))
                info["val_ssim"] = float(m_val.group(4))
                info["latest_line"] = l
                break
    except Exception:
        pass
    return info

def get_training_process():
    """Find the primary long-running training process (ignoring this monitor process)."""
    my_pid = str(os.getpid())
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, @{Name='Start';Expression={Get-Date $_.StartTime -Format 'yyyy-MM-dd HH:mm:ss'}}, CPU | Sort-Object Start"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    pid = parts[0]
                    if pid == my_pid:
                        continue
                    start_str = f"{parts[1]} {parts[2]}"
                    try:
                        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                        elapsed_sec = max(0, (datetime.now() - start_dt).total_seconds())
                        if elapsed_sec > 5:
                            return {
                                "pid": pid,
                                "status": "Running (Active GPU Training)",
                                "start_time": start_str,
                                "elapsed_sec": elapsed_sec
                            }
                    except Exception:
                        pass
    except Exception:
        pass
    return {"pid": "N/A", "status": "Completed / Idle", "elapsed_sec": 0, "start_time": "N/A"}

def get_checkpoint_info():
    """Safely inspect checkpoint metadata on CPU."""
    ckpt_path = os.path.join("weights", "nafnet_sr_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    
    mtime = os.path.getmtime(ckpt_path)
    last_mod = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
    info = {"last_updated": last_mod, "size_mb": f"{os.path.getsize(ckpt_path)/(1024*1024):.1f} MB"}
    
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            info["epoch"] = ckpt.get("epoch", 1)
            info["val_psnr"] = ckpt.get("val_psnr", 0.0)
            info["val_ssim"] = ckpt.get("val_ssim", 0.0)
            info["val_lpips"] = ckpt.get("val_lpips", 0.0)
    except Exception:
        pass
    return info

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

def build_dashboard_text(tick=0):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    proc = get_training_process()
    gpu = get_gpu_info()
    ckpt = get_checkpoint_info()
    log_info = parse_active_log()

    curr_epoch = log_info["epoch"]
    total_epochs = log_info["total_epochs"]
    curr_batch = log_info["batch"]
    total_batches = log_info["total_batches"]

    total_steps_all = total_epochs * total_batches
    completed_steps = (curr_epoch - 1) * total_batches + curr_batch
    overall_pct = min(99 if proc["status"].startswith("Running") else 100, int((completed_steps / total_steps_all) * 100)) if total_steps_all > 0 else 0
    epoch_pct = min(100, int((curr_batch / total_batches) * 100)) if total_batches > 0 else 0

    elapsed = proc.get("elapsed_sec", 0)
    if completed_steps > 0 and elapsed > 0:
        sec_per_step = elapsed / completed_steps
        rem_steps = max(0, total_steps_all - completed_steps)
        rem_sec = rem_steps * sec_per_step
    else:
        rem_sec = 0

    spinners = ["-", "\\", "|", "/"]
    spinner = spinners[tick % len(spinners)]

    bar_len = 25
    filled = int(bar_len * (overall_pct / 100.0))
    bar = "=" * filled + ">" if filled < bar_len else "=" * bar_len
    bar = bar.ljust(bar_len, " ")

    saved_epoch = ckpt.get("epoch", 0) if ckpt else 0

    lines = []
    lines.append("=" * 74)
    lines.append(f"  [KLA SEMICONDUCTOR RESTORATION -- LIVE FLICKER-FREE MONITOR] [{spinner}]")
    lines.append(f"  System Time : {now_str}  |  Training Start: {proc.get('start_time', 'N/A')}")
    lines.append("=" * 74)
    lines.append("")
    lines.append("[Process & Compute Status]")
    lines.append(f"  * Process Status   : {proc['status']} (PID: {proc['pid']})")
    if gpu:
        lines.append(f"  * GPU Util / Temp  : {gpu['util']} | {gpu['temp']}")
        lines.append(f"  * GPU VRAM Usage   : {gpu['mem_used']} / {gpu['mem_total']}")
    else:
        lines.append("  * GPU Stats        : [N/A]")
    lines.append("")
    lines.append("[Live Progress Tracking]")
    lines.append(f"  * Overall Progress : [{bar}] {overall_pct}% ({completed_steps}/{total_steps_all} total batches)")
    lines.append(f"  * Active Epoch     : Epoch {curr_epoch} of {total_epochs} (Batch: {curr_batch}/{total_batches} -> {epoch_pct}%)")
    lines.append(f"  * Current Loss     : Batch: {log_info['batch_loss']} | Running Epoch: {log_info['running_loss']}")
    lines.append(f"  * Time Elapsed     : {format_time(elapsed)}")
    lines.append(f"  * Est. Remaining   : ~{format_time(rem_sec)}")
    lines.append(f"  * Latest Log Line  : {log_info['latest_line']}")
    lines.append("")
    lines.append("[Best Validation Checkpoint Saved on Disk]")
    if ckpt and "val_psnr" in ckpt:
        psnr = ckpt.get("val_psnr", 0.0)
        ssim = ckpt.get("val_ssim", 0.0)
        lpips = ckpt.get("val_lpips", 0.0)
        lines.append(f"  * Latest Best Epoch: Epoch {saved_epoch} (Next disk checkpoint at end of Epoch {curr_epoch})")
        lines.append(f"  * Val PSNR         : {psnr:.4f} dB  (Baseline: 23.0065 dB -> +{psnr-23.0065:.2f} dB gain)")
        lines.append(f"  * Val SSIM         : {ssim:.4f}     (Baseline: 0.5286 -> +{ssim-0.5286:.4f} gain)")
        lines.append(f"  * Val LPIPS        : {lpips:.4f}    (Baseline: 0.4428 -> -{0.4428-lpips:.4f} error drop)")
        lines.append(f"  * Disk File Mod    : {ckpt.get('last_updated')} ({ckpt.get('size_mb')})")
    else:
        lines.append("  * Checkpoint       : Initializing...")
    lines.append("")
    lines.append("=" * 74)
    lines.append("  [Updates smoothly in-place | Press Ctrl + C to exit monitor]")
    lines.append("=" * 74)
    return "\n".join(lines)

def main():
    tick = 0
    # Clear screen once at startup
    os.system("cls" if os.name == "nt" else "clear")
    try:
        while True:
            text = build_dashboard_text(tick)
            # In-place ANSI cursor repositioning to top-left (zero flicker)
            sys.stdout.write("\033[H" + text)
            sys.stdout.flush()
            tick += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[Monitor Exited] Background training continues unaffected.")
        sys.exit(0)

if __name__ == "__main__":
    main()
