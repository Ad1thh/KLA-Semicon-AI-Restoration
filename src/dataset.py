import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor

class SemiconductorDataset(Dataset):
    """
    Dataset for Semiconductor Inspection Image Restoration.
    Loads paired (degraded, gt) .npy float32 arrays with fast concurrent in-memory caching.
    
    IMPORTANT INVARIANTS:
    1. Single-channel float32 grayscale images.
    2. Input pixel values can exceed [0, 1] due to speckle noise -> NEVER clip at dataloader!
    3. Ground truth (GT) images are strictly in [0, 1].
    """
    def __init__(self,
                 degraded_dir: str,
                 gt_dir: str,
                 augment: bool = False,
                 preload: bool = True,
                 scale_factor: int = 2):
        super().__init__()
        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir
        self.augment = augment
        self.preload = preload
        self.scale_factor = scale_factor

        # Find matching pairs
        degraded_files = {os.path.basename(f): f for f in glob.glob(os.path.join(degraded_dir, "*.npy"))}
        gt_files = {os.path.basename(f): f for f in glob.glob(os.path.join(gt_dir, "*.npy"))}

        common_filenames = sorted(list(set(degraded_files.keys()) & set(gt_files.keys())))
        if not common_filenames:
            raise ValueError(f"No matching pairs found between {degraded_dir} and {gt_dir}")

        self.degraded_paths = [degraded_files[fn] for fn in common_filenames]
        self.gt_paths = [gt_files[fn] for fn in common_filenames]
        self.filenames = common_filenames

        self.cached_deg = []
        self.cached_gt = []

        if self.preload:
            def _load_single(pair):
                d_path, g_path = pair
                deg = np.load(d_path).astype(np.float32)
                gt = np.load(g_path).astype(np.float32)

                if deg.ndim == 2:
                    deg = deg[np.newaxis, ...]
                elif deg.ndim == 3 and deg.shape[-1] == 1:
                    deg = deg.transpose(2, 0, 1)

                if gt.ndim == 2:
                    gt = gt[np.newaxis, ...]
                elif gt.ndim == 3 and gt.shape[-1] == 1:
                    gt = gt.transpose(2, 0, 1)

                # Domain Invariant: GT strictly clamped to [0, 1]; Degraded input NEVER clipped!
                gt = np.clip(gt, 0.0, 1.0)
                return torch.from_numpy(deg).float(), torch.from_numpy(gt).float()

            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4 if os.cpu_count() else 8)) as executor:
                results = list(executor.map(_load_single, zip(self.degraded_paths, self.gt_paths)))

            for d_tensor, g_tensor in results:
                self.cached_deg.append(d_tensor)
                self.cached_gt.append(g_tensor)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        if self.preload:
            deg_tensor = self.cached_deg[idx].clone()
            gt_tensor = self.cached_gt[idx].clone()
        else:
            deg = np.load(self.degraded_paths[idx]).astype(np.float32)
            gt = np.load(self.gt_paths[idx]).astype(np.float32)

            if deg.ndim == 2:
                deg = deg[np.newaxis, ...]
            elif deg.ndim == 3 and deg.shape[-1] == 1:
                deg = deg.transpose(2, 0, 1)

            if gt.ndim == 2:
                gt = gt[np.newaxis, ...]
            elif gt.ndim == 3 and gt.shape[-1] == 1:
                gt = gt.transpose(2, 0, 1)

            deg_tensor = torch.from_numpy(deg).float()
            gt_tensor = torch.clamp(torch.from_numpy(gt).float(), 0.0, 1.0)

        # Augmentation (Geometric symmetries: flips and 90-degree rotations)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                deg_tensor = torch.flip(deg_tensor, dims=[-1])
                gt_tensor = torch.flip(gt_tensor, dims=[-1])
            if torch.rand(1).item() > 0.5:
                deg_tensor = torch.flip(deg_tensor, dims=[-2])
                gt_tensor = torch.flip(gt_tensor, dims=[-2])
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                deg_tensor = torch.rot90(deg_tensor, k=k, dims=[-2, -1])
                gt_tensor = torch.rot90(gt_tensor, k=k, dims=[-2, -1])

        return {
            "degraded": deg_tensor,
            "gt": gt_tensor,
            "filename": self.filenames[idx]
        }

def get_dataloaders(train_degraded_dir: str,
                    train_gt_dir: str,
                    val_degraded_dir: str,
                    val_gt_dir: str,
                    batch_size: int = 8,
                    preload: bool = True,
                    num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders."""
    train_dataset = SemiconductorDataset(
        degraded_dir=train_degraded_dir,
        gt_dir=train_gt_dir,
        augment=True,
        preload=preload
    )
    val_dataset = SemiconductorDataset(
        degraded_dir=val_degraded_dir,
        gt_dir=val_gt_dir,
        augment=False,
        preload=preload
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
