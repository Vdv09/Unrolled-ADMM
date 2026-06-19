from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from src.datasets.digicam_dataset import DigiCamDataset

from lensless_helpers.preprocessor import get_dataset_object, get_roi
from src.utils.admm_utils import load_picture


class CustomDirDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        self.lensless_dir = self.data_dir / "lensless"
        self.masks_dir = self.data_dir / "masks"
        self.lensed_dir = self.data_dir / "lensed"

        self.image_ids = sorted(p.stem for p in self.lensless_dir.glob("*.png"))
        self.has_lensed = self.lensed_dir.is_dir()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        lensless_img = load_picture(self.lensless_dir / f"{image_id}.png")
        mask_vals = np.load(self.masks_dir / f"{image_id}.npy")

        if self.has_lensed:
            lensed_img = load_picture(self.lensed_dir / f"{image_id}.png")
        else:
            lensed_img = lensless_img

        lensed, lensless, psf = get_dataset_object(lensed_img, lensless_img, mask_vals)

        if psf.ndim == 4 and psf.shape[0] == 1:
            psf = psf.squeeze(0)

        item = {
            "image_id": image_id,
            "lensless": lensless,
            "psf": psf,
            "has_gt": torch.tensor(self.has_lensed),
        }

        if self.has_lensed:
            item["lensed_roi"] = torch.from_numpy(get_roi(lensed.numpy()))
        else:
            roi = get_roi(lensed.numpy())
            item["lensed_roi"] = torch.zeros_like(torch.from_numpy(roi))

        return item


class DigiCamEvalDataset(Dataset):
    def __init__(
        self,
        dataset_split = "test",
        dataset_name = "bezzam/DigiCam-Mirflickr-MultiMask-10K",
        masks_dir = "data/digicam_masks",
    ):
        self.dataset = DigiCamDataset(
            dataset_split=dataset_split,
            dataset_name=dataset_name,
            masks_dir=masks_dir,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {
            "image_id": f"{idx:06d}",
            "has_gt": torch.tensor(True),
            **sample,
        }
