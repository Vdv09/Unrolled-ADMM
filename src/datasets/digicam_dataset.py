from torch.utils.data import Dataset
from datasets import load_dataset
import os
import numpy as np
import torch

from lensless_helpers.preprocessor import get_dataset_object, get_roi

class DigiCamDataset(Dataset):
    def __init__(
        self,
        dataset_split,
        dataset_name = "bezzam/DigiCam-Mirflickr-MultiMask-10K",
        masks_dir = "data/digicam_masks",
    ):
        self.dataset = load_dataset(dataset_name, split=dataset_split)
        self.masks_dir = masks_dir
        self.masks_cache = {}
    
    def get_mask(self, mask_id):
        if mask_id not in self.masks_cache:
            mask_path = os.path.join(self.masks_dir, f"mask_{mask_id}.npy")
            self.masks_cache[mask_id] = np.load(mask_path)
        
        return self.masks_cache[mask_id]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset[idx]

        lensed, lensless, psf = get_dataset_object(
            row["lensed"],
            row["lensless"],
            self.get_mask(row["mask_label"])
        )

        return {
            "lensless": lensless,
            "lensed_roi": torch.from_numpy(get_roi(lensed.numpy())),
            "psf": psf
        }