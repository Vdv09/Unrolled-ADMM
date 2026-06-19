import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from lensless_helpers.preprocessor import get_dataset_object, get_roi


def load_rgb_chw(path):
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def list_reconstructions(pred_dir):
    return sorted(path for path in pred_dir.glob("*.png"))


def load_gt_roi_chw(dataset_dir, image_id):
    """GT ROI from homework layout (lensless + masks + lensed)."""
    root = Path(dataset_dir)
    lensless = np.array(Image.open(root / "lensless" / f"{image_id}.png").convert("RGB"))
    lensed = np.array(Image.open(root / "lensed" / f"{image_id}.png").convert("RGB"))
    mask = np.load(root / "masks" / f"{image_id}.npy")
    lensed_tensor, _, _ = get_dataset_object(lensed, lensless, mask)
    roi = get_roi(lensed_tensor.numpy())
    return torch.from_numpy(roi).permute(2, 0, 1)


def compute_metrics(gt_dir, pred_dir, device, dataset_dir=None):
    pred_paths = list_reconstructions(pred_dir)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device=device)

    psnr_vals, ssim_vals, mse_vals, lpips_vals = [], [], [], []

    for pred_path in pred_paths:
        image_id = pred_path.stem
        pred = load_rgb_chw(pred_path).unsqueeze(0).to(device=device)

        if dataset_dir is not None:
            gt = load_gt_roi_chw(dataset_dir, image_id).unsqueeze(0).to(device=device)
        else:
            gt_path = gt_dir / f"{image_id}.png"
            gt = load_rgb_chw(gt_path).unsqueeze(0).to(device=device)

        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for {image_id}: pred {tuple(pred.shape)} vs gt {tuple(gt.shape)}. "
                "Use --dataset-dir to compute GT ROI from lensless/masks/lensed."
            )

        psnr_vals.append(peak_signal_noise_ratio(pred, gt, data_range=1.0).item())
        ssim_vals.append(structural_similarity_index_measure(pred, gt, data_range=1.0).item())
        mse_vals.append(mean_squared_error(pred, gt).item())
        lpips_vals.append(lpips(pred, gt).item())

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "mse": float(np.mean(mse_vals)),
        "lpips": float(np.mean(lpips_vals)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="PSNR, SSIM, MSE, LPIPS between GT and reconstructions."
    )
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="GT PNGs (same size as predictions). Optional if --dataset-dir is set.")
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Homework data dir with lensless/, masks/, lensed/ — GT ROI is computed from it.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.dataset_dir is None and args.gt_dir is None:
        parser.error("Provide --gt-dir or --dataset-dir")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    metrics = compute_metrics(args.gt_dir, args.pred_dir, device, args.dataset_dir)

    print(f"PSNR: {metrics['psnr']} dB")
    print(f"SSIM: {metrics['ssim']}")
    print(f"MSE: {metrics['mse']}")
    print(f"LPIPS: {metrics['lpips']}")


if __name__ == "__main__":
    main()
