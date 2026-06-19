import argparse
import time

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.io_utils import ROOT_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model", default="modular_leadmm5")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--repeats", default=50, type=int)
    parser.add_argument("--height", default=380, type=int)
    parser.add_argument("--width", default=507, type=int)

    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    if args.device == "auto" and not torch.cuda.is_available():
        device = "cpu"

    config = OmegaConf.load(ROOT_PATH / "src/configs/model" / f"{args.model}.yaml")

    if args.variant is not None:
        config.variant = args.variant

    model = instantiate(config).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))

    model.eval()

    lensless = torch.rand(args.batch_size, args.height, args.width, 3, device=device)
    psf = torch.rand(args.batch_size, args.height, args.width, 3, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(lensless = lensless, psf = psf)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(args.repeats):
            model(lensless = lensless, psf = psf)

        if device == "cuda":
            torch.cuda.synchronize()

        all_time = time.perf_counter() - start

    processed = args.batch_size * args.repeats

    model_name = args.model
    
    if "variant" in config:
        model_name += f" ({config.variant})"

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Speed: {processed / all_time:.3f} images/sec")


if __name__ == "__main__":
    main()
