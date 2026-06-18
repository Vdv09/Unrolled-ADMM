import torch
import torch.nn as nn

from src.model.leadmm5 import LeADMM5
from src.model.drunet import DRUNet
from src.utils.admm_utils import swap_channels, crop_reconstruction_roi


inner_channels_by_size = [
    (32, 64, 116, 128),
    (64, 128, 232, 256)
]

class ModularLeADMM5(nn.Module):
    def __init__(
        self,
        variant,
        number_iterations = 5,
        mu_start = 1e-4,
        tau_start = 2e-4,
        coefficient_to_increase_dimension = 2
    ):
        super().__init__()

        self.preprocessor = self.postprocessor = None

        if variant in ("pre_post", "pre"):
            self.preprocessor = DRUNet(
                in_channels = 3,
                out_channels = 3,
                inner_channels = inner_channels_by_size[variant == "pre"],
            )

        if variant in ("pre_post", "post"):
            self.postprocessor = DRUNet(
                in_channels = 3,
                out_channels = 3,
                inner_channels = inner_channels_by_size[variant == "post"],
            )
        
        self.camera_inversion = LeADMM5(
            number_iterations = number_iterations,
            mu_start = mu_start,
            tau_start = tau_start,
            coefficient_to_increase_dimension = coefficient_to_increase_dimension,
        )

    def forward(self, lensless, psf, **batch):
        if self.preprocessor is not None:
            lensless = swap_channels(lensless)
            lensless = self.preprocessor(lensless)
            lensless = lensless.permute(0, 2, 3, 1)
        
        result = self.camera_inversion(lensless, psf)
        reconstruction = result["reconstruction"]

        if self.postprocessor is not None:
            reconstruction = self.postprocessor(reconstruction)

        return {
            "reconstruction": reconstruction,
            "reconstruction_roi": crop_reconstruction_roi(reconstruction),
        }