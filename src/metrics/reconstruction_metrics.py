from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from src.metrics.base_metric import BaseMetric
from src.utils import admm_utils


class Psnr(BaseMetric):
    def __call__(self, reconstruction_roi, lensed_roi, **batch):
        lensed_roi = admm_utils.swap_channels(lensed_roi)

        return peak_signal_noise_ratio(
            reconstruction_roi, lensed_roi, data_range=1.0
        ).item()


class Ssim(BaseMetric):
    def __call__(self, reconstruction_roi, lensed_roi, **batch):
        lensed_roi = admm_utils.swap_channels(lensed_roi)

        return structural_similarity_index_measure(
            reconstruction_roi, lensed_roi, data_range=1.0
        ).item()


class Mse(BaseMetric):
    def __call__(self, reconstruction_roi, lensed_roi, **batch):
        lensed_roi = admm_utils.swap_channels(lensed_roi)

        return mean_squared_error(reconstruction_roi, lensed_roi).item()
