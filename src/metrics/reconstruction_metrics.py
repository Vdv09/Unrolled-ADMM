from torchmetrics.functional import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from src.metrics.base_metric import BaseMetric
from src.utils import admm_utils


def roi_pair(reconstruction_roi, lensed_roi):
    return (reconstruction_roi.contiguous(), admm_utils.swap_channels(lensed_roi).contiguous())


class Psnr(BaseMetric):
    def __call__(self, reconstruction_roi, lensed_roi, **batch):
        pred, target = roi_pair(reconstruction_roi, lensed_roi)

        return peak_signal_noise_ratio(
            pred, target, data_range = 1.0
        ).item()


class Ssim(BaseMetric):
    def __call__(self, reconstruction_roi, lensed_roi, **batch):
        pred, target = roi_pair(reconstruction_roi, lensed_roi)

        return structural_similarity_index_measure(
            pred, target, data_range = 1.0
        ).item()


class Mse(BaseMetric):
    def __call__(self, reconstruction_roi, lensed_roi, **batch):
        pred, target = roi_pair(reconstruction_roi, lensed_roi)

        return mean_squared_error(pred, target).item()
