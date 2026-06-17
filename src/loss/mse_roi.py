import torch.nn.functional as F
from torch import nn

from src.utils import admm_utils


class MseRoiLoss(nn.Module):
    def forward(self, reconstruction_roi, lensed_roi, **batch):
        lensed_roi = admm_utils.swap_channels(lensed_roi)

        loss = F.mse_loss(reconstruction_roi, lensed_roi)
        
        return {
            "loss": loss
        }
