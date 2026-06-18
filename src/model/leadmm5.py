import torch
import torch.nn as nn

from src.utils import admm_utils


class LeADMM5(nn.Module):
    def __init__(
        self,
        number_iterations = 5,
        mu_start = 1e-4,
        tau_start = 2e-4,
        coefficient_to_increase_dimension = 2
    ):
        super().__init__()

        self.number_iterations = number_iterations
        self.coefficient_to_increase_dimension = coefficient_to_increase_dimension

        self.mu = nn.Parameter(
            torch.full((number_iterations,), mu_start)
        )
        self.tau = nn.Parameter(
            torch.full((number_iterations,), tau_start)
        )

    def reconstruct(self, lensless, psf):
        lensless = admm_utils.swap_channels(lensless)
        psf = admm_utils.swap_channels(psf)

        lensless_enlarged, increase_information = admm_utils.make_big_picture(
            lensless, self.coefficient_to_increase_dimension,
        )

        psf_enlarged, _ = admm_utils.make_big_picture(
            psf, self.coefficient_to_increase_dimension,
        )

        x = torch.zeros_like(lensless_enlarged)
        v = torch.zeros_like(lensless_enlarged)
        u = torch.zeros(
            x.shape + (2,),
            device=x.device,
            dtype=x.dtype,
        )
        w = torch.zeros_like(x)

        alpha_1 = torch.zeros_like(x)
        alpha_2 = torch.zeros_like(u)
        alpha_3 = torch.zeros_like(x)

        psf_fft = admm_utils.make_psf_fft(psf_enlarged)
        CT_C = admm_utils.make_CT_C(x, *increase_information)

        psiT_psi_fft = admm_utils.make_psiT_psi_fft(x.shape[-2], x.shape[-1], x.device, x.dtype)

        for i in range(self.number_iterations):
            current_mu = self.mu[i]
            current_tau = self.tau[i]

            denominator_fft_update_x = current_mu * torch.abs(psf_fft) ** 2 + current_mu * torch.abs(psiT_psi_fft) + current_mu

            x, v, u, w, alpha_1, alpha_2, alpha_3 = admm_utils.admm_step(
                x, v, u, w, alpha_1, alpha_2, alpha_3, psf_fft, lensless_enlarged, CT_C, denominator_fft_update_x, current_mu, current_tau
            )
        
        real_image = admm_utils.center_crop(x, *increase_information)

        return real_image
    
    def forward(self, lensless, psf, **batch):
        reconstruction = self.reconstruct(lensless, psf)
        reconstruction_roi = admm_utils.crop_reconstruction_roi(reconstruction)

        return {
            "reconstruction": reconstruction,
            "reconstruction_roi": reconstruction_roi,
        }
