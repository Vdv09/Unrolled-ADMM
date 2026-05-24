import torch
import torch.nn as nn
from lensless_helpers.preprocessor import get_roi


class ADMM100(nn.Module):
    def __init__(
        self,
        number_iterations,
        mu = 1e-4,
        tau = 2e-4,
        coefficient_to_increase_dimension = 2
    ):
        super().__init__()

        self.number_iterations = number_iterations
        self.mu = mu
        self.tau = tau
        self.coefficient_to_increase_dimension = coefficient_to_increase_dimension
    
    def swap_channels(self, x):  # [B, H, W, C] or [H, W, C] -> [B, C, H, W]
        if x.ndim == 3:
            x = x.unsqueeze(0)

        return x.permute(0, 3, 1, 2)
    
    def center_crop(self, x, top, left, h, w):  # x.shape == [B, C, H, W]
        return x[:, :, top:top + h, left:left + w]

    def make_big_picture(self, x):  # move x in center of big picture, x.shape == [B, C, H, W]
        h, w = x.shape[2], x.shape[3]

        new_h, new_w = h * self.coefficient_to_increase_dimension, w * self.coefficient_to_increase_dimension

        answer = torch.zeros(
            x.shape[0],
            x.shape[1],
            new_h,
            new_w,
            device=x.device,
            dtype=x.dtype,
        )

        top = (new_h - h) // 2
        left = (new_w - w) // 2

        answer[:, :, top:top + h, left:left + w] = x

        return answer, [top, left, h, w]  # information to center_crop in the end
    
    def image_gradient(self, x):  # x.shape == [B, C, H, W]
        gradient_x = torch.roll(x, 1, -1) - x
        gradient_y = torch.roll(x, 1, -2) - x

        return torch.stack([gradient_x, gradient_y], dim=-1)
    
    def image_gradient_transpose(self, x):  # x.shape == [B, C, H, W, 2]
        gradient_x = x[..., 0]
        gradient_y = x[..., 1]

        gradient_x = torch.roll(gradient_x, -1, -1) - gradient_x
        gradient_y = torch.roll(gradient_y, -1, -2) - gradient_y

        return gradient_x + gradient_y
    
    def make_soft_threshold(self, x, threshold):
        return torch.sign(x) * torch.max(torch.abs(x) - threshold, torch.zeros_like(x))
    
    def update_u(self, x, alpha_2):
        new_u = self.make_soft_threshold(
            self.image_gradient(x) + alpha_2 / self.mu,
            self.tau / self.mu,
        )

        return new_u
    
    def make_psf_fft(self, psf):
        psf_in_left_corner = torch.fft.ifftshift(psf, dim=(-2, -1))

        return torch.fft.fft2(psf_in_left_corner)
    
    def make_psiT_psi_fft(self, x):
        example = torch.zeros_like(x)
        example[..., 0, 0] = 1

        changes = self.image_gradient_transpose(self.image_gradient(example))

        return torch.fft.fft2(changes).real
    
    def fft_multiplication(self, x, psf_fft):
        return torch.fft.ifft2(torch.fft.fft2(x) * psf_fft).real
    
    def fft_multiplication_adj(self, x, psf_fft):
        return torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(psf_fft)).real
    
    def make_CT_C(self, x, top, left, h, w):
        answer = torch.zeros_like(x)

        answer[:, :, top:top + h, left:left + w] = 1

        return answer
    
    def update_v(self, x, psf_fft, alpha_1, lensless_enlarged, CT_C):
        Hx = self.fft_multiplication(x, psf_fft)

        new_v = (alpha_1 + self.mu * Hx + lensless_enlarged) / (CT_C + self.mu)

        return new_v
    
    def update_w(self, x, alpha_3):
        return torch.max(x + alpha_3 / self.mu, torch.zeros_like(x))
    
    def update_x(self, r, denominator_fft_update_x):
        r_fft = torch.fft.fft2(r)

        return torch.fft.ifft2(r_fft / denominator_fft_update_x).real
    
    def admm_step(self, x, v, u, w, alpha_1, alpha_2, alpha_3, psf_fft, lensless_enlarged, CT_C, denominator_fft_update_x):
        u = self.update_u(x, alpha_2)
        v = self.update_v(x, psf_fft, alpha_1, lensless_enlarged, CT_C)
        w = self.update_w(x, alpha_3)

        r = (self.mu * w - alpha_3) + self.image_gradient_transpose(self.mu * u - alpha_2) + \
            self.fft_multiplication_adj(self.mu * v - alpha_1, psf_fft)

        x = self.update_x(r, denominator_fft_update_x)

        alpha_1 = alpha_1 + self.mu * (self.fft_multiplication(x, psf_fft) - v)
        alpha_2 = alpha_2 + self.mu * (self.image_gradient(x) - u)
        alpha_3 = alpha_3 + self.mu * (x - w)

        return x, v, u, w, alpha_1, alpha_2, alpha_3

    
    def reconstruct(self, lensless, psf):
        lensless = self.swap_channels(lensless)
        psf = self.swap_channels(psf)

        lensless_enlarged, increase_information = self.make_big_picture(lensless)
        psf_enlarged, _ = self.make_big_picture(psf)

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

        psf_fft = self.make_psf_fft(psf_enlarged)
        CT_C = self.make_CT_C(x, *increase_information)

        psiT_psi_fft = self.make_psiT_psi_fft(x)
        denominator_fft_update_x = self.mu * torch.abs(psf_fft) ** 2 + self.mu * psiT_psi_fft + self.mu

        for _ in range(self.number_iterations):
            x, v, u, w, alpha_1, alpha_2, alpha_3 = self.admm_step(
                x, v, u, w, alpha_1, alpha_2, alpha_3, psf_fft, lensless_enlarged, CT_C, denominator_fft_update_x
            )
        
        real_image = self.center_crop(x, *increase_information)

        return real_image
    
    def forward(self, lensless, psf, **batch):
        reconstruction = self.reconstruct(lensless, psf)

        first_image = reconstruction[0].detach().permute(1, 2, 0).cpu().numpy()
        reconstruction_roi = get_roi(first_image)
        reconstruction_roi = torch.from_numpy(reconstruction_roi).to(lensless.device)

        return {
            "reconstruction": reconstruction,
            "reconstruction_roi": reconstruction_roi,
        }
