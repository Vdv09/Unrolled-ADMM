import torch
from lensless_helpers.preprocessor import ALIGNMENT


def swap_channels(x):  # [B, H, W, C] or [H, W, C] -> [B, C, H, W]
    if x.ndim == 3:
        x = x.unsqueeze(0)

    return x.permute(0, 3, 1, 2)


def center_crop(x, top, left, h, w):  # x.shape == [B, C, H, W]
    return x[:, :, top:top + h, left:left + w]


def make_big_picture(x, coefficient_to_increase_dimension):  # move x in center of big picture, x.shape == [B, C, H, W]
    h, w = x.shape[2], x.shape[3]

    new_h, new_w = h * coefficient_to_increase_dimension, w * coefficient_to_increase_dimension

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


def image_gradient(x):  # x.shape == [B, C, H, W]
    gradient_x = torch.roll(x, 1, -1) - x
    gradient_y = torch.roll(x, 1, -2) - x

    return torch.stack([gradient_x, gradient_y], dim=-1)


def image_gradient_transpose(x):  # x.shape == [B, C, H, W, 2]
    gradient_x = x[..., 0]
    gradient_y = x[..., 1]

    gradient_x = torch.roll(gradient_x, -1, -1) - gradient_x
    gradient_y = torch.roll(gradient_y, -1, -2) - gradient_y

    return gradient_x + gradient_y


def make_soft_threshold(x, threshold):
    return torch.sign(x) * torch.max(torch.abs(x) - threshold, torch.zeros_like(x))


def update_u(x, alpha_2, mu, tau):
    new_u = make_soft_threshold(
        image_gradient(x) + alpha_2 / mu,
        tau / mu,
    )

    return new_u


def make_psf_fft(psf):
    psf_in_left_corner = torch.fft.ifftshift(psf, dim=(-2, -1))

    return torch.fft.fft2(psf_in_left_corner)


def make_psiT_psi_fft(h, w, device, dtype):
    example = torch.zeros(1, 1, h, w, device = device, dtype = dtype)
    example[..., 0, 0] = 1

    changes = image_gradient_transpose(image_gradient(example))

    return torch.fft.fft2(changes)


def fft_multiplication(x, psf_fft):
    return torch.fft.ifft2(torch.fft.fft2(x) * psf_fft).real


def fft_multiplication_adj(x, psf_fft):
    return torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(psf_fft)).real


def make_CT_C(x, top, left, h, w):
    answer = torch.zeros_like(x)

    answer[:, :, top:top + h, left:left + w] = 1

    return answer


def update_v(x, psf_fft, alpha_1, lensless_enlarged, CT_C, mu):
    Hx = fft_multiplication(x, psf_fft)

    new_v = (alpha_1 + mu * Hx + lensless_enlarged) / (CT_C + mu)

    return new_v


def update_w(x, alpha_3, mu):
    return torch.max(x + alpha_3 / mu, torch.zeros_like(x))


def update_x(r, denominator_fft_update_x):
    r_fft = torch.fft.fft2(r)

    return torch.fft.ifft2(r_fft / denominator_fft_update_x).real


def admm_step(x, v, u, w, alpha_1, alpha_2, alpha_3, psf_fft, lensless_enlarged, CT_C, denominator_fft_update_x, mu, tau):
    u = update_u(x, alpha_2, mu, tau)
    v = update_v(x, psf_fft, alpha_1, lensless_enlarged, CT_C, mu)
    w = update_w(x, alpha_3, mu)

    r = (mu * w - alpha_3) + image_gradient_transpose(mu * u - alpha_2) + \
        fft_multiplication_adj(mu * v - alpha_1, psf_fft)

    x = update_x(r, denominator_fft_update_x)

    alpha_1 = alpha_1 + mu * (fft_multiplication(x, psf_fft) - v)
    alpha_2 = alpha_2 + mu * (image_gradient(x) - u)
    alpha_3 = alpha_3 + mu * (x - w)

    return x, v, u, w, alpha_1, alpha_2, alpha_3


def crop_reconstruction_roi(reconstruction):
    top, left = ALIGNMENT["top_left"]
    h = ALIGNMENT["height"]
    w = ALIGNMENT["width"]

    return center_crop(reconstruction, top, left, h, w)