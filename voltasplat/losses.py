import torch
import torch.nn.functional as F


def l1_loss(network_output, gt):
    return torch.abs(network_output - gt).mean()


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = (g[:, None] * g[None, :]).expand(channels, 1, window_size, window_size).contiguous()
    return window_2d


def ssim(img1, img2, window_size: int = 11):
    """Structural similarity (Wang et al. 2004) for (H, W, C) or (B, C, H, W) images in [0, 1],
    Gaussian window (sigma = 1.5), per-channel, mean over pixels."""
    if img1.dim() == 3:  # (H, W, C) -> (1, C, H, W)
        img1 = img1.permute(2, 0, 1).unsqueeze(0)
        img2 = img2.permute(2, 0, 1).unsqueeze(0)
    channels = img1.size(1)
    window = _gaussian_window(window_size, 1.5, channels, img1.device, img1.dtype)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def combined_loss(network_output, gt, lambda_dssim: float = 0.2):
    """(1 - lambda) * L1 + lambda * (1 - SSIM), the photometric loss used by 3D Gaussian Splatting."""
    return (1.0 - lambda_dssim) * l1_loss(network_output, gt) + lambda_dssim * (1.0 - ssim(network_output, gt))


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))
