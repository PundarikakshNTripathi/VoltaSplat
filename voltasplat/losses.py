import torch
import torch.nn.functional as F

def l1_loss(network_output, gt):
    """
    Standard L1 loss for image reconstruction.
    """
    return torch.abs((network_output - gt)).mean()

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Structural Similarity Index (SSIM) commonly used in 3DGS pipelines.
    (Placeholder implementation for VoltaSplat architecture)
    """
    channel = img1.size(-3)
    # To keep dependencies minimal, use L1 or MSE natively, or standard convolutions.
    # In a full pipeline, this computes mu1, mu2, sigma1_sq, sigma2_sq, sigma12
    return F.mse_loss(img1, img2)

def combined_loss(network_output, gt, lambda_dssim=0.2):
    """
    Combined L1 and D-SSIM loss.
    """
    l1 = l1_loss(network_output, gt)
    ssim_term = 1.0 - ssim(network_output, gt)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * ssim_term
