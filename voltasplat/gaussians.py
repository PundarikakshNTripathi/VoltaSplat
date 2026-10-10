"""Gaussian parameterisation helpers: covariance = R S S^T R^T from log-scales and quaternions."""
import torch
import torch.nn.functional as F


def quaternion_to_rotation(q):
    """Unit quaternion (w, x, y, z), shape (N, 4) -> rotation matrices (N, 3, 3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def build_cov3d(log_scales, quats):
    """Covariance entries ``[xx, yy, zz, xy, xz, yz]`` (N, 6) from per-axis log-scales (N, 3) and quaternions (N, 4)."""
    R = quaternion_to_rotation(quats)
    M = R * torch.exp(log_scales)[:, None, :]  # R @ diag(s)
    S = M @ M.transpose(1, 2)
    return torch.stack([S[:, 0, 0], S[:, 1, 1], S[:, 2, 2], S[:, 0, 1], S[:, 0, 2], S[:, 1, 2]], dim=-1)
