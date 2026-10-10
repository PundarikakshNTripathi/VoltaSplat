import math

import pytest
import torch

from voltasplat.cameras import Camera
from voltasplat.gaussians import build_cov3d

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)


def _rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def make_camera(W=80, H=64, fov_deg=60.0, yaw=0.15):
    fov = math.radians(fov_deg)
    R = _rotation_y(yaw)                      # camera-to-world rotation
    T = torch.tensor([0.1, -0.05, 0.2])       # world->view translation
    return Camera(R, T, fov, 2 * math.atan(math.tan(fov / 2) * H / W), W, H, device="cuda")


def make_scene(n, seed=0, scale_range=(-3.2, -1.2), device="cuda"):
    """Random Gaussians in front of the camera, with a few very large ones that span many tiles
    and a few behind the camera (must be culled)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(n, 3).uniform_(-1.2, 1.2, generator=g)
    means[:, 2] = torch.empty(n).uniform_(1.5, 5.0, generator=g)
    log_scales = torch.empty(n, 3).uniform_(*scale_range, generator=g)
    log_scales[: max(1, n // 12)] += 1.4                    # big splats covering several tiles
    quats = torch.randn(n, 4, generator=g)
    colors = torch.rand(n, 3, generator=g)
    opacities = torch.empty(n).uniform_(0.25, 0.95, generator=g)
    means[-2:, 2] = -1.0                                    # behind the camera -> culled
    cov3d = build_cov3d(log_scales, quats)
    return [x.to(device).contiguous() for x in (means, cov3d, colors, opacities)]
