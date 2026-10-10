import math

import torch


class Camera:
    """Pinhole camera. ``R`` (3x3) is the camera-to-world rotation and ``T`` (3,) the world->view translation,
    the same convention as the original 3D Gaussian Splatting code base.

    ``viewmatrix`` (world->view) and ``projmatrix`` (world->clip) are ordinary row-major 4x4 tensors."""

    def __init__(self, R, T, fovX, fovY, width, height, device="cuda"):
        self.device = device
        self.width = width
        self.height = height
        self.fovX, self.fovY = fovX, fovY
        self.tan_fovx = math.tan(fovX * 0.5)
        self.tan_fovy = math.tan(fovY * 0.5)
        self.focal_x = (0.5 * width) / self.tan_fovx
        self.focal_y = (0.5 * height) / self.tan_fovy

        view = torch.eye(4, dtype=torch.float32, device=device)
        view[:3, :3] = R.to(device=device, dtype=torch.float32).transpose(0, 1)
        view[:3, 3] = T.to(device=device, dtype=torch.float32)
        self.viewmatrix = view

        zfar, znear = 100.0, 0.01
        P = torch.zeros(4, 4, dtype=torch.float32, device=device)
        P[0, 0] = 1.0 / self.tan_fovx
        P[1, 1] = 1.0 / self.tan_fovy
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0  # clip-space w = view-space z
        self.projmatrix = P @ view
