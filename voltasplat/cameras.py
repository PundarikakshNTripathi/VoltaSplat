import torch
import math

class Camera:
    """
    VoltaSplat Camera object holding extrinsics and intrinsics for rendering.
    """
    def __init__(self, R, T, fovX, fovY, width, height, device="cuda"):
        self.device = device
        self.width = width
        self.height = height
        
        self.fovX = fovX
        self.fovY = fovY
        self.tan_fovx = math.tan(fovX * 0.5)
        self.tan_fovy = math.tan(fovY * 0.5)
        self.focal_x = (0.5 * width) / self.tan_fovx
        self.focal_y = (0.5 * height) / self.tan_fovy
        
        # Build view matrix
        viewmatrix = torch.eye(4, dtype=torch.float32, device=device)
        viewmatrix[:3, :3] = R.transpose(0, 1)
        viewmatrix[:3, 3] = T
        self.viewmatrix = viewmatrix
        
        # Build projection matrix
        zfar = 100.0
        znear = 0.01
        P = torch.zeros(4, 4, dtype=torch.float32, device=device)
        P[0, 0] = 1.0 / self.tan_fovx
        P[1, 1] = 1.0 / self.tan_fovy
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0
        
        self.projmatrix = (P @ viewmatrix).transpose(0, 1)
