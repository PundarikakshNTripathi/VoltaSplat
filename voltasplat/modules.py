"""PyTorch interface to the CUDA rasterizer.

Conventions
-----------
* ``viewmatrix`` is the ordinary row-major 4x4 world->view matrix, ``projmatrix`` the row-major 4x4
  world->clip matrix (``P @ viewmatrix``). The wrapper transposes both for the kernels.
* ``cov3d`` holds the six unique entries of each symmetric 3D covariance as ``[xx, yy, zz, xy, xz, yz]``.
* Gradients flow to ``means3d``, ``cov3d``, ``colors`` and ``opacities``. The camera (view/projection,
  focal lengths) is treated as fixed.
"""
import torch
import torch.nn as nn

import voltasplat._C as _C


class _RasterizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3d, cov3d, colors, opacities, viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy, W, H):
        view_cm = viewmatrix.t().contiguous()
        proj_cm = projmatrix.t().contiguous()
        (color, transmittance, contributors, radii, means2d, conics, _depths, sorted_ids, tile_ranges) = _C.rasterize_forward(
            means3d, cov3d, colors, opacities, view_cm, proj_cm, focal_x, focal_y, tan_fovx, tan_fovy, W, H
        )
        ctx.save_for_backward(means3d, cov3d, colors, opacities, view_cm, proj_cm, radii, tile_ranges, sorted_ids,
                              means2d, conics, transmittance, contributors)
        ctx.cam = (focal_x, focal_y, tan_fovx, tan_fovy, W, H)
        return color

    @staticmethod
    def backward(ctx, grad_out_color):
        (means3d, cov3d, colors, opacities, view_cm, proj_cm, radii, tile_ranges, sorted_ids,
         means2d, conics, transmittance, contributors) = ctx.saved_tensors
        focal_x, focal_y, tan_fovx, tan_fovy, W, H = ctx.cam
        g_means3d, g_cov3d, g_colors, g_opacities = _C.rasterize_backward(
            W, H, means3d, cov3d, view_cm, proj_cm, focal_x, focal_y, tan_fovx, tan_fovy, radii, tile_ranges,
            sorted_ids, means2d, conics, colors, opacities, transmittance, contributors, grad_out_color.contiguous(),
        )
        return g_means3d, g_cov3d, g_colors, g_opacities, None, None, None, None, None, None, None, None


def rasterize(means3d, cov3d, colors, opacities, viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy, W, H):
    """Differentiable rendering of N 3D Gaussians to an (H, W, 3) float32 image."""
    return _RasterizeFunction.apply(
        means3d.contiguous(), cov3d.contiguous(), colors.contiguous(), opacities.contiguous(),
        viewmatrix, projmatrix, float(focal_x), float(focal_y), float(tan_fovx), float(tan_fovy), int(W), int(H),
    )


class SplatRenderer(nn.Module):
    def __init__(self, W, H, focal_x, focal_y, tan_fovx, tan_fovy):
        super().__init__()
        self.W, self.H = W, H
        self.focal_x, self.focal_y = focal_x, focal_y
        self.tan_fovx, self.tan_fovy = tan_fovx, tan_fovy

    @classmethod
    def from_camera(cls, camera):
        return cls(camera.width, camera.height, camera.focal_x, camera.focal_y, camera.tan_fovx, camera.tan_fovy)

    def forward(self, means3d, cov3d, colors, opacities, viewmatrix, projmatrix):
        return rasterize(means3d, cov3d, colors, opacities, viewmatrix, projmatrix,
                         self.focal_x, self.focal_y, self.tan_fovx, self.tan_fovy, self.W, self.H)
