import torch
import torch.nn as nn
import voltasplat._C as _C

class _RasterizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3d, cov3d, colors, opacities, viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy, W, H):
        out_color, out_transmittance, final_index, radii, means2d, conics, depths, sorted_indices, tile_offsets = _C.rasterize_forward(
            means3d, cov3d, colors, opacities, viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy, W, H
        )
        ctx.save_for_backward(tile_offsets, sorted_indices, means2d, conics, colors, opacities, out_transmittance, final_index)
        ctx.W = W
        ctx.H = H
        return out_color

    @staticmethod
    def backward(ctx, grad_out_color):
        tile_offsets, sorted_indices, means2d, conics, colors, opacities, out_transmittance, final_index = ctx.saved_tensors
        W = ctx.W
        H = ctx.H
        
        grad_means2d, grad_conics, grad_colors, grad_opacities = _C.rasterize_backward(
            W, H, tile_offsets, sorted_indices, means2d, conics, colors, opacities, out_transmittance, final_index, grad_out_color.contiguous()
        )
        
        return None, None, grad_colors, grad_opacities, None, None, None, None, None, None, None, None

class SplatRenderer(nn.Module):
    def __init__(self, W, H, focal_x, focal_y, tan_fovx, tan_fovy):
        super().__init__()
        self.W = W
        self.H = H
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.tan_fovx = tan_fovx
        self.tan_fovy = tan_fovy
        
    def forward(self, means3d, cov3d, colors, opacities, viewmatrix, projmatrix):
        return _RasterizeFunction.apply(
            means3d.contiguous(), cov3d.contiguous(), colors.contiguous(), opacities.contiguous(), 
            viewmatrix.contiguous(), projmatrix.contiguous(), 
            self.focal_x, self.focal_y, self.tan_fovx, self.tan_fovy, self.W, self.H
        )
