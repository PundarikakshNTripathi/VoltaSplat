import torch
import pytest
from voltasplat.modules import SplatRenderer

def test_forward_and_backward():
    W, H = 256, 256
    N = 100
    try:
        means3d = torch.rand(N, 3, device='cuda', dtype=torch.float32)
        cov3d = torch.rand(N, 6, device='cuda', dtype=torch.float32)
        colors = torch.rand(N, 3, device='cuda', dtype=torch.float32, requires_grad=True)
        opacities = torch.rand(N, device='cuda', dtype=torch.float32, requires_grad=True)
        
        viewmatrix = torch.eye(4, device='cuda', dtype=torch.float32)
        projmatrix = torch.eye(4, device='cuda', dtype=torch.float32)
    except RuntimeError as e:
        if "no kernel image is available" in str(e):
            pytest.skip("PyTorch binary does not support this GPU architecture (sm_120).")
        raise
    
    renderer = SplatRenderer(W, H, 256.0, 256.0, 1.0, 1.0)
    
    # Forward Pass
    out_color = renderer(means3d, cov3d, colors, opacities, viewmatrix, projmatrix)
    
    assert out_color.shape == (H, W, 3), f"Expected shape {(H, W, 3)}, got {out_color.shape}"
    assert out_color.is_cuda, "Output should be on CUDA"
    
    # Backward Pass
    loss = out_color.sum()
    loss.backward()
    
    assert colors.grad is not None, "Gradients for colors were not computed."
    assert opacities.grad is not None, "Gradients for opacities were not computed."
    assert colors.grad.shape == (N, 3), "Incorrect gradient shape for colors."
    assert opacities.grad.shape == (N,), "Incorrect gradient shape for opacities."
