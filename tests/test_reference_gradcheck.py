"""CPU-only check of the *reference* renderer: float64 finite differences vs autograd.

The CUDA backward pass is validated against this reference's autograd (test_rasterizer.py); this test shows the
reference's own gradients are right, so the chain of evidence does not rest on the reference being "obviously" correct.
It needs no GPU, so CI can run it.
"""
import math

import torch

from voltasplat.reference import render_reference

W, H = 24, 16


def _scene(dtype=torch.float64):
    g = torch.Generator().manual_seed(11)
    n = 4
    means = torch.empty(n, 3, dtype=dtype).uniform_(-0.6, 0.6, generator=g)
    means[:, 2] = torch.empty(n, dtype=dtype).uniform_(2.0, 4.0, generator=g)
    cov = torch.zeros(n, 6, dtype=dtype)
    cov[:, :3] = torch.empty(n, 3, dtype=dtype).uniform_(0.01, 0.04, generator=g)
    cov[:, 3:] = torch.empty(n, 3, dtype=dtype).uniform_(-0.004, 0.004, generator=g)
    colors = torch.rand(n, 3, dtype=dtype, generator=g)
    opac = torch.empty(n, dtype=dtype).uniform_(0.4, 0.9, generator=g)
    return means, cov, colors, opac


def _camera(dtype=torch.float64):
    fov = math.radians(60.0)
    tan = math.tan(fov / 2)
    fx = 0.5 * W / tan
    fy = 0.5 * H / (tan * H / W)
    view = torch.eye(4, dtype=dtype)
    view[:3, 3] = torch.tensor([0.05, -0.02, 0.1], dtype=dtype)
    P = torch.zeros(4, 4, dtype=dtype)
    P[0, 0] = 1.0 / tan
    P[1, 1] = 1.0 / (tan * H / W)
    P[2, 2] = 100.0 / 99.99
    P[2, 3] = -(100.0 * 0.01) / 99.99
    P[3, 2] = 1.0
    return view, P @ view, fx, fy, tan, tan * H / W


def test_reference_gradients_match_finite_differences():
    view, proj, fx, fy, tx, ty = _camera()
    means, cov, colors, opac = [t.requires_grad_(True) for t in _scene()]

    def f(m, c, col, o):
        return render_reference(m, c, col, o, view, proj, fx, fy, tx, ty, W, H)

    assert f(means, cov, colors, opac).abs().sum() > 0
    assert torch.autograd.gradcheck(f, (means, cov, colors, opac), eps=1e-6, atol=1e-5, rtol=1e-4, nondet_tol=0.0)
