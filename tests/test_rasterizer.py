"""Correctness tests: CUDA rasterizer vs. the pure-PyTorch reference (forward image and autograd gradients)."""
import pytest
import torch

from conftest import make_camera, make_scene, requires_cuda
from voltasplat.modules import rasterize
from voltasplat.reference import render_reference

pytestmark = requires_cuda


def _render(fn, scene, cam, requires_grad=False):
    means, cov, colors, opac = [t.clone().requires_grad_(requires_grad) for t in scene]
    img = fn(means, cov, colors, opac, cam.viewmatrix, cam.projmatrix,
             cam.focal_x, cam.focal_y, cam.tan_fovx, cam.tan_fovy, cam.width, cam.height)
    return img, (means, cov, colors, opac)


@pytest.mark.parametrize("n,seed", [(24, 0), (60, 1), (150, 2)])
def test_forward_matches_reference(n, seed):
    cam = make_camera()
    scene = make_scene(n, seed)
    out, _ = _render(rasterize, scene, cam)
    ref, _ = _render(render_reference, scene, cam)
    assert out.shape == (cam.height, cam.width, 3)
    assert out.abs().sum() > 0
    assert (out - ref).abs().max().item() < 1e-5


@pytest.mark.parametrize("n,seed", [(24, 0), (60, 1), (150, 2)])
def test_gradients_match_autograd_reference(n, seed):
    cam = make_camera()
    scene = make_scene(n, seed)
    weights = torch.randn(cam.height, cam.width, 3, device="cuda", generator=torch.Generator("cuda").manual_seed(seed))

    out, p_cuda = _render(rasterize, scene, cam, requires_grad=True)
    (out * weights).sum().backward()
    ref, p_ref = _render(render_reference, scene, cam, requires_grad=True)
    (ref * weights).sum().backward()

    for name, a, b in zip(["means3d", "cov3d", "colors", "opacities"], p_cuda, p_ref):
        assert a.grad is not None and torch.isfinite(a.grad).all(), name
        denom = b.grad.norm().item() + 1e-8
        rel = (a.grad - b.grad).norm().item() / denom
        assert rel < 1e-4, f"{name}: relative gradient error {rel:.3e}"


def test_large_gaussian_spans_tiles():
    """A single big Gaussian must be rendered in every tile it overlaps, not only the tile holding its centre."""
    cam = make_camera(W=96, H=96, yaw=0.0)
    means = torch.tensor([[0.0, 0.0, 3.0]], device="cuda")
    big = torch.tensor([[0.35, 0.35, 0.35, 0.0, 0.0, 0.0]], device="cuda")   # sigma ~0.6 world units
    colors = torch.ones(1, 3, device="cuda")
    opac = torch.tensor([0.9], device="cuda")
    img = rasterize(means, big, colors, opac, cam.viewmatrix, cam.projmatrix,
                    cam.focal_x, cam.focal_y, cam.tan_fovx, cam.tan_fovy, 96, 96)
    lit_tiles = {(y // 16, x // 16) for y, x in (img[..., 0] > 0.05).nonzero().tolist()}
    assert len(lit_tiles) >= 9


def test_culled_gaussians_get_zero_gradient():
    cam = make_camera()
    means, cov, colors, opac = make_scene(10, seed=3)
    means = means.clone().requires_grad_(True)
    colors = colors.clone().requires_grad_(True)
    img = rasterize(means, cov, colors, opac, cam.viewmatrix, cam.projmatrix,
                    cam.focal_x, cam.focal_y, cam.tan_fovx, cam.tan_fovy, cam.width, cam.height)
    img.sum().backward()
    assert means.grad[-2:].abs().max().item() == 0.0   # the two Gaussians behind the camera
    assert colors.grad[-2:].abs().max().item() == 0.0


def test_empty_and_fully_culled_scenes():
    cam = make_camera()
    z = lambda *s: torch.zeros(*s, device="cuda")  # noqa: E731
    img = rasterize(z(0, 3), z(0, 6), z(0, 3), z(0), cam.viewmatrix, cam.projmatrix,
                    cam.focal_x, cam.focal_y, cam.tan_fovx, cam.tan_fovy, cam.width, cam.height)
    assert img.abs().sum().item() == 0.0
    behind = torch.tensor([[0.0, 0.0, -2.0]], device="cuda")
    cov = torch.tensor([[0.1, 0.1, 0.1, 0, 0, 0]], device="cuda")
    img = rasterize(behind, cov, torch.ones(1, 3, device="cuda"), torch.ones(1, device="cuda"), cam.viewmatrix,
                    cam.projmatrix, cam.focal_x, cam.focal_y, cam.tan_fovx, cam.tan_fovy, cam.width, cam.height)
    assert img.abs().sum().item() == 0.0


def test_fitting_reduces_loss():
    """End-to-end: gradients through means3d, cov3d, colors and opacities are good enough to fit a target image."""
    from voltasplat.gaussians import build_cov3d

    cam = make_camera(W=64, H=64, yaw=0.0)
    target_scene = make_scene(40, seed=7, scale_range=(-2.8, -1.8))
    target, _ = _render(rasterize, target_scene, cam)

    torch.manual_seed(0)
    n = 80
    means = torch.empty(n, 3, device="cuda").uniform_(-1.0, 1.0)
    means[:, 2] = torch.empty(n, device="cuda").uniform_(1.8, 4.5)
    means.requires_grad_(True)
    log_s = torch.full((n, 3), -2.2, device="cuda", requires_grad=True)
    quats = torch.randn(n, 4, device="cuda", requires_grad=True)
    colors = torch.rand(n, 3, device="cuda", requires_grad=True)
    opac = torch.full((n,), 0.5, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([{"params": [means], "lr": 2e-2}, {"params": [log_s, quats], "lr": 2e-2},
                            {"params": [colors, opac], "lr": 5e-2}])

    def step():
        img = rasterize(means, build_cov3d(log_s, quats), colors, opac.clamp(0.01, 0.99), cam.viewmatrix,
                        cam.projmatrix, cam.focal_x, cam.focal_y, cam.tan_fovx, cam.tan_fovy, 64, 64)
        loss = (img - target).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    first = step()
    for _ in range(150):
        last = step()
    assert last < 0.6 * first, (first, last)
