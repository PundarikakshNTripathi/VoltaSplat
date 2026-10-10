"""Pure-PyTorch reference renderer used as ground truth for the CUDA kernels.

It reproduces the kernel semantics step by step with differentiable tensor ops, so ``torch.autograd`` of this
function is the reference for the hand-derived CUDA backward pass:

  * EWA projection with the first-order Jacobian, 1.3x frustum clamp and +0.3 low-pass term,
  * 3-sigma radius -> bounding square -> set of 16x16 tiles; a Gaussian only contributes to pixels in those tiles,
  * front-to-back compositing in depth order with alpha = min(0.99, opacity * exp(power)), skipping alpha < 1/255,
    and stopping a pixel *before* a Gaussian that would push transmittance below 1e-4.

It is O(N * H * W) and only meant for small scenes.
"""
import torch

TILE = 16


def _cov3d_to_matrix(cov3d):
    c = cov3d
    row0 = torch.stack([c[:, 0], c[:, 3], c[:, 4]], dim=-1)
    row1 = torch.stack([c[:, 3], c[:, 1], c[:, 5]], dim=-1)
    row2 = torch.stack([c[:, 4], c[:, 5], c[:, 2]], dim=-1)
    return torch.stack([row0, row1, row2], dim=1)


def project(means3d, cov3d, viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy, W, H):
    """Returns means2d (N,2), conics (N,3), depths (N,), radii (N,) int64, tile rectangles (N,4) int64, valid (N,) bool."""
    R = viewmatrix[:3, :3]
    t = means3d @ R.T + viewmatrix[:3, 3]
    in_front = t[:, 2] >= 0.2
    tz = torch.where(in_front, t[:, 2], torch.ones_like(t[:, 2]))

    limx, limy = 1.3 * tan_fovx, 1.3 * tan_fovy
    tx = torch.clamp(t[:, 0] / tz, -limx, limx) * tz
    ty = torch.clamp(t[:, 1] / tz, -limy, limy) * tz
    zeros = torch.zeros_like(tz)
    J = torch.stack([
        torch.stack([focal_x / tz, zeros, -(focal_x * tx) / (tz * tz)], dim=-1),
        torch.stack([zeros, focal_y / tz, -(focal_y * ty) / (tz * tz)], dim=-1),
    ], dim=1)                                              # (N, 2, 3)
    T = J @ R                                              # (N, 2, 3)
    cov2 = T @ _cov3d_to_matrix(cov3d) @ T.transpose(1, 2)  # (N, 2, 2)
    cov_xx = cov2[:, 0, 0] + 0.3
    cov_yy = cov2[:, 1, 1] + 0.3
    cov_xy = cov2[:, 0, 1]
    det = cov_xx * cov_yy - cov_xy * cov_xy
    det_ok = det != 0
    det_safe = torch.where(det_ok, det, torch.ones_like(det))
    conics = torch.stack([cov_yy / det_safe, -cov_xy / det_safe, cov_xx / det_safe], dim=-1)

    hom = torch.cat([means3d, torch.ones_like(means3d[:, :1])], dim=-1) @ projmatrix.T
    pw = torch.where(in_front, hom[:, 3], torch.ones_like(hom[:, 3]))
    means2d = torch.stack([(hom[:, 0] / pw + 1.0) * W * 0.5, (hom[:, 1] / pw + 1.0) * H * 0.5], dim=-1)

    with torch.no_grad():
        mid = 0.5 * (cov_xx + cov_yy)
        disc = torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        lam_max = torch.maximum(mid + disc, mid - disc)
        radii = torch.ceil(3.0 * torch.sqrt(lam_max)).to(torch.int64)
        grid_x, grid_y = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
        r = radii.to(means2d.dtype)
        px, py = means2d[:, 0], means2d[:, 1]
        minx = torch.clamp(((px - r) / TILE).trunc().to(torch.int64), 0, grid_x)
        miny = torch.clamp(((py - r) / TILE).trunc().to(torch.int64), 0, grid_y)
        maxx = torch.clamp(((px + r + TILE - 1) / TILE).trunc().to(torch.int64), 0, grid_x)
        maxy = torch.clamp(((py + r + TILE - 1) / TILE).trunc().to(torch.int64), 0, grid_y)
        rects = torch.stack([minx, miny, maxx, maxy], dim=-1)
        touched = (maxx - minx) * (maxy - miny)
        valid = in_front & det_ok & (touched > 0)
    return means2d, conics, t[:, 2], radii, rects, valid


def render_reference(means3d, cov3d, colors, opacities, viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy, W, H):
    means2d, conics, depths, _radii, rects, valid = project(
        means3d, cov3d, viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy, W, H)
    device = means3d.device
    order = torch.argsort(depths.detach(), stable=True)
    order = order[valid[order]]

    ys = torch.arange(H, device=device, dtype=means3d.dtype)[:, None].expand(H, W)
    xs = torch.arange(W, device=device, dtype=means3d.dtype)[None, :].expand(H, W)
    tile_x = (torch.arange(W, device=device) // TILE)[None, :].expand(H, W)
    tile_y = (torch.arange(H, device=device) // TILE)[:, None].expand(H, W)

    dtype = means3d.dtype
    T = torch.ones(H, W, device=device, dtype=dtype)
    C = torch.zeros(H, W, 3, device=device, dtype=dtype)
    active = torch.ones(H, W, device=device, dtype=torch.bool)

    for g in order.tolist():
        member = (tile_x >= rects[g, 0]) & (tile_x < rects[g, 2]) & (tile_y >= rects[g, 1]) & (tile_y < rects[g, 3])
        dx = xs - means2d[g, 0]
        dy = ys - means2d[g, 1]
        power = -0.5 * (conics[g, 0] * dx * dx + 2.0 * conics[g, 1] * dx * dy + conics[g, 2] * dy * dy)
        power_c = torch.clamp(power, max=0.0)  # keeps exp() finite where the pixel is masked out anyway
        alpha = torch.clamp(opacities[g] * torch.exp(power_c), max=0.99)
        ok = member & active & (power <= 0) & (alpha >= 1.0 / 255.0)
        test_T = T * (1.0 - alpha)
        saturate = ok & (test_T < 1e-4)
        add = ok & ~saturate
        w = torch.where(add, alpha * T, torch.zeros_like(T))
        C = C + w[..., None] * colors[g]
        T = torch.where(add, test_T, T)
        active = active & ~saturate
    return C
