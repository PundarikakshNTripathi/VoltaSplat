"""End-to-end demo: fit 3D Gaussians to a target image through the differentiable CUDA rasterizer.

The target is a procedurally generated 256x256 image (smooth background, coloured discs, diagonal stripes, a ring),
so the demo needs no data download. The Gaussians are free in 3D (position, anisotropic scale, rotation, colour,
opacity) and are optimised with Adam on the 3DGS photometric loss (0.8 * L1 + 0.2 * (1 - SSIM)).

    python examples/fit_image.py --steps 800 --gaussians 4000 --out examples/fit_result.png
"""
import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from voltasplat.cameras import Camera  # noqa: E402
from voltasplat.gaussians import build_cov3d  # noqa: E402
from voltasplat.losses import combined_loss, psnr, ssim  # noqa: E402
from voltasplat.modules import rasterize  # noqa: E402


def make_target(size, device):
    ys, xs = torch.meshgrid(torch.linspace(0, 1, size, device=device), torch.linspace(0, 1, size, device=device), indexing="ij")
    img = torch.stack([0.25 + 0.5 * xs, 0.30 + 0.4 * ys, 0.55 - 0.3 * xs * ys], dim=-1)
    discs = [((0.25, 0.30, 0.13), (0.95, 0.25, 0.20)), ((0.70, 0.28, 0.10), (0.20, 0.80, 0.30)),
             ((0.35, 0.72, 0.16), (0.15, 0.35, 0.95)), ((0.78, 0.70, 0.12), (0.98, 0.85, 0.15))]
    for (cx, cy, r), col in discs:
        mask = ((xs - cx) ** 2 + (ys - cy) ** 2 < r * r).float()[..., None]
        img = img * (1 - mask) + torch.tensor(col, device=device) * mask
    stripes = (((xs + ys) * 14).floor() % 2 == 0).float()[..., None] * ((xs > 0.52) & (ys > 0.48) & (xs < 0.95)).float()[..., None]
    img = img * (1 - 0.5 * stripes) + 0.5 * stripes * torch.tensor([1.0, 1.0, 1.0], device=device)
    ring = (((xs - 0.5) ** 2 + (ys - 0.5) ** 2).sqrt() - 0.37).abs() < 0.012
    img = torch.where(ring[..., None], torch.tensor([0.05, 0.05, 0.1], device=device), img)
    return img.clamp(0, 1).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--gaussians", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_result.png"))
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    S = a.size
    fov = math.radians(50.0)
    cam = Camera(torch.eye(3), torch.zeros(3), fov, fov, S, S, device=dev)
    target = make_target(S, dev)

    n = a.gaussians
    half = math.tan(fov / 2) * 3.0                               # half extent of the view at depth 3
    means = torch.stack([(torch.rand(n, device=dev) * 2 - 1) * half, (torch.rand(n, device=dev) * 2 - 1) * half,
                         3.0 + 0.2 * torch.randn(n, device=dev)], dim=-1)
    px_world = 2 * half / S
    log_s = torch.full((n, 3), math.log(2.5 * px_world), device=dev)
    quats = torch.randn(n, 4, device=dev)
    colors = torch.rand(n, 3, device=dev) * 0.5 + 0.25
    logit_opacity = torch.full((n,), 0.0, device=dev)
    params = [means, log_s, quats, colors, logit_opacity]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": [means], "lr": 1.5 * px_world},
        {"params": [log_s], "lr": 0.02}, {"params": [quats], "lr": 0.01},
        {"params": [colors], "lr": 0.03}, {"params": [logit_opacity], "lr": 0.05},
    ])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.001 ** (1.0 / a.steps))

    def render():
        return rasterize(means, build_cov3d(log_s, quats), colors.clamp(0, 1), torch.sigmoid(logit_opacity),
                         cam.viewmatrix, cam.projmatrix, cam.focal_x, cam.focal_y, cam.tan_fovx, cam.tan_fovy, S, S)

    history = []
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(a.steps + 1):
        img = render()
        loss = combined_loss(img, target)
        if step % 50 == 0 or step == a.steps:
            with torch.no_grad():
                history.append({"step": step, "loss": loss.item(), "psnr": psnr(img, target).item(), "ssim": ssim(img, target).item()})
            print(f"step {step:4d}  loss {history[-1]['loss']:.4f}  PSNR {history[-1]['psnr']:.2f} dB  SSIM {history[-1]['ssim']:.4f}")
        if step == a.steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    with torch.no_grad():
        final = render().clamp(0, 1)
    result = {"gaussians": n, "steps": a.steps, "resolution": S, "seconds": elapsed, "ms_per_step": 1000 * elapsed / a.steps,
              "initial": history[0], "final": history[-1]}
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump({"result": result, "history": history}, f, indent=2)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
    axs[0].imshow(target.cpu()); axs[0].set_title("target")
    axs[1].imshow(final.cpu()); axs[1].set_title(f"{n:,} Gaussians, {a.steps} steps\nPSNR {history[-1]['psnr']:.1f} dB")
    axs[2].plot([h["step"] for h in history], [h["psnr"] for h in history], color="#4c6ef5")
    axs[2].set_xlabel("step"); axs[2].set_ylabel("PSNR (dB)"); axs[2].grid(alpha=0.25); axs[2].set_title("fitting curve")
    for ax in axs[:2]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
