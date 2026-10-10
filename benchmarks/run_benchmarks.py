"""Measured benchmark of the CUDA rasterizer (no hard-coded numbers).

For each scene size N a random Gaussian cloud is rendered at the chosen resolution:
  * forward time           : CUDA events around ``rasterize`` (median of --iters timed runs after warm-up)
  * forward + backward time: same, including ``loss.backward()`` through means3d/cov3d/colors/opacities
  * peak GPU memory        : ``torch.cuda.max_memory_allocated`` during one forward+backward (inputs included)
  * Gaussian-tile pairs    : number of (Gaussian, tile) entries sorted per frame

Scenes use Gaussians whose screen-space sigma is roughly 1.5-5 px (the regime of a trained 3DGS model), so the
tile-pair count per Gaussian is realistic. Results go to benchmarks/metrics.json, charts to benchmarks/images/, and the
README table between the BENCH markers is regenerated.
"""
import argparse
import json
import math
import os
import platform
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from voltasplat.cameras import Camera  # noqa: E402
from voltasplat.gaussians import build_cov3d  # noqa: E402
from voltasplat.modules import rasterize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def make_scene(n, W, H, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    depth = torch.empty(n).uniform_(2.0, 8.0, generator=g)
    half_w = depth * math.tan(math.radians(30.0))
    half_h = half_w * H / W
    means = torch.stack([(torch.rand(n, generator=g) * 2 - 1) * half_w,
                         (torch.rand(n, generator=g) * 2 - 1) * half_h, depth], dim=-1)
    px_per_unit = (0.5 * W / math.tan(math.radians(30.0))) / depth          # pixels per world unit at each depth
    sigma_px = torch.empty(n).uniform_(1.5, 5.0, generator=g)
    log_scales = torch.log(sigma_px / px_per_unit)[:, None].repeat(1, 3) + 0.3 * torch.randn(n, 3, generator=g)
    quats = torch.randn(n, 4, generator=g)
    cov3d = build_cov3d(log_scales, quats)
    colors = torch.rand(n, 3, generator=g)
    opac = torch.empty(n).uniform_(0.1, 0.9, generator=g)
    return [t.to(device).contiguous() for t in (means, cov3d, colors, opac)]


def median_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def bench(n, W, H, iters):
    cam = Camera(torch.eye(3), torch.zeros(3), math.radians(60), 2 * math.atan(math.tan(math.radians(30)) * H / W), W, H)
    means, cov, colors, opac = make_scene(n, W, H)
    args = (cam.viewmatrix, cam.projmatrix, cam.focal_x, cam.focal_y, cam.tan_fovx, cam.tan_fovy, W, H)

    def fwd():
        return rasterize(means, cov, colors, opac, *args)

    def fwd_bwd():
        m, c, col, o = [t.clone().requires_grad_(True) for t in (means, cov, colors, opac)]
        rasterize(m, c, col, o, *args).sum().backward()

    fwd_ms = median_ms(fwd, 3, iters)
    both_ms = median_ms(fwd_bwd, 2, max(5, iters // 2))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fwd_bwd()
    torch.cuda.synchronize()
    peak_mb = (torch.cuda.max_memory_allocated() - base) / 2**20

    from voltasplat import _C
    pairs = int(_C.rasterize_forward(means, cov, colors, opac, cam.viewmatrix.t().contiguous(),
                                     cam.projmatrix.t().contiguous(), cam.focal_x, cam.focal_y, cam.tan_fovx,
                                     cam.tan_fovy, W, H)[7].numel())
    return {"n": n, "forward_ms": fwd_ms, "forward_backward_ms": both_ms, "peak_extra_mb": peak_mb,
            "gaussian_tile_pairs": pairs, "fps_forward": 1000.0 / fwd_ms}


def update_readme(readme, rows, meta):
    start, end = "<!-- BENCHMARK_START -->", "<!-- BENCHMARK_END -->"
    if not os.path.exists(readme):
        print(f"{readme} not found; skipping README update")
        return
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    if start not in text:
        print("README markers not found; skipping README update")
        return
    lines = [
        f"Measured on {meta['gpu']} (torch {meta['torch']}, CUDA {meta['cuda']}), {meta['W']}x{meta['H']} output, "
        f"median of CUDA-event timings; scenes of random Gaussians with 1.5-5 px screen-space sigma.",
        "",
        "| Gaussians | forward (ms) | forward FPS | forward+backward (ms) | peak extra GPU memory (MB) | Gaussian-tile pairs |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['n']:,} | {r['forward_ms']:.2f} | {r['fps_forward']:.0f} | {r['forward_backward_ms']:.2f} | "
                     f"{r['peak_extra_mb']:.0f} | {r['gaussian_tile_pairs']:,} |")
    body = "\n".join(lines)
    text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: f"{start}\n{body}\n{end}", text, flags=re.S)
    with open(readme, "w", encoding="utf-8") as f:
        f.write(text)


def plot(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ns, [r["forward_ms"] for r in rows], "o-", color="#4c6ef5", label="forward")
    ax.plot(ns, [r["forward_backward_ms"] for r in rows], "s-", color="#e8590c", label="forward + backward")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Gaussians")
    ax.set_ylabel("Time (ms, median)")
    ax.set_title("VoltaSplat render time vs scene size")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "performance_scaling.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ns, [r["peak_extra_mb"] for r in rows], "^-", color="#2f9e44")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Gaussians")
    ax.set_ylabel("Peak extra GPU memory, forward+backward (MB)")
    ax.set_title("VoltaSplat memory vs scene size")
    ax.grid(True, which="both", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "memory_scaling.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 50_000, 100_000, 500_000, 1_000_000])
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--readme", default=os.path.join(HERE, "..", "README.md"))
    a = ap.parse_args()

    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(), "W": a.width, "H": a.height}
    rows = []
    for n in a.sizes:
        r = bench(n, a.width, a.height, a.iters)
        rows.append(r)
        print(f"N={n:>9,}  fwd {r['forward_ms']:8.2f} ms ({r['fps_forward']:7.1f} FPS)  fwd+bwd {r['forward_backward_ms']:8.2f} ms  "
              f"mem {r['peak_extra_mb']:8.0f} MB  pairs {r['gaussian_tile_pairs']:,}")
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump({"meta": meta, "results": rows}, f, indent=2)
    plot(rows, os.path.join(HERE, "images"))
    update_readme(a.readme, rows, meta)


if __name__ == "__main__":
    main()
