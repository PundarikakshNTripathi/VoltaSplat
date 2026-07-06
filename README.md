# VoltaSplat ⚡

A differentiable CUDA rasterization engine built from scratch for 3D Gaussian Splatting. 

Unlike standard implementations, VoltaSplat is designed as a bare-metal educational and experimental backend to understand how complex 3D representations mathematically map to GPU threads via PyTorch C++ extensions.

## Prerequisites
* CUDA Toolkit (11.8 or higher)
* PyTorch (2.0 or higher)
* Ninja build system (`uv pip install ninja`)

## Installation
VoltaSplat uses `setup.py` to compile custom CUDA kernels via PyTorch's extension API.
```bash
# Clone the repository
git clone [https://github.com/yourusername/VoltaSplat.git](https://github.com/yourusername/VoltaSplat.git)
cd VoltaSplat

# Install the extension in editable mode
uv pip install -e . --no-build-isolation --system
```

## Quick Start
```bash
import torch
from voltasplat.modules import SplatRenderer
from voltasplat.cameras import load_colmap_camera

# Initialize renderer and camera
renderer = SplatRenderer()
camera = load_colmap_camera("path/to/colmap/images")

# Render image from 3D Gaussians
gaussians = torch.rand((1000, 14), requires_grad=True, device="cuda")
rendered_image = renderer(gaussians, camera)
```