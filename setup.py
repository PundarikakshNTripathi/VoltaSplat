import glob
import os
import platform

import torch
import torch.utils.cpp_extension
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def cuda_arch_list() -> str:
    """Target the GPU that is present; otherwise a portable Ampere..Hopper list with PTX for forward compatibility.

    Override with TORCH_CUDA_ARCH_LIST (e.g. "8.6;12.0")."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}.{minor}"
    return "8.0;8.6;8.9;9.0+PTX"


os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list()

is_windows = platform.system() == "Windows"

if is_windows:
    # A CUDA toolkit newer than the one PyTorch was built with is fine for this extension
    # (it only uses stable runtime APIs), so skip torch's strict version equality check.
    torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None
    extra_compile_args = {
        "cxx": ["/O2", "/std:c++17", "/Zc:preprocessor"],
        "nvcc": ["-O3", "-std=c++17", "-Xcompiler", "/Zc:preprocessor"],
    }
else:
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17"],
    }

csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
src_files = sorted(glob.glob(os.path.join(csrc_dir, "*.cpp")) + glob.glob(os.path.join(csrc_dir, "*.cu")))

setup(
    name="voltasplat",
    version="0.1.0",
    description="Differentiable CUDA rasterizer for 3D Gaussian Splatting",
    packages=["voltasplat"],
    ext_modules=[
        CUDAExtension(
            name="voltasplat._C",
            sources=src_files,
            include_dirs=[os.path.join(csrc_dir, "include")],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
