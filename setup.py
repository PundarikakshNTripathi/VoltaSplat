import os
import platform
import glob
from setuptools import setup
import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Set target CUDA architectures
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6;9.0+PTX"

# Cross-platform specific configurations
is_windows = platform.system() == "Windows"

if is_windows:
    # Use user's specific CUDA path for their Windows machine if available
    cuda_dir = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3")
    if os.path.exists(cuda_dir):
        os.environ["CUDA_HOME"] = cuda_dir
        os.environ["CUDA_PATH"] = cuda_dir
        os.environ["NVCC"] = os.path.join(cuda_dir, "bin", "nvcc.exe")
        torch.utils.cpp_extension.CUDA_HOME = cuda_dir
    torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None
    
    # MSVC specific compiler arguments
    extra_compile_args = {
        'cxx': ['/O2', '/std:c++20', '/Zc:preprocessor'],
        'nvcc': ['-O3', '-std=c++20', '-Xcompiler', '/Zc:preprocessor']
    }
else:
    # Linux (GCC) standard compiler arguments
    extra_compile_args = {
        'cxx': ['-O3', '-std=c++20'],
        'nvcc': ['-O3', '-std=c++20']
    }

# Get all cpp and cu files in csrc
csrc_dir = os.path.join(os.path.dirname(__file__), 'csrc')
src_files = glob.glob(os.path.join(csrc_dir, '*.cpp')) + glob.glob(os.path.join(csrc_dir, '*.cu'))

setup(
    name='voltasplat',
    version='0.1.0',
    description='Differentiable CUDA Rasterizer for 3D Gaussian Splatting',
    ext_modules=[
        CUDAExtension(
            name='voltasplat._C',
            sources=src_files,
            include_dirs=[os.path.join(csrc_dir, 'include')],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
