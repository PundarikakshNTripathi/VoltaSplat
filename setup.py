import os
cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
os.environ["CUDA_HOME"] = cuda_dir
os.environ["CUDA_PATH"] = cuda_dir
os.environ["NVCC"] = cuda_dir + r"\bin\nvcc.exe"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0"

import torch
import torch.utils.cpp_extension
torch.utils.cpp_extension.CUDA_HOME = cuda_dir
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get all cpp and cu files in csrc
csrc_dir = os.path.join(os.path.dirname(__file__), 'csrc')
src_files = glob.glob(os.path.join(csrc_dir, '*.cpp')) + glob.glob(os.path.join(csrc_dir, '*.cu'))

# Extra compiler args
extra_compile_args = {
    'cxx': ['/O2', '/std:c++20', '/Zc:preprocessor'],
    'nvcc': ['-O3', '-std=c++20', '-Xcompiler', '/Zc:preprocessor']
}

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
