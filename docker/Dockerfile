FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip build-essential ninja-build git \
    && rm -rf /var/lib/apt/lists/*

# CUDA 12.8 wheels include Blackwell (sm_120) kernels; Ampere..Hopper are covered too.
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install --no-cache-dir pytest numpy matplotlib ninja

WORKDIR /app
COPY . .

# Compile for the common data-centre / desktop architectures (override with --build-arg for a single GPU).
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN python3 -m pip install --no-cache-dir -e . --no-build-isolation

CMD ["python3", "-m", "pytest", "tests", "-v"]
