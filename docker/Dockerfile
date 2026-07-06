FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Install essential build tools
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python dependency management
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install dependencies and build the CUDA extension
RUN uv pip install --system pytest
RUN uv pip install --system -e . --no-build-isolation

# Default command
CMD ["pytest", "tests/"]
