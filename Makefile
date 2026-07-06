# VoltaSplat Automation Makefile
# Use these commands to quickly set up, build, test, and benchmark the pipeline.

.PHONY: help setup build rebuild test benchmark clean

help:
	@echo "VoltaSplat Automation Commands:"
	@echo "  make setup      - Initializes uv virtual environment and installs PyTorch (cu124) & dependencies."
	@echo "  make build      - Compiles the custom CUDA extensions via setup.py."
	@echo "  make rebuild    - Cleans the build directory and recompiles from scratch."
	@echo "  make test       - Executes the pytest test suite for forward and backward passes."
	@echo "  make benchmark  - Runs the benchmarking script to generate performance stats and update README."
	@echo "  make clean      - Removes build artifacts, caches, and compiled extensions."
	@echo "  make all        - Runs setup, build, test, and benchmark sequentially."

all: setup build test benchmark

setup:
	@echo "Setting up uv virtual environment and PyTorch dependencies..."
	uv venv
	uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
	uv pip install pytest matplotlib

build:
	@echo "Building PyTorch CUDA Extension..."
	uv pip install -e . --no-build-isolation

rebuild: clean build

test:
	@echo "Executing Unit Tests..."
	pytest tests/ -v

benchmark:
	@echo "Running Benchmarks & Generating Visualizations..."
	python benchmarks/run_benchmarks.py

clean:
	@echo "Cleaning Build Artifacts..."
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf csrc/*.obj
	rm -rf voltasplat/*.pyd
	rm -rf voltasplat/*.so
