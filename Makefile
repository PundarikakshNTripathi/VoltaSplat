# VoltaSplat automation.
#   make all        setup -> build -> test -> benchmark -> example
#   TORCH_INDEX=cu124 make setup   to pick a different CUDA wheel index (cu128 is required for Blackwell / sm_120)
# On Windows run from an "x64 Native Tools" prompt (MSVC on PATH) with a POSIX shell providing `make`.

TORCH_INDEX ?= cu128
PY ?= python

.PHONY: help setup build rebuild test test-cpu benchmark example clean all

help:
	@echo "make setup      create a uv virtualenv and install PyTorch ($(TORCH_INDEX)) + test/plot dependencies"
	@echo "make build      compile the CUDA extension in place (voltasplat/_C)"
	@echo "make rebuild    clean + build"
	@echo "make test       full suite: CUDA rasterizer vs PyTorch reference (forward + gradients), fitting, edge cases"
	@echo "make test-cpu   GPU-free check: float64 gradcheck of the reference renderer"
	@echo "make benchmark  measure forward / forward+backward / memory, regenerate charts and README table"
	@echo "make example    fit 4,000 Gaussians to a procedural image (examples/fit_image.py)"
	@echo "make clean      remove build artifacts"

all: setup build test benchmark example

setup:
	uv venv
	uv pip install setuptools wheel ninja numpy pytest matplotlib
	uv pip install torch --index-url https://download.pytorch.org/whl/$(TORCH_INDEX) --extra-index-url https://pypi.org/simple

build:
	uv pip install -e . --no-build-isolation

rebuild: clean build

test:
	$(PY) -m pytest tests -v

test-cpu:
	$(PY) -m pytest tests/test_reference_gradcheck.py -v

benchmark:
	PYTHONPATH=. $(PY) benchmarks/run_benchmarks.py

example:
	PYTHONPATH=. $(PY) examples/fit_image.py --steps 800 --gaussians 4000

clean:
	rm -rf build/ *.egg-info/ .pytest_cache/
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -f voltasplat/*.pyd voltasplat/*.so csrc/*.obj csrc/*.exp csrc/*.lib
