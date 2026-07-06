.PHONY: all setup build test benchmark clean

all: setup build test benchmark

setup:
	uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
	uv pip install --system pytest matplotlib

build:
	uv pip install --system -e . --no-build-isolation

test:
	pytest tests/

benchmark:
	python benchmarks/run_benchmarks.py

clean:
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf __pycache__/
