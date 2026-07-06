#include <torch/extension.h>
#include "rasterizer.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &rasterize_forward, "VoltaSplat forward rasterization");
    m.def("rasterize_backward", &rasterize_backward, "VoltaSplat backward rasterization");
}
