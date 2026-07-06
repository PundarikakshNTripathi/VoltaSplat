#pragma once
#include <torch/extension.h>

void rasterize_forward(torch::Tensor dummy);
void rasterize_backward(torch::Tensor dummy);
