#pragma once
#include <torch/extension.h>
#include <vector>

// Torch-facing API (rasterizer.cpp). Matrices are COLUMN-MAJOR 4x4 tensors (the Python wrapper transposes).

// Forward: returns {color[H,W,3], transmittance[H,W], contributors[H,W] (int32), radii[N] (int32),
//                   means2d[N,2], conics[N,3], depths[N], sorted_gaussian_ids[R] (int32), tile_ranges[T,2] (int32)}
// `contributors[p]` is 1 + the position (in the sorted list) of the last Gaussian that touched pixel p, 0 if none.
std::vector<torch::Tensor> rasterize_forward(
    torch::Tensor means3d, torch::Tensor cov3d, torch::Tensor colors, torch::Tensor opacities,
    torch::Tensor viewmatrix_cm, torch::Tensor projmatrix_cm,
    float focal_x, float focal_y, float tan_fovx, float tan_fovy, int W, int H);

// Backward: returns {grad_means3d[N,3], grad_cov3d[N,6], grad_colors[N,3], grad_opacities[N]}
std::vector<torch::Tensor> rasterize_backward(
    int W, int H,
    torch::Tensor means3d, torch::Tensor cov3d,
    torch::Tensor viewmatrix_cm, torch::Tensor projmatrix_cm,
    float focal_x, float focal_y, float tan_fovx, float tan_fovy,
    torch::Tensor radii, torch::Tensor tile_ranges, torch::Tensor sorted_ids,
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors, torch::Tensor opacities,
    torch::Tensor out_transmittance, torch::Tensor contributors, torch::Tensor grad_out_color);
