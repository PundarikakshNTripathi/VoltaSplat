#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

#include <vector>

std::vector<torch::Tensor> rasterize_forward(
    torch::Tensor means3d,
    torch::Tensor cov3d,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    int W, int H);

std::vector<torch::Tensor> rasterize_backward(
    int W, int H,
    torch::Tensor tile_offsets,
    torch::Tensor sorted_indices,
    torch::Tensor means2d,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor out_transmittance,
    torch::Tensor final_index,
    torch::Tensor grad_out_color);

__global__ void preprocess_gaussians_kernel(
    const int num_gaussians,
    const float* __restrict__ means3d,
    const float* __restrict__ cov3d,
    const float* __restrict__ viewmatrix,
    const float* __restrict__ projmatrix,
    const float focal_x,
    const float focal_y,
    const float tan_fovx,
    const float tan_fovy,
    const int W,
    const int H,
    float* __restrict__ means2d,
    float* __restrict__ conics,
    float* __restrict__ depths,
    int* __restrict__ radii,
    const int grid_X,
    const int grid_Y);

__global__ void generate_keys_kernel(
    const int num_gaussians,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const float* __restrict__ means2d,
    const int grid_X,
    uint64_t* __restrict__ sort_keys,
    uint32_t* __restrict__ sort_values);

__global__ void render_tiles_kernel(
    const int W, const int H,
    const int grid_X, const int grid_Y,
    const uint2* __restrict__ tile_offsets,
    const uint32_t* __restrict__ sorted_indices,
    const float* __restrict__ means2d,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ out_color,
    float* __restrict__ out_transmittance,
    int* __restrict__ final_index);

__global__ void render_tiles_backward_kernel(
    const int W, const int H,
    const int grid_X, const int grid_Y,
    const uint2* __restrict__ tile_offsets,
    const uint32_t* __restrict__ sorted_indices,
    const float* __restrict__ means2d,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    const float* __restrict__ out_transmittance,
    const int* __restrict__ final_index,
    const float* __restrict__ grad_out_color,
    float* __restrict__ grad_means2d,
    float* __restrict__ grad_conics,
    float* __restrict__ grad_colors,
    float* __restrict__ grad_opacities);

void sort_gaussians(
    uint64_t* d_keys_in,
    uint64_t* d_keys_out,
    uint32_t* d_values_in,
    uint32_t* d_values_out,
    int num_items,
    cudaStream_t stream);

