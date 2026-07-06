#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

void rasterize_forward(torch::Tensor dummy);
void rasterize_backward(torch::Tensor dummy);

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

void sort_gaussians(
    uint64_t* d_keys_in,
    uint64_t* d_keys_out,
    uint32_t* d_values_in,
    uint32_t* d_values_out,
    int num_items,
    cudaStream_t stream);

