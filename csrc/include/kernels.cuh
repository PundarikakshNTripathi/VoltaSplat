#pragma once
// CUDA-side declarations: kernels, their launchers, and shared constants. Deliberately free of PyTorch
// headers so the .cu files compile with any nvcc/host-compiler pairing; the torch-facing code lives in
// rasterizer.cpp.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Every 4x4 matrix reaching the kernels is COLUMN-MAJOR: element (row r, col c) is m[c * 4 + r].
// The Python wrapper takes ordinary row-major tensors and transposes them before calling in.
#define VS_MAT(m, r, c) ((m)[(c) * 4 + (r)])

constexpr int TILE = 16;    // tile edge in pixels; one 16x16-thread block renders one tile
constexpr int BATCH = 256;  // Gaussians staged in shared memory at a time

// ---- kernels --------------------------------------------------------------------------------------
__global__ void preprocess_gaussians_kernel(
    int num_gaussians, const float* means3d, const float* cov3d, const float* viewmatrix, const float* projmatrix,
    float focal_x, float focal_y, float tan_fovx, float tan_fovy, int W, int H, int grid_X, int grid_Y,
    float* means2d, float* conics, float* depths, int* radii, int* rects, int* tiles_touched);

__global__ void duplicate_with_keys_kernel(
    int num_gaussians, const float* depths, const int* radii, const int* rects, const int* offsets_inclusive,
    int grid_X, uint64_t* keys, uint32_t* values);

__global__ void identify_tile_ranges_kernel(int num_items, const uint64_t* sorted_keys, int* tile_ranges);

__global__ void render_tiles_kernel(
    int W, int H, int grid_X, const int* tile_ranges, const uint32_t* sorted_ids,
    const float* means2d, const float* conics, const float* colors, const float* opacities,
    float* out_color, float* out_transmittance, int* contributors);

__global__ void render_tiles_backward_kernel(
    int W, int H, int grid_X, const int* tile_ranges, const uint32_t* sorted_ids,
    const float* means2d, const float* conics, const float* colors, const float* opacities,
    const float* out_transmittance, const int* contributors, const float* grad_out_color,
    float* grad_means2d, float* grad_conics, float* grad_colors, float* grad_opacities);

__global__ void preprocess_backward_kernel(
    int num_gaussians, const float* means3d, const float* cov3d, const float* viewmatrix, const float* projmatrix,
    float focal_x, float focal_y, float tan_fovx, float tan_fovy, int W, int H, const int* radii,
    const float* conics, const float* grad_means2d, const float* grad_conics,
    float* grad_means3d, float* grad_cov3d);

// ---- launchers (launch.cu): plain pointers, explicit stream, errors via cudaGetLastError -----------
void launch_preprocess(int N, const float* means3d, const float* cov3d, const float* view, const float* proj,
                       float fx, float fy, float tanx, float tany, int W, int H, int grid_X, int grid_Y,
                       float* means2d, float* conics, float* depths, int* radii, int* rects, int* tiles_touched,
                       cudaStream_t s);
void launch_duplicate(int N, const float* depths, const int* radii, const int* rects, const int* offsets,
                      int grid_X, uint64_t* keys, uint32_t* values, cudaStream_t s);
// CUB radix sort of (key, value) pairs on bits [0, end_bit). Pass temp == nullptr to query temp_bytes.
cudaError_t sort_pairs_launch(void* temp, size_t& temp_bytes, const uint64_t* keys_in, uint64_t* keys_out,
                              const uint32_t* vals_in, uint32_t* vals_out, int n, int end_bit, cudaStream_t s);
void launch_tile_ranges(int total, const uint64_t* sorted_keys, int* tile_ranges, cudaStream_t s);
void launch_render(int W, int H, int grid_X, int grid_Y, const int* tile_ranges, const uint32_t* sorted_ids,
                   const float* means2d, const float* conics, const float* colors, const float* opacities,
                   float* out_color, float* out_T, int* contributors, cudaStream_t s);
void launch_render_backward(int W, int H, int grid_X, int grid_Y, const int* tile_ranges, const uint32_t* sorted_ids,
                            const float* means2d, const float* conics, const float* colors, const float* opacities,
                            const float* out_T, const int* contributors, const float* grad_out,
                            float* g_means2d, float* g_conics, float* g_colors, float* g_opacities, cudaStream_t s);
void launch_preprocess_backward(int N, const float* means3d, const float* cov3d, const float* view, const float* proj,
                                float fx, float fy, float tanx, float tany, int W, int H, const int* radii,
                                const float* conics, const float* g_means2d, const float* g_conics,
                                float* g_means3d, float* g_cov3d, cudaStream_t s);
