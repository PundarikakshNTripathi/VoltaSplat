// Kernel launchers: the only place that uses <<<...>>> and CUB, so the torch-facing code can stay in plain C++.
#include <cub/cub.cuh>

#include "kernels.cuh"

static constexpr int THREADS = 256;
static inline int blocks_for(int n) { return (n + THREADS - 1) / THREADS; }

void launch_preprocess(int N, const float* means3d, const float* cov3d, const float* view, const float* proj,
                       float fx, float fy, float tanx, float tany, int W, int H, int grid_X, int grid_Y,
                       float* means2d, float* conics, float* depths, int* radii, int* rects, int* tiles_touched,
                       cudaStream_t s) {
    preprocess_gaussians_kernel<<<blocks_for(N), THREADS, 0, s>>>(N, means3d, cov3d, view, proj, fx, fy, tanx, tany, W, H,
                                                                  grid_X, grid_Y, means2d, conics, depths, radii, rects,
                                                                  tiles_touched);
}

void launch_duplicate(int N, const float* depths, const int* radii, const int* rects, const int* offsets, int grid_X,
                      uint64_t* keys, uint32_t* values, cudaStream_t s) {
    duplicate_with_keys_kernel<<<blocks_for(N), THREADS, 0, s>>>(N, depths, radii, rects, offsets, grid_X, keys, values);
}

cudaError_t sort_pairs_launch(void* temp, size_t& temp_bytes, const uint64_t* keys_in, uint64_t* keys_out,
                              const uint32_t* vals_in, uint32_t* vals_out, int n, int end_bit, cudaStream_t s) {
    return cub::DeviceRadixSort::SortPairs(temp, temp_bytes, keys_in, keys_out, vals_in, vals_out, n, 0, end_bit, s);
}

void launch_tile_ranges(int total, const uint64_t* sorted_keys, int* tile_ranges, cudaStream_t s) {
    identify_tile_ranges_kernel<<<blocks_for(total), THREADS, 0, s>>>(total, sorted_keys, tile_ranges);
}

void launch_render(int W, int H, int grid_X, int grid_Y, const int* tile_ranges, const uint32_t* sorted_ids,
                   const float* means2d, const float* conics, const float* colors, const float* opacities,
                   float* out_color, float* out_T, int* contributors, cudaStream_t s) {
    render_tiles_kernel<<<dim3(grid_X, grid_Y), dim3(TILE, TILE), 0, s>>>(
        W, H, grid_X, tile_ranges, sorted_ids, means2d, conics, colors, opacities, out_color, out_T, contributors);
}

void launch_render_backward(int W, int H, int grid_X, int grid_Y, const int* tile_ranges, const uint32_t* sorted_ids,
                            const float* means2d, const float* conics, const float* colors, const float* opacities,
                            const float* out_T, const int* contributors, const float* grad_out, float* g_means2d,
                            float* g_conics, float* g_colors, float* g_opacities, cudaStream_t s) {
    render_tiles_backward_kernel<<<dim3(grid_X, grid_Y), dim3(TILE, TILE), 0, s>>>(
        W, H, grid_X, tile_ranges, sorted_ids, means2d, conics, colors, opacities, out_T, contributors, grad_out,
        g_means2d, g_conics, g_colors, g_opacities);
}

void launch_preprocess_backward(int N, const float* means3d, const float* cov3d, const float* view, const float* proj,
                                float fx, float fy, float tanx, float tany, int W, int H, const int* radii,
                                const float* conics, const float* g_means2d, const float* g_conics,
                                float* g_means3d, float* g_cov3d, cudaStream_t s) {
    preprocess_backward_kernel<<<blocks_for(N), THREADS, 0, s>>>(N, means3d, cov3d, view, proj, fx, fy, tanx, tany, W, H,
                                                                 radii, conics, g_means2d, g_conics, g_means3d, g_cov3d);
}
