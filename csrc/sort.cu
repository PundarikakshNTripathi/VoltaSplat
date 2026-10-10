// Gaussian-to-tile binning: every Gaussian is duplicated once per 16x16 tile its bounding square overlaps,
// keyed by (tile id << 32 | depth bits) so a single global radix sort yields, per tile, a depth-ordered list.
#include "kernels.cuh"

__global__ void duplicate_with_keys_kernel(
    int num_gaussians, const float* __restrict__ depths, const int* __restrict__ radii,
    const int* __restrict__ rects, const int* __restrict__ offsets_inclusive, int grid_X,
    uint64_t* __restrict__ keys, uint32_t* __restrict__ values)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians || radii[idx] <= 0) return;

    int out = (idx == 0) ? 0 : offsets_inclusive[idx - 1];
    // Positive IEEE-754 floats order like their bit patterns, so the depth bits sort correctly as integers.
    const uint32_t depth_bits = __float_as_uint(depths[idx]);
    const int minx = rects[idx * 4 + 0], miny = rects[idx * 4 + 1];
    const int maxx = rects[idx * 4 + 2], maxy = rects[idx * 4 + 3];
    for (int y = miny; y < maxy; ++y) {
        for (int x = minx; x < maxx; ++x) {
            keys[out] = (static_cast<uint64_t>(y * grid_X + x) << 32) | depth_bits;
            values[out] = static_cast<uint32_t>(idx);
            ++out;
        }
    }
}

// After sorting, the entries of one tile are contiguous: record [start, end) for each tile.
__global__ void identify_tile_ranges_kernel(int num_items, const uint64_t* __restrict__ sorted_keys, int* __restrict__ tile_ranges)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_items) return;
    const uint32_t tile = static_cast<uint32_t>(sorted_keys[i] >> 32);
    if (i == 0 || tile != static_cast<uint32_t>(sorted_keys[i - 1] >> 32)) tile_ranges[tile * 2 + 0] = i;
    if (i == num_items - 1 || tile != static_cast<uint32_t>(sorted_keys[i + 1] >> 32)) tile_ranges[tile * 2 + 1] = i + 1;
}
