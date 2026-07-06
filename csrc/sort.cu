#include "rasterizer.cuh"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

__global__ void generate_keys_kernel(
    const int num_gaussians,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const float* __restrict__ means2d,
    const int grid_X,
    uint64_t* __restrict__ sort_keys,
    uint32_t* __restrict__ sort_values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    // Skip if culled
    if (radii[idx] <= 0) {
        // Place culled Gaussians at the end by assigning max key
        sort_keys[idx] = 0xFFFFFFFFFFFFFFFF;
        sort_values[idx] = idx;
        return;
    }

    float p_x = means2d[idx*2+0];
    float p_y = means2d[idx*2+1];
    
    // Assign to a single tile based on center for now
    // In full 3DGS, a Gaussian can touch multiple tiles and requires duplication.
    uint32_t tile_x = min(max((int)(p_x / 16.0f), 0), grid_X - 1);
    uint32_t tile_y = max((int)(p_y / 16.0f), 0);
    uint32_t tile_id = tile_y * grid_X + tile_x;

    float depth = depths[idx];
    uint32_t depth_int = *reinterpret_cast<uint32_t*>(&depth);
    
    // Key: Top 32 bits = Tile ID, Bottom 32 bits = Depth
    uint64_t sort_key = (static_cast<uint64_t>(tile_id) << 32) | depth_int;
    
    sort_keys[idx] = sort_key;
    sort_values[idx] = idx;
}

void sort_gaussians(
    uint64_t* d_keys_in,
    uint64_t* d_keys_out,
    uint32_t* d_values_in,
    uint32_t* d_values_out,
    int num_items,
    cudaStream_t stream)
{
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Pass 1: query temp storage size
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items,
        0, 64,
        stream
    );

    // Allocate temp storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Pass 2: perform sort
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items,
        0, 64,
        stream
    );

    // Cleanup
    cudaFree(d_temp_storage);
}
