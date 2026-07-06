# CUB DeviceRadixSort Implementation Specification for 3D Gaussian Splatting

## 1. Architectural Context
In 3D Gaussian Splatting (3DGS), rasterization requires millions of projected Gaussians to be sorted by their respective screen-space tiles and their view-space depth. This is necessary to perform correct front-to-back (or back-to-front) alpha compositing. 

To achieve real-time performance, sorting must be delegated to the GPU using `cub::DeviceRadixSort::SortPairs`. This specification outlines the strict memory layouts, bitwise key composition, and API execution patterns required for the CUDA implementation.

## 2. Key-Value Data Structure
Radix sort operates on key-value pairs. The rendering pipeline requires grouping Gaussians by Tile ID, and sorting them by Depth within those tiles.

* **Keys (`uint64_t`):** A 64-bit unsigned integer comprising two packed 32-bit values.
  * **Top 32 bits (MSB):** The Screen Tile ID.
  * **Bottom 32 bits (LSB):** The View-Space Depth.
  * *Sorting Logic:* Because radix sort processes bits from LSB to MSB, placing the Tile ID in the most significant bits ensures the final array is heavily grouped by tile. Within each tile block, the splats remain perfectly sorted by depth.
* **Values (`uint32_t`):** The original array index of the 3D Gaussian. This allows the rasterizer to fetch the correct covariance and color data after sorting.

## 3. Float-to-Integer Depth Conversion
Radix sort operates purely on bits. Standard IEEE-754 floating-point depth values cannot be directly bitwise-sorted if they contain negative numbers. However, in 3DGS, culled Gaussians generally have positive view-space depth (they are in front of the camera). 

To safely sort floating-point depth using an integer radix sort, the float must be reinterpreted as a `uint32_t`. 

### Bitwise Packing Implementation:
```cpp
// Assuming depth is a positive float and tile_id is a uint32_t
uint32_t depth_int = *reinterpret_cast<uint32_t*>(&depth);
uint64_t sort_key = (static_cast<uint64_t>(tile_id) << 32) | depth_int;

```

## 4. The Two-Pass CUB API Pattern

`cub::DeviceRadixSort::SortPairs` requires a strict two-pass execution model to calculate and manage its internal device memory.

### API Signature:

```cpp
cub::DeviceRadixSort::SortPairs(
    void* d_temp_storage,           // Device allocation for temporary memory
    size_t& temp_storage_bytes,     // Size of the temporary memory (output from pass 1, input to pass 2)
    const KeyT* d_keys_in,          // Pointer to unsorted 64-bit keys
    KeyT* d_keys_out,               // Pointer to sorted 64-bit keys
    const ValueT* d_values_in,      // Pointer to unsorted 32-bit indices
    ValueT* d_values_out,           // Pointer to sorted 32-bit indices
    int num_items,                  // Total number of overlapping Gaussians to sort
    int begin_bit = 0,              // Starting bit (usually 0)
    int end_bit = sizeof(KeyT) * 8, // Ending bit (usually 64)
    cudaStream_t stream = 0         // Target CUDA stream
);

```

### Execution Implementation:

```cpp
// 1. Initialize variables
void* d_temp_storage = nullptr;
size_t temp_storage_bytes = 0;

// 2. Pass 1: Query necessary temporary storage size
// Passing d_temp_storage as nullptr signals CUB to only calculate temp_storage_bytes
cub::DeviceRadixSort::SortPairs(
    d_temp_storage, temp_storage_bytes,
    d_keys_in, d_keys_out,
    d_values_in, d_values_out,
    num_instances, 
    0, 64
);

// 3. Allocate Temporary Storage
// Note: In a highly optimized training loop, this allocation should be cached or 
// handled by a custom memory pool to avoid cudaMalloc overhead on every frame.
cudaMalloc(&d_temp_storage, temp_storage_bytes);

// 4. Pass 2: Execute the Sort
cub::DeviceRadixSort::SortPairs(
    d_temp_storage, temp_storage_bytes,
    d_keys_in, d_keys_out,
    d_values_in, d_values_out,
    num_instances, 
    0, 64
);

// 5. Cleanup (If not using a persistent memory pool)
cudaFree(d_temp_storage);

```

## 5. Performance and Memory Management Rules

1. **Double Buffering:** CUB's `SortPairs` is out-of-place. It strictly requires separate, pre-allocated input and output buffers (`d_keys_in` vs `d_keys_out`, and `d_values_in` vs `d_values_out`). These buffers must be correctly sized to `num_instances * sizeof(type)` prior to kernel invocation.
2. **Allocation Overhead:** Repeatedly calling `cudaMalloc` and `cudaFree` for `d_temp_storage` during every forward pass will severely degrade FPS/training iterations. The `temp_storage_bytes` requirement scales predictably with `num_instances`. The engine should allocate a persistent block of VRAM for temporary storage during initialization and only reallocate if `num_instances` exceeds the capacity of the current buffer.
3. **Stream Synchronization:** If integrating into PyTorch, the CUB sort MUST be executed on the current PyTorch CUDA stream (`at::cuda::getCurrentCUDAStream()`), not the default stream (`0`), to prevent race conditions during the computational graph execution.