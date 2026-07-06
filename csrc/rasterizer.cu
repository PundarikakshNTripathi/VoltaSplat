#include "rasterizer.cuh"
#include <vector>

__global__ void identify_tile_ranges_kernel(
    int num_items,
    const uint64_t* __restrict__ keys,
    uint2* __restrict__ tile_offsets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_items) return;

    uint64_t key = keys[idx];
    uint32_t tile_id = (uint32_t)(key >> 32);
    
    // Max tile ID is 0xFFFFFFFF for culled Gaussians
    if (tile_id == 0xFFFFFFFF) return;

    if (idx == 0) {
        tile_offsets[tile_id].x = 0;
    } else {
        uint64_t prev_key = keys[idx - 1];
        uint32_t prev_tile_id = (uint32_t)(prev_key >> 32);
        
        if (tile_id != prev_tile_id) {
            tile_offsets[tile_id].x = idx; // start of this tile
            if (prev_tile_id != 0xFFFFFFFF) {
                tile_offsets[prev_tile_id].y = idx; // end of prev tile
            }
        }
    }
    
    if (idx == num_items - 1) {
        tile_offsets[tile_id].y = num_items;
    }
}

std::vector<torch::Tensor> rasterize_forward(
    torch::Tensor means3d,
    torch::Tensor cov3d,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    int W, int H)
{
    int num_gaussians = means3d.size(0);
    auto options = means3d.options();
    auto int_options = options.dtype(torch::kInt32);
    auto uint32_options = options.dtype(torch::kInt32); // Using int32 for compatibility
    auto uint64_options = options.dtype(torch::kInt64);
    
    int grid_X = (W + 15) / 16;
    int grid_Y = (H + 15) / 16;
    
    auto means2d = torch::zeros({num_gaussians, 2}, options);
    auto conics = torch::zeros({num_gaussians, 3}, options);
    auto depths = torch::zeros({num_gaussians}, options);
    auto radii = torch::zeros({num_gaussians}, int_options);
    
    int threads = 256;
    int blocks = (num_gaussians + threads - 1) / threads;
    
    preprocess_gaussians_kernel<<<blocks, threads>>>(
        num_gaussians,
        means3d.data_ptr<float>(),
        cov3d.data_ptr<float>(),
        viewmatrix.data_ptr<float>(),
        projmatrix.data_ptr<float>(),
        focal_x, focal_y, tan_fovx, tan_fovy,
        W, H,
        means2d.data_ptr<float>(),
        conics.data_ptr<float>(),
        depths.data_ptr<float>(),
        radii.data_ptr<int>(),
        grid_X, grid_Y
    );
    cudaDeviceSynchronize();
    
    auto keys_in = torch::zeros({num_gaussians}, uint64_options);
    auto values_in = torch::zeros({num_gaussians}, uint32_options);
    
    generate_keys_kernel<<<blocks, threads>>>(
        num_gaussians,
        depths.data_ptr<float>(),
        radii.data_ptr<int>(),
        means2d.data_ptr<float>(),
        grid_X,
        reinterpret_cast<uint64_t*>(keys_in.data_ptr<int64_t>()),
        reinterpret_cast<uint32_t*>(values_in.data_ptr<int32_t>())
    );
    cudaDeviceSynchronize();
    
    auto keys_out = torch::zeros_like(keys_in);
    auto values_out = torch::zeros_like(values_in);
    
    sort_gaussians(
        reinterpret_cast<uint64_t*>(keys_in.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(keys_out.data_ptr<int64_t>()),
        reinterpret_cast<uint32_t*>(values_in.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(values_out.data_ptr<int32_t>()),
        num_gaussians,
        0
    );
    cudaDeviceSynchronize();
    
    auto tile_offsets = torch::zeros({grid_X * grid_Y, 2}, int_options); // uint2 is 2x uint32
    identify_tile_ranges_kernel<<<blocks, threads>>>(
        num_gaussians,
        reinterpret_cast<uint64_t*>(keys_out.data_ptr<int64_t>()),
        reinterpret_cast<uint2*>(tile_offsets.data_ptr<int32_t>())
    );
    cudaDeviceSynchronize();
    
    auto out_color = torch::zeros({H, W, 3}, options);
    auto out_transmittance = torch::zeros({H, W}, options);
    auto final_index = torch::zeros({H, W}, int_options);
    
    dim3 grid(grid_X, grid_Y, 1);
    dim3 block(16, 16, 1);
    
    render_tiles_kernel<<<grid, block>>>(
        W, H, grid_X, grid_Y,
        reinterpret_cast<uint2*>(tile_offsets.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(values_out.data_ptr<int32_t>()),
        means2d.data_ptr<float>(),
        conics.data_ptr<float>(),
        colors.data_ptr<float>(),
        opacities.data_ptr<float>(),
        out_color.data_ptr<float>(),
        out_transmittance.data_ptr<float>(),
        final_index.data_ptr<int>()
    );
    cudaDeviceSynchronize();
    
    return {out_color, out_transmittance, final_index, radii, means2d, conics, depths, values_out, tile_offsets};
}

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
    torch::Tensor grad_out_color)
{
    int num_gaussians = means2d.size(0);
    auto options = means2d.options();
    
    auto grad_means2d = torch::zeros({num_gaussians, 2}, options);
    auto grad_conics = torch::zeros({num_gaussians, 3}, options);
    auto grad_colors = torch::zeros({num_gaussians, 3}, options);
    auto grad_opacities = torch::zeros({num_gaussians}, options);
    
    int grid_X = (W + 15) / 16;
    int grid_Y = (H + 15) / 16;
    dim3 grid(grid_X, grid_Y, 1);
    dim3 block(16, 16, 1);
    
    render_tiles_backward_kernel<<<grid, block>>>(
        W, H, grid_X, grid_Y,
        reinterpret_cast<uint2*>(tile_offsets.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(sorted_indices.data_ptr<int32_t>()),
        means2d.data_ptr<float>(),
        conics.data_ptr<float>(),
        colors.data_ptr<float>(),
        opacities.data_ptr<float>(),
        out_transmittance.data_ptr<float>(),
        final_index.data_ptr<int>(),
        grad_out_color.data_ptr<float>(),
        grad_means2d.data_ptr<float>(),
        grad_conics.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opacities.data_ptr<float>()
    );
    cudaDeviceSynchronize();
    
    // We only return gradients for things used in rendering loop
    return {grad_means2d, grad_conics, grad_colors, grad_opacities};
}
