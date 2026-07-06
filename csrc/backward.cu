#include "rasterizer.cuh"
#include <cuda_runtime.h>

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
    float* __restrict__ grad_opacities)
{
    // Block maps to a 16x16 tile
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int pix_x = tile_x * 16 + threadIdx.x;
    int pix_y = tile_y * 16 + threadIdx.y;
    
    bool inside = (pix_x < W && pix_y < H);
    int pix_idx = pix_y * W + pix_x;
    
    // Gradients from loss w.r.t the pixel color
    float dL_dpix_r = 0.0f, dL_dpix_g = 0.0f, dL_dpix_b = 0.0f;
    float final_T = 1.0f;
    int last_idx = 0;
    
    if (inside) {
        dL_dpix_r = grad_out_color[pix_idx*3+0];
        dL_dpix_g = grad_out_color[pix_idx*3+1];
        dL_dpix_b = grad_out_color[pix_idx*3+2];
        final_T = out_transmittance[pix_idx];
        last_idx = final_index[pix_idx];
    }
    
    uint2 range = tile_offsets[tile_y * grid_X + tile_x];
    uint32_t start_idx = range.x;
    
    // Collaborative max last_idx in this tile
    __shared__ int max_last_idx;
    if (threadIdx.x == 0 && threadIdx.y == 0) max_last_idx = 0;
    __syncthreads();
    
    if (inside && last_idx > 0) {
        atomicMax(&max_last_idx, last_idx);
    }
    __syncthreads();
    
    int end_idx_block = max_last_idx;
    
    // Back-to-front batched traversal
    int num_batches = (end_idx_block - start_idx + 255) / 256;
    int thread_linear_id = threadIdx.y * 16 + threadIdx.x;
    
    float T = final_T;
    float accum_rec_r = 0.0f;
    float accum_rec_g = 0.0f;
    float accum_rec_b = 0.0f;
    
    __shared__ float3 collected_means[256];
    __shared__ float3 collected_conics[256];
    __shared__ float3 collected_colors[256];
    __shared__ float collected_opacities[256];
    
    for (int b = num_batches - 1; b >= 0; --b) {
        int batch_start = start_idx + b * 256;
        int fetch_idx = batch_start + thread_linear_id;
        
        if (fetch_idx <= end_idx_block && fetch_idx >= start_idx) {
            uint32_t g_idx = sorted_indices[fetch_idx];
            collected_means[thread_linear_id] = make_float3(means2d[g_idx*2+0], means2d[g_idx*2+1], 0.0f);
            collected_conics[thread_linear_id] = make_float3(conics[g_idx*3+0], conics[g_idx*3+1], conics[g_idx*3+2]);
            collected_colors[thread_linear_id] = make_float3(colors[g_idx*3+0], colors[g_idx*3+1], colors[g_idx*3+2]);
            collected_opacities[thread_linear_id] = opacities[g_idx];
        }
        
        __syncthreads();
        
        int num_in_batch = min(256, end_idx_block - batch_start + 1);
        
        if (inside) {
            for (int i = num_in_batch - 1; i >= 0; --i) {
                int global_i = batch_start + i;
                if (global_i > last_idx) continue;
                if (global_i < start_idx) break;
                
                float3 mean = collected_means[i];
                float3 conic = collected_conics[i];
                float opacity = collected_opacities[i];
                float3 color = collected_colors[i];
                
                float dx = (float)pix_x - mean.x;
                float dy = (float)pix_y - mean.y;
                
                float power = -0.5f * (conic.x * dx * dx + 2.0f * conic.y * dx * dy + conic.z * dy * dy);
                if (power > 0.0f) continue;
                
                float alpha = min(0.99f, opacity * expf(power));
                if (alpha < 1.0f / 255.0f) continue;
                
                float T_before = T / (1.0f - alpha);
                
                float weight = alpha * T_before;
                float dL_dci_r = dL_dpix_r * weight;
                float dL_dci_g = dL_dpix_g * weight;
                float dL_dci_b = dL_dpix_b * weight;
                
                float dL_dalpha = 
                    dL_dpix_r * (color.x * T_before - accum_rec_r / (1.0f - alpha)) +
                    dL_dpix_g * (color.y * T_before - accum_rec_g / (1.0f - alpha)) +
                    dL_dpix_b * (color.z * T_before - accum_rec_b / (1.0f - alpha));
                
                accum_rec_r += color.x * weight;
                accum_rec_g += color.y * weight;
                accum_rec_b += color.z * weight;
                
                T = T_before;
                
                float dL_dopacity = dL_dalpha * expf(power);
                float dL_dpower = dL_dalpha * alpha;
                
                float dL_da = dL_dpower * (-0.5f * dx * dx);
                float dL_db = dL_dpower * (-1.0f * dx * dy);
                float dL_dc = dL_dpower * (-0.5f * dy * dy);
                
                float dL_ddx = dL_dpower * (-conic.x * dx - conic.y * dy);
                float dL_ddy = dL_dpower * (-conic.y * dx - conic.z * dy);
                
                float dL_dmean_x = -dL_ddx;
                float dL_dmean_y = -dL_ddy;
                
                uint32_t g_idx = sorted_indices[global_i];
                
                atomicAdd(&grad_colors[g_idx*3+0], dL_dci_r);
                atomicAdd(&grad_colors[g_idx*3+1], dL_dci_g);
                atomicAdd(&grad_colors[g_idx*3+2], dL_dci_b);
                
                atomicAdd(&grad_opacities[g_idx], dL_dopacity);
                
                atomicAdd(&grad_conics[g_idx*3+0], dL_da);
                atomicAdd(&grad_conics[g_idx*3+1], dL_db);
                atomicAdd(&grad_conics[g_idx*3+2], dL_dc);
                
                atomicAdd(&grad_means2d[g_idx*2+0], dL_dmean_x);
                atomicAdd(&grad_means2d[g_idx*2+1], dL_dmean_y);
            }
        }
        
        __syncthreads();
    }
}
