#include "rasterizer.cuh"

// Helper function to compute 3D-to-2D projection and covariance
__device__ void compute_cov2d(
    const float3& mean3d,
    const float* cov3d, // 6 elements: xx, yy, zz, xy, xz, yz
    const float* viewmatrix,
    const float focal_x,
    const float focal_y,
    const float tan_fovx,
    const float tan_fovy,
    float3& cov2d_abc) // output: a, b, c for 2D covariance
{
    // Transform point to camera space
    float t_x = viewmatrix[0] * mean3d.x + viewmatrix[4] * mean3d.y + viewmatrix[8] * mean3d.z + viewmatrix[12];
    float t_y = viewmatrix[1] * mean3d.x + viewmatrix[5] * mean3d.y + viewmatrix[9] * mean3d.z + viewmatrix[13];
    float t_z = viewmatrix[2] * mean3d.x + viewmatrix[6] * mean3d.y + viewmatrix[10] * mean3d.z + viewmatrix[14];

    // Frustum culling margin
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t_x / t_z;
    float tytz = t_y / t_z;
    t_x = min(limx, max(-limx, txtz)) * t_z;
    t_y = min(limy, max(-limy, tytz)) * t_z;

    // Jacobian J
    float J[9] = {
        focal_x / t_z, 0.0f, -(focal_x * t_x) / (t_z * t_z),
        0.0f, focal_y / t_z, -(focal_y * t_y) / (t_z * t_z),
        0.0f, 0.0f, 0.0f
    };

    // View matrix rotation part W
    float W[9] = {
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]
    };

    // T = J * W
    float T[9];
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            T[i*3+j] = J[i*3+0] * W[0*3+j] + J[i*3+1] * W[1*3+j] + J[i*3+2] * W[2*3+j];
        }
    }

    // 3D Covariance Sigma
    float Vrk[9] = {
        cov3d[0], cov3d[3], cov3d[4],
        cov3d[3], cov3d[1], cov3d[5],
        cov3d[4], cov3d[5], cov3d[2]
    };

    // T * Vrk
    float TVrk[9];
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            TVrk[i*3+j] = T[i*3+0] * Vrk[0*3+j] + T[i*3+1] * Vrk[1*3+j] + T[i*3+2] * Vrk[2*3+j];
        }
    }

    // cov2d = T * Vrk * T^T
    float cov_xx = TVrk[0*3+0] * T[0*3+0] + TVrk[0*3+1] * T[0*3+1] + TVrk[0*3+2] * T[0*3+2];
    float cov_yy = TVrk[1*3+0] * T[1*3+0] + TVrk[1*3+1] * T[1*3+1] + TVrk[1*3+2] * T[1*3+2];
    float cov_xy = TVrk[0*3+0] * T[1*3+0] + TVrk[0*3+1] * T[1*3+1] + TVrk[0*3+2] * T[1*3+2];

    // Low-pass filter to prevent aliasing
    cov_xx += 0.3f;
    cov_yy += 0.3f;

    cov2d_abc = {cov_xx, cov_yy, cov_xy};
}

__global__ void preprocess_gaussians_kernel(
    const int num_gaussians,
    const float* __restrict__ means3d,       // [N, 3]
    const float* __restrict__ cov3d,         // [N, 6]
    const float* __restrict__ viewmatrix,    // [4, 4]
    const float* __restrict__ projmatrix,    // [4, 4]
    const float focal_x,
    const float focal_y,
    const float tan_fovx,
    const float tan_fovy,
    const int W,
    const int H,
    float* __restrict__ means2d,             // [N, 2]
    float* __restrict__ conics,              // [N, 3]
    float* __restrict__ depths,              // [N]
    int* __restrict__ radii,                 // [N]
    const int grid_X,
    const int grid_Y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    float3 mean = {means3d[idx*3+0], means3d[idx*3+1], means3d[idx*3+2]};

    // Transform to view space
    float t_z = viewmatrix[2] * mean.x + viewmatrix[6] * mean.y + viewmatrix[10] * mean.z + viewmatrix[14];

    // Frustum culling: skip if behind near plane (e.g., z < 0.2)
    if (t_z < 0.2f) {
        radii[idx] = 0; // Mark as culled
        return;
    }

    // Compute 2D covariance
    float3 cov2d_abc;
    compute_cov2d(mean, &cov3d[idx*6], viewmatrix, focal_x, focal_y, tan_fovx, tan_fovy, cov2d_abc);
    
    float cov_xx = cov2d_abc.x;
    float cov_yy = cov2d_abc.y;
    float cov_xy = cov2d_abc.z;

    // Invert covariance (compute conic)
    float det = (cov_xx * cov_yy - cov_xy * cov_xy);
    if (det == 0.0f) {
        radii[idx] = 0;
        return;
    }
    float det_inv = 1.f / det;
    float conic_a = cov_yy * det_inv;
    float conic_b = -cov_xy * det_inv;
    float conic_c = cov_xx * det_inv;

    // Compute screen space position
    float p_w = projmatrix[3] * mean.x + projmatrix[7] * mean.y + projmatrix[11] * mean.z + projmatrix[15];
    float p_x = projmatrix[0] * mean.x + projmatrix[4] * mean.y + projmatrix[8] * mean.z + projmatrix[12];
    float p_y = projmatrix[1] * mean.x + projmatrix[5] * mean.y + projmatrix[9] * mean.z + projmatrix[13];
    
    p_x = (p_x / p_w + 1.0f) * W * 0.5f;
    p_y = (p_y / p_w + 1.0f) * H * 0.5f;

    // Bounding box (radius based on 3-sigma rule)
    float mid = 0.5f * (cov_xx + cov_yy);
    float lambda1 = mid + sqrtf(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrtf(max(0.1f, mid * mid - det));
    float my_radius = ceilf(3.f * sqrtf(max(lambda1, lambda2)));
    
    // Cull if outside screen
    if (p_x + my_radius < 0 || p_x - my_radius > W || p_y + my_radius < 0 || p_y - my_radius > H) {
        radii[idx] = 0;
        return;
    }

    // Write to global memory
    means2d[idx*2+0] = p_x;
    means2d[idx*2+1] = p_y;
    conics[idx*3+0] = conic_a;
    conics[idx*3+1] = conic_b;
    conics[idx*3+2] = conic_c;
    depths[idx] = t_z;
    radii[idx] = (int)my_radius;
}

__global__ void render_tiles_kernel(
    const int W, const int H,
    const int grid_X, const int grid_Y,
    const uint2* __restrict__ tile_offsets, // start and end indices in sorted arrays
    const uint32_t* __restrict__ sorted_indices,
    const float* __restrict__ means2d,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float* __restrict__ out_color,
    float* __restrict__ out_transmittance,
    int* __restrict__ final_index)
{
    // Block maps to a 16x16 tile
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int pix_x = tile_x * 16 + threadIdx.x;
    int pix_y = tile_y * 16 + threadIdx.y;
    
    // Each thread processes one pixel
    bool inside = (pix_x < W && pix_y < H);
    
    // Read tile offsets for this block
    uint2 range = tile_offsets[tile_y * grid_X + tile_x];
    uint32_t start_idx = range.x;
    uint32_t end_idx = range.y;
    
    // Shared memory for collaborative loading
    __shared__ float3 collected_means[256];
    __shared__ float3 collected_conics[256];
    __shared__ float3 collected_colors[256];
    __shared__ float collected_opacities[256];
    
    float T = 1.0f;
    float C[3] = {0.0f, 0.0f, 0.0f};
    int last_contributor = 0;
    
    int num_batches = (end_idx - start_idx + 255) / 256;
    int thread_linear_id = threadIdx.y * 16 + threadIdx.x;
    
    for (int b = 0; b < num_batches; ++b) {
        int batch_start = start_idx + b * 256;
        int fetch_idx = batch_start + thread_linear_id;
        
        // Collaborative load
        if (fetch_idx < end_idx) {
            uint32_t g_idx = sorted_indices[fetch_idx];
            collected_means[thread_linear_id] = make_float3(means2d[g_idx*2+0], means2d[g_idx*2+1], 0.0f);
            collected_conics[thread_linear_id] = make_float3(conics[g_idx*3+0], conics[g_idx*3+1], conics[g_idx*3+2]);
            collected_colors[thread_linear_id] = make_float3(colors[g_idx*3+0], colors[g_idx*3+1], colors[g_idx*3+2]);
            collected_opacities[thread_linear_id] = opacities[g_idx];
        }
        
        // Synchronize before reading shared memory
        __syncthreads();
        
        int num_in_batch = min(256, end_idx - batch_start);
        
        if (inside) {
            for (int i = 0; i < num_in_batch; ++i) {
                if (T < 0.0001f) break; // early stopping
                
                float3 mean = collected_means[i];
                float3 conic = collected_conics[i];
                float opacity = collected_opacities[i];
                float3 color = collected_colors[i];
                
                float dx = (float)pix_x - mean.x;
                float dy = (float)pix_y - mean.y;
                
                // Conic is a, b, c (where b is -cov_xy / det)
                // power = -0.5 * (a*dx*dx + 2*b*dx*dy + c*dy*dy)
                float power = -0.5f * (conic.x * dx * dx + 2.0f * conic.y * dx * dy + conic.z * dy * dy);
                if (power > 0.0f) continue;
                
                float alpha = min(0.99f, opacity * expf(power));
                if (alpha < 1.0f / 255.0f) continue;
                
                float weight = alpha * T;
                C[0] += weight * color.x;
                C[1] += weight * color.y;
                C[2] += weight * color.z;
                
                T *= (1.0f - alpha);
                last_contributor = batch_start + i;
            }
        }
        
        // Synchronize before overwriting shared memory in the next iteration
        __syncthreads();
        
        // Thread-wide stop if all pixels in the block are saturated
        // (A fully optimized version would use __syncthreads_count or __any_sync, but this works for basic requirement)
        if (inside && T < 0.0001f) {
            // We cannot just break here without causing thread divergence at __syncthreads!
            // Actually, we must ensure all threads reach __syncthreads(). 
            // So we just set a flag and stop accumulating.
        }
    }
    
    if (inside) {
        int pix_idx = pix_y * W + pix_x;
        out_color[pix_idx*3+0] = C[0];
        out_color[pix_idx*3+1] = C[1];
        out_color[pix_idx*3+2] = C[2];
        out_transmittance[pix_idx] = T;
        final_index[pix_idx] = last_contributor;
    }
}
