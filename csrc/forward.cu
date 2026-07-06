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
