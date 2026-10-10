// Forward pass: per-Gaussian preprocessing (EWA projection) and per-tile alpha compositing.
#include "kernels.cuh"

// Screen-space covariance of a 3D Gaussian under the local affine (first-order Taylor / Jacobian)
// approximation of the perspective map:  Sigma' = J W Sigma W^T J^T  (top-left 2x2 block).
__device__ __forceinline__ void compute_cov2d(
    const float* mean, const float* cov3d, const float* V, float focal_x, float focal_y,
    float tan_fovx, float tan_fovy, float& cov_xx, float& cov_xy, float& cov_yy)
{
    float t_x = VS_MAT(V, 0, 0) * mean[0] + VS_MAT(V, 0, 1) * mean[1] + VS_MAT(V, 0, 2) * mean[2] + VS_MAT(V, 0, 3);
    float t_y = VS_MAT(V, 1, 0) * mean[0] + VS_MAT(V, 1, 1) * mean[1] + VS_MAT(V, 1, 2) * mean[2] + VS_MAT(V, 1, 3);
    float t_z = VS_MAT(V, 2, 0) * mean[0] + VS_MAT(V, 2, 1) * mean[1] + VS_MAT(V, 2, 2) * mean[2] + VS_MAT(V, 2, 3);

    // Clamp the point to a slightly enlarged frustum before linearising, as in EWA splatting.
    const float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
    t_x = fminf(limx, fmaxf(-limx, t_x / t_z)) * t_z;
    t_y = fminf(limy, fmaxf(-limy, t_y / t_z)) * t_z;

    const float J[2][3] = {{focal_x / t_z, 0.f, -(focal_x * t_x) / (t_z * t_z)},
                           {0.f, focal_y / t_z, -(focal_y * t_y) / (t_z * t_z)}};
    float T[2][3];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            T[i][j] = J[i][0] * VS_MAT(V, 0, j) + J[i][1] * VS_MAT(V, 1, j) + J[i][2] * VS_MAT(V, 2, j);

    const float S[3][3] = {{cov3d[0], cov3d[3], cov3d[4]}, {cov3d[3], cov3d[1], cov3d[5]}, {cov3d[4], cov3d[5], cov3d[2]}};
    float TS[2][3];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) TS[i][j] = T[i][0] * S[0][j] + T[i][1] * S[1][j] + T[i][2] * S[2][j];

    cov_xx = TS[0][0] * T[0][0] + TS[0][1] * T[0][1] + TS[0][2] * T[0][2];
    cov_xy = TS[0][0] * T[1][0] + TS[0][1] * T[1][1] + TS[0][2] * T[1][2];
    cov_yy = TS[1][0] * T[1][0] + TS[1][1] * T[1][1] + TS[1][2] * T[1][2];
    cov_xx += 0.3f;  // low-pass filter: every splat covers at least ~one pixel
    cov_yy += 0.3f;
}

__global__ void preprocess_gaussians_kernel(
    int num_gaussians, const float* __restrict__ means3d, const float* __restrict__ cov3d,
    const float* __restrict__ viewmatrix, const float* __restrict__ projmatrix,
    float focal_x, float focal_y, float tan_fovx, float tan_fovy, int W, int H, int grid_X, int grid_Y,
    float* __restrict__ means2d, float* __restrict__ conics, float* __restrict__ depths,
    int* __restrict__ radii, int* __restrict__ rects, int* __restrict__ tiles_touched)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    const float mean[3] = {means3d[idx * 3 + 0], means3d[idx * 3 + 1], means3d[idx * 3 + 2]};
    const float t_z = VS_MAT(viewmatrix, 2, 0) * mean[0] + VS_MAT(viewmatrix, 2, 1) * mean[1] +
                      VS_MAT(viewmatrix, 2, 2) * mean[2] + VS_MAT(viewmatrix, 2, 3);
    if (t_z < 0.2f) return;  // behind / too close to the camera: culled (radii stays 0)

    float cov_xx, cov_xy, cov_yy;
    compute_cov2d(mean, &cov3d[idx * 6], viewmatrix, focal_x, focal_y, tan_fovx, tan_fovy, cov_xx, cov_xy, cov_yy);

    const float det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det == 0.0f) return;
    const float det_inv = 1.f / det;

    const float p_w = VS_MAT(projmatrix, 3, 0) * mean[0] + VS_MAT(projmatrix, 3, 1) * mean[1] + VS_MAT(projmatrix, 3, 2) * mean[2] + VS_MAT(projmatrix, 3, 3);
    const float p_x = VS_MAT(projmatrix, 0, 0) * mean[0] + VS_MAT(projmatrix, 0, 1) * mean[1] + VS_MAT(projmatrix, 0, 2) * mean[2] + VS_MAT(projmatrix, 0, 3);
    const float p_y = VS_MAT(projmatrix, 1, 0) * mean[0] + VS_MAT(projmatrix, 1, 1) * mean[1] + VS_MAT(projmatrix, 1, 2) * mean[2] + VS_MAT(projmatrix, 1, 3);
    const float px = (p_x / p_w + 1.0f) * W * 0.5f;
    const float py = (p_y / p_w + 1.0f) * H * 0.5f;

    // 3-sigma screen-space radius from the larger eigenvalue of the 2x2 covariance
    const float mid = 0.5f * (cov_xx + cov_yy);
    const float lambda1 = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
    const float lambda2 = mid - sqrtf(fmaxf(0.1f, mid * mid - det));
    const int radius = (int)ceilf(3.f * sqrtf(fmaxf(lambda1, lambda2)));

    // Tile rectangle [min, max) overlapped by the bounding square
    const int minx = min(grid_X, max(0, (int)((px - radius) / TILE)));
    const int miny = min(grid_Y, max(0, (int)((py - radius) / TILE)));
    const int maxx = min(grid_X, max(0, (int)((px + radius + TILE - 1) / TILE)));
    const int maxy = min(grid_Y, max(0, (int)((py + radius + TILE - 1) / TILE)));
    const int touched = (maxx - minx) * (maxy - miny);
    if (touched == 0) return;

    means2d[idx * 2 + 0] = px;
    means2d[idx * 2 + 1] = py;
    conics[idx * 3 + 0] = cov_yy * det_inv;
    conics[idx * 3 + 1] = -cov_xy * det_inv;
    conics[idx * 3 + 2] = cov_xx * det_inv;
    depths[idx] = t_z;
    radii[idx] = radius;
    rects[idx * 4 + 0] = minx;
    rects[idx * 4 + 1] = miny;
    rects[idx * 4 + 2] = maxx;
    rects[idx * 4 + 3] = maxy;
    tiles_touched[idx] = touched;
}

// One 16x16-thread block composites one 16x16-pixel tile, front to back, staging BATCH Gaussians
// at a time in shared memory. Transmittance below 1e-4 stops a pixel (early termination).
__global__ void render_tiles_kernel(
    int W, int H, int grid_X, const int* __restrict__ tile_ranges, const uint32_t* __restrict__ sorted_ids,
    const float* __restrict__ means2d, const float* __restrict__ conics, const float* __restrict__ colors,
    const float* __restrict__ opacities, float* __restrict__ out_color, float* __restrict__ out_transmittance,
    int* __restrict__ contributors)
{
    const int pix_x = blockIdx.x * TILE + threadIdx.x;
    const int pix_y = blockIdx.y * TILE + threadIdx.y;
    const bool inside = (pix_x < W && pix_y < H);

    const int tile = blockIdx.y * grid_X + blockIdx.x;
    const int start_idx = tile_ranges[tile * 2 + 0];
    const int end_idx = tile_ranges[tile * 2 + 1];

    __shared__ float2 s_mean[BATCH];
    __shared__ float3 s_conic[BATCH];
    __shared__ float3 s_color[BATCH];
    __shared__ float s_opacity[BATCH];

    float T = 1.0f;
    float C[3] = {0.f, 0.f, 0.f};
    int last = 0;
    bool done = false;

    const int tid = threadIdx.y * TILE + threadIdx.x;
    const int num_batches = (end_idx - start_idx + BATCH - 1) / BATCH;

    for (int b = 0; b < num_batches; ++b) {
        // Whole block stops once every pixel is saturated (or outside the image)
        if (__syncthreads_count(done || !inside) == TILE * TILE) break;

        const int batch_start = start_idx + b * BATCH;
        const int fetch = batch_start + tid;
        if (fetch < end_idx) {
            const uint32_t g = sorted_ids[fetch];
            s_mean[tid] = make_float2(means2d[g * 2 + 0], means2d[g * 2 + 1]);
            s_conic[tid] = make_float3(conics[g * 3 + 0], conics[g * 3 + 1], conics[g * 3 + 2]);
            s_color[tid] = make_float3(colors[g * 3 + 0], colors[g * 3 + 1], colors[g * 3 + 2]);
            s_opacity[tid] = opacities[g];
        }
        __syncthreads();

        const int n = min(BATCH, end_idx - batch_start);
        for (int i = 0; inside && !done && i < n; ++i) {
            const float dx = (float)pix_x - s_mean[i].x;
            const float dy = (float)pix_y - s_mean[i].y;
            const float3 con = s_conic[i];
            const float power = -0.5f * (con.x * dx * dx + 2.0f * con.y * dx * dy + con.z * dy * dy);
            if (power > 0.0f) continue;
            const float alpha = fminf(0.99f, s_opacity[i] * expf(power));
            if (alpha < 1.0f / 255.0f) continue;
            const float test_T = T * (1.0f - alpha);
            if (test_T < 0.0001f) {  // this Gaussian would saturate the pixel: stop *before* adding it
                done = true;
                break;
            }
            const float w = alpha * T;
            C[0] += w * s_color[i].x;
            C[1] += w * s_color[i].y;
            C[2] += w * s_color[i].z;
            T = test_T;
            last = batch_start + i + 1;
        }
        __syncthreads();
    }

    if (inside) {
        const int p = pix_y * W + pix_x;
        out_color[p * 3 + 0] = C[0];
        out_color[p * 3 + 1] = C[1];
        out_color[p * 3 + 2] = C[2];
        out_transmittance[p] = T;
        contributors[p] = last;
    }
}
