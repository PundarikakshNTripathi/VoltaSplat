// Backward pass.
//   render_tiles_backward_kernel : gradients of the compositing equation w.r.t. each Gaussian's 2D mean,
//                                  conic, color and opacity (walks every tile back to front).
//   preprocess_backward_kernel   : chain rule through the EWA projection to the 3D mean and 3D covariance.
#include "kernels.cuh"

__global__ void render_tiles_backward_kernel(
    int W, int H, int grid_X, const int* __restrict__ tile_ranges, const uint32_t* __restrict__ sorted_ids,
    const float* __restrict__ means2d, const float* __restrict__ conics, const float* __restrict__ colors,
    const float* __restrict__ opacities, const float* __restrict__ out_transmittance,
    const int* __restrict__ contributors, const float* __restrict__ grad_out_color,
    float* __restrict__ grad_means2d, float* __restrict__ grad_conics, float* __restrict__ grad_colors,
    float* __restrict__ grad_opacities)
{
    const int pix_x = blockIdx.x * TILE + threadIdx.x;
    const int pix_y = blockIdx.y * TILE + threadIdx.y;
    const bool inside = (pix_x < W && pix_y < H);
    const int p = pix_y * W + pix_x;

    const int tile = blockIdx.y * grid_X + blockIdx.x;
    const int start_idx = tile_ranges[tile * 2 + 0];

    float dL_dpix[3] = {0.f, 0.f, 0.f};
    float T = 1.0f;
    int last = 0;  // exclusive end of this pixel's contributor range
    if (inside) {
        dL_dpix[0] = grad_out_color[p * 3 + 0];
        dL_dpix[1] = grad_out_color[p * 3 + 1];
        dL_dpix[2] = grad_out_color[p * 3 + 2];
        T = out_transmittance[p];
        last = contributors[p];
    }

    // The block only needs to walk back from the furthest contributor of any of its pixels.
    __shared__ int block_last;
    if (threadIdx.x == 0 && threadIdx.y == 0) block_last = 0;
    __syncthreads();
    atomicMax(&block_last, last);
    __syncthreads();
    const int end_idx = block_last;
    if (end_idx <= start_idx) return;  // uniform across the block: nothing to differentiate

    __shared__ float2 s_mean[BATCH];
    __shared__ float3 s_conic[BATCH];
    __shared__ float3 s_color[BATCH];
    __shared__ float s_opacity[BATCH];
    __shared__ uint32_t s_id[BATCH];

    const int tid = threadIdx.y * TILE + threadIdx.x;
    const int num_batches = (end_idx - start_idx + BATCH - 1) / BATCH;
    float accum[3] = {0.f, 0.f, 0.f};  // color accumulated from Gaussians *behind* the current one

    for (int b = num_batches - 1; b >= 0; --b) {
        const int batch_start = start_idx + b * BATCH;
        const int fetch = batch_start + tid;
        if (fetch < end_idx) {
            const uint32_t g = sorted_ids[fetch];
            s_id[tid] = g;
            s_mean[tid] = make_float2(means2d[g * 2 + 0], means2d[g * 2 + 1]);
            s_conic[tid] = make_float3(conics[g * 3 + 0], conics[g * 3 + 1], conics[g * 3 + 2]);
            s_color[tid] = make_float3(colors[g * 3 + 0], colors[g * 3 + 1], colors[g * 3 + 2]);
            s_opacity[tid] = opacities[g];
        }
        __syncthreads();

        const int n = min(BATCH, end_idx - batch_start);
        for (int i = n - 1; inside && i >= 0; --i) {
            if (batch_start + i >= last) continue;  // beyond this pixel's last contributor

            const float dx = (float)pix_x - s_mean[i].x;
            const float dy = (float)pix_y - s_mean[i].y;
            const float3 con = s_conic[i];
            const float power = -0.5f * (con.x * dx * dx + 2.0f * con.y * dx * dy + con.z * dy * dy);
            if (power > 0.0f) continue;
            const float G = expf(power);
            const float raw_alpha = s_opacity[i] * G;
            const float alpha = fminf(0.99f, raw_alpha);
            if (alpha < 1.0f / 255.0f) continue;

            T = T / (1.0f - alpha);  // transmittance in front of this Gaussian
            const float weight = alpha * T;
            const uint32_t g = s_id[i];
            const float3 col = s_color[i];

            atomicAdd(&grad_colors[g * 3 + 0], dL_dpix[0] * weight);
            atomicAdd(&grad_colors[g * 3 + 1], dL_dpix[1] * weight);
            atomicAdd(&grad_colors[g * 3 + 2], dL_dpix[2] * weight);

            // d(pixel)/d(alpha_i) = c_i * T_i - (color of everything behind i) / (1 - alpha_i)
            const float inv = 1.0f / (1.0f - alpha);
            float dL_dalpha = dL_dpix[0] * (col.x * T - accum[0] * inv) +
                              dL_dpix[1] * (col.y * T - accum[1] * inv) +
                              dL_dpix[2] * (col.z * T - accum[2] * inv);
            accum[0] += col.x * weight;
            accum[1] += col.y * weight;
            accum[2] += col.z * weight;

            if (raw_alpha > 0.99f) continue;  // alpha was clamped: no gradient through opacity / geometry

            const float dL_dpower = dL_dalpha * alpha;  // alpha = opacity * exp(power)
            atomicAdd(&grad_opacities[g], dL_dalpha * G);
            atomicAdd(&grad_conics[g * 3 + 0], dL_dpower * (-0.5f * dx * dx));
            atomicAdd(&grad_conics[g * 3 + 1], dL_dpower * (-dx * dy));  // b appears twice in the quadratic form
            atomicAdd(&grad_conics[g * 3 + 2], dL_dpower * (-0.5f * dy * dy));
            atomicAdd(&grad_means2d[g * 2 + 0], dL_dpower * (con.x * dx + con.y * dy));
            atomicAdd(&grad_means2d[g * 2 + 1], dL_dpower * (con.y * dx + con.z * dy));
        }
        __syncthreads();
    }
}

// Backward of preprocess_gaussians_kernel: given dL/d(mean2d) and dL/d(conic), produce dL/d(mean3d) and dL/d(cov3d).
//
//   conic = inverse(Sigma2),  Sigma2 = T S T^T + 0.3 I,  T = J W (2x3),  mean2d = viewport(proj * mean3d)
//   dSigma2 = -Q G_Q Q                       (Q = conic matrix, G_Q = dL/dQ with the off-diagonal split in two)
//   dS      = sum_ij dSigma2_ij T_i^T T_j    -> grad_cov3d
//   dT_i    = 2 sum_j dSigma2_ij (S T_j^T)   -> dJ -> d(view-space position) through the frustum clamp
__global__ void preprocess_backward_kernel(
    int num_gaussians, const float* __restrict__ means3d, const float* __restrict__ cov3d,
    const float* __restrict__ V, const float* __restrict__ P, float focal_x, float focal_y,
    float tan_fovx, float tan_fovy, int W, int H, const int* __restrict__ radii,
    const float* __restrict__ conics, const float* __restrict__ grad_means2d, const float* __restrict__ grad_conics,
    float* __restrict__ grad_means3d, float* __restrict__ grad_cov3d)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians || radii[idx] <= 0) return;

    const float m[3] = {means3d[idx * 3 + 0], means3d[idx * 3 + 1], means3d[idx * 3 + 2]};
    const float* c3 = &cov3d[idx * 6];

    // ---- recompute the forward quantities --------------------------------------------------------
    float t[3];
    for (int r = 0; r < 3; ++r) t[r] = VS_MAT(V, r, 0) * m[0] + VS_MAT(V, r, 1) * m[1] + VS_MAT(V, r, 2) * m[2] + VS_MAT(V, r, 3);
    const float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
    const float rx = t[0] / t[2], ry = t[1] / t[2];
    const bool clamp_x = (rx < -limx || rx > limx), clamp_y = (ry < -limy || ry > limy);
    const float sgn_x = rx < 0 ? -1.f : 1.f, sgn_y = ry < 0 ? -1.f : 1.f;
    const float tx = clamp_x ? sgn_x * limx * t[2] : t[0];
    const float ty = clamp_y ? sgn_y * limy * t[2] : t[1];
    const float tz = t[2];
    const float iz = 1.f / tz, iz2 = iz * iz, iz3 = iz2 * iz;

    const float J[2][3] = {{focal_x * iz, 0.f, -focal_x * tx * iz2}, {0.f, focal_y * iz, -focal_y * ty * iz2}};
    float T[2][3];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) T[i][j] = J[i][0] * VS_MAT(V, 0, j) + J[i][1] * VS_MAT(V, 1, j) + J[i][2] * VS_MAT(V, 2, j);
    const float S[3][3] = {{c3[0], c3[3], c3[4]}, {c3[3], c3[1], c3[5]}, {c3[4], c3[5], c3[2]}};

    // ---- dL/dSigma2 -------------------------------------------------------------------------------
    const float qa = conics[idx * 3 + 0], qb = conics[idx * 3 + 1], qc = conics[idx * 3 + 2];
    const float ga = grad_conics[idx * 3 + 0], gb = 0.5f * grad_conics[idx * 3 + 1], gc = grad_conics[idx * 3 + 2];
    // M = G_Q Q
    const float m00 = ga * qa + gb * qb, m01 = ga * qb + gb * qc;
    const float m10 = gb * qa + gc * qb, m11 = gb * qb + gc * qc;
    // dSigma2 = -Q M
    const float d00 = -(qa * m00 + qb * m10), d01 = -(qa * m01 + qb * m11);
    const float d10 = -(qb * m00 + qc * m10), d11 = -(qb * m01 + qc * m11);
    const float dS2[2][2] = {{d00, 0.5f * (d01 + d10)}, {0.5f * (d01 + d10), d11}};

    // ---- dL/dcov3d: dS = sum_ij dS2_ij * T_i^T T_j ---------------------------------------------------
    float dS[3][3];
    for (int p = 0; p < 3; ++p)
        for (int q = 0; q < 3; ++q) {
            float acc = 0.f;
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) acc += dS2[i][j] * T[i][p] * T[j][q];
            dS[p][q] = acc;
        }
    float* gc3 = &grad_cov3d[idx * 6];
    gc3[0] = dS[0][0];
    gc3[1] = dS[1][1];
    gc3[2] = dS[2][2];
    gc3[3] = dS[0][1] + dS[1][0];
    gc3[4] = dS[0][2] + dS[2][0];
    gc3[5] = dS[1][2] + dS[2][1];

    // ---- dL/dT, dL/dJ ------------------------------------------------------------------------------
    float dT[2][3];
    for (int i = 0; i < 2; ++i)
        for (int p = 0; p < 3; ++p) {
            float acc = 0.f;
            for (int j = 0; j < 2; ++j) {
                float sT = S[p][0] * T[j][0] + S[p][1] * T[j][1] + S[p][2] * T[j][2];  // (S T_j^T)_p
                acc += dS2[i][j] * sT;
            }
            dT[i][p] = 2.f * acc;
        }
    float dJ[2][3];
    for (int i = 0; i < 2; ++i)
        for (int k = 0; k < 3; ++k) dJ[i][k] = dT[i][0] * VS_MAT(V, k, 0) + dT[i][1] * VS_MAT(V, k, 1) + dT[i][2] * VS_MAT(V, k, 2);

    // dL/d(t) through J00 = fx/tz, J02 = -fx tx/tz^2, J11 = fy/tz, J12 = -fy ty/tz^2 and the clamp
    float dt[3] = {0.f, 0.f, 0.f};
    if (!clamp_x) dt[0] += dJ[0][2] * (-focal_x * iz2);
    if (!clamp_y) dt[1] += dJ[1][2] * (-focal_y * iz2);
    float dtz = dJ[0][0] * (-focal_x * iz2) + dJ[1][1] * (-focal_y * iz2);
    dtz += dJ[0][2] * (2.f * focal_x * tx * iz3 + (clamp_x ? (-focal_x * iz2) * sgn_x * limx : 0.f));
    dtz += dJ[1][2] * (2.f * focal_y * ty * iz3 + (clamp_y ? (-focal_y * iz2) * sgn_y * limy : 0.f));
    dt[2] += dtz;

    // ---- dL/dmean3d from the view-space path and from mean2d --------------------------------------
    float dm[3];
    for (int c = 0; c < 3; ++c) dm[c] = VS_MAT(V, 0, c) * dt[0] + VS_MAT(V, 1, c) * dt[1] + VS_MAT(V, 2, c) * dt[2];

    const float px = VS_MAT(P, 0, 0) * m[0] + VS_MAT(P, 0, 1) * m[1] + VS_MAT(P, 0, 2) * m[2] + VS_MAT(P, 0, 3);
    const float py = VS_MAT(P, 1, 0) * m[0] + VS_MAT(P, 1, 1) * m[1] + VS_MAT(P, 1, 2) * m[2] + VS_MAT(P, 1, 3);
    const float pw = VS_MAT(P, 3, 0) * m[0] + VS_MAT(P, 3, 1) * m[1] + VS_MAT(P, 3, 2) * m[2] + VS_MAT(P, 3, 3);
    const float ipw = 1.f / pw;
    const float gx = grad_means2d[idx * 2 + 0], gy = grad_means2d[idx * 2 + 1];
    const float dpx = gx * W * 0.5f * ipw;
    const float dpy = gy * H * 0.5f * ipw;
    const float dpw = -(gx * W * 0.5f * px + gy * H * 0.5f * py) * ipw * ipw;
    for (int c = 0; c < 3; ++c) dm[c] += VS_MAT(P, 0, c) * dpx + VS_MAT(P, 1, c) * dpy + VS_MAT(P, 3, c) * dpw;

    grad_means3d[idx * 3 + 0] = dm[0];
    grad_means3d[idx * 3 + 1] = dm[1];
    grad_means3d[idx * 3 + 2] = dm[2];
}
