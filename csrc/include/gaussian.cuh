#pragma once
#include <cuda_runtime.h>

struct GaussianState {
    float3 mean3d;
    float* cov3d;
    float* sh_coeffs;
    float opacity;
};

// Helper for converting 3D covariance to 2D screen space
__device__ inline void compute_2d_covariance(
    const float3& mean, 
    const float* cov3d,
    const float* viewmatrix,
    const float focal_x, const float focal_y,
    const float tan_fovx, const float tan_fovy,
    float3& conic, float& radius)
{
    // Math logic for projection affine approximation (Jacobian formulation)
    // For VoltaSplat, the main projection logic currently lives in forward.cu.
    // This header provides modular types and inline utilities for potential extensions.
}
