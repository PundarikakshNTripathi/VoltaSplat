#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct CameraState {
    int image_width;
    int image_height;
    float focal_x;
    float focal_y;
    float tan_fovx;
    float tan_fovy;
    const float* viewmatrix;
    const float* projmatrix;
};

// Computes the 2D bounding box of a projected Gaussian given its 2D mean and covariance.
__device__ inline void compute_bounding_box(
    const float2& mean,
    const float3& conic,
    const float radius,
    int2& bbox_min,
    int2& bbox_max,
    const int W, const int H) 
{
    // The conic is the inverse 2D covariance. The radius is the footprint.
    bbox_min.x = max(0, (int)(mean.x - radius));
    bbox_max.x = min(W, (int)(mean.x + radius + 1.0f));
    bbox_min.y = max(0, (int)(mean.y - radius));
    bbox_max.y = min(H, (int)(mean.y + radius + 1.0f));
}
