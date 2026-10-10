// Host-side orchestration (compiled by the C++ compiler; kernels are launched through launch.cu).
//
// Forward:  preprocess -> per-Gaussian tile counts -> inclusive scan -> duplicate with (tile|depth) keys
//           -> CUB radix sort -> per-tile ranges -> per-tile alpha compositing.
// Backward: per-tile compositing gradients -> EWA-projection gradients to means3d / cov3d.
// Everything runs on PyTorch's current CUDA stream.
#include "rasterizer.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "kernels.cuh"

namespace {

void check_input(const torch::Tensor& t, const char* name, int64_t last_dim = -1) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    if (last_dim >= 0) TORCH_CHECK(t.dim() == 2 && t.size(1) == last_dim, name, " must have shape [N, ", last_dim, "]");
}

int bits_needed(uint32_t n) {  // ceil(log2(n)): bits to represent values in [0, n)
    int bits = 0;
    while (n > 1) { n = (n + 1) >> 1; ++bits; }
    return bits;
}

template <typename T>
T* ptr(torch::Tensor& t) { return t.data_ptr<T>(); }
template <typename T>
const T* cptr(const torch::Tensor& t) { return t.data_ptr<T>(); }

}  // namespace

std::vector<torch::Tensor> rasterize_forward(
    torch::Tensor means3d, torch::Tensor cov3d, torch::Tensor colors, torch::Tensor opacities,
    torch::Tensor viewmatrix_cm, torch::Tensor projmatrix_cm,
    float focal_x, float focal_y, float tan_fovx, float tan_fovy, int W, int H)
{
    check_input(means3d, "means3d", 3);
    check_input(cov3d, "cov3d", 6);
    check_input(colors, "colors", 3);
    check_input(opacities, "opacities");
    check_input(viewmatrix_cm, "viewmatrix");
    check_input(projmatrix_cm, "projmatrix");
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == means3d.size(0), "opacities must have shape [N]");
    TORCH_CHECK(cov3d.size(0) == means3d.size(0) && colors.size(0) == means3d.size(0), "inconsistent N");
    TORCH_CHECK(viewmatrix_cm.numel() == 16 && projmatrix_cm.numel() == 16, "view/proj matrices must be 4x4");
    TORCH_CHECK(W > 0 && H > 0, "image size must be positive");

    const c10::cuda::CUDAGuard device_guard(means3d.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = static_cast<int>(means3d.size(0));
    const int grid_X = (W + TILE - 1) / TILE;
    const int grid_Y = (H + TILE - 1) / TILE;
    const int num_tiles = grid_X * grid_Y;

    auto f32 = means3d.options();
    auto i32 = f32.dtype(torch::kInt32);
    auto i64 = f32.dtype(torch::kInt64);

    auto means2d = torch::zeros({N, 2}, f32);
    auto conics = torch::zeros({N, 3}, f32);
    auto depths = torch::zeros({N}, f32);
    auto radii = torch::zeros({N}, i32);
    auto rects = torch::zeros({N, 4}, i32);
    auto tiles_touched = torch::zeros({N}, i32);

    if (N > 0) {
        launch_preprocess(N, cptr<float>(means3d), cptr<float>(cov3d), cptr<float>(viewmatrix_cm),
                          cptr<float>(projmatrix_cm), focal_x, focal_y, tan_fovx, tan_fovy, W, H, grid_X, grid_Y,
                          ptr<float>(means2d), ptr<float>(conics), ptr<float>(depths), ptr<int>(radii),
                          ptr<int>(rects), ptr<int>(tiles_touched), stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // One (tile, depth) key per overlapped tile: size the lists with a prefix sum of tiles_touched.
    auto offsets = torch::cumsum(tiles_touched, 0, torch::kInt32);
    const int64_t total = N > 0 ? offsets[N - 1].item<int64_t>() : 0;
    TORCH_CHECK(total < (int64_t(1) << 31) - 1, "too many Gaussian-tile pairs (", total, "); reduce N or the resolution");

    auto tile_ranges = torch::zeros({num_tiles, 2}, i32);
    auto sorted_ids = torch::zeros({total}, i32);
    if (total > 0) {
        auto keys_in = torch::empty({total}, i64);
        auto vals_in = torch::empty({total}, i32);
        auto keys_out = torch::empty({total}, i64);
        launch_duplicate(N, cptr<float>(depths), cptr<int>(radii), cptr<int>(rects), cptr<int>(offsets), grid_X,
                         reinterpret_cast<uint64_t*>(ptr<int64_t>(keys_in)),
                         reinterpret_cast<uint32_t*>(ptr<int32_t>(vals_in)), stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        const int end_bit = 32 + bits_needed(static_cast<uint32_t>(num_tiles)) + 1;
        size_t temp_bytes = 0;
        auto k_in = reinterpret_cast<const uint64_t*>(cptr<int64_t>(keys_in));
        auto k_out = reinterpret_cast<uint64_t*>(ptr<int64_t>(keys_out));
        auto v_in = reinterpret_cast<const uint32_t*>(cptr<int32_t>(vals_in));
        auto v_out = reinterpret_cast<uint32_t*>(ptr<int32_t>(sorted_ids));
        C10_CUDA_CHECK(sort_pairs_launch(nullptr, temp_bytes, k_in, k_out, v_in, v_out, static_cast<int>(total), end_bit, stream));
        auto temp = torch::empty({static_cast<int64_t>(temp_bytes)}, f32.dtype(torch::kUInt8));
        C10_CUDA_CHECK(sort_pairs_launch(temp.data_ptr(), temp_bytes, k_in, k_out, v_in, v_out, static_cast<int>(total), end_bit, stream));

        launch_tile_ranges(static_cast<int>(total), reinterpret_cast<const uint64_t*>(cptr<int64_t>(keys_out)),
                           ptr<int>(tile_ranges), stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto out_color = torch::zeros({H, W, 3}, f32);
    auto out_transmittance = torch::ones({H, W}, f32);
    auto contributors = torch::zeros({H, W}, i32);
    launch_render(W, H, grid_X, grid_Y, cptr<int>(tile_ranges), reinterpret_cast<const uint32_t*>(cptr<int32_t>(sorted_ids)),
                  cptr<float>(means2d), cptr<float>(conics), cptr<float>(colors), cptr<float>(opacities),
                  ptr<float>(out_color), ptr<float>(out_transmittance), ptr<int>(contributors), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {out_color, out_transmittance, contributors, radii, means2d, conics, depths, sorted_ids, tile_ranges};
}

std::vector<torch::Tensor> rasterize_backward(
    int W, int H, torch::Tensor means3d, torch::Tensor cov3d, torch::Tensor viewmatrix_cm, torch::Tensor projmatrix_cm,
    float focal_x, float focal_y, float tan_fovx, float tan_fovy, torch::Tensor radii, torch::Tensor tile_ranges,
    torch::Tensor sorted_ids, torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, torch::Tensor out_transmittance, torch::Tensor contributors,
    torch::Tensor grad_out_color)
{
    check_input(grad_out_color, "grad_out_color");
    const c10::cuda::CUDAGuard device_guard(means3d.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = static_cast<int>(means3d.size(0));
    const int grid_X = (W + TILE - 1) / TILE;
    const int grid_Y = (H + TILE - 1) / TILE;
    auto f32 = means3d.options();

    auto grad_means2d = torch::zeros({N, 2}, f32);
    auto grad_conics = torch::zeros({N, 3}, f32);
    auto grad_colors = torch::zeros({N, 3}, f32);
    auto grad_opacities = torch::zeros({N}, f32);
    auto grad_means3d = torch::zeros({N, 3}, f32);
    auto grad_cov3d = torch::zeros({N, 6}, f32);

    launch_render_backward(W, H, grid_X, grid_Y, cptr<int>(tile_ranges),
                           reinterpret_cast<const uint32_t*>(cptr<int32_t>(sorted_ids)), cptr<float>(means2d),
                           cptr<float>(conics), cptr<float>(colors), cptr<float>(opacities),
                           cptr<float>(out_transmittance), cptr<int>(contributors), cptr<float>(grad_out_color),
                           ptr<float>(grad_means2d), ptr<float>(grad_conics), ptr<float>(grad_colors),
                           ptr<float>(grad_opacities), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    if (N > 0) {
        launch_preprocess_backward(N, cptr<float>(means3d), cptr<float>(cov3d), cptr<float>(viewmatrix_cm),
                                   cptr<float>(projmatrix_cm), focal_x, focal_y, tan_fovx, tan_fovy, W, H,
                                   cptr<int>(radii), cptr<float>(conics), cptr<float>(grad_means2d),
                                   cptr<float>(grad_conics), ptr<float>(grad_means3d), ptr<float>(grad_cov3d), stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {grad_means3d, grad_cov3d, grad_colors, grad_opacities};
}
