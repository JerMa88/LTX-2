#include <string>
#include <unordered_map>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include "nvfp4.h"

namespace ltx_nvfp4 {

void quantize_nvfp4_cuda(const torch::Tensor &x, const torch::Tensor &per_tensor_scale, torch::Tensor &out_packed,
                         torch::Tensor &out_scales, bool hi_first, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be 2-D contiguous CUDA");
  TORCH_CHECK(x.size(1) % kBlockSize == 0, "K must be a multiple of 16, got ", x.size(1));
  TORCH_CHECK(per_tensor_scale.is_cuda() && per_tensor_scale.scalar_type() == torch::kFloat32,
              "per_tensor_scale must be a CUDA fp32 scalar");
  TORCH_CHECK(per_tensor_scale.numel() == 1, "per_tensor_scale must hold one element");

  const int rows = static_cast<int>(x.size(0));
  const int blocks_per_row = static_cast<int>(x.size(1) / kBlockSize);
  const int padded_scale_cols = static_cast<int>(out_scales.size(1));
  TORCH_CHECK(padded_scale_cols % 4 == 0, "scale columns must be padded to a multiple of 4");

  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  DType dt;
  switch (x.scalar_type()) {
  case torch::kBFloat16:
    dt = DType::BF16;
    break;
  case torch::kHalf:
    dt = DType::FP16;
    break;
  case torch::kFloat32:
    dt = DType::FP32;
    break;
  default:
    TORCH_CHECK(false, "quantize_nvfp4 supports bf16/fp16/fp32, got ", x.scalar_type());
  }

  quantize_launch(dt, x.data_ptr(), per_tensor_scale.data_ptr<float>(),
                  out_packed.data_ptr<uint8_t>(), out_scales.data_ptr<uint8_t>(),
                  rows, blocks_per_row, padded_scale_cols, hi_first, variant, stream.stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dequantize_nvfp4_cuda(const torch::Tensor &packed, const torch::Tensor &per_tensor_scale,
                           const torch::Tensor &scales, torch::Tensor &out, bool hi_first) {
  TORCH_CHECK(packed.is_cuda() && packed.dim() == 2 && packed.is_contiguous(), "packed must be 2-D contiguous CUDA");
  const int rows = static_cast<int>(packed.size(0));
  const int blocks_per_row = static_cast<int>(packed.size(1) * 2 / kBlockSize);
  const int padded_scale_cols = static_cast<int>(scales.size(1));

  const c10::cuda::CUDAGuard guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  DType dt;
  switch (out.scalar_type()) {
  case torch::kBFloat16:
    dt = DType::BF16;
    break;
  case torch::kFloat32:
    dt = DType::FP32;
    break;
  default:
    TORCH_CHECK(false, "dequantize_nvfp4 supports bf16/fp32 out, got ", out.scalar_type());
  }

  dequantize_launch(dt, packed.data_ptr<uint8_t>(), per_tensor_scale.data_ptr<float>(),
                    scales.data_ptr<uint8_t>(), out.data_ptr(),
                    rows, blocks_per_row, padded_scale_cols, hi_first, stream.stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mul_scalars_cuda(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &out) {
  const c10::cuda::CUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  mul_scalars_launch(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), stream.stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void amax_scale_cuda(const torch::Tensor &x, torch::Tensor &out, double divisor) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "amax input must be contiguous CUDA");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.numel() == 1,
              "amax out must be a CUDA fp32 scalar");
  TORCH_CHECK(divisor > 0.0, "divisor must be positive, got ", divisor);

  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  float *out_ptr = out.data_ptr<float>();

  C10_CUDA_CHECK(cudaMemsetAsync(out_ptr, 0, sizeof(float), stream));

  const int64_t numel = x.numel();
  if (numel == 0)
    return;

  const float inv_divisor = static_cast<float>(1.0 / divisor);

  DType dt;
  switch (x.scalar_type()) {
  case torch::kBFloat16:
    dt = DType::BF16;
    break;
  case torch::kHalf:
    dt = DType::FP16;
    break;
  case torch::kFloat32:
    dt = DType::FP32;
    break;
  default:
    TORCH_CHECK(false, "amax_scale supports bf16/fp16/fp32, got ", x.scalar_type());
  }

  amax_scale_launch(dt, x.data_ptr(), numel, inv_divisor, out_ptr, stream.stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace ltx_nvfp4
