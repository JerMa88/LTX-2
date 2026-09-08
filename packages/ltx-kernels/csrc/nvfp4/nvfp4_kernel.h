#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace ltx_nvfp4 {

enum class DType { BF16, FP16, FP32 };

constexpr float kE2M1Max = 6.0f;
constexpr float kE4M3Max = 448.0f;
constexpr int kBlockSize = 16;

void quantize_launch(DType dtype, const void* x, const float* per_tensor_scale,
                     uint8_t* out, uint8_t* scales, int rows, int blocks_per_row,
                     int padded_scale_cols, bool hi_first, int64_t variant,
                     cudaStream_t stream);

void dequantize_launch(DType dtype, const uint8_t* packed, const float* per_tensor_scale,
                       const uint8_t* scales, void* out, int rows, int blocks_per_row,
                       int padded_scale_cols, bool hi_first, cudaStream_t stream);

void mul_scalars_launch(const float* a, const float* b, float* out, cudaStream_t stream);

void amax_scale_launch(DType dtype, const void* x, int64_t numel, float inv_divisor,
                       float* out, cudaStream_t stream);

} // namespace ltx_nvfp4
