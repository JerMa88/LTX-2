# LTX-2.5 Local NVFP4 Generation on NVIDIA GeForce RTX 5080 (SM 12.0)

This document provides comprehensive technical documentation for running **LTX-2.5** high-fidelity video generation on the **NVIDIA GeForce RTX 5080 (Blackwell SM 12.0, 16 GB GDDR7 VRAM, 64 GB DDR5 Host RAM)** under Windows 11 and Linux.

---

## 1. Architectural Overview & The Blackwell Advantage

### Native NVFP4 Tensor Core Acceleration
Prior generations (Ampere SM 8.0, Ada Lovelace SM 8.9) lack native hardware FP4 matrix multiply capability. The **RTX 5080 (Blackwell SM 12.0)** introduces hardware-accelerated **NVFP4 (E2M1)** tensor cores:
- **Format**: FP4 `E2M1` format (1 sign bit, 2 exponent bits, 1 mantissa bit; range $[-6.0, 6.0]$).
- **Block Scaling**: FP8 `E4M3` block scales (1 scale factor per 16 elements).
- **Global Scaling**: FP32 tensor scale factor.
- **Compute Engine**: `cuBLASLt` block-scaled GEMM (`cublasLtMatmul`) directly executes against 4-bit weights without decompression into BF16/FP16 before calculation.

### Memory Footprint Comparison

| Component | Unquantized BF16 | NVFP4 Pre-Quantized | Savings |
|---|---|---|---|
| **Diffusion Transformer (22B)** | ~42.0 GB | **17.44 GB** | **~58% reduction** |
| **Gemma 4 12B Text Encoder** | ~24.5 GB | **24.46 GB** (BF16 in RAM) | Loaded once, discarded from GPU |
| **Spatial Upscaler (2x)** | ~0.93 GB | **0.93 GB** | In VRAM (~0.9 GB) |
| **Video VAE (BF16)** | ~1.37 GB | **1.37 GB** | Decoded in temporal/spatial chunks |
| **Audio VAE (BF16)** | ~0.34 GB | **0.34 GB** | Decoded in continuous latent space |

Because the 22B transformer is pre-quantized to NVFP4, its static footprint in VRAM is ~17.4 GB. When combined with Windows WDDM commit headroom and CUDA virtual memory paging, generation runs completely in VRAM without requiring CPU block-streaming.

---

## 2. Kernel Toolchain & MSVC Windows Porting

Compiling PyTorch C++/CUDA extensions on Windows with MSVC 14.43 and CUDA 12.8 exposed several compiler bugs that required specific architecture decoupling.

### A. MSVC C1001 Internal Compiler Error in PyTorch 2.11
- **Root Cause**: PyTorch 2.11's `<torch/extension.h>` transitively pulls in the entire dynamo autograd and c10 dispatcher template graph. MSVC's compiler front-end (`walk.cpp`) crashes with `fatal error C1001: Internal compiler error` when processing these deeply nested templates alongside nvcc host stubs.
- **Resolution**:
  1. Decoupled device kernels from host bindings:
     - `packages/ltx-kernels/csrc/nvfp4/quantize.cu` contains **pure CUDA device code** (only standard C++ and CUDA runtime headers, zero PyTorch/ATen dependencies).
     - `packages/ltx-kernels/csrc/nvfp4/quantize_host.cpp` contains the PyTorch ATen host wrappers and `pybind11` entrypoints.
  2. In host files, replaced `#include <torch/extension.h>` with minimal headers:
     ```cpp
     #include <torch/csrc/utils/pybind.h>
     #include <torch/types.h>
     ```
  3. Included standard library containers (`<string>`, `<vector>`, `<unordered_map>`, `<optional>`) **before** any PyTorch headers.

### B. Windows Linker Library Alignment
On Windows, PyTorch Python C-extensions cannot rely on ELF dynamic symbol resolution. `setup.py` was updated to explicitly add `torch_python.lib` and CUDA runtime import libraries:
```python
if is_win:
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    cuda_link_dirs.append(str(torch_lib_dir))
    libraries = ["torch", "torch_python", "c10", "cublasLt"]
```

### C. Architecture Code Flags (`TORCH_CUDA_ARCH_LIST`)
CUDA 12.8 supports Blackwell SM 12.0 (`sm_120`). In `setup.py`, the arch discovery was adapted to accept `12.0` / `120`:
```python
_NVFP4_ARCHES = ["100", "100a", "110", "110a", "120", "120a"]
```

---

## 3. Checkpoint Pipeline & Format Normalization

### Gemma 4 12B Dequantization (`scripts/dequantize_gemma_nvfp4.py`)
- **Problem**: Community checkpoint `gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors` packages weights using ComfyUI's custom packed format (`.comfy_quant`), which creates rank and dimension mismatches with Hugging Face's standard `AutoModelForImageTextToText`.
- **Solution**: Built a standalone GPU conversion script `scripts/dequantize_gemma_nvfp4.py`. It reads the packed U8 tensors, uses the compiled RTX 5080 `ltx_kernels.nvfp4.dequantize_nvfp4` kernel to convert 328 layers back to native BF16, and writes `models/ltx25/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (24.46 GB) in **9.88 seconds**.

---

## 4. Pipeline Execution & Autograd Lifecycle

### A. Quantization Policy vs. Offload Mode
In `ltx-pipelines`, the quantization policy `QuantizationKind.NVFP4_PREQUANT` is incompatible with `OffloadMode.CPU` or `DISK`:
```python
# Fails with ValueError: Block streaming is not supported with this quantization policy:
pipeline = TI2VidTwoStagesHQPipeline(..., quantization=quant_policy, offload_mode=OffloadMode.CPU)

# Correct configuration:
pipeline = TI2VidTwoStagesHQPipeline(..., quantization=quant_policy, offload_mode=OffloadMode.NONE)
```
With `OffloadMode.NONE`, weights remain natively in NVFP4 on the GPU.

### B. Generator Autograd Context in Video Encoding
The DiffVAE video decoder produces a Python generator yielding decoded pixel chunks:
```python
decoded_video = self.video_decoder(video_state.latent, tiling_config, generator, dtype=vae_dtype)
```
When `encode_video()` consumes this iterator (`next(video, None)`), if the outer function is not inside `@torch.inference_mode()`, PyTorch autograd resumes tracking inside the generator on inference tensors, throwing:
```
RuntimeError: Inference tensors cannot be saved for backward.
```
**Fix**: `run_rtx5080.py` applies `@torch.inference_mode()` to `run_pipeline()`, ensuring the full lifecycle of diffusion, decoding, and ffmpeg muxing stays strictly in inference mode.

---

## 5. Telemetry & Verification Benchmarks

### Test Run: 9 Frames @ 1280x768 (2-Stage HQ, 8 Steps Stage 1, 3 Steps Stage 2)
```
GPU: NVIDIA GeForce RTX 5080 (15.89 GB Physical VRAM)
Host Memory: 63.91 GB DDR5 (85.98 GB Commit Limit with Pagefile)

[Watermark] Initial Baseline          | VRAM Alloc:  0.00 GB | Res:  0.00 GB | Host RAM: 19.64 GB
[Watermark] Pipeline Initialized      | VRAM Alloc:  0.00 GB | Res:  0.00 GB | Host RAM: 19.68 GB
Stage 1 (640x384, 8 steps):            28.0 seconds (~3.50s / step)
Spatial Upscaler (2x):                 2.1 seconds
Stage 2 (1280x768, 3 steps):           14.0 seconds (~4.66s / step)
DiffVAE Video & Audio Decode:          5.3 seconds
Video File Encoding:                   5.3 seconds
[Watermark] Generation Complete       | Peak VRAM: 23.93 GB  | Host RAM: 21.65 GB
Total Wall Time:                      236.33 seconds (~3.9 minutes)
Output:                               outputs/test_verification_9f.mp4 (1280x768, 24 fps, YUV420p + AAC)
```

---

## 6. Pre-Flight Verification & Execution Guide

### Running Pre-Flight Diagnostics
Always verify hardware and environment readiness before long generation runs:
```powershell
.\.venv\Scripts\python.exe preflight_check.py
```

### Running Pipeline Dry-Run
Validates that model weights, memory allocation, and graph construction succeed without rendering diffusion steps:
```powershell
.\.venv\Scripts\python.exe run_rtx5080.py --prompt "Test prompt" --dry-run
```

### Running Production Video Generation
```powershell
.\.venv\Scripts\python.exe run_rtx5080.py `
  --prompt "A cinematic shot of a majestic waterfall in a lush tropical forest with sunlight rays" `
  --num-frames 121 `
  --width 1920 `
  --height 1088 `
  --steps 15 `
  --output outputs/waterfall_121f.mp4
```
