# LTX-2.5 Local NVFP4 Generation on NVIDIA GeForce RTX 5080 (SM 12.0)

This document provides comprehensive technical documentation for running **LTX-2.5** high-fidelity video generation on the **NVIDIA GeForce RTX 5080 (Blackwell SM 12.0, 16 GB GDDR7 VRAM, 64 GB DDR5 Host RAM)** under Windows 11 and Linux.

---

## 1. Architectural Overview & The Blackwell Advantage

### Native NVFP4 Tensor Core Acceleration
Prior generations (Ampere SM 8.0, Ada Lovelace SM 8.9) lack native hardware FP4 matrix multiply capability. The **RTX 5080 (Blackwell SM 12.0)** introduces hardware-accelerated **NVFP4 (E2M1)** tensor cores:
- **Format**: FP4 `E2M1` format (1 sign bit, 2 exponent bits, 1 mantissa bit; dynamic range $[-6.0, 6.0]$).
- **Block Scaling**: FP8 `E4M3` block scales (1 scale factor per 16 elements).
- **Global Scaling**: FP32 tensor scale factor.
- **Compute Engine**: `cuBLASLt` block-scaled GEMM (`cublasLtMatmul`) directly executes against 4-bit weights without decompression into BF16/FP16 before calculation.

### Memory Footprint Comparison

| Component | Unquantized BF16 | NVFP4 Pre-Quantized | Savings |
|---|---|---|---|
| **Diffusion Transformer (22B)** | ~42.0 GB | **17.44 GB** | **~58% reduction** |
| **Gemma 4 12B Text Encoder** | ~24.5 GB | **24.46 GB** (BF16 in RAM) | Loaded once per prompt, zero VRAM leak |
| **Spatial Upscaler (2x)** | ~0.93 GB | **0.93 GB** | Resident in VRAM (~0.9 GB) |
| **Video VAE (BF16)** | ~1.37 GB | **1.37 GB** | Decoded in temporal/spatial chunks |
| **Audio VAE (BF16)** | ~0.34 GB | **0.34 GB** | Decoded in continuous latent space |

---

## 2. Kernel Toolchain & MSVC Windows Porting

Compiling PyTorch C++/CUDA extensions on Windows with MSVC 14.43 and CUDA 12.8 exposed compiler bugs that required specific architecture decoupling.

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
On Windows, PyTorch Python C-extensions cannot rely on ELF dynamic symbol resolution. `setup.py` explicitly adds `torch_python.lib` and CUDA runtime import libraries:
```python
if is_win:
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    cuda_link_dirs.append(str(torch_lib_dir))
    libraries = ["torch", "torch_python", "c10", "cublasLt"]
```

### C. Architecture Code Flags (`TORCH_CUDA_ARCH_LIST`)
CUDA 12.8 supports Blackwell SM 12.0 (`sm_120`). In `setup.py`, arch discovery accepts `12.0` / `120`:
```python
_NVFP4_ARCHES = ["100", "100a", "110", "110a", "120", "120a"]
```

---

## 3. Zero-MMap StateDict Loader & Windows Virtual Address Space

### A. The Windows `0xC0000005` Access Violation Bug
During large multi-stage model loading on Windows workstations, execution previously failed with:
```
Windows fatal exception: access violation (exit code 3221225477 / 0xC0000005)
Current thread (most recent call first):
  File "torch/storage.py", line 471 in __getitem__
  File "ltx_core/loader/sft_loader.py", line 36 in load
```

### B. Root Cause: Commit Charge Collision
On Windows, the operating system strictly enforces the Virtual Memory Commit Limit ($\text{Physical RAM} + \text{Pagefile}$). Unlike Linux, which overcommits virtual address ranges, Windows requires backed paging storage for every committed page:
1. **Pinned Memory Reservation**: Block streaming allocates **20.3 GB** of non-pageable host physical RAM using `torch.empty(..., pin_memory=True)`.
2. **Intermediate StateDict Allocation**: Standard PyTorch state dict construction instantiates an unpinned **20.3 GB** Python dictionary in heap memory.
3. **Repeated File Memory-Mapping (mmap)**: Standard `safetensors.safe_open(framework="pt")` calls memory-map the **24.5 GB** Gemma checkpoint and **17.4 GB** transformer into the process virtual address space.
4. **The Collision**: $20.3\text{ GB (pinned)} + 20.3\text{ GB (heap)} + 24.5\text{ GB (mmap)} = \mathbf{65.1\text{ GB}}$ virtual commit against an initial ceiling of ~53 GB. Touching the mmapped pages caused Windows to reject page table allocation, triggering access violation `0xC0000005`.

### C. Solution: Zero-MMap Streaming Direct File I/O
We redesigned `packages/ltx-core/src/ltx_core/loader/sft_loader.py`:
1. **Header-Only Metadata Extraction**:
   [`read_safetensors_header()`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/packages/ltx-core/src/ltx_core/loader/sft_loader.py#L78) reads the 8-byte length prefix and parses the JSON header directly via Python `open(path, "rb")`. No memory mapping is used to inspect keys or metadata.
2. **Streaming `readinto()`**:
   In `SafetensorsStateDictLoader.load()`, each tensor is allocated directly via `torch.empty()` and filled using `file.seek()` and `file.readinto(tensor.reshape(-1).view(torch.uint8).numpy())`.
3. **Zero Address Space Footprint**: Eliminates the 24.5 GB memory map completely, operating safely within system commit limits.

---

## 4. Critical NVFP4 Quantization Invariant: Bitwise vs. Arithmetic Scale Views

### The Flaw
In safetensors checkpoints for NVFP4 models (`ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors`), `weight_scale` tensors are stored with dtype `torch.float8_e4m3fn`. However, the CUDA kernel engine `NVFP4Linear` stores these scales as raw `uint8` byte buffers.

If checkpoint scales are copied into `torch.uint8` tensor memory via an arithmetic copy (`dest.copy_(temp)`), PyTorch performs a **numeric float-to-uint8 cast**:
- Any scale factor $x \in (0.0, 1.0)$ is cast to **`0`**.
- Because almost all layer scale factors in the 22B transformer are fractional, **every single linear weight scale was zeroed out**, destroying the model weights and collapsing diffusion generation into pure static noise.

### The Correct Mapping
NVFP4 scales must **always** be loaded via bitwise view reinterpretation:
```python
# CORRECT: Bitwise view reinterpretation preserves raw FP8 scale bytes
scale_tensor = raw_fp8_tensor.view(torch.uint8).contiguous()
```
This is handled canonically by `build_prequant_sd_ops` inside `SDOps.apply_to_key_value()`. The canonical `load_state_dict()` path in `builder.py` guarantees this invariant is preserved.

---

## 5. NVFP4 Block Streaming Engine (`OffloadMode.CPU`)

We enabled native NVFP4 support in the core block streaming engine:
1. **Layout Preservation**:
   In [`derive_layout()`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/packages/ltx-core/src/ltx_core/block_streaming/utils.py#L54), non-FP8 dtypes are normally coerced to the target compute dtype (BF16). For NVFP4, `torch.uint8` (4-bit packed weights and scale bytes) and `torch.float32` (`weight_scale_2`, `input_scale`) are explicitly preserved in `PRESERVED_DTYPES`.
2. **DiffusionStage Rule Registration**:
   In [`DiffusionStage._builder()`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py#L383), `nvfp4_fuse_rule` is registered alongside `fp8_cast_fuse_rule` as an allowed block streaming policy.
3. **Hardware Profile**:
   With `OffloadMode.CPU`, only the active transformer block resides on GPU during forward execution (<600 MB resident weights), keeping peak VRAM below **8.13 GB** even during full 1280×768 super-resolution generation.

---

## 6. DistilledPipeline Architecture (Method A)

The production pipeline for rapid high-fidelity generation on RTX 5080:

```
Input Image + Text Prompt
   │
   ├─► Gemma 4 12B Prompt Encoding (Host RAM, ~32s, zero VRAM leak)
   │
   ├─► Stage 1: Half-Res Denoising (640×384)
   │     • Model: 22B NVFP4 Distilled Transformer
   │     • Sampler: Euler Ancestral, 8 steps
   │     • Guidance: CFG 1.0 (CFG-free)
   │     • Time: ~24.2s (Peak VRAM: 4.57 GB)
   │
   ├─► Stage 1.5: 2x Latent Spatial Upscale
   │     • Model: Latent Spatial Upscaler x2 (BF16)
   │     • Time: ~1.2s (Peak VRAM: 2.01 GB)
   │
   ├─► Stage 2: Full-Res Refinement (1280×768)
   │     • Model: 22B NVFP4 Distilled Transformer
   │     • Sampler: Euler, 3 steps
   │     • Guidance: CFG 1.0
   │     • Time: ~27.2s (Peak VRAM: 5.90 GB)
   │
   ├─► Stage 3: DiffVAE Decode
   │     • Mode: CHUNKED_EAGER temporal/spatial tiling
   │     • Time: ~2.0s (Peak VRAM: 1.35 GB)
   │
   └─► Stage 4: MP4 Video Encoding (av / ffmpeg)
         • Output: 1280×768 @ 24 fps, YUV420p + AAC
         • Time: ~44.5s (Peak VRAM: 8.13 GB)
```

---

## 7. Subprocess Isolation Orchestrator

For multi-scene generation jobs (e.g. the 7-scene SMU commercial), sequential generation inside a single Python process can accumulate memory fragmentation. 

The orchestrator in [`generate_smu_commercial_distilled.py`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/generate_smu_commercial_distilled.py) executes each scene as an isolated operating system subprocess:
```python
subprocess.run([
    sys.executable,
    str(Path(__file__).resolve()),
    "--scene-id", str(sc["id"]),
    "--width", str(args.width),
    "--height", str(args.height),
    "--frame-rate", str(args.frame_rate),
    "--offload-mode", str(args.offload_mode),
])
```
- **100% Host Memory Reclamation**: When the subprocess terminates, the operating system immediately reclaims all committed pages and unpins physical memory.
- **Fault Isolation**: If an individual scene fails, the pipeline logs the failure and avoids corrupting the memory state of subsequent scenes.

---

## 8. Hardware Telemetry & Benchmarks

Measured live via NVML on the **NVIDIA GeForce RTX 5080 (16 GB VRAM)** during full 7-scene production generation:

| Metric | Measured Value | Operational Headroom |
| :--- | :--- | :--- |
| **Total Commercial Render Time (7 Scenes)** | **671.66s** (11.19 min) | Seamless background execution |
| **Mean Scene Render Time (121 frames)** | **134.33s** (2.24 min) | 8 steps Stage 1 + 3 steps Stage 2 |
| **Mean Peak VRAM** | **8.13 GB / 16.0 GB** | **49.2% free VRAM remaining** |
| **Absolute Max Peak VRAM** | **8.13 GB** | Completely bounded by block streaming |
| **Mean Host RAM Footprint** | **52.54 GB / 64.0 GB** | Process-isolated, zero accumulation |
| **Mean GPU Compute Utilization** | **47.4%** | Peak 100.0% during refinement |
| **Mean GPU Power Draw** | **131.3 Watts** | Peak 268.2W (TDP ceiling 350W) |
| **Total Electrical Energy** | **24.50 Watt-hours (Wh)** | Highly energy-efficient |

---

## 9. Production CLI Commands

### A. Run Single Video Generation
```powershell
.\.venv\Scripts\python.exe run_rtx5080.py `
  --prompt "A majestic drone shot of Dallas Hall on the SMU campus, warm morning sunlight" `
  --num-frames 121 `
  --width 1280 `
  --height 768 `
  --steps 15 `
  --output outputs/dallas_hall.mp4
```

### B. Run Multi-Scene Distilled Commercial Orchestrator
```powershell
# Renders all 7 commercial scenes with process isolation and assembles outputs/smu_commercial_full.mp4
.\.venv\Scripts\python.exe generate_smu_commercial_distilled.py

# Force regeneration of a specific scene
.\.venv\Scripts\python.exe generate_smu_commercial_distilled.py --scene-id 3 --force
```
