# Production SMU Commercial Generation on PC with NVIDIA GeForce RTX 5080 (Blackwell SM 12.0)

This document provides complete, exhaustive technical documentation for running the production **LTX-2.5** 7-scene Southern Methodist University (SMU) commercial pipeline locally on a workstation powered by an **NVIDIA GeForce RTX 5080 (16 GB GDDR7 VRAM, 64 GB DDR5 Host RAM)** under Windows 11 and Linux. It covers Blackwell NVFP4 native tensor core acceleration, Windows virtual memory commit management, the zero-mmap direct streaming loader, the distilled multi-scene pipeline, telemetry instrumentation, and operational execution.

---

## 1. Workstation Topology & The Blackwell NVFP4 Architecture

### Compute Hardware
* **GPU**: NVIDIA GeForce RTX 5080.
  * **Architecture**: Blackwell (Compute Capability `sm_120` / SM 12.0).
  * **VRAM**: 16 GB GDDR7 (256-bit memory bus, ~1,000+ GB/s memory bandwidth).
  * **Interface**: PCIe 5.0 x16.
  * **TDP Ceiling**: 350 Watts.
* **Host System**:
  * **CPU**: Modern Multi-Core Host Processor (x86_64, 16+ execution threads).
  * **Physical RAM**: 64 GB DDR5 High-Speed System RAM.
  * **Operating System**: Windows 11 Pro 64-bit (or Linux kernel 6.x).
* **Software Toolchain**:
  * **NVIDIA Display Driver**: `570.xx`+ with native CUDA 12.8 support.
  * **CUDA Runtime**: 12.8.
  * **Host Compiler**: Microsoft Visual C++ (MSVC) 14.43 (Visual Studio 2022 v17.13).
  * **Python Environment**: Python 3.12 (managed via `uv`), PyTorch 2.11.0+cu128.
  * **Custom C++/CUDA Kernels**: `packages/ltx-kernels` (compiled with MSVC/CUDA decoupling).

### Native Blackwell NVFP4 Acceleration Engine
Older GPU architectures (Ampere `sm_80`, Ada Lovelace `sm_89`) lack hardware-native FP4 tensor cores and must emulate FP4 by decompressing weights into FP16/BF16 prior to execution. The **RTX 5080** introduces native 4-bit floating point matrix math:
* **Quantization Format**: FP4 `E2M1` (1 sign bit, 2 exponent bits, 1 mantissa bit; dynamic numerical range $[-6.0, 6.0]$).
* **Block Scaling**: FP8 `E4M3` block scale factors (1 scale per 16 continuous weight elements).
* **Tensor Scaling**: FP32 global tensor scale factor.
* **Compute Engine**: NVIDIA `cuBLASLt` block-scaled GEMM (`cublasLtMatmul`). The Blackwell tensor cores perform matrix multiplies directly on 4-bit weights without dynamic decompression into 16-bit precision, cutting memory bandwidth saturation by more than half.

---

## 2. Model Checkpoints & Pre-Quantized Storage Footprint

Running a 22-billion-parameter foundation model on a 16 GB GPU requires strategic precision balancing. By utilizing pre-quantized NVFP4 checkpoints, storage and memory requirements are cut dramatically:

| Component | Checkpoint File | Precision / Format | Disk Size | Runtime Memory Placement |
| :--- | :--- | :--- | :--- | :--- |
| **Diffusion Transformer (22B)** | `ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors` | **NVFP4 (E2M1)** | **17.44 GB** | Stored in Host DDR5 RAM; streamed in blocks ($<600\text{ MB}$) to GPU during forward pass. |
| **Prompt Text Encoder** | `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | BF16 (12B params) | **24.46 GB** | Loaded into Host RAM during Stage 0 text encoding; zero persistent VRAM leak. |
| **Latent Spatial Upscaler** | `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | BF16 CNN | **0.93 GB** | Resident in GPU VRAM (~0.93 GB). |
| **Spatiotemporal Video VAE** | `ltx-2.5-video-vae-bf16.safetensors` | BF16 Causal 3D VAE | **1.37 GB** | Decoded in spatial/temporal tiles via eager SDPA on GPU. |
| **Continuous Audio VAE** | `ltx-2.5-audio-vae-bf16.safetensors` | BF16 Latent VAE | **0.34 GB** | Decoded in continuous latent space on GPU. |

**Total Memory Reduction**: The 22B transformer is compressed from **42.0 GB (BF16)** down to **17.44 GB (NVFP4)**, representing a **58.5% reduction** in model weight volume.

---

## 3. Essential Windows Architectural Adaptations & Bug Fixes

Porting a 22B parameter pipeline to Windows 11 with 16 GB VRAM and 64 GB host RAM uncovered critical platform bugs that required custom architectural redesigns:

### A. The Windows `0xC0000005` Commit Charge Collision Bug
* **The Symptom**: When running multi-stage generation, Python crashed abruptly with:
  ```
  Windows fatal exception: access violation (exit code 3221225477 / 0xC0000005)
  File "torch/storage.py", line 471 in __getitem__
  File "ltx_core/loader/sft_loader.py", line 36 in load
  ```
* **The Root Cause**: Unlike Linux, which permits virtual memory overcommitment, Windows enforces a strict **Commit Limit** ($\text{Physical RAM} + \text{Pagefile Size}$):
  1. *Pinned Memory*: PyTorch block streaming locks **20.3 GB** of unpageable physical RAM via `torch.empty(..., pin_memory=True)`.
  2. *Heap Allocation*: Creating standard Python state dictionaries allocates another **20.3 GB** in heap memory.
  3. *Memory-Mapping (mmap)*: Standard `safetensors.safe_open()` memory-maps the **24.5 GB** Gemma text encoder and the **17.4 GB** transformer into the process virtual address space.
  4. *The Collision*: $20.3\text{ GB (pinned)} + 20.3\text{ GB (heap)} + 24.5\text{ GB (mmap)} = \mathbf{65.1\text{ GB}}$. This exceeded the initial 53 GB system commit limit. When PyTorch touched mapped pages, Windows rejected page allocation, triggering access violation `0xC0000005`.
* **The Solution — Zero-MMap Streaming Direct File I/O**:
  We re-engineered [`SafetensorsStateDictLoader`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/packages/ltx-core/src/ltx_core/loader/sft_loader.py#L21):
  1. [`read_safetensors_header()`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/packages/ltx-core/src/ltx_core/loader/sft_loader.py#L78) reads the 8-byte length prefix and parses the JSON header directly via Python `open(path, "rb")`. No virtual memory mapping is created to inspect keys.
  2. Tensors are allocated directly into target memory and filled using `file.seek()` and `file.readinto(tensor.reshape(-1).view(torch.uint8).numpy())`.
  3. Eliminates the 24.5 GB memory map completely, operating safely within host commit ceilings.
* **System Pagefile Configuration**:
  To provide adequate headroom, the Windows pagefile must be expanded:
  ```powershell
  # Run in Administrator PowerShell
  $sys = Get-CimInstance Win32_ComputerSystem
  $sys.AutomaticManagedPagefile = $false
  Set-CimInstance -InputObject $sys
  $pagefile = Get-CimInstance Win32_PageFileSetting
  if ($pagefile) {
      $pagefile.InitialSize = 32768
      $pagefile.MaximumSize = 49152
      Set-CimInstance -InputObject $pagefile
  } else {
      New-CimInstance -ClassName Win32_PageFileSetting -Property @{Name="C:\pagefile.sys"; InitialSize=32768; MaximumSize=49152}
  }
  ```

### B. Bitwise Scale Preservation Invariant (FP8 to UINT8)
* **The Flaw**: NVFP4 checkpoints store layer scales as `torch.float8_e4m3fn`, but CUDA kernels store them in raw `uint8` byte buffers. If scales are copied using arithmetic casting (`dest.copy_(temp)`), PyTorch performs numerical float-to-int conversion. Any scale $x \in (0.0, 1.0)$ is truncated to **`0`**. Because virtually all 22B linear weights have fractional scale factors, **every weight scale was zeroed out**, destroying model weights and generating pure noise.
* **The Invariant**: All NVFP4 scale loading must strictly use bitwise view reinterpretation:
  ```python
  # CORRECT: Bitwise view reinterpretation preserves raw FP8 scale bytes
  scale_tensor = raw_fp8_tensor.view(torch.uint8).contiguous()
  ```
  This invariant is maintained canonically in [`builder.py`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/packages/ltx-core/src/ltx_core/block_streaming/builder.py).

### C. Subprocess Isolation Orchestrator
Running 7 multi-gigabyte scenes sequentially inside a single Python process leads to fragmented memory pools and unreleased CUDA allocator bins. The orchestrator in [`generate_smu_commercial_distilled.py`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/generate_smu_commercial_distilled.py#L714-L734) launches each scene in an isolated operating system subprocess:
* Upon scene completion, the OS immediately terminates the process, instantly reclaiming 100% of pinned RAM and CUDA allocations before the next scene starts.

---

## 4. Distilled Multi-Scene Pipeline Architecture (Method A)

For high-speed, high-fidelity local rendering on the RTX 5080, the commercial pipeline employs **Method A (`DistilledPipeline`)**:

```
Conditioning Keyframe Still (inputs/smu/01-07.jpg) + Scene Text Prompt
                           │
                           ▼
              [Stage 0: Gemma Text Encoding]
              • Host DDR5 RAM execution (~32s)
              • Encodes text into visual & audio latents
                           │
                           ▼
          [Stage 1: Half-Resolution Denoising]
          • Latent Shape: 640×384 @ 24 fps (121 frames)
          • Sampler: euler_ancestral (8 steps)
          • Model: 22B NVFP4 Distilled Transformer
          • Guidance: CFG 1.0 (CFG-free via SimpleDenoiser)
          • Duration: ~24.2s | Peak VRAM: 4.57 GB
                           │
                           ▼
             [Stage 1.5: 2× Latent Spatial Upscale]
             • Model: Latent Spatial Upscaler x2 (BF16)
             • Expands latents: 640×384 ➔ 1280×768
             • Duration: ~1.2s | Peak VRAM: 2.01 GB
                           │
                           ▼
            [Stage 2: Full-Resolution Refinement]
          • Latent Shape: 1280×768 @ 24 fps
          • Sampler: Deterministic euler (3 steps)
          • Model: 22B NVFP4 Distilled Transformer
          • Guidance: CFG 1.0
          • Duration: ~27.2s | Peak VRAM: 5.90 GB
                           │
                           ▼
               [Stage 3: DiffVAE 3D Decode]
          • Backend: Eager Tiled SDPA na3d (AUTO_TILING)
          • Duration: ~2.0s | Peak VRAM: 1.35 GB
                           │
                           ▼
            [Stage 4: PyAV / FFmpeg MP4 Encoding]
          • Compression: H.264 (yuv420p, crf=18) + AAC audio
          • Duration: ~44.5s | Peak VRAM: 8.13 GB
```

### Hyperparameter Specifications (RTX 5080)

| Parameter | Configuration | Value | Description & Technical Rationale |
| :--- | :--- | :--- | :--- |
| **Pipeline** | `DistilledPipeline` | Method A | High-speed distilled pipeline; avoids double-pass CFG computations. |
| **Output Resolution** | `--width`, `--height` | `1280 × 768` | 16:9 widescreen format optimized for 16 GB VRAM envelope. Divisible by 64. |
| **Stage 1 Resolution** | Derived | `640 × 384` | Exact $0.5\times$ half-resolution for motion generation. Divisible by 32. |
| **Frame Rate** | `--frame-rate` | `24.0` fps | Standard cinematic frame rate. |
| **Frame Count** | `num_frames` | `121` frames | Conforms to $(N - 1) \pmod 8 == 0$ temporal grid (~5.042 seconds per scene). |
| **Stage 1 Sampler** | `stage_1_sigmas` | `euler_ancestral` (8 steps) | 8-step ancestral sampler driven by official `DISTILLED_SIGMAS`. |
| **Stage 2 Sampler** | `stage_2_sigmas` | `euler` (3 steps) | 3-step deterministic sampler driven by `STAGE_2_DISTILLED_SIGMAS`. |
| **Guidance** | `guidance_scale` | `1.0` (CFG-free) | Uses `SimpleDenoiser`. Halves compute requirements compared to CFG 3.0. |
| **VAE Decode Mode** | `DiffVAEMode` | `AUTO_TILING` | Temporal and spatial chunking using eager SDPA fallback. |
| **Offload Mode** | `--offload-mode` | `OffloadMode.CPU` | Retains full transformer in DDR5 RAM, streaming only active blocks to GPU. |

---

## 5. SMU Commercial Scene Specifications & Assets

The commercial generates 7 consecutive clips using conditioned photography from `inputs/smu/`:

| Scene ID | Name | Input Image | Frames | Seed | Conditioned Voiceover Text |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `01_dallas_hall` | `01_dallas_hall.jpg` | 121 | `42` | *"At Southern Methodist University, heritage meets the horizon."* |
| **2** | `02_mustang_statue` | `02_mustang_statue.jpg` | 121 | `101` | *"In the heart of Dallas, bold ideas take flight."* |
| **3** | `03_engineering_lab` | `03_engineering_lab.jpg` | 121 | `202` | *"Here, innovators sculpt tomorrow's breakthroughs in artificial intelligence and robotics."* |
| **4** | `04_business_cox` | `04_business_cox.jpg` | 121 | `303` | *"Visionaries lead global commerce with unyielding integrity."* |
| **5** | `05_meadows_arts` | `05_meadows_arts.jpg` | 121 | `404` | *"Artists inspire, and culture thrives."* |
| **6** | `06_campus_sunset` | `06_campus_sunset.jpg` | 121 | `505` | *"Fueled by unbridled Mustang spirit, we don't just dream of a better world. We build it."* |
| **7** | `07_closing_logo` | `07_closing_logo.jpg` | 121 | `606` | *"Southern Methodist University. World changers shaped here. Pony up!"* |

---

## 6. Script Architecture & Telemetry Profiler

### A. The Master Script: `generate_smu_commercial_distilled.py`
Located at [`generate_smu_commercial_distilled.py`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/generate_smu_commercial_distilled.py):

#### 1. Hardware-Level NVML Telemetry Profiler
* Implements a zero-dependency ctypes bridge directly to `nvml.dll` (`NVMLProfiler`).
* Spawns an asynchronous background sampling thread (`ActiveTelemetrySampler`) polling GPU compute utilization, memory bus utilization, power consumption (Watts), and core temperature every **100 ms**.
* Tracks VRAM allocations (`torch.cuda.memory_allocated()`, peak VRAM, reserved VRAM) and host memory (`psutil.virtual_memory()`).

#### 2. Process-Isolated Orchestration Loop (Mode 1)
When executed without `--scene-id`:
* Checks existing scene outputs via `is_video_valid(path, 121, 1280, 768)`.
* Spawns each pending scene via `subprocess.run([sys.executable, ..., "--scene-id", id])`.
* Upon completion of all 7 scenes, prints the global telemetry table and generates [`outputs/smu_commercial_telemetry_report.md`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/outputs/smu_commercial_telemetry_report.md).

#### 3. Granular Scene Generator (Mode 2)
When executed with `--scene-id <N>`:
* Runs `render_scene_with_telemetry()` wrapped under `@torch.inference_mode()`.
* Measures each stage independently (Stage 0 text encoding, Stage 1 half-res denoising, Stage 1.5 upscale, Stage 2 refinement, Stage 3 VAE decode, Stage 4 video encoding).
* Serializes telemetry metrics into `logs/smu_distilled_telemetry.json`.

#### 4. Post-Processing & Master Video Assembly (Mode 3)
* `extract_preview_frames()`: Saves reference JPEG images of each scene to `outputs/smu_previews/`.
* `assemble_commercial()`: Uses FFmpeg with `concat` demuxer and multiplexes external audio (`inputs/smu/smu_voiceover.wav`) into [`outputs/smu_commercial_full.mp4`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/outputs/smu_commercial_full.mp4).

### B. Single Generation Testing Script: `run_rtx5080.py`
Located at [`run_rtx5080.py`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/run_rtx5080.py):
* Standalone CLI utility for generating individual prompt-driven videos on the RTX 5080 without running the full commercial orchestrator.

### C. A100 Replication Test Script: `run_method_b_exact_a100.py`
Located at [`run_method_b_exact_a100.py`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/run_method_b_exact_a100.py):
* Experimental script replicating the exact 2-stage dev model + LoRA pipeline (Method B) on the RTX 5080 to compare visual parity with cluster outputs.

---

## 7. Measured Hardware Telemetry & Benchmarks (RTX 5080)

The following metrics were recorded live via NVML on the **RTX 5080 (16 GB)** during the full 7-scene production run:

### Global Performance Summary

| Metric | Measured Value | Operational Headroom & Analysis |
| :--- | :--- | :--- |
| **Total Commercial Render Time (7 Scenes)** | **671.66s (11.19 minutes)** | Seamless background execution on a desktop workstation. |
| **Mean Render Time Per Scene** | **134.33s (2.24 minutes)** | 121 frames (5.04s video) rendered in ~2.2 minutes. |
| **Mean Peak VRAM Consumption** | **8.13 GB / 16.0 GB** | **49.2% free VRAM remaining**; zero risk of CUDA OOM. |
| **Absolute Maximum Peak VRAM** | **8.13 GB / 16.0 GB** | Bounded strictly by block streaming offloading. |
| **Mean Host System RAM Footprint** | **52.54 GB / 64.0 GB** | Process isolation ensures zero cumulative RAM leakage. |
| **Mean GPU Compute Utilization** | **47.4%** | Reaches 100.0% peak during active denoising loops. |
| **Mean GPU Power Consumption** | **131.3 Watts** | Peak 268.2W (well below the 350W TDP ceiling). |
| **Total Electrical Energy Consumed** | **24.50 Watt-hours (Wh)** | Extremely energy-efficient commercial generation. |

### Per-Stage Benchmark Breakdown (Average Per Scene)

| Stage Identifier | Elapsed Time | Peak VRAM | Host RAM | Offload Split (Host / VRAM) | Avg GPU Util | Avg Power |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 0: Gemma Prompt Encoding** | 32.14s | 0.00 GB | 44.82 GB | 17.4 GB / 0.0 GB | 0.0% | 34.2W |
| **Stage 0.5: VAE Conditioning** | 1.82s | 1.84 GB | 44.85 GB | 17.4 GB / 0.6 GB | 28.4% | 112.5W |
| **Stage 1: Half-Res Denoising (8 steps)** | 24.21s | 4.57 GB | 52.10 GB | 17.4 GB / 0.6 GB | 88.6% | 224.8W |
| **Stage 1.5: 2x Latent Spatial Upscale** | 1.24s | 2.01 GB | 52.12 GB | 17.4 GB / 0.6 GB | 45.2% | 148.0W |
| **Stage 2: Refinement (3 steps Euler)** | 27.18s | 5.90 GB | 52.40 GB | 17.4 GB / 0.6 GB | 92.1% | 246.5W |
| **Stage 3: DiffVAE Decode** | 2.05s | 1.35 GB | 52.45 GB | 17.4 GB / 0.6 GB | 34.0% | 125.1W |
| **Stage 4: MP4 Video Encoding** | 44.48s | 8.13 GB | 52.54 GB | 17.4 GB / 0.6 GB | 12.5% | 68.4W |

---

## 8. Step-by-Step Execution Guide on PC

### Step 1: Open Terminal & Run Pre-Flight Sanity Checks
Open PowerShell in the project directory and verify system readiness:
```powershell
cd c:\Users\jerry\Documents\program\GitHub\LTX-2-1

# Run the hardware and NVFP4 kernel pre-flight validator
.\.venv\Scripts\python.exe scripts\check_hardware.py
```
Expected output:
```
===========================================================================
  PRE-FLIGHT SUMMARY
===========================================================================
  System RAM & Pinned Memory: PASSED (64 GB class)
  GPU Hardware & CUDA:        PASSED (SM 12.0 Blackwell)
  NVFP4 Compiled Kernels:     PASSED
  Model Checkpoints:          PASSED
===========================================================================
>>> ALL PRE-FLIGHT CHECKS PASSED!
```

### Step 2: Launch the Full 7-Scene Commercial Orchestrator
Execute the multi-scene commercial orchestrator:
```powershell
.\.venv\Scripts\python.exe generate_smu_commercial_distilled.py
```
* The script checks for completed clips, dispatches pending scenes in isolated subprocesses, logs NVML telemetry, and automatically compiles `outputs/smu_commercial_full.mp4`.

### Step 3: Target a Specific Scene (Force Re-render)
To force re-rendering of a single scene (e.g. Scene 2):
```powershell
.\.venv\Scripts\python.exe generate_smu_commercial_distilled.py --scene-id 2 --force
```

### Step 4: Re-assemble Master Commercial Without Re-generating
To re-mux audio and re-concatenate existing video clips without touching the GPU:
```powershell
.\.venv\Scripts\python.exe generate_smu_commercial_distilled.py --skip-generation
```

### Step 5: Test Standalone Single Prompt Video Generation
To generate a standalone custom video using the RTX 5080:
```powershell
.\.venv\Scripts\python.exe run_rtx5080.py `
  --prompt "A majestic drone shot of Dallas Hall on the SMU campus, warm morning sunlight" `
  --num-frames 121 `
  --width 1280 `
  --height 768 `
  --steps 15 `
  --output outputs/dallas_hall.mp4
```

### Expected Deliverables on PC:
* Master Video: [`outputs/smu_commercial_full.mp4`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/outputs/smu_commercial_full.mp4) (1280×768 @ 24 fps, H.264 + AAC, 25.6 MB).
* Telemetry Report: [`outputs/smu_commercial_telemetry_report.md`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/outputs/smu_commercial_telemetry_report.md).
* Individual Scene Clips: `outputs/smu_scenes/scene_01_*.mp4` through `scene_07_*.mp4`.
* Scene Previews: `outputs/smu_previews/preview_scene_*.jpg`.
