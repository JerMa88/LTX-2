# Running LTX-2.5 Local NVFP4 Generation on NVIDIA RTX 5080 (16GB VRAM / 64GB RAM)

This guide provides the battle-tested, production-ready blueprint to run the official **LTX-2.5** high-quality video generation pipeline on a personal workstation equipped with:
- **GPU**: NVIDIA GeForce RTX 5080 (16 GB GDDR7 VRAM, Blackwell Architecture, Compute Capability **SM 12.0** / `sm_120`)
- **System Memory**: 64 GB DDR5 Host RAM (with >= 80 GB pagefile commit limit)
- **Operating System**: Windows 11 64-bit (or Linux / Ubuntu 22.04+ / WSL2)

---

## 1. Hardware Architecture & Feasibility Math

### Can 16 GB VRAM Run LTX-2.5 1080p?
**YES, 100% operational via native Blackwell NVFP4 quantization and block streaming.**

| Component | Baseline BF16 | RTX 5080 NVFP4 Setup | Memory Location & Peak Footprint |
|---|---|---|---|
| **Diffusion Transformer (22B)** | 42 GB (OOM on 16GB) | **NVFP4** (17.44 GB) | **Native VRAM** (`OffloadMode.NONE`) or **CPU Block Streaming** (`OffloadMode.CPU`, <600MB resident on GPU). |
| **Gemma 4 12B Text Encoder** | 24.5 GB (OOM on 16GB) | Gemma 4 12B BF16 + Projection | **Host RAM**. Loaded during prompt encoding, immediately freed. Zero-mmap streaming prevents commit spikes. |
| **Spatial Upscaler (2x)** | 0.93 GB | LTX-2.5 Latent Upscaler | **VRAM** (~0.9 GB). |
| **Stage 1 (640×384)** | ~12 GB VRAM | Active Denoising State | **~4.57 GB VRAM** peak with NVFP4. |
| **Stage 2 Super-Resolution (1280×768)** | ~28 GB VRAM (OOM on 16GB) | Latent Spatial Upscale + Euler | **~5.90 GB VRAM** peak. |
| **DiffVAE Video Decoder** | ~34 GB VRAM (OOM on 16GB) | `CHUNKED_EAGER` tiling | **~1.35 GB VRAM** peak. |
| **Total VRAM Utilization** | OOM on 16GB | **DistilledPipeline NVFP4** | **8.13 GB Peak VRAM** (50.8% of 16 GB capacity). |

### The Blackwell Advantage
Unlike Ampere (A100) or Ada Lovelace (RTX 4090), the **RTX 5080** features **Blackwell SM 12.0 NVFP4 hardware Tensor Cores**:
- **E2M1 FP4** tensor weights.
- **E4M3 FP8** block scales (1 scale per 16 elements).
- **FP32** global tensor scale.
- Native **cuBLASLt** execution without unpacking overhead.

---

## 2. Workstation Prerequisites

### A. Drivers & CUDA
1. **NVIDIA Display Driver**: Version **572.61** or higher.
2. **CUDA Toolkit**: Version **12.8** (e.g., installed at `C:\Users\<user>\cuda_12.8` or standard CUDA path).
3. **C++ Compiler Tools**:
   - **Windows 11**: Visual Studio 2022 Build Tools (MSVC v143 - VS 2022 C++ x64/x86 build tools + Windows 10/11 SDK).
   - **Linux**: `sudo apt update && sudo apt install -y build-essential ninja-build git git-lfs ffmpeg`
4. **Python**: Python **3.12** 64-bit.
5. **Windows Pagefile / Virtual Memory**:
   - Total commit limit must be at least **80 GB** (Physical RAM + Pagefile).
   - Verify via PowerShell:
     ```powershell
     Get-CimInstance Win32_PageFileUsage | Select-Object AllocatedBaseSize, CurrentUsage
     ```

---

## 3. Step-by-Step Installation

### Step 1: Environment Setup
```powershell
# Create Python 3.12 virtual environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install PyTorch with CUDA 12.8
```powershell
.\.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Install Core Packages & Dependencies
```powershell
.\.venv\Scripts\pip.exe install -e packages/ltx-core
.\.venv\Scripts\pip.exe install -e packages/ltx-pipelines
.\.venv\Scripts\pip.exe install imageio[ffmpeg] av decord safetensors huggingface_hub psutil
```

### Step 4: Compile Blackwell NVFP4 Kernels (`ltx-kernels`)
```powershell
# Option A: Run automated build script
.\scripts\build_ltx_kernels.bat

# Option B: Manual pip installation
$env:CUDA_HOME = "C:\Users\jerry\cuda_12.8"
$env:PATH = "$env:CUDA_HOME\bin;" + $env:PATH
$env:TORCH_CUDA_ARCH_LIST = "12.0"
.\.venv\Scripts\pip.exe install -e packages/ltx-kernels --no-build-isolation
```

**Verify Kernel Availability**:
```powershell
.\.venv\Scripts\python.exe -c "import ltx_kernels.nvfp4 as nvfp4; print('NVFP4 Available:', nvfp4.is_available())"
# Output must be: NVFP4 Available: True
```

---

## 4. Checkpoint Organization

All models reside under `models/ltx25`:
```
models/ltx25/
├── diffusion_models/
│   └── ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors (17.44 GB)
├── latent_upscale_models/
│   └── ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors (0.93 GB)
├── vae/
│   ├── ltx-2.5-video-vae-bf16.safetensors (1.37 GB)
│   └── ltx-2.5-audio-vae-bf16.safetensors (0.34 GB)
└── text_encoders/
    └── gemma4-12b-with-proj-ltx-2.5-bf16.safetensors (24.46 GB)
```

---

## 5. Pre-Flight Diagnostic Check

Verify all hardware, kernels, and files before launching long runs:
```powershell
.\.venv\Scripts\python.exe preflight_check.py
```
Ensure all 4 sections pass:
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

---

## 6. Running Video Generations

### Method A: Ultra-Fast Distilled Pipeline (Recommended)
Generates high-definition (1280×768) video in **~134 seconds (~2.2 minutes)** using the 22B NVFP4 distilled model:

```powershell
# Run the multi-scene commercial orchestrator (isolated subprocesses per scene)
.\.venv\Scripts\python.exe generate_smu_commercial_distilled.py

# Or render a single scene
.\.venv\Scripts\python.exe generate_smu_commercial_distilled.py --scene-id 1 --force
```

### Method B: Full Production TI2VidTwoStagesHQPipeline (Dev Model + LoRA)
For custom CFG guidance and negative prompt enforcement:
```powershell
.\.venv\Scripts\python.exe run_rtx5080.py `
  --prompt "A cinematic drone shot of Dallas Hall on the SMU campus, warm morning sunlight" `
  --num-frames 121 `
  --width 1280 `
  --height 768 `
  --steps 15 `
  --output outputs/dallas_hall.mp4
```

---

## 7. Important Technical Rules & Constraints

1. **NVFP4 Weight Scale Invariant**:
   - In safetensors, NVFP4 `weight_scale` tensors are stored as `float8_e4m3fn`.
   - They **must** be viewed in memory as raw bytes via `.view(torch.uint8)`.
   - **Never** perform an arithmetic copy (`copy_()`) between `float8` and `uint8`, as values $< 1.0$ will truncate to 0 and destroy generation quality.
2. **Zero-MMap File Loader**:
   - To prevent Windows `0xC0000005` virtual address space commit collisions between pinned buffers (20.3 GB) and checkpoints (24.5 GB), all weights are streamed directly via `f.readinto()` rather than `safetensors.safe_open()`.
3. **Subprocess Isolation**:
   - Multi-scene generation batches should run each scene in a dedicated process (`subprocess.run([sys.executable, ...])`) so host RAM is 100% reclaimed by the operating system between scenes.
4. **Resolution Divisibility**:
   - Width and height must be multiples of 64 (e.g. `1280x768`, `1920x1088`).
5. **Frame Count**:
   - Total frames must be `8 * k + 1` (e.g. 9, 17, 25, 33, 41, ..., 121).
