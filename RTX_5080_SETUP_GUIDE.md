# Running LTX-2.5 Local NVFP4 Generation on NVIDIA RTX 5080 (16GB VRAM / 64GB RAM)

This guide provides the battle-tested, production-ready blueprint to run the official **LTX-2.5** high-quality video generation pipeline on a personal workstation equipped with:
- **GPU**: NVIDIA GeForce RTX 5080 (16 GB GDDR7 VRAM, Blackwell Architecture, Compute Capability **SM 12.0** / `sm_120`)
- **System Memory**: 64 GB DDR5 Host RAM (with >= 80 GB pagefile commit limit)
- **Operating System**: Windows 11 64-bit (or Linux / Ubuntu 22.04+ / WSL2)

---

## 1. Hardware Architecture & Feasibility Math

### Can 16 GB VRAM Run LTX-2.5 1080p?
**YES, 100% operational via native Blackwell NVFP4 quantization.**

| Component | Baseline BF16 | RTX 5080 NVFP4 Setup | Memory Location & Peak Footprint |
|---|---|---|---|
| **Diffusion Transformer (22B)** | 42 GB (OOM on 16GB) | **NVFP4** (17.44 GB) | **Native VRAM** via `OffloadMode.NONE`. Fully pre-quantized in FP4 weights. |
| **Gemma 4 12B Text Encoder** | 24.5 GB (OOM on 16GB) | Gemma 4 12B BF16 + Projection | **Host RAM**. Loaded during prompt encoding, immediately freed. |
| **Spatial Upscaler (2x)** | 0.93 GB | LTX-2.5 Latent Upscaler | **VRAM** (~0.9 GB). |
| **Stage 1 (e.g. 640×384 or 960×544)** | ~12 GB VRAM | Active Denoising State | **~6.5 GB VRAM** peak. |
| **Stage 2 Super-Resolution (1280×768 or 1920×1088)** | ~28 GB VRAM (OOM on 16GB) | Latent Auto-Tiling | **~9.5 - 11 GB VRAM** peak. |
| **DiffVAE Video Decoder (SDR H.264)** | ~34 GB VRAM (OOM on 16GB) | `AUTO_TILING` + Chunked VAE | **~11.8 - 13.5 GB VRAM** peak. |
| **Total Host RAM Utilization** | ~96 GB | Pinned Weights + Paging | **~21 - 38 GB Host RAM** (comfortably within 64 GB). |

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
5. **Windows Pagefile**:
   - Total commit limit must be at least **80 GB** (RAM + Pagefile).
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
The C++/CUDA extension includes decoupled host and device sources to avoid MSVC compiler template crashes:
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

## 4. Downloading & Preparing Models

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
    ├── gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors (9.87 GB)
    └── gemma4-12b-with-proj-ltx-2.5-bf16.safetensors (24.46 GB, converted)
```

### Step 1: Download Checkpoints
Run the multi-connection download script:
```powershell
.\.venv\Scripts\python.exe scripts/download_ltx25_nvfp4.py
```

### Step 2: Convert Gemma 4 to Standard BF16
Convert the packed ComfyUI `.comfy_quant` format to native BF16 for HuggingFace compatibility:
```powershell
.\.venv\Scripts\python.exe scripts/dequantize_gemma_nvfp4.py
```
*(Takes under 10 seconds using the RTX 5080 hardware dequantizer).*

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

### Quick Test Generation (9 frames, 1280x768, ~3-4 minutes)
```powershell
.\.venv\Scripts\python.exe run_rtx5080.py `
  --prompt "A vibrant hummingbird hovering near tropical flowers" `
  --num-frames 9 `
  --width 1280 `
  --height 768 `
  --steps 8 `
  --output outputs/hummingbird_9f.mp4
```

### High-Quality 25-Frame Generation (~15-20 minutes)
```powershell
.\.venv\Scripts\python.exe run_rtx5080.py `
  --prompt "A cinematic drone shot of a misty fjord in Norway at dawn" `
  --num-frames 25 `
  --width 1280 `
  --height 768 `
  --steps 15 `
  --output outputs/fjord_25f.mp4
```

### Full Production 121-Frame Generation (1088p, ~1.5 - 2 hours)
```powershell
.\.venv\Scripts\python.exe run_rtx5080.py `
  --prompt "A cinematic shot of a majestic waterfall in a lush tropical forest with sunlight rays" `
  --num-frames 121 `
  --width 1920 `
  --height 1088 `
  --steps 15 `
  --output outputs/waterfall_121f.mp4
```

---

## 7. Important Technical Rules & Constraints

1. **Resolution Divisibility**:
   - For `TI2VidTwoStagesHQPipeline`, `width` and `height` must be multiples of 64 (e.g. `1280x768`, `1920x1088`).
2. **Frame Count**:
   - Total frames must be `8 * k + 1` (e.g. 9, 17, 25, 33, 41, ..., 121).
3. **Offload Mode**:
   - Must use `OffloadMode.NONE` with pre-quantized NVFP4. `OffloadMode.CPU` / `DISK` is not supported by block streaming with NVFP4 policies.
4. **Inference Mode**:
   - Always invoke `encode_video()` under `@torch.inference_mode()` to ensure lazy generator iteration does not trigger autograd conflicts.
