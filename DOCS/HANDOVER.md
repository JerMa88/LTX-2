# LTX-2.5 HPC Cluster Deployment & Production SMU Commercial Pipeline Handover

This document provides a comprehensive technical handover for the LTX-2.5 video generation stack on the SMU HPC DGX A100 cluster, detailing our implementation, the upgraded 7-scene Southern Methodist University commercial pipeline, recommended production hyperparameters, bug fixes, system architecture constraints, and step-by-step instructions for replication.

---

## 1. System Architecture & Cluster Setup

### Compute Environment
- **Cluster**: Southern Methodist University (SMU) ManeFrame / DGX A100 cluster.
- **Compute Nodes**: `bcm-dgxa100-[0001-0020]`.
- **GPU Accelerator**: NVIDIA A100-SXM4-80GB (80 GB HBM2e VRAM, PCIe Gen4 / NVLink).
- **NVIDIA Driver**: `570.158.01` (supports up to **CUDA 12.8**).
- **Host Memory**: 2.06 TB host RAM per node, 128 AMD EPYC CPU cores.
- **SLURM Partition**: `short` (walltime up to 4:00:00) or `batch` (walltime up to 24:00:00).
- **SLURM Allocation Account**: `mhahsler_course_recomm_0001`.

### Filesystem & Quotas
- **Project Directory (`/work`)**: Located at `/work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2`. High storage capacity (~300+ GB free), shared across cluster compute nodes via parallel NFS/Lustre.
- **Home Directory (`/users/jerryma`)**: Has strict quota limits.
  - **CRITICAL**: Never allow caches (e.g. Hugging Face, uv, pip) to default to `~/.cache`. Always set `export UV_CACHE_DIR="$(pwd)/.uv_cache"`.

---

## 2. Models & Weights Configuration

All LTX-2.5 model weights are centralized under `models/ltx-2.5/` (~115 GiB total):

| Component | Path | Size | Description |
|-----------|------|------|-------------|
| **Base Foundation Transformer** | `models/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | ~40.0 GB | Full 22B parameter dev transformer (CFG & negative prompt capable) |
| **Distilled Refinement LoRA** | `models/ltx-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | ~8.3 GB | Distilled LoRA for Stage 2 high-resolution refinement |
| **Distilled Transformer (Draft)** | `models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | ~39.1 GB | Fast 8-step unguided draft model (CFG=1.0) |
| **Text Encoder** | `models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | ~24.5 GB | Gemma 4 12B encoder with projection layer |
| **Video VAE** | `models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors` | ~1.4 GB | Spatiotemporal causal video VAE |
| **Audio VAE** | `models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors` | ~0.34 GB | Continuous latent audio VAE |
| **Spatial Upscaler** | `models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | ~0.93 GB | 2x latent spatial upscaler for Stage 2 refinement |

*Download script*: [download_models.sh](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2/download_models.sh) (uses `hf download` via `huggingface_hub`).

---

## 3. Production Hyperparameters & Architectural Upgrades

To eliminate blurry, deformed faces and low-frequency artifacts from initial 8-step distilled drafts, we upgraded our pipeline to the community and official production standard:

| Hyperparameter | Value | Description & Technical Rationale |
|---|---|---|
| **Pipeline** | `TI2VidTwoStagesHQPipeline` | Official two-stage high-quality image-to-video pipeline. Stage 1 computes motion/structure at half resolution with full CFG; Stage 2 refines high frequencies at full resolution. |
| **Base Model** | `dev-transformer-bf16` | 22B foundation model capable of true Classifier-Free Guidance (CFG). |
| **Refinement LoRA** | `distilled-lora-450-bf16` | Applied in Stage 2 (strength: 0.8) with distilled sigmas `[0.91, 0.725, 0.42, 0.0]`. |
| **Resolution** | `1088×1920` (1080p Full HD) | Divisible by 64 (17×64, 30×64). Stage 1 runs at `544×960` (divisible by 32). Minimal crop from 16:9 inputs. |
| **Sampler** | `res2s` second-order | Runge-Kutta / Adams-Bashforth second-order sampler with 15 steps (30 evaluations). |
| **Video CFG** | `3.0` | Eliminates hallucinations, sharpens faces, guarantees tight prompt adherence. |
| **Video Rescale** | `0.7` | Rescales high-CFG dynamic range to prevent color clipping and burn-in. |
| **Negative Prompt** | `DEFAULT_NEGATIVE_PROMPT` | Rigorously penalizes blur, bad facial geometry, jitter, artifacts, and low resolution. |
| **Audio CFG** | `7.0` | Guided continuous latent audio synthesis. |
| **Offload Mode** | `OffloadMode.CPU` | Retains all weights in CPU RAM, offloading modules to GPU during forward passes (~44 GB host RAM, ~31 GB peak VRAM). |

---

## 4. Bugs Encountered & Technical Solutions

### Bug 1: CUDA 13.2 Driver Incompatibility
- **Symptom**: Upstream repository configured `cu132` wheels (`torch==2.13.0+cu132`). Failed with `CUDA driver version is insufficient for CUDA runtime version`.
- **Root Cause**: The DGX node driver `570.158.01` supports CUDA up to 12.8, not 13.2.
- **Resolution**: Aligned all dependencies to `torch==2.11.0+cu128`, `torchaudio==2.11.0+cu128`, `torchvision==0.26.0+cu128`.

### Bug 2: PyTorch / TorchAudio Minor Version Mismatch
- **Symptom**: Undefined symbol errors in `_torchaudio_so`.
- **Resolution**: Explicitly locked both `torch` and `torchaudio` to version `2.11.0`.

### Bug 3: PyTorch Inference Mode vs Lazy Generator Decode
- **Symptom**: `RuntimeError: Inference tensors cannot be saved for backward. Please do not use Tensors created in inference mode in computation tracked by autograd.`
- **Root Cause**: `pipeline(...)` returns a lazy generator for `result.video`. Iterating inside `encode_video(...)` outside inference mode triggered autograd checks.
- **Resolution**: Decorated `render_scene(...)` with `@torch.inference_mode()`.

### Bug 4: Repeated 66 GB Model Reload Overhead
- **Symptom**: Running sequential CLI commands reloads the model from the network filesystem for every scene, adding 10–15 minutes per scene.
- **Resolution**: [generate_smu_commercial.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2/generate_smu_commercial.py) loads the model once and processes all scenes in memory, dropping render time to ~330s per scene.

### Bug 5: Aspect Ratio & Resolution Drift
- **Symptom**: Incomplete runs produced mixed resolutions (1536×1024 vs 1920×1088).
- **Resolution**: Updated `is_video_valid()` to verify width, height, and frame count, ensuring outdated clips are automatically detected and regenerated to Full HD.

---

## 5. SMU Commercial Pipeline

### Inputs (`inputs/smu/`)
- `01_dallas_hall.jpg`
- `02_mustang_statue.jpg`
- `03_engineering_lab.jpg`
- `04_business_cox.jpg`
- `05_meadows_arts.jpg`
- `06_campus_sunset.jpg`
- `07_closing_logo.jpg`
- `smu_voiceover.wav` (35.589s duration, 22050 Hz, 1 channel PCM)

### Mathematical Synchronization
- Video FPS: `24.0`
- Scenes 1–6: `121` frames each (5.042s each) = 726 frames
- Scene 7 (Closing Logo): `129` frames = 5.375s
- Total Video Frames: `855` frames (at `8k + 1` grid) = **35.625 seconds**
- Audio Match: 35.589s audio completes seamlessly over the final title card (0.036s delta).

### Outputs (`outputs/`)
- `outputs/smu_commercial_full.mp4`: Final master commercial (1920×1088, 24 fps, H.264 + AAC 22050Hz, 35.58s duration, 47 MB).
- `outputs/smu_scenes/scene_*.mp4`: Individual scene clips in Full HD 1080p.
- `outputs/smu_previews/preview_scene_*.jpg`: Extracted high-resolution preview snapshots.

---

## 6. How to Replicate & Execute

### 1. Activating the Environment
```bash
cd /work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export PATH="/users/jerryma/.local/bin:$PATH"
```

### 2. Running the Full Commercial
```bash
sbatch generate_smu_commercial.slurm
```
Monitor logs:
```bash
squeue -u $USER
tail -f logs/slurm_smu_<JOBID>.out
```

### 3. Re-running a Specific Scene
To force regeneration of a specific scene (e.g. Scene 3):
```bash
sbatch generate_smu_commercial.slurm --scene-id 3 --force
```

### 4. Re-assembling the Commercial Video
To re-run video concatenation and audio multiplexing without re-rendering:
```bash
./.venv/bin/python generate_smu_commercial.py --skip-generation
```

---

## 7. System Constraints & Rules of Thumb
1. **Resolution Divisibility**: Both height and width must be strictly divisible by **64** for the 2-stage pipeline (e.g. `1088×1920`).
2. **Temporal Frame Count**: Must follow `(num_frames - 1) % 8 == 0` (e.g. 121, 129, 161).
3. **Memory Allocation**: Request at least `--mem=160G` in SLURM to hold the dev transformer (40GB), text encoder (25GB), and LoRA (8.3GB) in CPU RAM for offloading.
4. **Clean VRAM**: Always invoke `gc.collect()` and `torch.cuda.empty_cache()` between consecutive generations.
