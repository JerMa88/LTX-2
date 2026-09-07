# LTX-2.5 HPC Cluster Deployment & SMU Commercial Pipeline Handover

This document provides a comprehensive technical handover for the LTX-2.5 video generation stack on the SMU HPC DGX A100 cluster, detailing our implementation, the 7-scene Southern Methodist University commercial pipeline, bug fixes, system architecture constraints, and step-by-step instructions for replication.

---

## 1. System Architecture & Cluster Setup

### Compute Environment
- **Cluster**: Southern Methodist University (SMU) ManeFrame / DGX A100 cluster.
- **Compute Nodes**: `bcm-dgxa100-[0001-0020]`.
- **GPU Accelerator**: NVIDIA A100-SXM4-80GB (80 GB HBM2e VRAM, PCIe Gen4 / NVLink).
- **NVIDIA Driver**: `570.158.01` (supports up to **CUDA 12.8**).
- **SLURM Partition**: `short` (walltime up to 4:00:00) or `batch` (walltime up to 24:00:00).
- **SLURM Allocation Account**: `mhahsler_course_recomm_0001`.

### Filesystem & Quotas
- **Project Directory (`/work`)**: Located at `/work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2`. High storage capacity (~340+ GB free), shared across cluster compute nodes via parallel NFS/Lustre.
- **Home Directory (`/users/jerryma`)**: Has strict quota limits.
  - **CRITICAL**: Never allow caches (e.g. Hugging Face, uv, pip) to default to `~/.cache`. Always set `export UV_CACHE_DIR="$(pwd)/.uv_cache"`.

---

## 2. Models & Weights Configuration

All LTX-2.5 model weights are centralized under `models/ltx-2.5/` (~66 GiB total):

| Component | Path | Size | Description |
|-----------|------|------|-------------|
| **Diffusion Transformer** | `models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | ~39.1 GB | 22B parameter distilled video/audio transformer |
| **Text Encoder** | `models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | ~24.5 GB | Gemma 4 12B encoder with projection layer |
| **Video VAE** | `models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors` | ~1.4 GB | Spatiotemporal causal video VAE |
| **Audio VAE** | `models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors` | ~0.34 GB | Continuous latent audio VAE |
| **Spatial Upscaler** | `models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | ~0.93 GB | 2x latent spatial upscaler for Stage 2 refinement |

*Download script*: [download_models.sh](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2/download_models.sh) (uses `hf download` via `huggingface_hub`).

---

## 3. Bugs Encountered & Technical Solutions

### Bug 1: CUDA 13.2 Driver Incompatibility
- **Symptom**: Upstream repository configured `cu132` wheels (`torch==2.13.0+cu132`). When executed on the cluster nodes, PyTorch failed with:
  `CUDA driver version is insufficient for CUDA runtime version`.
- **Root Cause**: The DGX node driver `570.158.01` supports CUDA up to 12.8, not 13.2.
- **Resolution**:
  - Aligned all `pyproject.toml` configurations to PyTorch CUDA 12.8:
    - `torch==2.11.0+cu128`
    - `torchaudio==2.11.0+cu128`
    - `torchvision==0.26.0+cu128`
  - Switched the index URLs from `https://download.pytorch.org/whl/cu132` to `https://download.pytorch.org/whl/cu128`.

### Bug 2: PyTorch / TorchAudio Minor Version Mismatch
- **Symptom**: Installing mismatched minor versions resulted in binary C++ symbol loading failures (`_torchaudio_so` undefined symbols).
- **Resolution**: Explicitly locked both `torch` and `torchaudio` to version `2.11.0`.

### Bug 3: PyTorch Inference Mode vs Lazy Generator Decode
- **Symptom**: During video export, PyTorch crashed with:
  `RuntimeError: Inference tensors cannot be saved for backward. Please do not use Tensors created in inference mode in computation tracked by autograd.`
- **Root Cause**: `pipeline(...)` returns a lazy generator for `result.video`. When `with torch.inference_mode():` exits, subsequent iteration inside `encode_video(...)` runs with autograd enabled. Since the tensors inside the generator were created under inference mode, autograd rejected them.
- **Resolution**: Wrapped the entire `render_scene(...)` function (covering both `pipeline(...)` and `encode_video(...)`) with the `@torch.inference_mode()` decorator.

### Bug 4: Repeated 66 GB Model Reload Overhead
- **Symptom**: Running sequential CLI commands (`python -m ltx_pipelines.distilled`) reloads the 66 GB model weights from the network filesystem for every scene, taking 10–15 minutes per reload.
- **Resolution**: Created [generate_smu_commercial.py](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2/generate_smu_commercial.py), which initializes `DistilledPipeline` **once** in host memory and iterates across all 7 scenes. Generation time per scene dropped to ~125–135 seconds.

### Bug 5: SLURM Stdout Buffering
- **Symptom**: Python output was buffered by block buffering in non-interactive SLURM runs, delaying logs.
- **Resolution**: Set `export PYTHONUNBUFFERED=1` in SLURM scripts and passed `flush=True` in all Python print statements.

---

## 4. SMU Commercial Pipeline

### Inputs
Located in `inputs/smu/`:
- `01_dallas_hall.jpg` (1376×768)
- `02_mustang_statue.jpg` (1376×768)
- `03_engineering_lab.jpg` (1376×768)
- `04_business_cox.jpg` (1376×768)
- `05_meadows_arts.jpg` (1376×768)
- `06_campus_sunset.jpg` (1376×768)
- `07_closing_logo.jpg` (1376×768)
- `smu_voiceover.wav` (35.589s duration, 22050 Hz, 1 channel PCM)

### Mathematical Synchronization
- Video FPS: `24.0`
- Scenes 1–6: `121` frames each (5.042s each) = 726 frames
- Scene 7 (Closing Logo): `129` frames = 5.375s
- Total Frames: `855` frames (satisfying `8k + 1` grid) = **35.625 seconds**
- Audio Match: The 35.589-second voiceover finishes with less than 0.036s discrepancy, allowing the audio to complete cleanly over the final title card.

### Outputs
Located in `outputs/`:
- `outputs/smu_commercial_full.mp4`: Final 35.58-second commercial (1536×1024, 24 fps, H.264 + AAC 22050Hz).
- `outputs/smu_scenes/scene_*.mp4`: Individual generated clips (121/129 frames each).
- `outputs/smu_previews/preview_scene_*.jpg`: Extracted middle-frame snapshots of each scene.

---

## 5. How to Replicate & Execute

### 1. Activating the Environment
```bash
cd /work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export PATH="/users/jerryma/.local/bin:$PATH"
```

### 2. Running the Full Commercial
To run the end-to-end generation and assembly on the A100 GPU:
```bash
sbatch generate_smu_commercial.slurm
```
Monitor the job with:
```bash
squeue -u $USER
tail -f logs/slurm_smu_<JOBID>.out
```

### 3. Re-running a Specific Scene
The script is idempotent and checks existing scenes. To force regeneration of a single scene (e.g. Scene 3):
```bash
./.venv/bin/python generate_smu_commercial.py --scene-id 3 --force
```

### 4. Re-assembling the Video
To re-run video concatenation and audio multiplexing without re-rendering:
```bash
./.venv/bin/python generate_smu_commercial.py --skip-generation
```

---

## 6. System Constraints & Rules of Thumb
1. **Resolution Divisibility**: For the two-stage distilled pipeline, both height and width must be strictly divisible by **64** (e.g. 1024×1536, 896×1536, 768×1344).
2. **Temporal Frame Count**: Must follow `(num_frames - 1) % 8 == 0` (e.g. 49, 97, 121, 129, 161).
3. **Memory Offload**: Always pass `offload_mode=OffloadMode.CPU` to `DistilledPipeline`. This keeps peak VRAM usage around ~31 GB (well within the 80 GB limit) during DiffVAE decode.
4. **Always Clean Memory**: Call `gc.collect()` and `torch.cuda.empty_cache()` between consecutive generations.
