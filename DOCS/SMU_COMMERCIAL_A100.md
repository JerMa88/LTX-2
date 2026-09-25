# Production SMU Commercial Generation on NVIDIA DGX A100 (80GB VRAM)

This document provides complete, exhaustive technical documentation for running the production **LTX-2.5** 7-scene Southern Methodist University (SMU) commercial pipeline on an **NVIDIA DGX A100 (A100-SXM4-80GB)** cluster environment. It covers system topology, checkpoint management, two-stage mathematical hyperparameters, conditioning metadata, script architectures, and execution workflows.

---

## 1. Compute Infrastructure & Cluster Architecture

### Hardware Configuration
* **Cluster Environment**: Southern Methodist University (SMU) ManeFrame / DGX A100 Supercomputing Cluster.
* **Compute Nodes**: `bcm-dgxa100-[0001-0020]`.
* **Accelerator**: $1 \times$ NVIDIA A100-SXM4-80GB.
  * **Architecture**: Ampere (Compute Capability `sm_80`).
  * **VRAM**: 80 GB HBM2e (high-bandwidth stacked memory, ~2,039 GB/s memory bandwidth).
  * **Interconnect**: NVIDIA NVLink Gen3 (600 GB/s bidirectional) / PCIe Gen4.
* **Host Compute**:
  * **CPU**: Dual AMD EPYC 7742 (128 physical cores, 256 threads @ 2.25 GHz base).
  * **Host RAM**: 2.06 TB DDR4-3200 ECC Registered RAM.
* **Operating System & Driver Stack**:
  * **OS**: Linux x86_64 (Enterprise Linux kernel).
  * **NVIDIA Driver**: `570.158.01` (supports up to CUDA 12.8 runtime).
  * **CUDA Runtime**: 12.8.
  * **PyTorch Ecosystem**: `torch==2.11.0+cu128`, `torchaudio==2.11.0+cu128`, `torchvision==0.26.0+cu128`.

### Storage Topology & Quotas
* **Work Allocation Path**: `/work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2`.
  * High-throughput parallel Lustre/NFS filesystem (~300+ GB available capacity).
  * All virtual environments, checkpoints, and generated media reside strictly on `/work`.
* **Home Directory (`/users/jerryma`) Quota Isolation**:
  * The user home directory has strict disk quotas. To prevent Hugging Face and uv from overflowing `~/.cache`, cache paths must be explicitly redirected:
    ```bash
    export UV_CACHE_DIR="$(pwd)/.uv_cache"
    export HF_HOME="$(pwd)/.hf_cache"
    ```

---

## 2. Model Checkpoints & Storage Footprint

The production pipeline utilizes the complete, unquantized BF16 model suite of **LTX-2.5** (~115 GiB total storage footprint under `models/ltx-2.5/`):

| Component | Checkpoint File | Format / Dtype | Disk Size | Role in Two-Stage Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Base Diffusion Transformer** | `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | BF16 (22B params) | **~40.0 GB** | Stage 1 primary denoising engine; supports true Classifier-Free Guidance (CFG) & negative prompting. |
| **Distilled Refinement LoRA** | `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | BF16 (Low-Rank) | **~8.3 GB** | Applied in Stage 2 super-resolution refinement at weight strength `0.8`. |
| **Prompt Text Encoder** | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | BF16 (12B params) | **~24.5 GB** | Gemma 4 12B LLM with cross-attention projection; encodes positive & negative prompts into conditioning latents. |
| **Spatiotemporal Video VAE** | `vae/ltx-2.5-video-vae-bf16.safetensors` | BF16 Causal 3D VAE | **~1.4 GB** | Encodes conditioning images into latent space and decodes final denoised latents into RGB pixels via Triton `na3d`. |
| **Continuous Audio VAE** | `vae/ltx-2.5-audio-vae-bf16.safetensors` | BF16 Latent VAE | **~0.34 GB** | Synthesizes continuous latent audio channels aligned with visual dynamics. |
| **Latent Spatial Upscaler** | `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | BF16 CNN | **~0.93 GB** | 2× spatial interpolation between Stage 1 latents ($960\times544$) and Stage 2 latents ($1920\times1088$). |

---

## 3. Production Generation Hyperparameters & Mathematical Pipeline

The commercial is rendered using the official `TI2VidTwoStagesHQPipeline`. Unlike fast 1-stage distilled pipelines, this two-stage architecture separates motion synthesis from high-frequency surface refinement:

```
Conditioning Keyframe (1080p) + Prompt + Negative Prompt
                      │
                      ▼
         [Stage 0: Gemma Text Encoder]
         (Positive & Negative Latent Embeddings)
                      │
                      ▼
       [Stage 1: Low-Resolution Denoising]
       • Geometry: 960×544 @ 24 fps (121 / 129 frames)
       • Engine: 22B Base Dev Transformer (BF16)
       • Sampler: res2s 2nd-order ODE (15 steps = 30 evals)
       • Guidance: Video CFG = 3.0, Rescale = 0.7
       • Distilled LoRA Strength: 0.0 (Pure Base Model)
                      │
                      ▼
         [Stage 1.5: Latent Spatial Upscale]
         • Model: 2x Latent Spatial Upscaler (BF16)
         • Output Latent Grid: 1920×1088
                      │
                      ▼
      [Stage 2: Super-Resolution Refinement]
       • Geometry: 1920×1088 @ 24 fps
       • Engine: 22B Base Dev + Distilled LoRA (Strength 0.8)
       • Sampler: Distilled sigmas [0.91, 0.725, 0.42, 0.0] (3 steps)
       • Target: High-frequency texture, text, edges, faces
                      │
                      ▼
          [Stage 3: DiffVAE 3D Decode]
       • Neighborhood Attention: Triton na3d fallback
       • Output: Full HD 1080p 24 fps RGB Video Stream
                      │
                      ▼
       [Stage 4: Audio Multiplex & Assembly]
       • External Voiceover Muxing (smu_voiceover.wav)
       • Final Master: outputs/smu_commercial_full.mp4
```

### Detailed Hyperparameter Specification

| Parameter | Configuration | Value | Technical Rationale & Impact |
| :--- | :--- | :--- | :--- |
| **Pipeline Class** | `pipeline` | `TI2VidTwoStagesHQPipeline` | Two-stage hierarchical synthesis prevents hallucinations and guarantees sharp details. |
| **Output Resolution** | `--width`, `--height` | `1920 × 1088` | Full HD widescreen. Must be strictly divisible by **64** ($1920/64=30$, $1088/64=17$). |
| **Stage 1 Resolution** | Derived | `960 × 544` | Exact $0.5\times$ downscale. Divisible by **32**, allowing efficient spatial-temporal patchification. |
| **Frame Rate** | `--frame-rate` | `24.0` fps | Cinematic broadcast standard. |
| **Temporal Frame Grid** | `num_frames` | `121` & `129` | Follows the required temporal causality equation: $(N - 1) \pmod 8 == 0$. |
| **Sampler** | `sampler` | `res2s` | Runge-Kutta / Adams-Bashforth 2nd-order ODE solver. Computes 2 function evaluations per step. |
| **Inference Steps** | `--num-inference-steps` | `15` | Total of $15 \times 2 = \mathbf{30}$ function evaluations in Stage 1 for full mathematical convergence. |
| **Video CFG Guidance** | `--video-cfg-guidance-scale` | `3.0` | Forces adherence to prompt semantics and locks camera trajectory. |
| **Video Rescale Scale** | `--video-rescale-scale` | `0.7` | Dynamic range rescale factor; prevents color burnout, contrast clipping, and oversaturation. |
| **Video STG Guidance** | `--video-stg-guidance-scale` | `0.0` | Spatio-temporal guidance scale disabled to prevent temporal stutter. |
| **Audio CFG Guidance** | `--audio-cfg-guidance-scale` | `7.0` | High guidance scale for continuous latent audio generation. |
| **Audio-to-Video (A2V)** | `--a2v-guidance-scale` | `3.0` | Cross-attention alignment between audio envelope and visual motion. |
| **Distilled LoRA Stage 1** | `--distilled-lora-strength-stage-1` | `0.0` | Stage 1 uses pure foundation weights to preserve unconstrained semantic diversity. |
| **Distilled LoRA Stage 2** | `--distilled-lora-strength-stage-2` | `0.8` | Distilled LoRA applied at 80% strength to inject distilled sharpening without artifacting. |
| **Offload Mode** | `offload_mode` | `OffloadMode.CPU` | Keeps model weights in 2TB host RAM, dynamically streaming active blocks to A100 VRAM. |
| **Quantization** | `--quantization` | `fp8-cast` | Performs runtime casting of linear layers to FP8, bounding peak VRAM to ~31 GB. |

### Anti-Distortion Negative Prompt
To eliminate human deformities, blurry textures, and camera jitter, the pipeline enforces `DEFAULT_NEGATIVE_PROMPT` on every scene:
```
worst quality, inconsistent motion, blurry, jittery, distorted, low quality, 
compression artifacts, noise, banding, jitter, warped faces, unnatural limbs, 
flickering, text artifacts, watermark, low resolution
```

---

## 4. SMU Commercial Scene Metadata & Audio Synchronization

The commercial consists of 7 conditioned scenes generated from high-resolution still photographs located in `inputs/smu/`:

| Scene ID | Scene Name | Keyframe Image (`inputs/smu/`) | Frames | Time | Seed | Voiceover Transcript Conditioned in Prompt |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | `01_dallas_hall` | `01_dallas_hall.jpg` | 121 | 5.042s | `42` | *"At Southern Methodist University, heritage meets the horizon."* |
| **2** | `02_mustang_statue` | `02_mustang_statue.jpg` | 121 | 5.042s | `101` | *"In the heart of Dallas, bold ideas take flight."* |
| **3** | `03_engineering_lab` | `03_engineering_lab.jpg` | 121 | 5.042s | `202` | *"Here, innovators sculpt tomorrow's breakthroughs in artificial intelligence and robotics."* |
| **4** | `04_business_cox` | `04_business_cox.jpg` | 121 | 5.042s | `303` | *"Visionaries lead global commerce with unyielding integrity."* |
| **5** | `05_meadows_arts` | `05_meadows_arts.jpg` | 121 | 5.042s | `404` | *"Artists inspire, and culture thrives."* |
| **6** | `06_campus_sunset` | `06_campus_sunset.jpg` | 121 | 5.042s | `505` | *"Fueled by unbridled Mustang spirit, we don't just dream of a better world. We build it."* |
| **7** | `07_closing_logo` | `07_closing_logo.jpg` | 129 | 5.375s | `606` | *"Southern Methodist University. World changers shaped here. Pony up!"* |

### Temporal Audio Synchronization Math
* **Frame Rate**: Exactly $24.0$ frames per second.
* **Scenes 1 through 6**: $6 \times 121\text{ frames} = 726\text{ frames} = \mathbf{30.250\text{ seconds}}$.
* **Scene 7 (Closing Logo Title Card)**: $129\text{ frames} = \mathbf{5.375\text{ seconds}}$.
* **Total Video Timeline**: $726 + 129 = \mathbf{855\text{ frames}} = \mathbf{35.625\text{ seconds}}$.
* **Voiceover File (`inputs/smu/smu_voiceover.wav`)**:
  * Audio Duration: **35.589 seconds** (22,050 Hz, 1-channel mono PCM).
  * Delta: $|35.625 - 35.589| = \mathbf{0.036\text{ seconds}}$ ($<1$ video frame). The narrator finishes cleanly over the final Mustang logo animation.

---

## 5. Script Architecture & Implementation Details

The A100 execution pipeline is governed by two core files:

### A. The SLURM Batch Script: `generate_smu_commercial.slurm`
Located at [`generate_smu_commercial.slurm`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/generate_smu_commercial.slurm):
* **Resource Allocation**:
  * `#SBATCH --partition=short`: Queues into the high-priority short partition (up to 4 hours walltime).
  * `#SBATCH --gres=gpu:1`: Requests 1 dedicated physical NVIDIA A100-SXM4-80GB GPU.
  * `#SBATCH --cpus-per-task=16`: Allocates 16 AMD EPYC CPU cores for data loading and PyAV video encoding.
  * `#SBATCH --mem=64G`: Requests 64 GB of host RAM from the node's 2 TB pool.
* **Environment Controls**:
  * `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Prevents VRAM fragmentation during large 1080p latent allocations.
  * `export UV_CACHE_DIR="$(pwd)/.uv_cache"`: Prevents quota exhaustion in `/users/jerryma`.
  * `export PYTHONUNBUFFERED=1`: Ensures real-time streaming of stdout/stderr to SLURM logs.

### B. The Production Pipeline Driver: `generate_smu_commercial.py`
Located at [`generate_smu_commercial.py`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/generate_smu_commercial.py):

#### 1. Validation & Integrity Checker (`is_video_valid`)
Verifies if an existing output video meets exact production specifications:
* Checks file existence and minimum size threshold ($>10\text{ KB}$).
* Inspects container headers via PyAV to verify `video_stream.frames == sc["num_frames"]`, `width == 1920`, and `height == 1088`. If an incomplete or old resolution render is detected, it flags the scene for automatic regeneration.

#### 2. Subprocess Isolation Orchestrator (Mode 1)
When invoked without `--scene-id`:
* Scans all 7 scenes and compiles `scenes_pending`.
* For each pending scene, dispatches an isolated subprocess:
  ```python
  cmd = [
      sys.executable, str(Path(__file__).resolve()),
      "--scene-id", str(sc["id"]),
      "--width", str(args.width),
      "--height", str(args.height),
      "--num-inference-steps", str(args.num_inference_steps),
      ...
  ]
  subprocess.run(cmd)
  ```
* Ensures that memory accumulated during forward passes is 100% purged by the operating system between scene renders.

#### 3. Single-Scene Generator (Mode 2)
When executed with `--scene-id <N>`:
* Constructs `ModelPaths` pointing to the dev transformer, Gemma text encoder, video VAE, and audio VAE.
* Initializes `TI2VidTwoStagesHQPipeline` with `OffloadMode.CPU` and `QuantizationKind("fp8-cast")`.
* Enforces `@torch.inference_mode()` on `render_scene()` to prevent PyTorch from retaining autograd computation graphs during lazy generator decoding.
* Saves rendered video directly to `outputs/smu_scenes/scene_<ID>_<NAME>.mp4`.

#### 4. Post-Processing & Master Video Assembly (Mode 3)
* `extract_preview_frames()`: Extracts high-resolution JPEG frames from frame index $N/2$ of each scene to `outputs/smu_previews/`.
* `assemble_commercial()`: Generates an FFmpeg demuxer manifest (`concat_list.txt`) and executes:
  ```bash
  ffmpeg -y -f concat -safe 0 -i concat_list.txt -i inputs/smu/smu_voiceover.wav \
    -c:v libx264 -pix_fmt yuv420p -preset medium -crf 18 \
    -c:a aac -b:a 192k -shortest outputs/smu_commercial_full.mp4
  ```

---

## 6. Step-by-Step Execution Guide on A100

### Step 1: Connect to Cluster & Set Up Environment
```bash
ssh <username>@superpod.smu.edu
cd /work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2

# Set quota-safe cache redirects
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export PATH="/users/jerryma/.local/bin:$PATH"

# Verify virtual environment
source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Step 2: Launch the Full 7-Scene Commercial Job
Submit the SLURM batch job to render all 7 scenes and assemble the final master video:
```bash
sbatch generate_smu_commercial.slurm
```

Monitor job status and log outputs:
```bash
# Check queue position
squeue -u $USER

# Follow live output logs
tail -f logs/slurm_smu_<JOBID>.out
```

### Step 3: Target a Specific Scene (Force Regeneration)
If an individual scene needs regeneration (e.g. Scene 3):
```bash
sbatch generate_smu_commercial.slurm --scene-id 3 --force
```

### Step 4: Assemble Without Re-rendering
To re-run video concatenation, voiceover multiplexing, and preview extraction without re-generating existing videos:
```bash
./.venv/bin/python generate_smu_commercial.py --skip-generation
```

### Expected Deliverables on A100:
* Final Master Video: [`outputs/smu_commercial_full.mp4`](file:///c:/Users/jerry/Documents/program/GitHub/LTX-2-1/outputs/smu_commercial_full.mp4) (1920×1088 @ 24 fps, H.264 + AAC, ~47 MB).
* Scene Clips: `outputs/smu_scenes/scene_01_*.mp4` through `scene_07_*.mp4`.
* Scene Previews: `outputs/smu_previews/preview_scene_*.jpg`.
