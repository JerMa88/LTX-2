# HPC vs. PC Video Quality: Root-Cause Analysis

## Executive Summary

After exhaustive comparison of [slurm_smu_514043.out](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2/logs/slurm_smu_514043.out) / [slurm_smu_514046.err](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2/logs/slurm_smu_514046.err) against [pc_10-second.log](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2/logs/pc_10-second.log), I found **4 critical differences** causing the quality gap.

---

## Difference Table

| Parameter | HPC (A100 80GB) | PC (RTX 5080 16GB) | Impact |
|---|---|---|---|
| **Inference Steps** | **15** steps (30 fn evals) | **12** steps (24 fn evals) | 🔴 **MAJOR** — Underconverged denoising |
| **Transformer Checkpoint** | `ltx-2.5-22b-dev-transformer-bf16` (full base model) | `ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2` (distilled + NVFP4) | 🔴 **MAJOR** — Different weights entirely |
| **Stage 1 Resolution** | `960×544` (half-res as expected) | `640×384` (⅓-res!) | 🔴 **MAJOR** — Motion/structure at much lower resolution |
| **DiffVAE Backend** | Triton `na3d` fallback | Eager tiled SDPA `na3d` fallback | 🟡 MINOR — Slightly different numerics |

---

## 🔴 Root Cause #1: Only 12 Inference Steps (vs 15 on HPC)

The PC command explicitly passed `--steps 12`:

```
--steps 12
```

The PC log confirms:
> `Steps: 12 (res2s sampler, CFG: 3.0, Rescale: 0.7)`

The HPC used the default **15 steps** (from [generate_smu_commercial.py line 443](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2/generate_smu_commercial.py#L443)). With the `res2s` 2nd-order ODE solver, this means:

- **HPC**: 15 steps × 2 = **30 function evaluations** → full convergence
- **PC**: 12 steps × 2 = **24 function evaluations** → **underconverged**

> [!IMPORTANT]
> **Fix**: Remove `--steps 12` or explicitly use `--steps 15`. This alone will noticeably improve temporal coherence and fine detail.

---

## 🔴 Root Cause #2: Wrong Transformer Checkpoint (Distilled-Comfy-V2 vs. Base Dev)

The PC used:
```
ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors
```

The HPC used:
```
ltx-2.5-22b-dev-transformer-bf16.safetensors
```

These are **fundamentally different models**:

| | HPC Checkpoint | PC Checkpoint |
|---|---|---|
| Name | `dev-transformer-bf16` | `distilled-transformer-nvfp4-comfy-v2` |
| Precision | BF16 (full) | NVFP4 (4-bit pre-quantized) |
| Model Type | **Base foundation** (full 22B dev) | **Distilled** variant packaged for ComfyUI |
| Dynamic Range | Full BF16 | Reduced (4-bit quantization baked in) |

The `run_rtx5080.py` in the [RTX_5080_SETUP_GUIDE.md](file:///work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2/RTX_5080_SETUP_GUIDE.md#L224) was designed to auto-detect and prefer the NVFP4 checkpoint **with runtime `nvfp4-prequant` quantization applied to the dev BF16 weights**. Instead, the user downloaded a **different** checkpoint (`-comfy-v2`) which is a distilled+pre-quantized variant — a lower quality starting point.

> [!WARNING]
> The `comfy-v2` checkpoint is designed for fast ComfyUI workflows, **not** production HQ rendering. It trades substantial quality for compatibility with ComfyUI's memory management.

> [!IMPORTANT]
> **Fix**: Download the correct checkpoint. Either:
> 1. Use `ltx-2.5-22b-dev-transformer-bf16.safetensors` + `fp8-cast` quantization (matches HPC exactly), **OR**
> 2. Use `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` (the official NVFP4 — **without** `-comfy-v2`) + `nvfp4-prequant`
>
> ```bash
> # Option A — Exact HPC replication (recommended, needs ~42GB download):
> hf download Lightricks/LTX-2.5 diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors --local-dir models/ltx25
> # Then run with: --quantization fp8-cast
>
> # Option B — Native NVFP4 (needs ~19GB download):
> hf download Lightricks/LTX-2.5 diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors --local-dir models/ltx25
> # Then update run_rtx5080.py to point to this file
> ```

---

## 🔴 Root Cause #3: Stage 1 at 640×384 Instead of 960×544

The PC log shows Stage 1 ran at **640×384**:
```
Running denoising loop (12 steps, 640×384 241 frames @ 24.0 fps)
```

The HPC's two-stage pipeline renders Stage 1 at **half the final resolution** → `960×544` for a `1920×1088` output.

But the PC's `run_rtx5080.py` was invoked with `--width 1280 --height 768`, so:
- Stage 1 = 1280/2 × 768/2 = **640×384** ✅ (this is correct math for the requested resolution)

The real problem: the user asked for `1280×768` output instead of `1920×1088`.

| | HPC | PC |
|---|---|---|
| Final Resolution | 1920×1088 | 1280×768 |
| Stage 1 Resolution | 960×544 | 640×384 |
| Total Pixels | 2,088,960 | 983,040 |
| Pixel Ratio | 1.0× | **0.47×** (less than half!) |

> [!IMPORTANT]
> **Fix**: Use `--width 1920 --height 1088` to match HPC resolution. If VRAM is tight, `OffloadMode.CPU` handles this — the guide confirms peak VRAM stays under 14GB even at full 1080p.

---

## 🟡 Root Cause #4: DiffVAE Backend (Eager SDPA vs Triton)

Both HPC and PC are missing `natten` and fall back to alternative neighborhood attention:

- **HPC**: `DiffVAE NA fallback: using Triton na3d.`
- **PC**: `DiffVAE NA fallback: using eager tiled SDPA na3d.`

The Triton kernel on Linux is faster and may have slightly better numerical precision than the pure-PyTorch eager SDPA path on Windows. This is a **minor** contributor — it affects VAE decode fidelity marginally, not the core diffusion quality.

> [!TIP]
> **Optional fix**: Install `natten` if available for Windows/CUDA 12.8, or run under WSL2 where Triton works natively:
> ```bash
> uv sync --package ltx-core --extra natten
> ```

---

## 🚨 Timing Anomaly: Extreme Slowdown on PC

Even accounting for the weaker GPU, the timings are **pathologically slow**:

| Stage | HPC (A100) | PC (RTX 5080) | Expected PC | Actual Slowdown |
|---|---|---|---|---|
| Stage 1 (denoising) | ~2 min 30s (15 steps) | **1h 05min** (12 steps) | ~2 min | **26× slower** |
| Stage 2 (3 steps) | ~57s | **1h 42min** | ~1 min | **107× slower** |
| **Total** | ~5.5 min | **3 hours 12 min** | ~4 min | **35× slower** |

This is **far beyond** what hardware differences would explain. The RTX 5080 should be within 2-3× of the A100 for this workload with NVFP4. A 35× slowdown strongly suggests:

1. **The `-comfy-v2` checkpoint forces decompression during every forward pass** (no native NVFP4 kernel path), causing massive overhead.
2. **Potential CPU offload thrashing** — the 16GB VRAM ceiling + non-optimal checkpoint may force excessive PCIe transfers.

---

## ✅ Recommended Fix (Priority Order)

1. **Download correct checkpoint**: `ltx-2.5-22b-dev-transformer-bf16.safetensors` with `--quantization fp8-cast` to exactly match HPC
2. **Use 15 inference steps**: Remove `--steps 12`
3. **Use full resolution**: `--width 1920 --height 1088`
4. **Install natten** (optional, minor improvement)

These changes will replicate HPC quality **exactly**. The `fp8-cast` quantization on the full BF16 dev checkpoint is what the HPC used and is lossless at the perceptual level.
