"""
LTX-2.5 NVFP4-Cast Production Generation Pipeline for NVIDIA RTX 5080 (16GB VRAM / 64GB RAM).

Uses the full non-distilled 22B `dev-transformer-bf16` checkpoint with online BF16->NVFP4
quantization (nvfp4-cast) applied at load time.  This yields:
  - Full-quality foundation model weights (no distillation quality loss)
  - Blackwell FP4 tensor core acceleration for inference
  - ~11 GB resident VRAM for the transformer (no block streaming required)
  - OffloadMode.NONE: all weights stay in VRAM after quantization load

Checkpoint:  diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors
Quantization: nvfp4-cast (online BF16->NVFP4, ActScale.FIXED_1)
LoRA:         loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors
              Stage 1 strength = 0.0 (pure dev model denoising)
              Stage 2 strength = 0.8 (HPC-verified distilled refinement in upscaling)
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.types import OffloadMode

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RTX5080_NVFP4")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "ltx25"

# HPC-verified anti-distortion negative prompt (from generate_smu_commercial.py)
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "watermark, low resolution, artifacts, oversaturated, flicker"
)


@dataclass
class MemoryWatermark:
    stage_name: str
    vram_allocated_gb: float
    vram_reserved_gb: float
    vram_max_gb: float
    host_ram_used_gb: float
    host_ram_pct: float

    @classmethod
    def record(cls, stage_name: str) -> MemoryWatermark:
        vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
        vram_max = torch.cuda.max_memory_allocated() / (1024 ** 3)
        vm = psutil.virtual_memory()
        host_used = vm.used / (1024 ** 3)
        host_pct = vm.percent
        wm = cls(stage_name, vram_alloc, vram_res, vram_max, host_used, host_pct)
        logger.info(
            f"[Watermark] {stage_name:<25} | "
            f"VRAM Alloc: {vram_alloc:5.2f} GB | Res: {vram_res:5.2f} GB | Peak: {vram_max:5.2f} GB | "
            f"Host RAM: {host_used:5.2f} GB ({host_pct:.1f}%)"
        )
        return wm


def clean_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LTX-2.5 RTX 5080 Dev+NVFP4-Cast Production Pipeline")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    parser.add_argument("--input-image", type=str, default=None, help="Optional input image path for image-to-video")
    parser.add_argument("--output", type=str, default="outputs/generated_video.mp4", help="Output MP4 file path")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR), help="Root models directory")

    # Production defaults matching HPC (A100) configuration
    parser.add_argument("--width", type=int, default=1920, help="Output width (default 1920, matches HPC)")
    parser.add_argument("--height", type=int, default=1088, help="Output height (default 1088, matches HPC)")
    parser.add_argument("--num-frames", type=int, default=121, help="Total frames (default 121, ~5s at 24fps)")
    parser.add_argument("--frame-rate", type=float, default=24.0, help="Video frame rate (default 24.0)")
    parser.add_argument("--steps", type=int, default=15, help="Diffusion steps (default 15, matches HPC)")
    parser.add_argument("--video-cfg", type=float, default=3.0, help="Video CFG scale (default 3.0)")
    parser.add_argument("--rescale", type=float, default=0.7, help="Rescale factor (default 0.7)")
    parser.add_argument("--audio-cfg", type=float, default=7.0, help="Audio CFG scale (default 7.0)")
    parser.add_argument("--a2v-guidance", type=float, default=3.0, help="Audio-to-video guidance (default 3.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random generation seed")

    # Distilled LoRA strengths per stage (HPC-verified defaults)
    # Stage 1: pure dev model denoising (0.0 = no distillation LoRA)
    # Stage 2: blended refinement during super-resolution upscaling (0.8 = HPC default)
    parser.add_argument("--distilled-lora-strength-stage-1", type=float, default=0.0,
                        help="Distilled LoRA strength in Stage 1 denoising (default 0.0 = pure dev model)")
    parser.add_argument("--distilled-lora-strength-stage-2", type=float, default=0.8,
                        help="Distilled LoRA strength in Stage 2 super-res refinement (default 0.8, HPC-matched)")

    # Execution modes
    parser.add_argument("--transformer-path", type=str, default=None,
                        help="Transformer checkpoint path (default: auto-detects dev-transformer-nvfp4 or dev-transformer-bf16)")
    parser.add_argument("--offload-mode", type=str, default="cpu", choices=["none", "cpu", "disk"],
                        help="Weight offload mode: 'cpu' (pinned RAM streaming, recommended for 16GB VRAM), 'none', or 'disk'")
    parser.add_argument("--quantization", type=str, default="auto", choices=["auto", "nvfp4-prequant", "nvfp4-cast", "fp8-cast", "none"],
                        help="Quantization policy (default: auto-detects prequant vs cast based on checkpoint)")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline load and memory without rendering")
    parser.add_argument("--safe-mode", action="store_true", help="Aggressive memory cleanup after every operation")

    return parser.parse_args()


@torch.inference_mode()
def run_pipeline(args: argparse.Namespace) -> Path:
    models_root = Path(args.models_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve transformer path
    if args.transformer_path:
        transformer_path = Path(args.transformer_path)
    else:
        nvfp4_dev = models_root / "diffusion_models" / "ltx-2.5-22b-dev-transformer-nvfp4.safetensors"
        bf16_dev = models_root / "diffusion_models" / "ltx-2.5-22b-dev-transformer-bf16.safetensors"
        transformer_path = nvfp4_dev if nvfp4_dev.exists() else bf16_dev

    # Distilled LoRA adapter (BF16, ~8.3GB) — used at strength=0.0/0.8 per stage
    distilled_lora_path = models_root / "loras" / "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
    # Text encoder: use dequantized BF16 if available, else fall back to nvfp4 original
    text_encoder_path = models_root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    if not text_encoder_path.exists():
        text_encoder_path = models_root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors"
    video_vae_path = models_root / "vae" / "ltx-2.5-video-vae-bf16.safetensors"
    audio_vae_path = models_root / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
    spatial_upscaler_path = models_root / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

    # Validate required files exist before loading anything
    missing = []
    for label, p in [
        ("Transformer", transformer_path),
        ("Distilled LoRA", distilled_lora_path),
        ("Text Encoder", text_encoder_path),
        ("Video VAE", video_vae_path),
        ("Audio VAE", audio_vae_path),
        ("Spatial Upscaler", spatial_upscaler_path),
    ]:
        if not p.exists():
            missing.append(f"  [{label}]: {p}")
    if missing:
        logger.error("MISSING CHECKPOINT FILES — verify model paths:")
        for m in missing:
            logger.error(m)
        raise FileNotFoundError(f"Missing {len(missing)} checkpoint(s). See above.")

    # Resolve offload mode and quantization policy
    if args.offload_mode == "cpu":
        offload_mode = OffloadMode.CPU
    elif args.offload_mode == "disk":
        offload_mode = OffloadMode.DISK
    else:
        offload_mode = OffloadMode.NONE

    is_nvfp4_prequant = "nvfp4" in transformer_path.name
    if args.quantization == "auto":
        if is_nvfp4_prequant:
            quant_kind_str = "nvfp4-prequant"
            quant_policy = QuantizationKind.NVFP4_PREQUANT.to_policy(checkpoint_path=str(transformer_path))
        else:
            quant_kind_str = "nvfp4-cast"
            quant_policy = QuantizationKind.NVFP4_CAST.to_policy()
    elif args.quantization == "nvfp4-prequant":
        quant_kind_str = "nvfp4-prequant"
        quant_policy = QuantizationKind.NVFP4_PREQUANT.to_policy(checkpoint_path=str(transformer_path))
    elif args.quantization == "nvfp4-cast":
        quant_kind_str = "nvfp4-cast"
        quant_policy = QuantizationKind.NVFP4_CAST.to_policy()
    elif args.quantization == "fp8-cast":
        quant_kind_str = "fp8-cast"
        quant_policy = QuantizationKind.FP8_CAST.to_policy(checkpoint_path=str(transformer_path))
    else:
        quant_kind_str = "none"
        quant_policy = None

    logger.info("=" * 80)
    logger.info("  NVIDIA GEFORCE RTX 5080 (16GB VRAM) - LTX-2.5 NVFP4 PRODUCTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Transformer:   {transformer_path.name} ({transformer_path.stat().st_size / (1024**3):.2f} GB)")
    logger.info(f"Quantization:  {quant_kind_str} (Blackwell FP4 tensor cores)")
    logger.info(f"Offload Mode:  {offload_mode.name} ({'<600MB GPU resident weights with CPU block streaming' if offload_mode != OffloadMode.NONE else 'Resident in VRAM'})")
    logger.info(f"LoRA Stage 1:  {args.distilled_lora_strength_stage_1} (pure dev model denoising)")
    logger.info(f"LoRA Stage 2:  {args.distilled_lora_strength_stage_2} (blended super-res refinement)")
    logger.info(f"Resolution:    {args.width}x{args.height} @ {args.frame_rate} fps")
    logger.info(f"Total Frames:  {args.num_frames} ({args.num_frames / args.frame_rate:.2f} seconds)")
    logger.info(f"Steps:         {args.steps} (res2s sampler, CFG: {args.video_cfg}, Rescale: {args.rescale})")
    logger.info(f"Prompt:        {args.prompt}")
    logger.info("=" * 80)

    # Initial telemetry baseline
    torch.cuda.reset_peak_memory_stats()
    watermarks = [MemoryWatermark.record("Initial Baseline")]

    # Build ModelPaths in split-file mode
    model_paths = ModelPaths.from_split(
        transformer_path=str(transformer_path),
        text_encoder_path=str(text_encoder_path),
        video_vae_path=str(video_vae_path),
        audio_vae_path=str(audio_vae_path),
    )

    # Distilled LoRA with ComfyUI-to-ltx-core key renaming.
    # At strength 0.0 (Stage 1), the LoRA deltas are zero-weighted — pure dev model.
    # At strength 0.8 (Stage 2), the LoRA guides the super-resolution refinement.
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            path=str(distilled_lora_path),
            strength=1.0,                          # base weight; per-stage strengths set below
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,  # renames ComfyUI LoRA keys to ltx-core keys
        )
    ]

    # Initialize Pipeline
    logger.info("[+] Initializing TI2VidTwoStagesHQPipeline...")
    t0 = time.time()

    pipeline = TI2VidTwoStagesHQPipeline(
        model_paths=model_paths,
        distilled_lora=distilled_lora,
        distilled_lora_strength_stage_1=args.distilled_lora_strength_stage_1,
        distilled_lora_strength_stage_2=args.distilled_lora_strength_stage_2,
        spatial_upsampler_path=str(spatial_upscaler_path),
        loras=(),
        device=torch.device("cuda:0"),
        quantization=quant_policy,
        offload_mode=offload_mode,
        diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
        alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
    )
    init_duration = time.time() - t0
    watermarks.append(MemoryWatermark.record("Pipeline Initialized"))
    logger.info(f"[+] Pipeline loaded in {init_duration:.2f} seconds.")

    if args.dry_run:
        logger.info("[Dry Run] Pipeline successfully constructed and verified in memory!")
        logger.info("[Dry Run] Peak VRAM: %.2f GB | Host RAM: %.2f GB",
                    watermarks[-1].vram_max_gb, watermarks[-1].host_ram_used_gb)
        logger.info("[Dry Run] Exiting without rendering full video.")
        return output_path

    # Image conditioning input (optional)
    images = []
    if args.input_image and Path(args.input_image).exists():
        logger.info(f"[+] Using Input Conditioning Image: {args.input_image}")
        images = [ImageConditioningInput(path=args.input_image, frame_index=0)]

    # Generate video via two-stage HQ pipeline
    logger.info("[+] Starting Two-Stage HQ Generation...")
    logger.info("    Stage 1: Full denoising at half-res (%dx%d, %d steps)",
                args.width // 2, args.height // 2, args.steps)
    logger.info("    Latent upscale: 2x spatial upsampler")
    logger.info("    Stage 2: Refinement at full-res (%dx%d, 3 steps)", args.width, args.height)
    gen_start = time.time()

    try:
        output = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.steps,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=args.video_cfg,
                stg_scale=0.0,
                rescale_scale=args.rescale,
                modality_scale=args.a2v_guidance,
            ),
            audio_guider_params=MultiModalGuiderParams(
                cfg_scale=args.audio_cfg,
                stg_scale=0.0,
                rescale_scale=0.0,
                modality_scale=0.0,
            ),
            images=images,
            tiling_config=AUTO_TILING,
        )
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"[-] CUDA OUT OF MEMORY ERROR: {e}")
        logger.error("VRAM Allocated: %.2f GB | Reserved: %.2f GB | Peak: %.2f GB",
                     torch.cuda.memory_allocated() / (1024 ** 3),
                     torch.cuda.memory_reserved() / (1024 ** 3),
                     torch.cuda.max_memory_allocated() / (1024 ** 3))
        logger.error("HINT: Try --num-frames 9 or --width 1280 --height 768 to reduce VRAM usage.")
        clean_memory()
        raise e

    gen_duration = time.time() - gen_start
    watermarks.append(MemoryWatermark.record("Generation Complete"))
    logger.info(f"[+] Generation finished in {gen_duration:.2f} seconds.")

    # Encode and save output video
    logger.info(f"[+] Encoding output video to {output_path}...")
    encode_video(
        video=output.video,
        audio=output.audio,
        output_path=str(output_path),
        fps=int(args.frame_rate),
        video_chunks_number=get_video_chunks_number(output.num_frames, output.tiling_config),
    )
    watermarks.append(MemoryWatermark.record("Video Encoded"))

    logger.info("=" * 80)
    logger.info(f"  GENERATION SUCCESSFUL: {output_path}")
    logger.info(f"  Total Video Duration: {args.num_frames / args.frame_rate:.2f}s ({args.num_frames} frames)")
    logger.info(f"  Peak VRAM Consumption: {max(w.vram_max_gb for w in watermarks):.2f} GB / 16.0 GB (WDDM limit)")
    logger.info(f"  Peak Host RAM Footprint: {max(w.host_ram_used_gb for w in watermarks):.2f} GB / 64.0 GB")
    logger.info("=" * 80)

    return output_path


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(args)
        return 0
    except FileNotFoundError as e:
        logger.error(str(e))
        return 2
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
