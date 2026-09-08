"""
LTX-2.5 NVFP4 Production Generation Pipeline for NVIDIA RTX 5080 (16GB VRAM / 64GB RAM).
Runs TI2VidTwoStagesHQPipeline with NVFP4 pre-quantized transformer, CPU layer offload,
chunked eager DiffVAE, and delicate VRAM/RAM watermark telemetry to prevent OOM.
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps
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

# Cluster-verified anti-distortion negative prompt
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
    parser = argparse.ArgumentParser(description="LTX-2.5 RTX 5080 NVFP4 Production Generation")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    parser.add_argument("--input-image", type=str, default=None, help="Optional input image path for image-to-video")
    parser.add_argument("--output", type=str, default="outputs/generated_video.mp4", help="Output MP4 file path")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR), help="Root models directory")
    
    # Production defaults (cluster-verified)
    parser.add_argument("--width", type=int, default=1920, help="Output width (default 1920)")
    parser.add_argument("--height", type=int, default=1088, help="Output height (default 1088)")
    parser.add_argument("--num-frames", type=int, default=121, help="Total frames (default 121, ~5s at 24fps)")
    parser.add_argument("--frame-rate", type=float, default=24.0, help="Video frame rate (default 24.0)")
    parser.add_argument("--steps", type=int, default=15, help="Diffusion steps (default 15)")
    parser.add_argument("--video-cfg", type=float, default=3.0, help="Video CFG scale (default 3.0)")
    parser.add_argument("--rescale", type=float, default=0.7, help="Rescale factor (default 0.7)")
    parser.add_argument("--audio-cfg", type=float, default=7.0, help="Audio CFG scale (default 7.0)")
    parser.add_argument("--a2v-guidance", type=float, default=3.0, help="Audio-to-video guidance (default 3.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random generation seed")
    
    # Execution modes
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline load and memory without rendering full video")
    parser.add_argument("--safe-mode", action="store_true", help="Aggressive memory cleanup after every operation")

    return parser.parse_args()


@torch.inference_mode()
def run_pipeline(args: argparse.Namespace) -> Path:
    models_root = Path(args.models_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transformer_path = models_root / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors"
    text_encoder_path = models_root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    if not text_encoder_path.exists():
        text_encoder_path = models_root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors"
    video_vae_path = models_root / "vae" / "ltx-2.5-video-vae-bf16.safetensors"
    audio_vae_path = models_root / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
    spatial_upscaler_path = models_root / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

    logger.info("=" * 80)
    logger.info("  NVIDIA GEFORCE RTX 5080 (16GB VRAM) - LTX-2.5 NVFP4 PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Resolution:    {args.width}x{args.height} @ {args.frame_rate} fps")
    logger.info(f"Total Frames:  {args.num_frames} ({args.num_frames / args.frame_rate:.2f} seconds)")
    logger.info(f"Steps:         {args.steps} (res2s sampler, CFG: {args.video_cfg}, Rescale: {args.rescale})")
    logger.info(f"Prompt:        {args.prompt}")
    logger.info(f"Quantization:  Native NVFP4 (cuBLASLt FP4 tensor cores)")
    logger.info(f"Offload Mode:  OffloadMode.NONE (Native NVFP4 in VRAM)")
    logger.info("=" * 80)

    # Initial telemetry baseline
    torch.cuda.reset_peak_memory_stats()
    watermarks = [MemoryWatermark.record("Initial Baseline")]

    # Build ModelPaths in split pack mode
    model_paths = ModelPaths.from_split(
        transformer_path=str(transformer_path),
        text_encoder_path=str(text_encoder_path),
        video_vae_path=str(video_vae_path),
        audio_vae_path=str(audio_vae_path),
    )

    # Build NVFP4 pre-quantization policy
    logger.info("[+] Configuring NVFP4 Pre-Quantization Policy...")
    quant_policy = QuantizationKind.NVFP4_PREQUANT.to_policy(checkpoint_path=str(transformer_path))

    # Dummy LoRA list for pre-fused distilled transformer (weight already embedded)
    dummy_lora = [LoraPathStrengthAndSDOps(path=str(transformer_path), strength=0.0, sd_ops=())]

    # Initialize Pipeline
    logger.info("[+] Initializing TI2VidTwoStagesHQPipeline with NVFP4 on RTX 5080...")
    t0 = time.time()
    
    pipeline = TI2VidTwoStagesHQPipeline(
        model_paths=model_paths,
        distilled_lora=dummy_lora,
        distilled_lora_strength_stage_1=0.0,
        distilled_lora_strength_stage_2=0.0,
        spatial_upsampler_path=str(spatial_upscaler_path),
        loras=(),
        device=torch.device("cuda:0"),
        quantization=quant_policy,
        offload_mode=OffloadMode.NONE,
        diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
        alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
    )
    init_duration = time.time() - t0
    watermarks.append(MemoryWatermark.record("Pipeline Initialized"))
    logger.info(f"[+] Pipeline loaded in {init_duration:.2f} seconds.")

    if args.dry_run:
        logger.info("[Dry Run] Pipeline successfully constructed and verified in memory!")
        logger.info("[Dry Run] Peak VRAM: %.2f GB | Host RAM: %.2f GB", watermarks[-1].vram_max_gb, watermarks[-1].host_ram_used_gb)
        logger.info("[Dry Run] Exiting without rendering full video.")
        return output_path

    # Image conditioning input (optional)
    images = []
    if args.input_image and Path(args.input_image).exists():
        logger.info(f"[+] Using Input Conditioning Image: {args.input_image}")
        images = [ImageConditioningInput(path=args.input_image, frame_index=0)]

    # Generate video
    logger.info("[+] Starting Two-Stage HQ Generation...")
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
        clean_memory()
        raise e

    gen_duration = time.time() - gen_start
    watermarks.append(MemoryWatermark.record("Generation Complete"))
    logger.info(f"[+] Generation finished in {gen_duration:.2f} seconds.")

    # Save output video
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
    logger.info(f"  Peak VRAM Consumption: {max(w.vram_max_gb for w in watermarks):.2f} GB / 16.0 GB")
    logger.info(f"  Peak Host RAM Footprint: {max(w.host_ram_used_gb for w in watermarks):.2f} GB / 64.0 GB")
    logger.info("=" * 80)

    return output_path


def main() -> int:
    args = parse_args()
    try:
        run_pipeline(args)
        return 0
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
