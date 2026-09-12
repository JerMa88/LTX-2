"""
Method B: True Exact A100 HPC Setup Replication for NVIDIA RTX 5080 (16GB VRAM).
Replicates the exact A100 commercial pipeline configuration:
  - Checkpoint: ltx-2.5-22b-dev-transformer-bf16.safetensors (Full 42GB foundation dev model)
  - Quantization: fp8-cast (runtime FP8 dynamic casting, matching A100)
  - Pipeline: TI2VidTwoStagesHQPipeline
  - Sampler: res2s 2nd-order ODE solver (15 steps Stage 1 = 30 evals + 3 steps Stage 2 = 6 evals)
  - LoRA: ltx-2.5-22b-distilled-lora-450-bf16.safetensors (Stage 1: 0.0, Stage 2: 0.8)
  - Guidance: CFG 3.0, Rescale 0.7, Anti-distortion Negative Prompt
  - Offload: OffloadMode.CPU (FP8 block streaming)
  - Host RAM optimization: NVFP4 text encoder (10.6GB) to keep total commit comfortably within 64GB
"""

import argparse
import gc
import logging
import os
from pathlib import Path
import time

import psutil
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
from ltx_pipelines.utils.types import OffloadMode

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("MethodB_ExactA100")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "ltx25"
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "watermark, low resolution, artifacts, oversaturated, flicker"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Method B: True Exact A100 Replication")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", type=str, default="outputs/method_b_exact_a100_25f_1080p.mp4")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--video-cfg", type=float, default=3.0)
    parser.add_argument("--rescale", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload-mode", type=str, default="disk", choices=["none", "cpu", "disk"])
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    models_root = Path(args.models_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transformer_path = models_root / "diffusion_models" / "ltx-2.5-22b-dev-transformer-bf16.safetensors"
    distilled_lora_path = models_root / "loras" / "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
    text_encoder_path = models_root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    if not text_encoder_path.exists():
        text_encoder_path = models_root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors"

    video_vae_path = models_root / "vae" / "ltx-2.5-video-vae-bf16.safetensors"
    audio_vae_path = models_root / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
    spatial_upscaler_path = models_root / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

    if args.offload_mode == "disk":
        offload_mode = OffloadMode.DISK
    elif args.offload_mode == "cpu":
        offload_mode = OffloadMode.CPU
    else:
        offload_mode = OffloadMode.NONE
    quant_policy = QuantizationKind.FP8_CAST.to_policy(checkpoint_path=str(transformer_path))

    logger.info("=" * 80)
    logger.info("  METHOD B: EXACT A100 HPC REPLICATION (BASE DEV BF16 + FP8-CAST + RES2S)")
    logger.info("=" * 80)
    logger.info(f"Transformer:   {transformer_path.name} (42GB Foundation Dev Model)")
    logger.info(f"Quantization:  fp8-cast (matches A100 run)")
    logger.info(f"Text Encoder:  {text_encoder_path.name}")
    logger.info(f"LoRA Stage 1:  0.0 (Pure Base Dev Denoising)")
    logger.info(f"LoRA Stage 2:  0.8 (Super-Resolution Refinement with Distilled LoRA)")
    logger.info(f"Resolution:    {args.width}x{args.height} @ {args.frame_rate} fps")
    logger.info(f"Total Frames:  {args.num_frames} ({args.num_frames / args.frame_rate:.2f} seconds)")
    logger.info(f"Seed:          {args.seed}")
    logger.info(f"Sampler:       res2s ({args.steps} steps = 30 fn evaluations)")
    logger.info(f"Guidance:      CFG {args.video_cfg}, Rescale {args.rescale}")
    logger.info("=" * 80)

    model_paths = ModelPaths.from_split(
        transformer_path=str(transformer_path),
        text_encoder_path=str(text_encoder_path),
        video_vae_path=str(video_vae_path),
        audio_vae_path=str(audio_vae_path),
    )

    distilled_lora = [
        LoraPathStrengthAndSDOps(
            path=str(distilled_lora_path),
            strength=1.0,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]

    t0 = time.time()
    pipeline = TI2VidTwoStagesHQPipeline(
        model_paths=model_paths,
        distilled_lora=distilled_lora,
        distilled_lora_strength_stage_1=0.0,
        distilled_lora_strength_stage_2=0.8,
        spatial_upsampler_path=str(spatial_upscaler_path),
        loras=(),
        device=torch.device("cuda:0"),
        quantization=quant_policy,
        offload_mode=offload_mode,
        diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
        alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
    )
    logger.info(f"[+] Pipeline constructed in {time.time() - t0:.2f}s")

    gen_start = time.time()
    result = pipeline(
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
            modality_scale=3.0,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=0.0,
            rescale_scale=0.0,
            modality_scale=0.0,
        ),
        images=[],
        tiling_config=AUTO_TILING,
    )
    gen_duration = time.time() - gen_start
    logger.info(f"[+] Generation complete in {gen_duration:.2f}s")

    logger.info(f"[+] Encoding video to {output_path}...")
    encode_video(
        video=result.video,
        fps=args.frame_rate,
        audio=result.audio,
        output_path=str(output_path),
        video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
    )
    logger.info(f"[+] Video successfully encoded -> {output_path}")


if __name__ == "__main__":
    main()
