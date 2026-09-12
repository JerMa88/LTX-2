"""
Method A: Official DistilledPipeline for NVIDIA RTX 5080 (16GB VRAM).
Uses the official Lightricks DistilledPipeline:
  - Checkpoint: ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors
  - Sampler: euler_ancestral for Stage 1 (8 steps), deterministic euler for Stage 2 (3 steps)
  - Guidance: CFG-free (CFG = 1.0) via SimpleDenoiser
  - Offload: OffloadMode.CPU (NVFP4 block streaming)
  - Total function evaluations: 11 (fastest and cleanest motion)
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import psutil
import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import OffloadMode

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("MethodA_Distilled")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "ltx25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Method A: Official DistilledPipeline")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, default="outputs/method_a_distilled_25f_1080p.mp4")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload-mode", type=str, default="cpu", choices=["none", "cpu"])
    parser.add_argument("--transformer-path", type=str, default=None)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    models_root = Path(args.models_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transformer_path = (
        Path(args.transformer_path)
        if args.transformer_path
        else models_root / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors"
    )

    text_encoder_path = models_root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    if not text_encoder_path.exists():
        text_encoder_path = models_root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors"

    video_vae_path = models_root / "vae" / "ltx-2.5-video-vae-bf16.safetensors"
    audio_vae_path = models_root / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
    spatial_upscaler_path = models_root / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

    offload_mode = OffloadMode.CPU if args.offload_mode == "cpu" else OffloadMode.NONE
    quant_policy = QuantizationKind.NVFP4_PREQUANT.to_policy(checkpoint_path=str(transformer_path))

    logger.info("=" * 80)
    logger.info("  METHOD A: OFFICIAL DISTILLED PIPELINE (EULER ANCESTRAL, CFG 1.0, 11 EVALS)")
    logger.info("=" * 80)
    logger.info(f"Transformer:   {transformer_path.name}")
    logger.info(f"Text Encoder:  {text_encoder_path.name}")
    logger.info(f"Resolution:    {args.width}x{args.height} @ {args.frame_rate} fps")
    logger.info(f"Total Frames:  {args.num_frames} ({args.num_frames / args.frame_rate:.2f} seconds)")
    logger.info(f"Seed:          {args.seed}")
    logger.info(f"Sampler:       Stage 1: euler_ancestral (8 steps) | Stage 2: euler (3 steps)")
    logger.info(f"Guidance:      CFG 1.0 (CFG-Free native distilled)")
    logger.info("=" * 80)

    model_paths = ModelPaths.from_split(
        transformer_path=str(transformer_path),
        text_encoder_path=str(text_encoder_path),
        video_vae_path=str(video_vae_path),
        audio_vae_path=str(audio_vae_path),
    )

    t0 = time.time()
    pipeline = DistilledPipeline(
        model_paths=model_paths,
        spatial_upsampler_path=str(spatial_upscaler_path),
        loras=[],
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
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
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
