#!/usr/bin/env python3
"""
Generate a 7-scene Southern Methodist University (SMU) commercial using LTX-2.5.
Each scene is conditioned on an input photograph from inputs/smu/ with tailored
cinematic prompts, rendered via DistilledPipeline on an A100 GPU, and assembled
with inputs/smu/smu_voiceover.wav into outputs/smu_commercial_full.mp4.
"""

import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import av
import imageio_ffmpeg
import soundfile as sf
import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import OffloadMode

# Scene specifications
SCENES = [
    {
        "id": 1,
        "name": "01_dallas_hall",
        "image": "inputs/smu/01_dallas_hall.jpg",
        "prompt": (
            "A majestic, slow, cinematic forward camera dolly toward Dallas Hall on the Southern Methodist University "
            "campus, lush green lawn, warm bright morning sunlight casting gentle shadows on the historic red brick "
            "and neoclassical white columns and dome, vibrant blue sky, gentle breeze swaying tree branches, collegiate excellence."
        ),
        "num_frames": 121,
        "seed": 42,
    },
    {
        "id": 2,
        "name": "02_mustang_statue",
        "image": "inputs/smu/02_mustang_statue.jpg",
        "prompt": (
            "A dynamic, heroic low-angle orbital camera pan around the bronze Mustang statue fountain on the SMU campus, "
            "crystal clear water splashing dynamically in slow motion, glistening water droplets, powerful bronze horse "
            "sculptures conveying unstoppable energy and Mustang pride, dramatic cinematic lighting."
        ),
        "num_frames": 121,
        "seed": 101,
    },
    {
        "id": 3,
        "name": "03_engineering_lab",
        "image": "inputs/smu/03_engineering_lab.jpg",
        "prompt": (
            "A smooth cinematic tracking shot inside the modern SMU Lyle School of Engineering robotics laboratory, "
            "engaged students collaborating with robotic arms and advanced technology, illuminated LED displays, "
            "futuristic computer monitors showing engineering data, clean high-tech research atmosphere."
        ),
        "num_frames": 121,
        "seed": 202,
    },
    {
        "id": 4,
        "name": "04_business_cox",
        "image": "inputs/smu/04_business_cox.jpg",
        "prompt": (
            "An elegant, steady gliding camera shot through the sunlit atrium of the SMU Cox School of Business, "
            "polished limestone floors reflecting natural light, soaring glass windows, professional students engaged "
            "in dynamic discussion, inspiring academic and leadership atmosphere, warm architectural lighting."
        ),
        "num_frames": 121,
        "seed": 303,
    },
    {
        "id": 5,
        "name": "05_meadows_arts",
        "image": "inputs/smu/05_meadows_arts.jpg",
        "prompt": (
            "A graceful, fluid cinematic camera crane shot across the SMU Meadows School of the Arts plaza, "
            "contemporary outdoor sculptures, vibrant artistic architecture, creative energy, gentle afternoon golden "
            "sunlight highlighting modern art installations and students walking by."
        ),
        "num_frames": 121,
        "seed": 404,
    },
    {
        "id": 6,
        "name": "06_campus_sunset",
        "image": "inputs/smu/06_campus_sunset.jpg",
        "prompt": (
            "A breathtaking wide cinematic aerial gliding shot over the Southern Methodist University campus during a "
            "spectacular golden hour sunset, rich amber and purple sky reflecting off campus pathways, tree-lined walkways "
            "illuminated by warm campus lampposts, timeless collegiate beauty and wonder."
        ),
        "num_frames": 121,
        "seed": 505,
    },
    {
        "id": 7,
        "name": "07_closing_logo",
        "image": "inputs/smu/07_closing_logo.jpg",
        "prompt": (
            "A sleek, premium broadcast commercial title card animation, slow subtle zoom into the official Southern "
            "Methodist University Mustang logo and typography, soft cinematic lens flare and ambient light rays dancing "
            "across the iconic blue and red collegiate crest, pristine and inspiring conclusion."
        ),
        "num_frames": 129,  # 129 frames brings total duration to 855 frames (35.625s), perfectly syncing with 35.59s voiceover
        "seed": 606,
    },
]


def is_video_valid(
    file_path: Path,
    expected_frames: int | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> bool:
    """Check if video file exists, can be opened, and has valid frames and matching resolution."""
    if not file_path.is_file() or file_path.stat().st_size < 10000:
        return False
    try:
        container = av.open(str(file_path))
        video_stream = container.streams.video[0]
        frames = video_stream.frames
        width = video_stream.width
        height = video_stream.height
        container.close()
        if expected_frames is not None and frames != expected_frames:
            print(f"Warning: {file_path} has {frames} frames, expected {expected_frames}")
            return False
        if expected_width is not None and width != expected_width:
            print(f"Info: {file_path} width {width} != expected {expected_width} (requires regeneration)")
            return False
        if expected_height is not None and height != expected_height:
            print(f"Info: {file_path} height {height} != expected {expected_height} (requires regeneration)")
            return False
        return frames > 0
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return False


def assemble_commercial(
    scenes_dir: Path,
    voiceover_path: Path,
    output_path: Path,
    scenes: list[dict[str, Any]],
) -> None:
    """Concatenate scenes and multiplex voiceover track using imageio_ffmpeg binary."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    concat_list_file = scenes_dir / "concat_list.txt"

    print("\n" + "=" * 60)
    print("Assembling final SMU commercial...")
    print("=" * 60)

    # Verify all scene files exist
    scene_files = []
    for sc in scenes:
        scene_file = scenes_dir / f"scene_{sc['id']:02d}_{sc['name']}.mp4"
        if not scene_file.is_file():
            raise FileNotFoundError(f"Missing required scene file: {scene_file}")
        scene_files.append(scene_file.resolve())

    # Write concat list
    with open(concat_list_file, "w") as f:
        for sf in scene_files:
            f.write(f"file '{sf}'\n")

    print(f"Concat list written to {concat_list_file}")

    # Run FFmpeg concat and audio mux
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_file),
        "-i",
        str(voiceover_path.resolve()),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(output_path.resolve()),
    ]

    print("Running FFmpeg command:")
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FFmpeg stdout:\n{res.stdout}")
        print(f"FFmpeg stderr:\n{res.stderr}")
        raise RuntimeError(f"FFmpeg assembly failed with exit code {res.returncode}")

    print(f"Successfully assembled full commercial to: {output_path}")

    # Validate output
    container = av.open(str(output_path))
    v_stream = container.streams.video[0]
    a_stream = container.streams.audio[0]
    duration = float(v_stream.duration * v_stream.time_base)
    print(f"Output Video Resolution: {v_stream.width}x{v_stream.height}")
    print(f"Output Video Frames: {v_stream.frames} ({duration:.2f} seconds)")
    print(f"Output Audio Sample Rate: {a_stream.rate}Hz, Channels: {a_stream.channels}")
    container.close()


def extract_preview_frames(
    scenes_dir: Path, scenes: list[dict[str, Any]], previews_dir: Path
) -> None:
    """Extract representative preview frames from each generated scene."""
    previews_dir.mkdir(parents=True, exist_ok=True)
    for sc in scenes:
        scene_file = scenes_dir / f"scene_{sc['id']:02d}_{sc['name']}.mp4"
        if not scene_file.is_file():
            continue
        try:
            container = av.open(str(scene_file))
            target_frame_idx = sc["num_frames"] // 2
            curr = 0
            for frame in container.decode(video=0):
                if curr == target_frame_idx:
                    img = frame.to_image()
                    out_img_path = previews_dir / f"preview_scene_{sc['id']:02d}_{sc['name']}.jpg"
                    img.save(out_img_path, quality=90)
                    print(f"Saved preview frame: {out_img_path}")
                    break
                curr += 1
            container.close()
        except Exception as e:
            print(f"Could not extract preview for scene {sc['id']}: {e}")


@torch.inference_mode()
def render_scene(
    pipeline: TI2VidTwoStagesHQPipeline,
    sc: dict[str, Any],
    args: argparse.Namespace,
    scene_output: Path,
) -> None:
    """Render a single scene using LTX-2.5 TI2VidTwoStagesHQPipeline and encode to MP4."""
    scene_t0 = time.time()
    images = [
        ImageConditioningInput(
            path=str(Path(sc["image"]).resolve()),
            frame_idx=0,
            strength=1.0,
        )
    ]

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=args.video_cfg_guidance_scale,
        stg_scale=args.video_stg_guidance_scale,
        rescale_scale=args.video_rescale_scale,
        modality_scale=args.a2v_guidance_scale,
        skip_step=0,
        stg_blocks=[28] if args.video_stg_guidance_scale > 0 else [],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=args.audio_cfg_guidance_scale,
        stg_scale=0.0,
        rescale_scale=1.0,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = pipeline(
        prompt=sc["prompt"],
        negative_prompt=args.negative_prompt,
        seed=sc["seed"],
        height=args.height,
        width=args.width,
        num_frames=sc["num_frames"],
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=images,
        enhance_prompt=False,
        tiling_config=AUTO_TILING,
    )

    encode_video(
        video=result.video,
        fps=int(args.frame_rate),
        audio=result.audio,
        output_path=str(scene_output),
        video_chunks_number=get_video_chunks_number(
            result.num_frames, result.tiling_config
        ),
    )

    elapsed = time.time() - scene_t0
    vram_str = ""
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
        res_gb = torch.cuda.max_memory_reserved() / (1024**3)
        vram_str = f" [Peak VRAM: {alloc_gb:.2f} GB allocated, {res_gb:.2f} GB reserved]"
    print(f"Scene {sc['id']} generated in {elapsed:.2f}s -> {scene_output}{vram_str}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full SMU commercial with LTX-2.5 (High Quality Two-Stage)")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/ltx-2.5"),
        help="Root directory for downloaded LTX-2.5 weights",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=Path("inputs/smu"),
        help="Directory containing input images and voiceover",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for generated scenes and master commercial",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path("outputs/smu_scenes"),
        help="Directory where individual scene video clips are stored",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1088,
        help="Output video height (divisible by 64, default 1088 for 1080p widescreen)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Output video width (divisible by 64, default 1920 for 1080p widescreen)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=15,
        help="Number of res2s inference steps (default: 15, equivalent to 30 function evaluations)",
    )
    parser.add_argument(
        "--video-cfg-guidance-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance (CFG) scale for video (default: 3.0)",
    )
    parser.add_argument(
        "--video-stg-guidance-scale",
        type=float,
        default=0.0,
        help="Spatio-temporal guidance (STG) scale for video (default: 0.0)",
    )
    parser.add_argument(
        "--video-rescale-scale",
        type=float,
        default=0.7,
        help="Video guidance rescale scale to prevent oversaturation (default: 0.7)",
    )
    parser.add_argument(
        "--audio-cfg-guidance-scale",
        type=float,
        default=7.0,
        help="CFG scale for audio (default: 7.0)",
    )
    parser.add_argument(
        "--a2v-guidance-scale",
        type=float,
        default=3.0,
        help="Audio-to-video cross-modality guidance scale (default: 3.0)",
    )
    parser.add_argument(
        "--distilled-lora-strength-stage-1",
        type=float,
        default=0.0,
        help="Distilled LoRA strength in Stage 1 (default: 0.0, uses pure base dev model)",
    )
    parser.add_argument(
        "--distilled-lora-strength-stage-2",
        type=float,
        default=0.8,
        help="Distilled LoRA strength in Stage 2 super-resolution refinement (default: 0.8)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for CFG denoising",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=24.0,
        help="Video frame rate in FPS",
    )
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
        help="Optional: Run only a specific scene ID (1-7)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip video generation and only run assembly",
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default=None,
        help="Optional override for transformer checkpoint path (defaults to dev transformer bf16)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="fp8-cast",
        choices=["fp8-cast", "fp8-scaled-mm", "nvfp4-cast", "nvfp4-prequant"],
        help="Quantization policy: fp8-cast (default, 64GB RAM/32GB VRAM), fp8-scaled-mm, nvfp4-cast, nvfp4-prequant",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of existing scene videos",
    )
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.scenes_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.outputs_dir / "smu_previews"

    scenes_to_run = SCENES
    if args.scene_id is not None:
        scenes_to_run = [s for s in SCENES if s["id"] == args.scene_id]
        if not scenes_to_run:
            raise ValueError(f"Invalid scene ID: {args.scene_id}. Must be 1-7.")

    voiceover_path = args.inputs_dir / "smu_voiceover.wav"
    if not voiceover_path.is_file():
        raise FileNotFoundError(f"Voiceover not found at {voiceover_path}")

    master_output = args.outputs_dir / "smu_commercial_full.mp4"

    if not args.skip_generation:
        # Check which scenes need generation
        scenes_pending = []
        for sc in scenes_to_run:
            scene_output = args.scenes_dir / f"scene_{sc['id']:02d}_{sc['name']}.mp4"
            if args.force or not is_video_valid(
                scene_output, sc["num_frames"], args.width, args.height
            ):
                scenes_pending.append(sc)
            else:
                print(
                    f"[SKIP] Scene {sc['id']}: {sc['name']} already exists at {args.width}x{args.height} and is valid.",
                    flush=True,
                )

        if scenes_pending:
            print(f"\nInitializing LTX-2.5 TI2VidTwoStagesHQPipeline on {torch.cuda.get_device_name(0)}...", flush=True)
            t0 = time.time()

            transformer_path = (
                args.transformer_path
                if args.transformer_path
                else str(args.models_dir / "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors")
            )
            model_paths = ModelPaths.from_split(
                transformer_path=transformer_path,
                text_encoder_path=str(
                    args.models_dir / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
                ),
                video_vae_path=str(args.models_dir / "vae/ltx-2.5-video-vae-bf16.safetensors"),
                audio_vae_path=str(args.models_dir / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
            )
            spatial_upsampler_path = str(
                args.models_dir
                / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
            )
            distilled_lora_path = str(
                args.models_dir
                / "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
            )
            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    path=distilled_lora_path,
                    strength=1.0,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]

            quant_policy = None
            if args.quantization:
                if args.quantization in ("nvfp4-cast", "nvfp4-prequant"):
                    if torch.cuda.is_available():
                        major, minor = torch.cuda.get_device_capability(0)
                        if major < 10:
                            raise RuntimeError(
                                f"NVFP4 quantization requires NVIDIA Blackwell architecture (SM >= 10.0) with hardware FP4 tensor cores. "
                                f"Current device is {torch.cuda.get_device_name(0)} (sm_{major}{minor}). "
                                f"Please use '--quantization fp8-cast' or '--quantization fp8-scaled-mm' instead."
                            )
                from ltx_pipelines.utils.quantization_factory import QuantizationKind
                quant_kind = QuantizationKind(args.quantization)
                quant_policy = quant_kind.to_policy(checkpoint_path=transformer_path)
                print(f"Applying quantization policy: {args.quantization}", flush=True)

            pipeline = TI2VidTwoStagesHQPipeline(
                model_paths=model_paths,
                distilled_lora=distilled_lora,
                distilled_lora_strength_stage_1=args.distilled_lora_strength_stage_1,
                distilled_lora_strength_stage_2=args.distilled_lora_strength_stage_2,
                spatial_upsampler_path=spatial_upsampler_path,
                loras=(),
                quantization=quant_policy,
                offload_mode=OffloadMode.CPU,
            )
            print(f"HQ Pipeline initialized in {time.time() - t0:.2f}s.", flush=True)

            total_pending = len(scenes_pending)
            for idx, sc in enumerate(scenes_pending, 1):
                scene_output = args.scenes_dir / f"scene_{sc['id']:02d}_{sc['name']}.mp4"
                print("\n" + "=" * 60, flush=True)
                print(
                    f"[{idx}/{total_pending}] Generating Scene {sc['id']}: {sc['name']} ({sc['num_frames']} frames)...",
                    flush=True,
                )
                print(f"Image: {sc['image']}", flush=True)
                print(f"Prompt: {sc['prompt']}", flush=True)
                print("=" * 60, flush=True)

                render_scene(pipeline, sc, args, scene_output)

                # Clean up memory between scenes
                gc.collect()
                torch.cuda.empty_cache()

            print("\nAll pending scenes generated successfully!", flush=True)
        else:
            print("\n" + "=" * 70, flush=True)
            print("NOTICE: All requested scenes already exist in outputs/smu_scenes/ and are valid.", flush=True)
            print("Skipping AI diffusion generation. (Job did not fail; cache validation succeeded).", flush=True)
            print("To force re-rendering of existing scenes, run with: --force", flush=True)
            print("Or to re-render a specific scene, run with: --scene-id <1-7> --force", flush=True)
            print("=" * 70 + "\n", flush=True)

    # Extract preview frames
    extract_preview_frames(args.scenes_dir, SCENES, previews_dir)

    # Assemble full commercial if all 7 scenes exist
    all_scenes_exist = all(
        (args.scenes_dir / f"scene_{sc['id']:02d}_{sc['name']}.mp4").is_file() for sc in SCENES
    )
    if all_scenes_exist:
        assemble_commercial(args.scenes_dir, voiceover_path, master_output, SCENES)
    else:
        print("Note: Not all 7 scenes exist yet; skipping assembly.")


if __name__ == "__main__":
    main()
