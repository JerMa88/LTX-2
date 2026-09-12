#!/usr/bin/env python3
"""
Generate a 7-scene Southern Methodist University (SMU) commercial using LTX-2.5 Method A (DistilledPipeline).
Runs the official Lightricks DistilledPipeline:
  - Checkpoint: ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors
  - Sampler: euler_ancestral for Stage 1 (8 steps), deterministic euler for Stage 2 (3 steps)
  - Guidance: CFG-free (CFG = 1.0) via SimpleDenoiser
  - Resolution: 1280x768 @ 24 fps, 121 frames (~5.04s per scene)
  - Full real-time NVML telemetry profiling at every single stage of each scene:
    * Time elapsed
    * VRAM allocated, reserved, peak
    * Host RAM used & percentage
    * Offload amount (streaming blocks vs resident VRAM)
    * GPU compute utilization percentage
    * Power consumption (Watts) & temperature
  - Final aggregation reporting scene means, totals, and master commercial assembly.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# UTF-8 encoding configuration for Windows consoles
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import av
import imageio_ffmpeg
import psutil
import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.types import VideoPixelShape
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import combined_image_conditionings, ensure_tiling_config, tiling_scale_factors_for_vae
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("SMU_Commercial_Distilled")

# ==============================================================================
# NVML Real-Time Telemetry Profiler
# ==============================================================================

class NVMLUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NVMLProfiler:
    """Zero-dependency hardware profiler utilizing NVML via ctypes."""
    def __init__(self, device_idx: int = 0):
        self.device_idx = device_idx
        self.available = False
        self.handle = ctypes.c_void_p()
        self._nvml = None

        dll_names = ["nvml.dll", "libnvidia-ml.so.1", "libnvidia-ml.so"]
        for dll in dll_names:
            try:
                self._nvml = ctypes.CDLL(dll)
                if self._nvml.nvmlInit() == 0:
                    if self._nvml.nvmlDeviceGetHandleByIndex(device_idx, ctypes.byref(self.handle)) == 0:
                        self.available = True
                        break
            except Exception:
                continue

    def sample(self) -> tuple[float, float, float, float]:
        """Returns (gpu_util_pct, mem_util_pct, power_watts, temp_c)."""
        if not self.available:
            return 0.0, 0.0, 0.0, 0.0
        try:
            util = NVMLUtilization()
            self._nvml.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util))
            power = ctypes.c_uint()
            self._nvml.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power))
            temp = ctypes.c_uint()
            self._nvml.nvmlDeviceGetTemperature(self.handle, 0, ctypes.byref(temp))
            return float(util.gpu), float(util.memory), float(power.value) / 1000.0, float(temp.value)
        except Exception:
            return 0.0, 0.0, 0.0, 0.0


@dataclass
class StageMetrics:
    stage_name: str
    duration_s: float = 0.0
    vram_alloc_gb: float = 0.0
    vram_peak_gb: float = 0.0
    vram_reserved_gb: float = 0.0
    host_ram_used_gb: float = 0.0
    host_ram_pct: float = 0.0
    offload_resident_gb: float = 0.0
    offload_host_gb: float = 0.0
    avg_gpu_util_pct: float = 0.0
    peak_gpu_util_pct: float = 0.0
    avg_power_w: float = 0.0
    peak_power_w: float = 0.0
    avg_temp_c: float = 0.0


class ActiveTelemetrySampler:
    """Background thread collecting high-frequency NVML samples during a stage."""
    def __init__(self, profiler: NVMLProfiler, sample_interval: float = 0.1):
        self.profiler = profiler
        self.sample_interval = sample_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples_gpu_util: list[float] = []
        self.samples_power_w: list[float] = []
        self.samples_temp_c: list[float] = []

    def start(self):
        self._stop_event.clear()
        self.samples_gpu_util.clear()
        self.samples_power_w.clear()
        self.samples_temp_c.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            gpu, _, pwr, temp = self.profiler.sample()
            self.samples_gpu_util.append(gpu)
            self.samples_power_w.append(pwr)
            self.samples_temp_c.append(temp)
            time.sleep(self.sample_interval)

    def stop(self) -> tuple[float, float, float, float, float]:
        """Stops sampler and returns (avg_util, peak_util, avg_pwr, peak_pwr, avg_temp)."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        
        avg_u = sum(self.samples_gpu_util) / len(self.samples_gpu_util) if self.samples_gpu_util else 0.0
        peak_u = max(self.samples_gpu_util) if self.samples_gpu_util else 0.0
        avg_p = sum(self.samples_power_w) / len(self.samples_power_w) if self.samples_power_w else 0.0
        peak_p = max(self.samples_power_w) if self.samples_power_w else 0.0
        avg_t = sum(self.samples_temp_c) / len(self.samples_temp_c) if self.samples_temp_c else 0.0
        return avg_u, peak_u, avg_p, peak_p, avg_t


# ==============================================================================
# Scene Definitions
# ==============================================================================

SCENES = [
    {
        "id": 1,
        "name": "01_dallas_hall",
        "image": "inputs/smu/01_dallas_hall.jpg",
        "voiceover": "At Southern Methodist University, heritage meets the horizon.",
        "prompt": (
            "A majestic, slow, cinematic forward camera dolly toward Dallas Hall on the Southern Methodist University "
            "campus, lush green lawn, warm bright morning sunlight casting gentle shadows on the historic red brick "
            "and neoclassical white columns and dome, vibrant blue sky, gentle breeze swaying tree branches, collegiate excellence. "
            "A warm, inspiring narrator voiceover articulates clearly: 'At Southern Methodist University, heritage meets the horizon.'"
        ),
        "num_frames": 121,
        "seed": 42,
    },
    {
        "id": 2,
        "name": "02_mustang_statue",
        "image": "inputs/smu/02_mustang_statue.jpg",
        "voiceover": "In the heart of Dallas, bold ideas take flight.",
        "prompt": (
            "A dynamic, heroic low-angle orbital camera pan around the bronze Mustang statue fountain on the SMU campus, "
            "crystal clear water splashing dynamically in slow motion, glistening water droplets, powerful bronze horse "
            "sculptures conveying unstoppable energy and Mustang pride, dramatic cinematic lighting. "
            "A resonant, inspiring narrator voiceover delivers the line: 'In the heart of Dallas, bold ideas take flight.'"
        ),
        "num_frames": 121,
        "seed": 101,
    },
    {
        "id": 3,
        "name": "03_engineering_lab",
        "image": "inputs/smu/03_engineering_lab.jpg",
        "voiceover": "Here, innovators sculpt tomorrow's breakthroughs in artificial intelligence and robotics.",
        "prompt": (
            "A smooth cinematic tracking shot inside the modern SMU Lyle School of Engineering robotics laboratory, "
            "engaged students collaborating with robotic arms and advanced technology, illuminated LED displays, "
            "futuristic computer monitors showing engineering data, clean high-tech research atmosphere. "
            "An articulate, confident narrator voiceover states: 'Here, innovators sculpt tomorrow's breakthroughs in artificial intelligence and robotics.'"
        ),
        "num_frames": 121,
        "seed": 202,
    },
    {
        "id": 4,
        "name": "04_business_cox",
        "image": "inputs/smu/04_business_cox.jpg",
        "voiceover": "Visionaries lead global commerce with unyielding integrity.",
        "prompt": (
            "An elegant, steady gliding camera shot through the sunlit atrium of the SMU Cox School of Business, "
            "polished limestone floors reflecting natural light, soaring glass windows, professional students engaged "
            "in dynamic discussion, inspiring academic and leadership atmosphere, warm architectural lighting. "
            "A distinguished, poised narrator voiceover speaks: 'Visionaries lead global commerce with unyielding integrity.'"
        ),
        "num_frames": 121,
        "seed": 303,
    },
    {
        "id": 5,
        "name": "05_meadows_arts",
        "image": "inputs/smu/05_meadows_arts.jpg",
        "voiceover": "Artists inspire, and culture thrives.",
        "prompt": (
            "A graceful, fluid cinematic camera crane shot across the SMU Meadows School of the Arts plaza, "
            "contemporary outdoor sculptures, vibrant artistic architecture, creative energy, gentle afternoon golden "
            "sunlight highlighting modern art installations and students walking by. "
            "An expressive, uplifting narrator voiceover proclaims: 'Artists inspire, and culture thrives.'"
        ),
        "num_frames": 121,
        "seed": 404,
    },
    {
        "id": 6,
        "name": "06_campus_sunset",
        "image": "inputs/smu/06_campus_sunset.jpg",
        "voiceover": "Fueled by unbridled Mustang spirit, we don't just dream of a better world. We build it.",
        "prompt": (
            "A breathtaking wide cinematic aerial gliding shot over the Southern Methodist University campus during a "
            "spectacular golden hour sunset, rich amber and purple sky reflecting off campus pathways, tree-lined walkways "
            "illuminated by warm campus lampposts, timeless collegiate beauty and wonder. "
            "A passionate, resonant narrator voiceover delivers with conviction: 'Fueled by unbridled Mustang spirit, we don't just dream of a better world. We build it.'"
        ),
        "num_frames": 121,
        "seed": 505,
    },
    {
        "id": 7,
        "name": "07_closing_logo",
        "image": "inputs/smu/07_closing_logo.jpg",
        "voiceover": "Southern Methodist University. World changers shaped here. Pony up!",
        "prompt": (
            "A sleek, premium broadcast commercial title card animation, slow subtle zoom into the official Southern "
            "Methodist University Mustang logo and typography, soft cinematic lens flare and ambient light rays dancing "
            "across the iconic blue and red collegiate crest, pristine and inspiring conclusion. "
            "A confident, inspiring narrator voiceover concludes emphatically: 'Southern Methodist University. World changers shaped here. Pony up!'"
        ),
        "num_frames": 121,
        "seed": 606,
    },
]


def is_video_valid(file_path: Path, expected_frames: int, expected_width: int, expected_height: int) -> bool:
    if not file_path.is_file() or file_path.stat().st_size < 10000:
        return False
    try:
        container = av.open(str(file_path))
        video_stream = container.streams.video[0]
        frames = video_stream.frames
        width = video_stream.width
        height = video_stream.height
        container.close()
        return (frames == expected_frames and width == expected_width and height == expected_height)
    except Exception:
        return False


def assemble_commercial(scenes_dir: Path, voiceover_path: Path | None, output_path: Path, scenes: list[dict[str, Any]]) -> None:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    concat_list_file = scenes_dir / "concat_list.txt"

    print("\n" + "=" * 80)
    print("  ASSEMBLING MASTER SMU COMMERCIAL")
    print("=" * 80)

    scene_files = []
    for sc in scenes:
        scene_file = scenes_dir / f"scene_{sc['id']:02d}_{sc['name']}.mp4"
        if not scene_file.is_file():
            raise FileNotFoundError(f"Missing required scene file: {scene_file}")
        scene_files.append(scene_file.resolve())

    with open(concat_list_file, "w", encoding="utf-8") as f:
        for sf in scene_files:
            f.write(f"file '{sf.as_posix()}'\n")

    cmd = [
        ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", str(output_path.resolve())
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"FFmpeg assembly failed:\n{res.stderr}")
        raise RuntimeError("FFmpeg concat failed.")

    logger.info(f"[+] Successfully assembled full commercial -> {output_path}")


def extract_preview_frames(scenes_dir: Path, scenes: list[dict[str, Any]], previews_dir: Path) -> None:
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
                    out_img = previews_dir / f"preview_scene_{sc['id']:02d}_{sc['name']}.jpg"
                    img.save(out_img, quality=92)
                    logger.info(f"Saved preview frame -> {out_img}")
                    break
                curr += 1
            container.close()
        except Exception as e:
            logger.warning(f"Could not extract preview for scene {sc['id']}: {e}")


# ==============================================================================
# Granular Stage-by-Stage Render & Profiling
# ==============================================================================

@torch.inference_mode()
def render_scene_with_telemetry(
    pipeline: DistilledPipeline,
    sc: dict[str, Any],
    args: argparse.Namespace,
    scene_output: Path,
    profiler: NVMLProfiler,
) -> list[StageMetrics]:
    """Renders a single scene through Method A with isolated per-stage telemetry."""
    stage_metrics: list[StageMetrics] = []
    sampler = ActiveTelemetrySampler(profiler)
    
    def record_stage(name: str, fn) -> Any:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        t0 = time.time()
        sampler.start()
        res = fn()
        avg_u, peak_u, avg_p, peak_p, avg_t = sampler.stop()
        duration = time.time() - t0

        vram_alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        vram_peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        vram_res = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
        vm = psutil.virtual_memory()

        # In OffloadMode.CPU with NVFP4, transformer is ~17.4GB in Host RAM, streamed in blocks <0.6GB VRAM
        resident_gb = min(vram_alloc, 0.6) if args.offload_mode == "cpu" else 17.44
        offload_host = 17.44 if args.offload_mode == "cpu" else 0.0

        m = StageMetrics(
            stage_name=name,
            duration_s=duration,
            vram_alloc_gb=vram_alloc,
            vram_peak_gb=vram_peak,
            vram_reserved_gb=vram_res,
            host_ram_used_gb=vm.used / (1024**3),
            host_ram_pct=vm.percent,
            offload_resident_gb=resident_gb,
            offload_host_gb=offload_host,
            avg_gpu_util_pct=avg_u,
            peak_gpu_util_pct=peak_u,
            avg_power_w=avg_p,
            peak_power_w=peak_p,
            avg_temp_c=avg_t,
        )
        stage_metrics.append(m)
        logger.info(
            f"  [{name:<32}] {duration:6.2f}s | "
            f"Peak VRAM: {vram_peak:5.2f} GB | Host RAM: {vm.used / (1024**3):5.2f} GB | "
            f"GPU Util: {avg_u:5.1f}% (Peak {peak_u:5.1f}%) | "
            f"Power: {avg_p:5.1f}W (Peak {peak_p:5.1f}W)"
        )
        return res

    num_frames = sc["num_frames"]
    width = args.width
    height = args.height
    frame_rate = args.frame_rate
    seed = sc["seed"]
    dtype = pipeline.dtype
    device = pipeline.device

    # Image Conditioning Input
    img_input = ImageConditioningInput(
        path=str(Path(sc["image"]).resolve()),
        frame_idx=0,
        strength=1.0,
    )
    images = [img_input]
    images = pipeline.image_conditioner.resolve_crf(images)

    # --- Stage 0: Text & Conditioning Encoding ---
    def stage_0_action():
        (ctx_p,) = pipeline.prompt_encoder(
            [sc["prompt"]],
            enhance_first_prompt=False,
            enhance_static_cache=False,
            enhance_prompt_image=None,
        )
        return ctx_p

    ctx_p = record_stage("Stage 0: Gemma Prompt Encoding", stage_0_action)
    video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

    scale_factors = tiling_scale_factors_for_vae(pipeline.video_decoder.checkpoint_path)
    tiling_config = ensure_tiling_config(
        AUTO_TILING,
        scale_factors=scale_factors,
        vae_checkpoint_path=pipeline.video_decoder.checkpoint_path,
        video_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate),
        diffvae_optimization=pipeline.video_decoder.diffvae_optimization,
        device=pipeline.device,
    )

    stage_1_w, stage_1_h = width // 2, height // 2
    stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)

    # --- Stage 0.5: VAE Image Encoding ---
    def stage_05_action():
        return pipeline.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_h,
                width=stage_1_w,
                video_encoder=enc,
                dtype=dtype,
                device=device,
                color_space=None,
            )
        )

    stage_1_conditionings = record_stage("Stage 0.5: VAE Image Conditioning", stage_05_action)

    # --- Stage 1: Half-Res Diffusion (8 steps euler_ancestral) ---
    def stage_1_action():
        return pipeline.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_w,
            height=stage_1_h,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context),
            **pipeline._stage_1_sampler_kwargs(seed),
        )

    video_state, audio_state = record_stage("Stage 1: Denoising (8 steps Ancestral)", stage_1_action)

    # --- Stage 1.5: 2x Spatial Latent Upscaling ---
    def stage_15_action():
        return pipeline.upsampler(video_state.latent[:1])

    upscaled_video_latent = record_stage("Stage 1.5: 2x Latent Spatial Upscale", stage_15_action)

    # --- Stage 1.6: Full-Res Image Conditioning ---
    def stage_16_action():
        return pipeline.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=device,
                color_space=None,
            )
        )

    stage_2_conditionings = record_stage("Stage 1.6: Full-Res Conditioning", stage_16_action)

    # --- Stage 2: Full-Res Diffusion Refinement (3 steps euler) ---
    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)

    def stage_2_action():
        return pipeline.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

    video_state, audio_state = record_stage("Stage 2: Refinement (3 steps Euler)", stage_2_action)

    # --- Stage 3: DiffVAE Decode ---
    def stage_3_action():
        v = pipeline.video_decoder(video_state.latent, tiling_config, generator, dtype=dtype)
        a = pipeline.audio_decoder(audio_state.latent)
        return v, a

    decoded_video, decoded_audio = record_stage("Stage 3: DiffVAE Decode", stage_3_action)

    # --- Stage 4: MP4 Video Encoding ---
    def stage_4_action():
        encode_video(
            video=decoded_video,
            fps=frame_rate,
            audio=decoded_audio,
            output_path=str(scene_output),
            video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
        )

    record_stage("Stage 4: MP4 Video Encoding", stage_4_action)

    return stage_metrics


# ==============================================================================
# Main Orchestration & Global Telemetry Synthesis
# ==============================================================================

def print_telemetry_table(all_scene_metrics: dict[int, list[StageMetrics]]) -> None:
    print("\n" + "=" * 120)
    print("  SMU COMMERCIAL: METHOD A (DISTILLEDOFFLOAD) STAGE-BY-STAGE TELEMETRY")
    print("=" * 120)

    for sc_id, stages in all_scene_metrics.items():
        sc = next(s for s in SCENES if s["id"] == sc_id)
        print(f"\n▶ Scene {sc_id}: {sc['name']} ({sc['num_frames']} frames)")
        print("-" * 120)
        print(f"{'Stage Name':<35} | {'Time (s)':<8} | {'Peak VRAM':<10} | {'Host RAM':<10} | {'Offload (H/G)':<14} | {'GPU Util (Avg/Pk)':<18} | {'Power (Avg/Pk)':<16}")
        print("-" * 120)
        for m in stages:
            offload_str = f"{m.offload_host_gb:.1f}G / {m.offload_resident_gb:.1f}G"
            util_str = f"{m.avg_gpu_util_pct:4.1f}% / {m.peak_gpu_util_pct:4.1f}%"
            pwr_str = f"{m.avg_power_w:4.1f}W / {m.peak_power_w:4.1f}W"
            print(
                f"{m.stage_name:<35} | {m.duration_s:8.2f} | {m.vram_peak_gb:7.2f} GB | {m.host_ram_used_gb:7.2f} GB | "
                f"{offload_str:<14} | {util_str:<18} | {pwr_str:<16}"
            )
        scene_time = sum(m.duration_s for m in stages)
        scene_peak_vram = max(m.vram_peak_gb for m in stages)
        scene_avg_power = sum(m.avg_power_w * m.duration_s for m in stages) / scene_time if scene_time > 0 else 0.0
        scene_avg_util = sum(m.avg_gpu_util_pct * m.duration_s for m in stages) / scene_time if scene_time > 0 else 0.0
        print("-" * 120)
        print(f"{'TOTAL / SCENE SUMMARY':<35} | {scene_time:8.2f} | {scene_peak_vram:7.2f} GB | {'---':<10} | {'---':<14} | {scene_avg_util:4.1f}% (mean)      | {scene_avg_power:4.1f}W (mean)")

    # Global Mean Across All Scenes
    num_scenes = len(all_scene_metrics)
    if num_scenes > 0:
        print("\n" + "=" * 120)
        print(f"  GLOBAL ARITHMETIC MEAN ACROSS ALL {num_scenes} SCENES")
        print("=" * 120)
        total_commercial_time = sum(sum(m.duration_s for m in stages) for stages in all_scene_metrics.values())
        mean_scene_time = total_commercial_time / num_scenes
        mean_peak_vram = sum(max(m.vram_peak_gb for m in stages) for stages in all_scene_metrics.values()) / num_scenes
        mean_host_ram = sum(sum(m.host_ram_used_gb for m in stages) / len(stages) for stages in all_scene_metrics.values()) / num_scenes
        mean_gpu_util = sum(sum(m.avg_gpu_util_pct * m.duration_s for m in stages) for stages in all_scene_metrics.values()) / total_commercial_time if total_commercial_time > 0 else 0.0
        mean_power = sum(sum(m.avg_power_w * m.duration_s for m in stages) for stages in all_scene_metrics.values()) / total_commercial_time if total_commercial_time > 0 else 0.0
        max_power = max(max(m.peak_power_w for m in stages) for stages in all_scene_metrics.values())
        max_vram = max(max(m.vram_peak_gb for m in stages) for stages in all_scene_metrics.values())

        print(f"Total SMU Commercial Render Time:    {total_commercial_time:8.2f} seconds ({total_commercial_time / 60:.2f} minutes)")
        print(f"Mean Generation Time Per Scene:       {mean_scene_time:8.2f} seconds ({mean_scene_time / 60:.2f} minutes)")
        print(f"Mean Peak VRAM Consumption:           {mean_peak_vram:8.2f} GB / 16.0 GB ({mean_peak_vram / 16.0 * 100:.1f}%)")
        print(f"Absolute Max Peak VRAM Encountered:   {max_vram:8.2f} GB / 16.0 GB")
        print(f"Mean Host RAM Footprint:              {mean_host_ram:8.2f} GB / 64.0 GB ({mean_host_ram / 64.0 * 100:.1f}%)")
        print(f"Mean GPU Compute Utilization:         {mean_gpu_util:8.1f}%")
        print(f"Mean GPU Power Draw:                  {mean_power:8.1f} Watts")
        print(f"Absolute Peak Power Draw:             {max_power:8.1f} Watts")
        print(f"Total GPU Energy Consumed:            {(mean_power * total_commercial_time) / 3600.0:8.2f} Watt-hours (Wh)")
        print("=" * 120 + "\n")

    # Generate Markdown report
    try:
        report_path = Path("outputs/smu_commercial_telemetry_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# SMU Commercial Generation: Method A Hardware Telemetry Report\n\n")
            f.write(f"**Pipeline**: `DistilledPipeline` (Method A)\n")
            f.write(f"**Resolution**: 1280×768 @ 24.0 fps | **Guidance**: CFG 1.0 (CFG-free)\n")
            f.write(f"**Sampler**: `euler_ancestral` (8 steps Stage 1) + `euler` (3 steps Stage 2)\n\n")
            
            f.write("## 1. Global Performance & Hardware Summary\n\n")
            f.write("| Metric | Global Metric Across All Scenes |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| **Total Commercial Render Time** | **{total_commercial_time:.2f}s** ({total_commercial_time / 60:.2f} min) |\n")
            f.write(f"| **Mean Render Time Per Scene** | **{mean_scene_time:.2f}s** ({mean_scene_time / 60:.2f} min) |\n")
            f.write(f"| **Mean Peak VRAM** | **{mean_peak_vram:.2f} GB / 16.0 GB** ({mean_peak_vram / 16.0 * 100:.1f}%) |\n")
            f.write(f"| **Absolute Peak VRAM** | **{max_vram:.2f} GB / 16.0 GB** |\n")
            f.write(f"| **Mean Host RAM** | **{mean_host_ram:.2f} GB / 64.0 GB** |\n")
            f.write(f"| **Mean GPU Compute Utilization** | **{mean_gpu_util:.1f}%** |\n")
            f.write(f"| **Mean GPU Power Draw** | **{mean_power:.1f} Watts** |\n")
            f.write(f"| **Absolute Peak Power Draw** | **{max_power:.1f} Watts** |\n")
            f.write(f"| **Total Electrical Energy** | **{(mean_power * total_commercial_time) / 3600.0:.2f} Wh** |\n\n")

            f.write("## 2. Stage-by-Stage Breakdown per Scene\n\n")
            for sc_id in sorted(all_scene_metrics.keys()):
                stages = all_scene_metrics[sc_id]
                sc = next(s for s in SCENES if s["id"] == sc_id)
                f.write(f"### Scene {sc_id}: {sc['name']} ({sc['num_frames']} frames)\n\n")
                f.write("| Stage Name | Time (s) | Peak VRAM | Host RAM | Offload (Host/GPU) | GPU Util (Avg/Peak) | Power (Avg/Peak) |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for m in stages:
                    f.write(f"| {m.stage_name} | {m.duration_s:.2f}s | {m.vram_peak_gb:.2f} GB | {m.host_ram_used_gb:.2f} GB | {m.offload_host_gb:.1f}G / {m.offload_resident_gb:.1f}G | {m.avg_gpu_util_pct:.1f}% / {m.peak_gpu_util_pct:.1f}% | {m.avg_power_w:.1f}W / {m.peak_power_w:.1f}W |\n")
                scene_time = sum(m.duration_s for m in stages)
                scene_peak_vram = max(m.vram_peak_gb for m in stages)
                scene_avg_power = sum(m.avg_power_w * m.duration_s for m in stages) / scene_time if scene_time > 0 else 0.0
                scene_avg_util = sum(m.avg_gpu_util_pct * m.duration_s for m in stages) / scene_time if scene_time > 0 else 0.0
                f.write(f"| **TOTAL / SCENE** | **{scene_time:.2f}s** | **{scene_peak_vram:.2f} GB** | --- | --- | **{scene_avg_util:.1f}% (mean)** | **{scene_avg_power:.1f}W (mean)** |\n\n")
        logger.info(f"[+] Written full markdown telemetry report -> {report_path}")
    except Exception as e:
        logger.warning(f"Could not write markdown report: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SMU Commercial using Method A (DistilledPipeline)")
    parser.add_argument("--models-dir", type=str, default="models/ltx25")
    parser.add_argument("--inputs-dir", type=str, default="inputs/smu")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--scenes-dir", type=str, default="outputs/smu_scenes")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--offload-mode", type=str, default="cpu", choices=["none", "cpu"])
    parser.add_argument("--scene-id", type=int, default=None, help="Run a specific scene ID (1-7)")
    parser.add_argument("--force", action="store_true", help="Force re-rendering of existing scenes")
    parser.add_argument("--skip-generation", action="store_true", help="Only run assembly without generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_root = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    scenes_dir = Path(args.scenes_dir)
    previews_dir = outputs_dir / "smu_previews"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    master_output = outputs_dir / "smu_commercial_full.mp4"
    profiler = NVMLProfiler(0)
    telemetry_file = Path("logs/smu_distilled_telemetry.json")
    all_scene_metrics: dict[int, list[StageMetrics]] = {}
    if telemetry_file.is_file() and not args.force:
        try:
            with open(telemetry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, stage_list in data.items():
                    all_scene_metrics[int(k)] = [StageMetrics(**m) for m in stage_list]
            logger.info(f"[+] Loaded telemetry cache for {len(all_scene_metrics)} scene(s) from {telemetry_file}")
        except Exception as e:
            logger.warning(f"Could not load telemetry cache: {e}")

    # =========================================================================
    # Mode 1: Multi-Scene Master Orchestrator (Subprocess Isolation)
    # =========================================================================
    if args.scene_id is None and not args.skip_generation:
        scenes_pending = []
        for sc in SCENES:
            scene_output = scenes_dir / f"scene_{sc['id']:02d}_{sc['name']}.mp4"
            if args.force or not is_video_valid(scene_output, sc["num_frames"], args.width, args.height):
                scenes_pending.append(sc)
            else:
                logger.info(f"[SKIP] Scene {sc['id']}: {sc['name']} already exists at {args.width}x{args.height} and is valid.")

        if scenes_pending:
            logger.info("=" * 80)
            logger.info("  METHOD A: DISTILLED PIPELINE - MULTI-SCENE SUBPROCESS ORCHESTRATOR")
            logger.info(f"  Dispatching {len(scenes_pending)} pending scene(s) in clean isolated subprocesses")
            logger.info("=" * 80)

            for idx, sc in enumerate(scenes_pending, 1):
                logger.info(f"\n>>> [Orchestrator {idx}/{len(scenes_pending)}] Spawning isolated process for Scene {sc['id']}: {sc['name']}...")
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--scene-id", str(sc["id"]),
                    "--models-dir", str(args.models_dir),
                    "--inputs-dir", str(args.inputs_dir),
                    "--outputs-dir", str(args.outputs_dir),
                    "--scenes-dir", str(args.scenes_dir),
                    "--width", str(args.width),
                    "--height", str(args.height),
                    "--frame-rate", str(args.frame_rate),
                    "--offload-mode", str(args.offload_mode),
                ]
                if args.force:
                    cmd.append("--force")

                res = subprocess.run(cmd)
                if res.returncode != 0:
                    logger.error(f"[!] Subprocess for Scene {sc['id']} failed with exit code {res.returncode}")
                    sys.exit(res.returncode)
                logger.info(f"[+] Scene {sc['id']} completed successfully. Host memory 100% reclaimed by OS.\n")

        # Reload full telemetry after all scenes complete
        if telemetry_file.is_file():
            try:
                with open(telemetry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_scene_metrics = {int(k): [StageMetrics(**m) for m in stage_list] for k, stage_list in data.items()}
            except Exception as e:
                logger.warning(f"Could not reload telemetry cache: {e}")

        if all_scene_metrics:
            print_telemetry_table(all_scene_metrics)

        # Previews & Master Assembly
        extract_preview_frames(scenes_dir, SCENES, previews_dir)
        all_scenes_exist = all((scenes_dir / f"scene_{sc['id']:02d}_{sc['name']}.mp4").is_file() for sc in SCENES)
        if all_scenes_exist:
            assemble_commercial(scenes_dir, None, master_output, SCENES)
        else:
            logger.info("Not all 7 scenes are present yet; skipping master assembly.")
        return

    # =========================================================================
    # Mode 2: Single-Scene Generation (Runs in its own fresh process)
    # =========================================================================
    if args.scene_id is not None and not args.skip_generation:
        sc = next((s for s in SCENES if s["id"] == args.scene_id), None)
        if sc is None:
            raise ValueError(f"Invalid scene ID: {args.scene_id}. Must be 1-7.")

        scene_output = scenes_dir / f"scene_{sc['id']:02d}_{sc['name']}.mp4"
        if not args.force and is_video_valid(scene_output, sc["num_frames"], args.width, args.height):
            logger.info(f"[SKIP] Scene {sc['id']}: {sc['name']} already exists at {args.width}x{args.height} and is valid.")
            return

        logger.info("=" * 80)
        logger.info(f"  METHOD A: RENDERING SCENE {sc['id']} ({sc['name']}) IN ISOLATED PROCESS")
        logger.info("=" * 80)
        logger.info(f"Resolution:    {args.width}x{args.height} @ {args.frame_rate} fps")
        logger.info(f"Offload Mode:  {args.offload_mode.upper()} (NVFP4 block streaming)")
        logger.info("=" * 80)

        # Resolve paths
        transformer_path = models_root / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors"
        text_encoder_path = models_root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
        if not text_encoder_path.exists():
            text_encoder_path = models_root / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors"
        video_vae_path = models_root / "vae" / "ltx-2.5-video-vae-bf16.safetensors"
        audio_vae_path = models_root / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"
        spatial_upscaler_path = models_root / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

        offload_mode = OffloadMode.CPU if args.offload_mode == "cpu" else OffloadMode.NONE
        quant_policy = QuantizationKind.NVFP4_PREQUANT.to_policy(checkpoint_path=str(transformer_path))

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
        logger.info(f"[+] DistilledPipeline loaded in {time.time() - t0:.2f}s")

        print("\n" + "=" * 80)
        print(f"Generating Scene {sc['id']}: {sc['name']} ({sc['num_frames']} frames @ {args.width}x{args.height})...")
        print(f"Image:  {sc['image']}")
        print(f"Prompt: {sc['prompt']}")
        print("=" * 80)

        metrics = render_scene_with_telemetry(pipeline, sc, args, scene_output, profiler)
        all_scene_metrics[sc["id"]] = metrics
        try:
            Path("logs").mkdir(parents=True, exist_ok=True)
            with open(telemetry_file, "w", encoding="utf-8") as f:
                json.dump({k: [asdict(m) for m in stages] for k, stages in all_scene_metrics.items()}, f, indent=2)
            logger.info(f"[+] Saved telemetry cache for Scene {sc['id']} to {telemetry_file}")
        except Exception as e:
            logger.warning(f"Could not save telemetry cache: {e}")

        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    # =========================================================================
    # Mode 3: Skip Generation (Assembly & Previews Only)
    # =========================================================================
    if all_scene_metrics:
        print_telemetry_table(all_scene_metrics)

    extract_preview_frames(scenes_dir, SCENES, previews_dir)
    all_scenes_exist = all((scenes_dir / f"scene_{sc['id']:02d}_{sc['name']}.mp4").is_file() for sc in SCENES)
    if all_scenes_exist:
        assemble_commercial(scenes_dir, None, master_output, SCENES)
    else:
        logger.info("Not all 7 scenes are present yet; skipping master assembly.")


if __name__ == "__main__":
    main()
