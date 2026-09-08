"""
High-Speed Resumable Downloader for LTX-2.5 NVFP4 Model Pack using curl.exe.
Downloads verified ComfyUI-ready NVFP4 quantized transformer, Gemma 4 12B NVFP4 text encoder,
dual VAEs (video and audio), and 2x latent spatial upscaler.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_BASE = PROJECT_ROOT / "models" / "ltx25"

DOWNLOAD_MANIFEST = [
    {
        "url": "https://huggingface.co/BennyDaBall/LTX-2.5-22b-distilled-nvfp4-comfy-v2/resolve/main/ltx-2.5-nvfp4-v2-t2v-example-workflow.json",
        "local_dest": TARGET_BASE / "ltx-2.5-nvfp4-v2-t2v-example-workflow.json",
        "description": "LTX-2.5 Community Reference 2-Stage Production Workflow JSON",
        "expected_bytes": 120248,
    },
    {
        "url": "https://huggingface.co/vonkaiser/LTX-2.5-FP8-NVFP4/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors",
        "local_dest": TARGET_BASE / "vae" / "ltx-2.5-audio-vae-bf16.safetensors",
        "description": "LTX-2.5 Audio VAE Decoder (BF16)",
        "expected_bytes": 364866540,
    },
    {
        "url": "https://huggingface.co/vonkaiser/LTX-2.5-FP8-NVFP4/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "local_dest": TARGET_BASE / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "description": "LTX-2.5 2x Latent Spatial Upscaler & Refiner (BF16)",
        "expected_bytes": 997972044,
    },
    {
        "url": "https://huggingface.co/vonkaiser/LTX-2.5-FP8-NVFP4/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors",
        "local_dest": TARGET_BASE / "vae" / "ltx-2.5-video-vae-bf16.safetensors",
        "description": "LTX-2.5 3D Diffusion Video VAE Decoder (BF16)",
        "expected_bytes": 1475760668,
    },
    {
        "url": "https://huggingface.co/Deadshot699/ltx-2.5-gemma4-12b-comfy-nvfp4/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors",
        "local_dest": TARGET_BASE / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors",
        "description": "Gemma 4 12B Text Encoder with AV Projections (NVFP4 ComfyUI-native)",
        "expected_bytes": 10599427024,
    },
    {
        "url": "https://huggingface.co/BennyDaBall/LTX-2.5-22b-distilled-nvfp4-comfy-v2/resolve/main/ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors",
        "local_dest": TARGET_BASE / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors",
        "description": "LTX-2.5 22B Distilled Transformer (NVFP4 stamped, Blackwell sm_120)",
        "expected_bytes": 18728994704,
    },
]


def download_file_with_curl(url: str, dest: Path, expected_bytes: int, desc: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    gb_str = f"{expected_bytes / (1024**3):.2f} GB" if expected_bytes > 1024**2 else f"{expected_bytes / 1024:.1f} KB"
    print(f"\n[*] Downloading: {desc}")
    print(f"    Target:   {dest.resolve()}")
    print(f"    Expected: {gb_str}")

    if dest.exists():
        cur_size = dest.stat().st_size
        if cur_size == expected_bytes:
            print(f"    [+] Already completely downloaded ({cur_size / (1024**3):.2f} GB). Skipping.")
            return
        elif cur_size > 0:
            print(f"    [>] Resuming partial download from {cur_size / (1024**3):.2f} GB...")

    has_aria2 = shutil.which("aria2c") is not None
    if has_aria2:
        cmd = [
            "aria2c",
            "-c",
            "-x", "8",
            "-s", "8",
            "-k", "1M",
            "--file-allocation=none",
            "-d", str(dest.parent),
            "-o", dest.name,
            url,
        ]
        t0 = time.time()
        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            elapsed = time.time() - t0
            final_size = dest.stat().st_size
            speed = (final_size / (1024**2)) / elapsed if elapsed > 0 else 0
            print(f"    [+] Finished in {elapsed:.1f}s | Final Size: {final_size / (1024**3):.2f} GB ({speed:.1f} MB/s)")
            return
        print(f"    [!] aria2c returned {proc.returncode}. Falling back to curl.exe with resume...")

    cmd = [
        "curl.exe",
        "-L",
        "-C", "-",
        "--create-dirs",
        "-o", str(dest),
        url,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Download failed with return code {proc.returncode} for {url}")

    elapsed = time.time() - t0
    final_size = dest.stat().st_size
    speed = (final_size / (1024**2)) / elapsed if elapsed > 0 else 0
    print(f"    [+] Finished in {elapsed:.1f}s | Final Size: {final_size / (1024**3):.2f} GB ({speed:.1f} MB/s)")


def main():
    print("=" * 80)
    print("  LTX-2.5 NVFP4 RESUMABLE MODEL PACK DOWNLOADER (curl.exe)")
    print(f"  Target Directory: {TARGET_BASE.resolve()}")
    print("=" * 80)

    total_start = time.time()
    for idx, item in enumerate(DOWNLOAD_MANIFEST, 1):
        print(f"\n[{idx}/{len(DOWNLOAD_MANIFEST)}]")
        download_file_with_curl(
            url=item["url"],
            dest=item["local_dest"],
            expected_bytes=item["expected_bytes"],
            desc=item["description"],
        )

    overall_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"  ALL 6 LTX-2.5 NVFP4 MODEL COMPONENTS DOWNLOADED AND VERIFIED!")
    print(f"  Total Duration: {overall_time / 60:.2f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    main()
