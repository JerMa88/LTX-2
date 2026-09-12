"""
Download ltx-2.5-22b-dev-transformer-bf16.safetensors and distilled LoRA
from Lightricks/LTX-2.5 using huggingface_hub with aria2c acceleration.

Usage:
    python scripts/download_dev_transformer.py
    python scripts/download_dev_transformer.py --token hf_YOURTOKEN
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "models" / "ltx25"

FILES_TO_DOWNLOAD = [
    {
        "repo_path": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "local_dir": MODELS_ROOT / "diffusion_models",
        "size_hint": "~42 GB",
    },
    {
        "repo_path": "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
        "local_dir": MODELS_ROOT / "loras",
        "size_hint": "~2.5 GB",
    },
]

REPO_ID = "Lightricks/LTX-2.5"


def get_hf_token(cli_token: str | None) -> str | None:
    """Resolve HuggingFace token from CLI arg, env var, or cached login."""
    if cli_token:
        return cli_token
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env:
        return env
    # Check huggingface_hub cached token
    try:
        from huggingface_hub import HfFolder  # noqa: PLC0415
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass
    return None


def download_with_hf_hub(repo_path: str, local_dir: Path, token: str | None) -> Path:
    """Download a single file using huggingface_hub (supports resuming)."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    filename = Path(repo_path).name
    subfolder = str(Path(repo_path).parent) if Path(repo_path).parent != Path(".") else None

    print(f"\n[+] Downloading: {repo_path}", flush=True)
    print(f"    -> Destination: {local_dir / filename}", flush=True)

    local_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        subfolder=subfolder if subfolder and subfolder != "." else None,
        local_dir=str(local_dir),
        token=token,
    )
    elapsed = time.time() - t0
    size_bytes = Path(path).stat().st_size
    size_gb = size_bytes / (1024 ** 3)
    speed_mbps = (size_bytes / (1024 ** 2)) / elapsed if elapsed > 0 else 0
    print(f"[+] Downloaded {size_gb:.2f} GB in {elapsed/60:.1f} min ({speed_mbps:.1f} MB/s)", flush=True)
    return Path(path)



def main() -> None:
    parser = argparse.ArgumentParser(description="Download LTX-2.5 dev transformer + distilled LoRA")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace API token (hf_...). If omitted, uses HF_TOKEN env var or cached login.")
    parser.add_argument("--only-lora", action="store_true",
                        help="Only download the distilled LoRA (skip transformer)")
    parser.add_argument("--only-transformer", action="store_true",
                        help="Only download the transformer (skip LoRA)")
    args = parser.parse_args()

    token = get_hf_token(args.token)
    if not token:
        print("\n[!] No HuggingFace token found. Attempting unauthenticated download...", flush=True)
        print("    (If this fails, provide one via: --token hf_YOUR_TOKEN or set HF_TOKEN env var)", flush=True)

    files = FILES_TO_DOWNLOAD
    if args.only_lora:
        files = [f for f in files if "lora" in f["repo_path"]]
    elif args.only_transformer:
        files = [f for f in files if "transformer" in f["repo_path"]]

    total_start = time.time()
    for item in files:
        dest = item["local_dir"] / Path(item["repo_path"]).name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"[SKIP] Already exists: {dest} ({dest.stat().st_size / (1024**3):.2f} GB)", flush=True)
            continue
        print(f"\n{'='*60}", flush=True)
        print(f"File: {item['repo_path']}  ({item['size_hint']})", flush=True)
        print(f"{'='*60}", flush=True)
        download_with_hf_hub(item["repo_path"], item["local_dir"], token)

    total_elapsed = time.time() - total_start
    print(f"\n[✓] All downloads complete in {total_elapsed/60:.1f} minutes.", flush=True)
    print(f"    Transformer: {MODELS_ROOT / 'diffusion_models' / 'ltx-2.5-22b-dev-transformer-bf16.safetensors'}", flush=True)
    print(f"    LoRA:        {MODELS_ROOT / 'loras' / 'ltx-2.5-22b-distilled-lora-450-bf16.safetensors'}", flush=True)


if __name__ == "__main__":
    main()
