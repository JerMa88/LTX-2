"""
Quantize ltx-2.5-22b-dev-transformer-bf16.safetensors to native Blackwell NVFP4.
Uses the compiled RTX 5080 ltx_kernels.nvfp4 extension for ultra-fast GPU quantization.

Produces a full foundation model checkpoint in NVFP4 format (~12.5 GB),
eliminating both the 40GB BF16 load bottleneck and memory paging thrashing.
"""

from __future__ import annotations

import argparse
import gc
import re
import time
from pathlib import Path

import safetensors.torch as st
import torch

from ltx_core.quantization.nvfp4.convert import (
    _LTX2_FV_NVFP4_SUFFIXES,
    quantize_bf16_weight_to_nvfp4,
)

# Match allowlisted linear weights regardless of prefix (e.g. model.diffusion_model.transformer_blocks.X...)
_SUFFIX_PATTERN = re.compile(
    r"\.transformer_blocks\.\d+(?:\._orig_mod)?\.(" + "|".join(re.escape(s) for s in _LTX2_FV_NVFP4_SUFFIXES) + r")\.weight$"
)


def is_nvfp4_target(key: str) -> bool:
    return _SUFFIX_PATTERN.search(key) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize LTX-2.5 Dev Transformer to NVFP4")
    parser.add_argument(
        "--src",
        type=str,
        default="models/ltx25/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        help="Path to source BF16 safetensors file",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="models/ltx25/diffusion_models/ltx-2.5-22b-dev-transformer-nvfp4.safetensors",
        help="Path to destination NVFP4 safetensors file",
    )
    parser.add_argument("--batch-size", type=int, default=12, help="Number of layers to process before CUDA sync/GC")
    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_path}")

    print("=" * 80)
    print("  LTX-2.5 22B DEV TRANSFORMER -> BLACKWELL NVFP4 QUANTIZER")
    print("=" * 80)
    print(f"Source Checkpoint: {src_path} ({src_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"Target Checkpoint: {dst_path}")
    print(f"CUDA Device:       {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    t0 = time.time()
    f = st.safe_open(str(src_path), framework="pt")
    raw_metadata = f.metadata() or {}
    keys = list(f.keys())
    print(f"[*] Total tensors in source checkpoint: {len(keys)}")

    target_keys = [k for k in keys if is_nvfp4_target(k)]
    print(f"[*] Identified {len(target_keys)} allowlisted Linear weights to quantize to NVFP4.")

    output_tensors: dict[str, torch.Tensor] = {}
    quantized_count = 0
    passthrough_count = 0

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for i, k in enumerate(keys):
            if is_nvfp4_target(k):
                tensor_bf16 = f.get_tensor(k)
                if tensor_bf16.dim() == 2 and tensor_bf16.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    w_fp4, w_scale, decode = quantize_bf16_weight_to_nvfp4(tensor_bf16.cuda())

                    base = k[: -len(".weight")]
                    output_tensors[k] = w_fp4.cpu()
                    # Store weight_scale as F8_E4M3 for safetensors spec compatibility
                    output_tensors[f"{base}.weight_scale"] = w_scale.view(torch.float8_e4m3fn).cpu()
                    output_tensors[f"{base}.weight_scale_2"] = decode.to(dtype=torch.float32, device="cpu").reshape(())
                    # Provide 1.0 input_scale so both ActScale.STATIC and ActScale.FIXED_1 work seamlessly
                    output_tensors[f"{base}.input_scale"] = torch.tensor(1.0, dtype=torch.float32)

                    quantized_count += 1
                    if quantized_count % 48 == 0 or quantized_count == len(target_keys):
                        elapsed = time.time() - t0
                        rate = quantized_count / elapsed
                        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
                        print(
                            f"    [Quantized {quantized_count:3d}/{len(target_keys)}] "
                            f"({elapsed:.1f}s, {rate:.1f} layers/s, Peak VRAM: {peak_vram:.1f} MB)"
                        )
                else:
                    output_tensors[k] = tensor_bf16
                    passthrough_count += 1
            else:
                output_tensors[k] = f.get_tensor(k)
                passthrough_count += 1

            if i % args.batch_size == 0:
                gc.collect()
                torch.cuda.empty_cache()

    quantize_elapsed = time.time() - t0
    print(f"\n[+] Quantization completed in {quantize_elapsed:.2f}s.")
    print(f"[*] Saving {len(output_tensors)} tensors to {dst_path}...")

    save_t0 = time.time()
    st.save_file(output_tensors, str(dst_path), metadata=raw_metadata)
    save_elapsed = time.time() - save_t0

    total_elapsed = time.time() - t0
    dst_size_gb = dst_path.stat().st_size / (1024**3)

    print("=" * 80)
    print("  QUANTIZATION COMPLETE")
    print("=" * 80)
    print(f"Output File:       {dst_path}")
    print(f"File Size:         {dst_size_gb:.2f} GB (compressed from {src_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"Quantized Layers:  {quantized_count}")
    print(f"Passthrough Tensors: {passthrough_count}")
    print(f"Total Time:        {total_elapsed:.2f}s (Quant: {quantize_elapsed:.2f}s, Save: {save_elapsed:.2f}s)")
    print("=" * 80)


if __name__ == "__main__":
    main()
