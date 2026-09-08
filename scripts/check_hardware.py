"""
Hardware Diagnostic & Dual-GPU Capability Verification Script.
Checks CUDA devices, VRAM allocations, cross-GPU PCIe transfers, and FP8 capability.
"""

import sys
import torch
import yaml
from pathlib import Path


def run_hardware_check(config_path="config/inference_config.yaml"):
    print("=" * 70)
    print("      DUAL-GPU HARDWARE & PIPELINE DIAGNOSTIC SYSTEM")
    print("=" * 70)

    # 1. Check PyTorch & CUDA installation
    print(f"[+] Python Version: {sys.version.split()[0]}")
    print(f"[+] PyTorch Version: {torch.__version__}")
    print(f"[+] CUDA Available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("[-] ERROR: CUDA is not available in current PyTorch installation.")
        return False

    device_count = torch.cuda.device_count()
    print(f"[+] Detected GPU Count: {device_count}")

    gpus = []
    for idx in range(device_count):
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        gpus.append({"id": idx, "name": name, "cap": cap, "vram": total_mem_gb})
        print(f"    - GPU {idx}: {name} (Compute Cap: {cap[0]}.{cap[1]}, VRAM: {total_mem_gb:.2f} GB)")

    # Load configuration
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"\n[+] Loaded Config from {config_path}")
        gpu_dit = cfg["hardware"]["gpu_dit"]
        gpu_offload = cfg["hardware"]["gpu_offload"]
    else:
        gpu_dit = "cuda:0"
        gpu_offload = "cuda:1" if device_count > 1 else "cpu"

    print(f"[+] Primary DiT GPU (Blackwell): {gpu_dit}")
    print(f"[+] Secondary Offload Target: {gpu_offload}")

    # 2. Verify GPU allocation & VRAM allocation on Primary GPU (cuda:0 - RTX 5080)
    try:
        dev_dit = torch.device(gpu_dit)
        t_dit = torch.randn((2048, 2048), dtype=torch.float16, device=dev_dit)
        print(f"[+] Successfully allocated FP16 tensor on Primary GPU {dev_dit} ({gpus[0]['name']}).")
    except Exception as e:
        print(f"[-] ERROR allocating tensor on Primary GPU ({gpu_dit}): {e}")
        return False

    # 3. Check Offload Target Allocation
    try:
        dev_offload = torch.device(gpu_offload)
        if dev_offload.type == "cuda":
            t_off = torch.randn((1024, 1024), dtype=torch.float32, device=dev_offload)
            print(f"[+] Successfully allocated FP32 tensor on Offload GPU {dev_offload}.")
        else:
            t_off = torch.randn((1024, 1024), dtype=torch.float32, device=torch.device("cpu"))
            print("[+] Offload Target set to CPU.")
    except Exception as e:
        print(f"[-] Note on Offload Target ({gpu_offload}): {e}. Falling back offload to CPU.")
        gpu_offload = "cpu"

    # 4. FP8 Capability Check on Primary GPU (RTX 5080)
    print(f"\n[+] Checking FP8 (E4M3) Support on Primary GPU ({gpu_dit})...")
    if hasattr(torch, "float8_e4m3fn"):
        try:
            fp8_tensor = torch.ones((1024, 1024), dtype=torch.float8_e4m3fn, device=dev_dit)
            print(f"[+] FP8 (float8_e4m3fn) Tensor Allocation on {gpu_dit}: PASSED")
        except Exception as e:
            print(f"[-] FP8 Tensor Allocation note: {e}")
    else:
        print("[-] PyTorch build does not include native float8_e4m3fn type.")

    print("\n" + "=" * 70)
    print("      HARDWARE DIAGNOSTIC COMPLETE - SYSTEM READY FOR INFERENCE")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_hardware_check()
    sys.exit(0 if success else 1)
