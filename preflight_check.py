"""
Pre-Flight Diagnostic and System Readiness Check for RTX 5080 LTX-2.5 NVFP4.
Validates hardware, drivers, NVFP4 kernels, model checkpoints, host RAM,
and offload memory allocation.
"""

import os
import sys
from pathlib import Path
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "ltx25"

def print_header(title: str) -> None:
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)

def check_system_memory() -> bool:
    print_header("1. HOST SYSTEM RAM & PAGEFILE VERIFICATION")
    
    total_ram_gb = 0.0
    avail_ram_gb = 0.0
    total_pagefile_gb = 0.0
    free_pagefile_gb = 0.0

    if sys.platform == "win32":
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_ram_gb = stat.ullTotalPhys / (1024 ** 3)
            avail_ram_gb = stat.ullAvailPhys / (1024 ** 3)
            total_pagefile_gb = stat.ullTotalPageFile / (1024 ** 3)
            free_pagefile_gb = stat.ullAvailPageFile / (1024 ** 3)
    
    if total_ram_gb == 0.0:
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024 ** 3)
        avail_ram_gb = vm.available / (1024 ** 3)

    print(f"[+] Total System RAM:        {total_ram_gb:.2f} GB")
    print(f"[+] Available System RAM:    {avail_ram_gb:.2f} GB")
    if total_pagefile_gb > 0:
        print(f"[+] Total Commit Limit:      {total_pagefile_gb:.2f} GB (RAM + Pagefile)")
        print(f"[+] Available Commit Limit:  {free_pagefile_gb:.2f} GB")

    passed = True
    if total_ram_gb < 60.0:
        print(f"[-] WARNING: Total host RAM is {total_ram_gb:.1f} GB (< 64 GB).")
        passed = False
    else:
        print("[+] Host RAM verification:    PASSED (64 GB class)")

    # Pinned memory allocation test (test allocating 2 GB pinned buffer to verify WDDM behavior)
    try:
        t_pinned = torch.empty((1024, 1024, 512), dtype=torch.float32, pin_memory=True)
        print("[+] Pinned CPU memory test:   PASSED (2 GB pinned buffer allocated)")
        del t_pinned
    except Exception as e:
        print(f"[-] WARNING: Pinned CPU memory allocation failed: {e}")
        passed = False

    return passed

def check_gpu() -> bool:
    print_header("2. GPU HARDWARE & CUDA VERIFICATION")
    print(f"[+] PyTorch Version:   {torch.__version__}")
    print(f"[+] CUDA Available:    {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("[-] FATAL: CUDA is not available in PyTorch!")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)

    print(f"[+] Primary Device:    {gpu_name}")
    print(f"[+] Compute Capability: SM {cap[0]}.{cap[1]}")
    print(f"[+] Total VRAM:        {vram_gb:.2f} GB")

    passed = True
    if cap[0] < 10:
        print(f"[-] ERROR: Compute capability {cap[0]}.{cap[1]} < 10.0. Blackwell required for NVFP4.")
        passed = False
    else:
        print(f"[+] Blackwell Architecture Verified: PASSED (SM {cap[0]}.{cap[1]})")

    # Verify tensor allocation on GPU
    try:
        t = torch.zeros((1024, 1024), dtype=torch.bfloat16, device="cuda:0")
        print("[+] BF16 GPU Allocation: PASSED")
        del t
    except Exception as e:
        print(f"[-] ERROR allocating BF16 tensor on GPU: {e}")
        passed = False

    # Check FP8 datatype support
    if hasattr(torch, "float8_e4m3fn"):
        try:
            fp8_t = torch.zeros((512, 512), dtype=torch.float8_e4m3fn, device="cuda:0")
            print("[+] FP8 (float8_e4m3fn) Tensor Allocation: PASSED")
            del fp8_t
        except Exception as e:
            print(f"[-] WARNING: FP8 allocation failed: {e}")
    else:
        print("[-] WARNING: torch.float8_e4m3fn not present in PyTorch build.")

    return passed

def check_nvfp4_kernels() -> bool:
    print_header("3. NVFP4 COMPILED KERNEL VERIFICATION")
    try:
        import ltx_kernels.nvfp4 as nvfp4
        reason = nvfp4.unavailable_reason()
        if reason is not None:
            print(f"[-] NVFP4 Kernels Unavailable: {reason}")
            return False
        
        print("[+] ltx_kernels.nvfp4 is importable and available!")
        
        # Test NVFP4 GEMM support probe if available
        if hasattr(nvfp4, "probe_gemm_support"):
            probe = nvfp4.probe_gemm_support(128, 128, 128, 0)
            print(f"[+] GEMM Support Probe: {probe}")
        
        return True
    except ImportError as e:
        print(f"[-] ltx_kernels.nvfp4 is NOT installed: {e}")
        print("    Run: uv pip install -e packages/ltx-kernels --no-build-isolation")
        return False
    except Exception as e:
        print(f"[-] Error testing NVFP4 kernels: {e}")
        return False

def check_checkpoints() -> bool:
    print_header("4. MODEL CHECKPOINTS VERIFICATION")
    expected_files = [
        ("Diffusion Transformer (NVFP4)", MODELS_DIR / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors", 18721741072),
        ("Gemma 4 12B Text Encoder (NVFP4)", MODELS_DIR / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-nvfp4.safetensors", 10595300646),
        ("Video VAE Decoder (BF16)", MODELS_DIR / "vae" / "ltx-2.5-video-vae-bf16.safetensors", 1472223346),
        ("Audio VAE Decoder (BF16)", MODELS_DIR / "vae" / "ltx-2.5-audio-vae-bf16.safetensors", 364866540),
        ("Spatial Upscaler x2 (BF16)", MODELS_DIR / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", 995778752),
    ]

    all_present = True
    for desc, path, expected_bytes in expected_files:
        if not path.exists():
            print(f"[-] MISSING:  {desc}")
            print(f"              Path: {path}")
            all_present = False
        else:
            actual_bytes = path.stat().st_size
            pct = (actual_bytes / expected_bytes) * 100
            gb = actual_bytes / (1024 ** 3)
            if actual_bytes == expected_bytes:
                print(f"[+] VERIFIED: {desc} ({gb:.2f} GB)")
            else:
                print(f"[~] PARTIAL:  {desc} ({gb:.2f} GB / {pct:.1f}%)")
                all_present = False

    return all_present

def main() -> int:
    print("=" * 75)
    print("      LTX-2.5 NVFP4 RTX 5080 PRE-FLIGHT SYSTEM READINESS CHECK")
    print("=" * 75)

    mem_ok = check_system_memory()
    gpu_ok = check_gpu()
    kernel_ok = check_nvfp4_kernels()
    models_ok = check_checkpoints()

    print_header("PRE-FLIGHT SUMMARY")
    print(f"  System RAM & Pinned Memory: {'PASSED' if mem_ok else 'FAILED / WARNING'}")
    print(f"  GPU Hardware & CUDA:        {'PASSED' if gpu_ok else 'FAILED'}")
    print(f"  NVFP4 Compiled Kernels:     {'PASSED' if kernel_ok else 'PENDING COMPILATION'}")
    print(f"  Model Checkpoints:          {'PASSED' if models_ok else 'DOWNLOADING / INCOMPLETE'}")
    print("=" * 75)

    if mem_ok and gpu_ok and kernel_ok and models_ok:
        print("\n>>> ALL PRE-FLIGHT CHECKS PASSED! Ready for video generation.\n")
        return 0
    else:
        print("\n>>> Pre-flight checks incomplete. Complete remaining prerequisites above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
