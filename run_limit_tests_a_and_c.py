"""
Runner script to execute Method A followed by Method C for the 10-second limit test (241 frames @ 24 fps, 1280x768).
Runs each method in its own subprocess to ensure complete GPU/VRAM reclamation between runs.
Fully UTF-8 hardened to avoid Windows charmap encoding errors.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "limit_test_a_and_c.log"

PROMPT = (
    "Cinematic macro slow-motion footage of a vibrant hummingbird hovering near exotic flowers, "
    "rhythmic wing beats, crisp high-shutter detail, golden sunlight"
)
SEED = "42"
NUM_FRAMES = "241"
WIDTH = "1280"
HEIGHT = "768"

def safe_print(text: str):
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except Exception:
        # Fallback to buffer write with replacement
        try:
            sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        except Exception:
            pass

def log(msg: str):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}\n"
    safe_print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            f.write(line)
    except Exception:
        pass

def run_cmd(cmd: list[str], label: str) -> bool:
    log("=" * 80)
    log(f"STARTING {label}")
    log(f"Command: {' '.join(cmd)}")
    log("=" * 80)
    t0 = time.time()
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    
    with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
        for out_line in proc.stdout:
            safe_print(out_line)
            f.write(out_line)
            f.flush()
        proc.wait()
    
    duration = time.time() - t0
    if proc.returncode == 0:
        log(f"[+] {label} COMPLETED SUCCESSFULLY in {duration:.2f}s ({duration / 60:.2f} min)")
        return True
    else:
        log(f"[-] {label} FAILED with exit code {proc.returncode} after {duration:.2f}s")
        return False

def main():
    log("=" * 80)
    log("STARTING SEQUENTIAL 10-SECOND LIMIT BENCHMARK: METHOD A -> METHOD C")
    log(f"Prompt:     {PROMPT}")
    log(f"Resolution: {WIDTH}x{HEIGHT} | Frames: {NUM_FRAMES} (~10.04s @ 24fps) | Seed: {SEED}")
    log("=" * 80)

    # 1. Method A: DistilledPipeline (Euler Ancestral, CFG 1.0, 11 evals)
    cmd_a = [
        str(PYTHON_EXE),
        "run_method_a_distilled.py",
        "--prompt", PROMPT,
        "--seed", SEED,
        "--num-frames", NUM_FRAMES,
        "--width", WIDTH,
        "--height", HEIGHT,
        "--output", "outputs/method_a_limit_test_10s_241f.mp4"
    ]
    ok_a = run_cmd(cmd_a, "METHOD A (Official Distilled Pipeline)")

    log("\nWaiting 10 seconds before starting Method C for cooldown...")
    time.sleep(10)

    # 2. Method C: Tuned Two-Stage HQ Pipeline (res2s, CFG 1.0, 15+3 steps)
    cmd_c = [
        str(PYTHON_EXE),
        "run_rtx5080.py",
        "--prompt", PROMPT,
        "--transformer-path", "models/ltx25/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4-comfy-v2.safetensors",
        "--offload-mode", "cpu",
        "--video-cfg", "1.0",
        "--rescale", "0.0",
        "--distilled-lora-strength-stage-1", "0.0",
        "--distilled-lora-strength-stage-2", "0.0",
        "--num-frames", NUM_FRAMES,
        "--width", WIDTH,
        "--height", HEIGHT,
        "--steps", "15",
        "--seed", SEED,
        "--output", "outputs/method_c_limit_test_10s_241f.mp4"
    ]
    ok_c = run_cmd(cmd_c, "METHOD C (Tuned Two-Stage Pipeline)")

    log("=" * 80)
    log("ALL BENCHMARK RUNS FINISHED")
    log(f"Method A Status: {'SUCCESS' if ok_a else 'FAILED'}")
    log(f"Method C Status: {'SUCCESS' if ok_c else 'FAILED'}")
    log("=" * 80)

if __name__ == "__main__":
    main()
