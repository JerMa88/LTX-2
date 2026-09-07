#!/bin/bash
#SBATCH --job-name=smu_commercial
#SBATCH --partition=short
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm_smu_%j.out
#SBATCH --error=logs/slurm_smu_%j.err

echo "=== SLURM Job Started: SMU Commercial (HQ 2-Stage) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CUDA Devices:"
nvidia-smi

export PATH="/users/jerryma/.local/bin:$PATH"
cd /work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_CACHE_DIR="$(pwd)/.uv_cache"
export PYTHONUNBUFFERED=1

mkdir -p logs outputs/smu_scenes outputs/smu_previews

echo "Launching SMU commercial generation script with args: $@"
./.venv/bin/python generate_smu_commercial.py "$@"

STATUS=$?
echo "=== SLURM Job Finished with exit code: ${STATUS} ==="
exit ${STATUS}
