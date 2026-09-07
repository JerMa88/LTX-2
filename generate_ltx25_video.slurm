#!/bin/bash
#SBATCH --job-name=ltx25_video
#SBATCH --partition=short
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_ltx25_%j.out
#SBATCH --error=slurm_ltx25_%j.err

echo "=== SLURM Job Started ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "CUDA Devices:"
nvidia-smi

export PATH="/users/jerryma/.local/bin:$PATH"
cd /work/projects/mhahsler/course_recomm/allocation001/AI_Club/projects/LTX-2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p outputs

echo "Launching LTX-2.5 DistilledPipeline video generation with FP8 Cast & 64GB RAM..."
./.venv/bin/python -m ltx_pipelines.distilled \
    --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --spatial-upsampler-path models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --quantization fp8-cast \
    --num-frames 121 \
    --seed 42 \
    --offload cpu \
    --output-path outputs/ltx25_sample.mp4 \
    --prompt "A cinematic slow-motion tracking shot of a majestic golden eagle soaring gracefully above snow-capped mountain peaks at sunset, dramatic golden hour lighting illuminating its feathers, crisp alpine air, camera panning smoothly with the eagle's flight." \
    "$@"

STATUS=$?
echo "=== SLURM Job Finished with exit code: ${STATUS} ==="
exit ${STATUS}
