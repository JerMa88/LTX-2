#!/bin/bash
set -e

echo "Starting download of LTX-2.5 model weights..."
mkdir -p models/ltx-2.5

hf download Lightricks/LTX-2.5 \
    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
    diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
    loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    vae/ltx-2.5-video-vae-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --local-dir models/ltx-2.5

echo "Download completed. Verifying files:"
ls -lh models/ltx-2.5/*/*.safetensors
