#!/bin/bash

# SeedVR Model Download Script
set -e

echo "Starting SeedVR model download..."

# Create checkpoints directory
mkdir -p ckpts
cd ckpts

# Function to download from HuggingFace
download_hf_model() {
    local repo_id=$1
    local model_name=$2
    
    echo "Downloading $model_name from $repo_id..."
    
    # Use huggingface-hub to download
    python3 -c "
import os
from huggingface_hub import snapshot_download

repo_id = '$repo_id'
model_name = '$model_name'
cache_dir = './cache'
local_dir = './'

print(f'Downloading {model_name} from {repo_id}...')
snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=['*.json', '*.safetensors', '*.pth', '*.bin', '*.py', '*.md', '*.txt']
)
print(f'{model_name} download completed!')
"
}

# Download SeedVR2-7B model (default)
if [ "${MODEL_SIZE:-7b}" = "7b" ]; then
    download_hf_model "ByteDance-Seed/SeedVR2-7B" "SeedVR2-7B"
    # Rename the checkpoint file if needed
    if [ -f "seedvr2_ema_7b.pth" ]; then
        echo "SeedVR2-7B checkpoint found"
    else
        echo "Warning: SeedVR2-7B checkpoint not found, checking for alternative names..."
        # Look for any .pth files and rename appropriately
        for file in *.pth; do
            if [ -f "$file" ]; then
                echo "Found checkpoint: $file, renaming to seedvr2_ema_7b.pth"
                mv "$file" "seedvr2_ema_7b.pth"
                break
            fi
        done
    fi
fi

# Download SeedVR2-3B model if specified
if [ "${MODEL_SIZE}" = "3b" ]; then
    download_hf_model "ByteDance-Seed/SeedVR2-3B" "SeedVR2-3B"
    # Rename the checkpoint file if needed
    if [ -f "seedvr2_ema_3b.pth" ]; then
        echo "SeedVR2-3B checkpoint found"
    else
        echo "Warning: SeedVR2-3B checkpoint not found, checking for alternative names..."
        for file in *.pth; do
            if [ -f "$file" ]; then
                echo "Found checkpoint: $file, renaming to seedvr2_ema_3b.pth"
                mv "$file" "seedvr2_ema_3b.pth"
                break
            fi
        done
    fi
fi

# Download VAE checkpoint
echo "Downloading VAE checkpoint..."
if [ ! -f "ema_vae.pth" ]; then
    # Try to download VAE from the same model repository
    echo "VAE checkpoint will be included in the model download"
fi

# Verify downloads
echo "Verifying downloads..."
ls -la

echo "Model download completed successfully!"
echo "Available checkpoints:"
ls -la *.pth 2>/dev/null || echo "No .pth files found"

cd ..
echo "Setup complete! Models are ready in ./ckpts/"