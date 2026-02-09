#!/bin/bash
# install.sh

if [ "$1" == "rocm" ]; then
    echo "Installing for ROCm..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1
    uv pip install -e .
elif [ "$1" == "cpu" ]; then
    echo "Installing for CPU..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    uv pip install -e .
else
    echo "Installing for CUDA (default)..."
    uv pip install -e .
fi
