#!/bin/bash
# Installation script for SMURF Ultrasound
# Usage: ./install.sh [cuda_version]
# Example: ./install.sh cu118 (for CUDA 11.8)

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   SMURF Ultrasound - Installation Script                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Detect Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if Python 3.8+
MIN_VERSION="3.8"
if [ "$(printf '%s\n' "$MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    echo "✗ Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
fi

echo ""

# Create virtual environment
echo "📦 Setting up virtual environment..."
python3 -m venv smurf_env
source smurf_env/bin/activate

echo "✓ Virtual environment created"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "✓ pip upgraded"
echo ""

# Install PyTorch
echo "📦 Installing PyTorch with GPU support..."
CUDA_VERSION="${1:-cu118}"  # Default to CUDA 11.8

if [ "$CUDA_VERSION" = "cu118" ]; then
    echo "Installing PyTorch for CUDA 11.8..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "$CUDA_VERSION" = "cu121" ]; then
    echo "Installing PyTorch for CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [ "$CUDA_VERSION" = "cpu" ]; then
    echo "Installing PyTorch for CPU..."
    pip install torch torchvision torchaudio
else
    echo "Invalid CUDA version: $CUDA_VERSION"
    echo "Use: cu118 (default), cu121, or cpu"
    exit 1
fi

echo "✓ PyTorch installed"
echo ""

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed"
echo ""

# Verify installation
echo "🔍 Verifying installation..."
python3 << 'EOF'
import torch
import numpy as np
import scipy
import matplotlib

print(f"  ✓ PyTorch {torch.__version__}")
print(f"  ✓ NumPy {np.__version__}")
print(f"  ✓ SciPy {scipy.__version__}")
print(f"  ✓ Matplotlib {matplotlib.__version__}")
print()
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Installation Complete!                                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "To activate the environment, run:"
echo "  source smurf_env/bin/activate"
echo ""
echo "To start training, run:"
echo "  python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 50"
echo ""
echo "To run inference, run:"
echo "  python3 test_gpu.py --gpu 0"
echo ""
