#!/bin/bash
# SMURF Ultrasound - GPU Testing/Inference Script
# Runs inference on test data and generates DeepUse-format outputs
# Run on: GPU server with PyTorch installed

echo "======================================================================"
echo "SMURF ULTRASOUND - GPU INFERENCE/TESTING"
echo "======================================================================"
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 (change if needed)

# Check for GPU
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

echo ""
echo "Running inference on test data..."
echo ""

python3 << 'EOF'
import sys
sys.path.insert(0, '/path/to/SMURF')  # Update path

from test_inference import run_inference_on_test_data, plot_training_history

# Run inference
run_inference_on_test_data()

# Plot training history if available
plot_training_history()

print("\n" + "=" * 70)
print("INFERENCE COMPLETE!")
print("=" * 70)
print("\nOutputs saved to: test_results/")
print("- Displacement/strain heatmaps")
print("- Results in DeepUse format (.mat files)")
print("- Training history plots")
EOF

echo ""
echo "======================================================================"
echo "Testing completed!"
echo "Results saved to: test_results/"
echo "======================================================================"
