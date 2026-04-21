#!/bin/bash
# SMURF Ultrasound - GPU Training Script
# Trains SMURF model on ultrasound data with GPU acceleration
# Run on: GPU server with PyTorch installed

echo "======================================================================"
echo "SMURF ULTRASOUND - GPU TRAINING"
echo "======================================================================"
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 (change if needed)
BATCH_SIZE=16
NUM_EPOCHS=50
LEARNING_RATE=1e-4

# Check for GPU
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

echo ""

# Run training
echo "Starting training on GPU..."
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo ""

python3 << 'EOF'
import sys
sys.path.insert(0, '/path/to/SMURF')  # Update path

from train_real_data import TrainingConfig, UltrasoundTrainer

# Override config for GPU
config = TrainingConfig()
config.batch_size = 16
config.num_epochs = 50
config.learning_rate = 1e-4
config.num_workers = 8
config.train_data_dir = "/path/to/training/data"  # Update path
config.test_data_dir = "/path/to/test/data"       # Update path

print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)
print(f"Batch size: {config.batch_size}")
print(f"Epochs: {config.num_epochs}")
print(f"Learning rate: {config.learning_rate}")
print(f"Workers: {config.num_workers}")
print()

trainer = UltrasoundTrainer(config)
trainer.train()
EOF

echo ""
echo "======================================================================"
echo "Training completed!"
echo "Checkpoints saved to: checkpoints/"
echo "======================================================================"
