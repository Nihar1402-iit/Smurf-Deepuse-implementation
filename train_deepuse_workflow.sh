#!/bin/bash
# SMURF Ultrasound - Complete GPU Training & Testing Workflow
# Execute this on GPU server to train with DeepUse-inspired improvements

set -e

echo "================================================================================"
echo "SMURF ULTRASOUND - DEEPUSE-INSPIRED TRAINING WORKFLOW"
echo "================================================================================"
echo ""

# Configuration
GPU_ID=${1:-0}
BATCH_SIZE=${2:-16}
EPOCHS=${3:-100}
CHECKPOINT_DIR="checkpoints"
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

echo "Configuration:"
echo "  GPU ID: $GPU_ID"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo "  Log File: $LOG_FILE"
echo ""

# Step 1: Pull latest changes from GitHub
echo "================================================================================"
echo "STEP 1: Pull Latest Changes from GitHub"
echo "================================================================================"
echo ""
git pull origin master 2>&1 | tee -a "$LOG_FILE"
echo ""

# Step 2: Verify changes are in place
echo "================================================================================"
echo "STEP 2: Verify DeepUse Integration"
echo "================================================================================"
echo ""
python3 -c "
import torch
from smurf_ultrasound_wrapper import SMURFUltrasoundWithLosses
from deepuse_utils import ncc_similarity, crop_boundaries
print('✓ All modules imported successfully')
print('✓ NCC similarity function available')
print('✓ Boundary cropping function available')
" 2>&1 | tee -a "$LOG_FILE"
echo ""

# Step 3: Run integration tests
echo "================================================================================"
echo "STEP 3: Run DeepUse Integration Tests"
echo "================================================================================"
echo ""
python3 test_deepuse_integration.py 2>&1 | tee -a "$LOG_FILE"
echo ""

# Step 4: Start training
echo "================================================================================"
echo "STEP 4: Start GPU Training with DeepUse-Inspired Losses"
echo "================================================================================"
echo ""
echo "Training will use:"
echo "  - NCC-based similarity loss (replaces photometric L1)"
echo "  - Boundary cropping (removes 143-pixel artifacts)"
echo "  - Displacement regularization (prevents trivial solution)"
echo "  - Strain regularization (smooth strain field)"
echo ""
python3 train_gpu.py \
  --gpu "$GPU_ID" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  2>&1 | tee -a "$LOG_FILE"
echo ""

# Step 5: Verify training completed
echo "================================================================================"
echo "STEP 5: Training Complete - Verify Results"
echo "================================================================================"
echo ""
if [ -f "$CHECKPOINT_DIR/best_model.pt" ]; then
    echo "✓ Best model saved: $CHECKPOINT_DIR/best_model.pt"
    ls -lh "$CHECKPOINT_DIR/best_model.pt" 2>&1 | tee -a "$LOG_FILE"
else
    echo "✗ Best model not found!"
fi

if [ -f "$CHECKPOINT_DIR/history.json" ]; then
    echo "✓ Training history saved: $CHECKPOINT_DIR/history.json"
else
    echo "✗ Training history not found!"
fi

echo ""
echo "================================================================================"
echo "TRAINING WORKFLOW COMPLETE"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. Review loss curves in: $CHECKPOINT_DIR/history.json"
echo "2. Run inference: python3 test_gpu.py --gpu $GPU_ID --checkpoint $CHECKPOINT_DIR/best_model.pt"
echo "3. Compare results with baseline model"
echo ""
echo "Full log: $LOG_FILE"
