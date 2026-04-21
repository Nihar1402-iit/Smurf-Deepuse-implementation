# SMURF Ultrasound - Training & Testing Guide

## Overview
This guide explains all available scripts for training and testing the SMURF ultrasound model with DeepUse-inspired improvements.

---

## 📋 Available Scripts

### Training Scripts

#### 1. **Python Script (Recommended)**
**File:** `train_gpu.py`

**Usage:**
```bash
python3 train_gpu.py [OPTIONS]
```

**Options:**
- `--gpu GPU_ID` - GPU device ID (default: 0)
- `--batch-size SIZE` - Batch size (default: 16)
- `--epochs NUM` - Number of epochs (default: 50)
- `--lr RATE` - Learning rate (default: 1e-4)
- `--workers NUM` - Number of data loading workers (default: 8)
- `--train-data PATH` - Training data directory
- `--test-data PATH` - Test data directory
- `--checkpoint-dir PATH` - Checkpoint directory (default: checkpoints)

**Examples:**
```bash
# Train for 100 epochs with batch size 16 on GPU 0
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100

# Train with custom learning rate
python3 train_gpu.py --gpu 0 --epochs 100 --lr 5e-5

# Train with specific data paths
python3 train_gpu.py --gpu 0 --epochs 100 \
  --train-data /path/to/train \
  --test-data /path/to/test
```

---

#### 2. **Bash Script (Alternative)**
**File:** `train_gpu.sh`

**Usage:**
```bash
bash train_gpu.sh
```

**Configuration:** Edit variables at the top:
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU device
BATCH_SIZE=16
NUM_EPOCHS=50
LEARNING_RATE=1e-4
```

---

#### 3. **Complete Workflow Script (Recommended for Full Pipeline)**
**File:** `train_deepuse_workflow.sh`

**Usage:**
```bash
bash train_deepuse_workflow.sh [GPU_ID] [BATCH_SIZE] [EPOCHS]
```

**What it does:**
1. ✅ Pulls latest changes from GitHub
2. ✅ Installs/updates Python dependencies
3. ✅ Trains the model with DeepUse-inspired improvements
4. ✅ Tests on validation data
5. ✅ Generates performance report
6. ✅ Uploads results

**Examples:**
```bash
# Train on GPU 0, 100 epochs, batch size 16
bash train_deepuse_workflow.sh 0 16 100

# Default (GPU 0, batch size 16, 50 epochs)
bash train_deepuse_workflow.sh

# Train on GPU 1 with custom settings
bash train_deepuse_workflow.sh 1 32 150
```

---

### Testing/Inference Scripts

#### 1. **Python Script (Recommended)**
**File:** `test_gpu.py`

**Usage:**
```bash
python3 test_gpu.py [OPTIONS]
```

**Options:**
- `--gpu GPU_ID` - GPU device ID (default: 0)
- `--checkpoint PATH` - Path to checkpoint (.pt file)
- `--test-data PATH` - Test data directory
- `--output-dir PATH` - Output directory (default: test_results)

**Examples:**
```bash
# Test with checkpoint
python3 test_gpu.py --gpu 0 --checkpoint checkpoints/best_model.pt

# Test with custom output directory
python3 test_gpu.py --gpu 0 \
  --checkpoint checkpoints/best_model.pt \
  --output-dir my_results

# Test with specific test data
python3 test_gpu.py --gpu 0 \
  --checkpoint checkpoints/best_model.pt \
  --test-data /teamspace/studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/test_data_deepuse \
  --output-dir test_results_v3
```

---

#### 2. **Bash Script (Alternative)**
**File:** `test_gpu.sh`

**Usage:**
```bash
bash test_gpu.sh
```

**Configuration:** Edit path references at the top:
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU device
```

---

#### 3. **Integration Test**
**File:** `test_deepuse_integration.py`

**Usage:**
```bash
python3 test_deepuse_integration.py
```

**What it tests:**
- DeepUse utilities integration
- NCC similarity computation
- Boundary cropping
- Strain computation

---

### Diagnostic Scripts

#### Debug Inference
**File:** `debug_inference.py`

**Usage:**
```bash
python3 debug_inference.py
```

**What it shows:**
- Model architecture
- Input/output shapes
- Displacement statistics
- Strain statistics
- Boundary artifacts analysis

---

## 🚀 Recommended Workflow

### For Initial Training:

```bash
# Step 1: Install dependencies
bash install.sh

# Step 2: Start training (recommended: complete workflow)
bash train_deepuse_workflow.sh 0 16 100

# This will:
# - Train the model for 100 epochs
# - Save checkpoints every epoch
# - Test on validation data
# - Generate performance report
```

### For Quick Training (Development):

```bash
# Just train without full workflow
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 50
```

### For Testing/Inference:

```bash
# Test trained model
python3 test_gpu.py --gpu 0 \
  --checkpoint checkpoints/best_model.pt \
  --test-data /path/to/test_data \
  --output-dir test_results_v1
```

---

## 📊 Monitoring Training

The training scripts automatically:
- Log to `training_YYYYMMDD_HHMMSS.log`
- Save checkpoints to `checkpoints/`
- Display real-time loss curves
- Save training history to `training_history.json`

**Monitor training:**
```bash
# Watch the log file
tail -f training_*.log

# Or check training history
python3 << 'EOF'
import json
with open('training_history.json') as f:
    history = json.load(f)
    print(f"Epochs: {len(history['loss'])}")
    print(f"Best loss: {min(history['loss']):.4f}")
    print(f"Latest loss: {history['loss'][-1]:.4f}")
EOF
```

---

## 💾 Output Files

### After Training:
- `checkpoints/` - Model checkpoints
- `checkpoints/best_model.pt` - Best model (lowest validation loss)
- `training_*.log` - Training logs
- `training_history.json` - Loss history

### After Testing:
- `test_results/` - Test outputs
- `test_results/displacements/` - Displacement heatmaps
- `test_results/strains/` - Strain heatmaps
- `test_results/comparison.png` - Before/after comparison

---

## 🔧 Advanced Options

### Custom Data Paths

```bash
# Use specific training/test data
python3 train_gpu.py \
  --gpu 0 \
  --batch-size 16 \
  --epochs 100 \
  --train-data /path/to/train_data \
  --test-data /path/to/test_data
```

### Multiple GPUs

```bash
# Use GPU 1 instead of 0
python3 train_gpu.py --gpu 1 --epochs 100

# Or via environment variable
CUDA_VISIBLE_DEVICES=1 python3 train_gpu.py --epochs 100
```

### Resume Training

```bash
# The scripts automatically resume from best checkpoint
python3 train_gpu.py --gpu 0 --epochs 150
# Will load checkpoints/best_model.pt and continue training
```

---

## ⚠️ Troubleshooting

### GPU Not Found
```bash
# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, check CUDA installation
# Use CPU instead (slower):
python3 train_gpu.py --gpu -1 --epochs 10
```

### Out of Memory
```bash
# Reduce batch size
python3 train_gpu.py --gpu 0 --batch-size 8 --epochs 100

# Or reduce model size in config
```

### Data Not Found
```bash
# Check data directory exists
ls -la /path/to/test_data

# Update path in scripts or use --train-data/--test-data flags
```

---

## 📈 Expected Results

### Training
- Initial loss: ~0.5-1.0
- After 10 epochs: ~0.3-0.5
- After 50 epochs: ~0.1-0.3
- After 100 epochs: ~0.05-0.15

### Testing
- Displacement maps: Clear vessel structures
- Strain maps: Smooth without boundary artifacts
- No trivial solution (near-zero outputs)

---

## 📝 Summary

| Task | Command | Time |
|------|---------|------|
| **Install** | `bash install.sh` | ~5 min |
| **Train (50 epochs)** | `python3 train_gpu.py --epochs 50` | ~30 min |
| **Train (100 epochs)** | `python3 train_gpu.py --epochs 100` | ~60 min |
| **Full workflow (100 epochs)** | `bash train_deepuse_workflow.sh 0 16 100` | ~90 min |
| **Test/Inference** | `python3 test_gpu.py --checkpoint checkpoints/best_model.pt` | ~5 min |
| **Debug/Diagnose** | `python3 debug_inference.py` | ~2 min |

---

**Next Step:** Choose your training command from the above and run it! 🚀
