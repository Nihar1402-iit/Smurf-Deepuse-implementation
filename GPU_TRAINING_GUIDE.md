# SMURF Ultrasound - GPU Training & Testing Guide

## Quick Start (GPU Server)

### 1. Clone Repository
```bash
cd /path/to/workspace
git clone https://github.com/Nihar1402-iit/Smurf-Deepuse-implementation.git
cd Smurf-Deepuse-implementation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
```bash
# Training data (MAT files)
ln -s /path/to/training/data ./data/train
# or set in config

# Test data (RAW files)
ln -s /path/to/test/data ./data/test
# or set in config
```

### 4. GPU Training

#### Option A: Python script with arguments
```bash
# Single GPU (GPU 0)
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 50 --lr 1e-4 \
  --train-data /path/to/train/data \
  --test-data /path/to/test/data

# Multiple GPUs (distributed training - set up in code)
python3 train_gpu.py --gpu 0 --batch-size 32 --epochs 50
```

#### Option B: Bash script
```bash
chmod +x train_gpu.sh
./train_gpu.sh
```

#### Option C: Direct training
```bash
python3 train_real_data.py
```

### 5. GPU Inference/Testing

#### Option A: Python script with arguments
```bash
# Run inference with best checkpoint
python3 test_gpu.py --gpu 0 \
  --checkpoint checkpoints/best_model.pt \
  --test-data /path/to/test/data \
  --output-dir test_results
```

#### Option B: Bash script
```bash
chmod +x test_gpu.sh
./test_gpu.sh
```

#### Option C: Direct testing
```bash
python3 test_inference.py
```

---

## Hardware Requirements

### Minimum
- GPU: 4GB VRAM (NVIDIA GPU with CUDA support)
- RAM: 16GB
- Storage: 50GB (for data + checkpoints)

### Recommended
- GPU: 16GB+ VRAM (e.g., RTX 3090, A100)
- RAM: 32GB+
- Storage: 100GB+
- Multiple GPUs for faster training

---

## GPU Setup on Server

### 1. Verify CUDA Installation
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Expected output:
# CUDA Version: 11.8 or higher
```

### 2. Create Virtual Environment
```bash
python3 -m venv smurf_env
source smurf_env/bin/activate  # On macOS/Linux
# or
smurf_env\Scripts\activate  # On Windows
```

### 3. Install PyTorch with GPU Support
```bash
# For CUDA 11.8+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify GPU in PyTorch
```bash
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF
```

---

## Training Configuration

Edit these parameters in `train_gpu.py` or pass as command-line arguments:

```bash
python3 train_gpu.py \
  --gpu 0 \                          # GPU device ID
  --batch-size 16 \                  # Batch size (increase for faster training)
  --epochs 50 \                      # Number of training epochs
  --lr 1e-4 \                        # Learning rate
  --workers 8 \                      # Number of data loading workers
  --train-data /path/to/train/data \ # Training data directory
  --test-data /path/to/test/data \   # Test data directory
  --checkpoint-dir checkpoints       # Where to save checkpoints
```

### Recommended Settings

#### Small Dataset (< 1000 samples)
```bash
python3 train_gpu.py --gpu 0 --batch-size 8 --epochs 100 --lr 5e-5
```

#### Medium Dataset (1000-10000 samples)
```bash
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 50 --lr 1e-4
```

#### Large Dataset (> 10000 samples)
```bash
python3 train_gpu.py --gpu 0 --batch-size 32 --epochs 30 --lr 1e-3
```

#### Multiple GPUs (if available)
```bash
# Set CUDA_VISIBLE_DEVICES for multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0,1,2,3
python3 train_gpu.py --batch-size 64 --epochs 50
```

---

## Monitoring Training

### TensorBoard
```bash
# In separate terminal
tensorboard --logdir checkpoints/tensorboard
# Open browser to http://localhost:6006
```

### Check GPU Usage
```bash
# Watch GPU memory and utilization
watch -n 1 nvidia-smi

# Or single query
nvidia-smi
```

### Check Training Logs
```bash
# Watch training log in real-time
tail -f training.log

# Or print last 50 lines
tail -50 training.log
```

---

## Output Files

### After Training
```
checkpoints/
├── best_model.pt           # Best model on validation set
├── checkpoint_epoch_5.pt   # Checkpoint after epoch 5
├── checkpoint_epoch_10.pt  # Checkpoint after epoch 10
├── history.json            # Training history
└── tensorboard/            # TensorBoard logs
    ├── training/
    └── validation/
```

### After Testing
```
test_results/
├── results.json            # Test results (JSON)
├── training_history.png    # Training curves
├── pair_000/               # Results for test pair 0
│   ├── result_pair_000.mat # DeepUse format (.mat)
│   ├── displacement_heatmap.png
│   ├── strain_heatmap.png
│   └── strain_histogram.png
├── pair_001/               # Results for test pair 1
│   └── ...
└── ...
```

---

## Expected Outputs (Matching DeepUse)

### MAT File Format
Each `.mat` file contains:
- `displacement`: [H, W] - Axial displacement (pixels)
- `strain`: [H, W] - Axial strain (dimensionless)
- `bmode`: [H, W] - B-mode ultrasound image

Example:
```python
import scipy.io as sio
result = sio.loadmat('test_results/pair_000/result_pair_000.mat')
print(result.keys())  # ['displacement', 'strain', 'bmode', ...]
print(result['displacement'].shape)  # (256, 256)
print(result['strain'].shape)         # (256, 256)
```

### Visualization Outputs
- `displacement_heatmap.png`: Axial displacement visualization
- `strain_heatmap.png`: Axial strain visualization
- `strain_histogram.png`: Strain value distribution

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python3 train_gpu.py --batch-size 8

# Or reduce number of workers
python3 train_gpu.py --workers 2
```

### GPU Not Detected
```bash
# Verify NVIDIA driver
nvidia-smi

# Verify PyTorch GPU support
python3 -c "import torch; print(torch.cuda.is_available())"

# Set CUDA device explicitly
export CUDA_VISIBLE_DEVICES=0
python3 train_gpu.py --gpu 0
```

### Slow Training
```bash
# Increase batch size if you have GPU memory
python3 train_gpu.py --batch-size 32

# Increase workers
python3 train_gpu.py --workers 16

# Use mixed precision (add to code)
# pip install torch-tb-profiler
```

### Data Not Found
```bash
# Verify data paths
ls -la /path/to/train/data
ls -la /path/to/test/data

# Pass correct paths
python3 train_gpu.py --train-data /correct/path/to/train
```

---

## Performance Benchmarks

### Expected Speed (RTX 3090, batch_size=16)
- Training: ~100-200 ms/batch
- Inference: ~50-100 ms/frame pair
- Epoch time: ~5-10 minutes (with 1000 training samples)

### Expected Accuracy
- Photometric loss: Decreases from ~0.1 to ~0.01
- Strain similarity to DeepUse: ~90-95%

---

## Complete Training Workflow

### Step 1: Start with small dataset
```bash
python3 train_gpu.py --gpu 0 --batch-size 8 --epochs 10
```

### Step 2: Monitor training
```bash
# In another terminal
tensorboard --logdir checkpoints/tensorboard
```

### Step 3: Run full training
```bash
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 50 --lr 1e-4
```

### Step 4: Test on validation data
```bash
python3 test_gpu.py --gpu 0 --checkpoint checkpoints/best_model.pt
```

### Step 5: Analyze results
```bash
python3 << 'EOF'
import json
with open('test_results/results.json') as f:
    results = json.load(f)
print(f"Average inference time: {sum(r['inference_time_ms'] for r in results) / len(results):.2f} ms")
print(f"Average strain: {sum(r['strain_mean'] for r in results) / len(results):.6f}")
EOF
```

---

## Advanced Options

### Distributed Training (Multi-GPU)
```bash
# Will be added in future version
# For now, set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 train_gpu.py --batch-size 64
```

### Mixed Precision Training
```bash
# Reduces memory usage, speeds up training
# Add to train_gpu.py:
# from torch.cuda.amp import autocast, GradScaler
```

### Custom Loss Weights
Edit `train_real_data.py`:
```python
losses = model.compute_losses(I_t, I_t1, output)
# Modify loss weights here
total_loss = (
    1.5 * losses["photometric"] +  # Increase photometric weight
    0.05 * losses["smoothness"] +
    0.1 * losses["strain_reg"]
)
```

---

## Useful Commands

### Check GPU Status
```bash
nvidia-smi -l 1  # Refresh every 1 second
```

### Profile Training
```bash
python3 -m torch.utils.bottleneck train_gpu.py --gpu 0
```

### Export Model for Inference Only
```bash
python3 << 'EOF'
import torch
from smurf_ultrasound_wrapper import SMURFUltrasoundWrapper
from smurf_core import SMURFModel

model = SMURFUltrasoundWrapper(SMURFModel())
model.eval()
torch.save(model.state_dict(), 'model_inference_only.pt')
EOF
```

---

## Next Steps

1. **Prepare your data**: Ensure training/test data are in correct format
2. **Run training**: Start with `python3 train_gpu.py --gpu 0`
3. **Monitor**: Check TensorBoard and GPU usage
4. **Test**: Run inference with `python3 test_gpu.py --gpu 0`
5. **Compare**: Compare outputs with DeepUse results

---

## Support & Issues

- Check `training.log` for errors
- Verify data format matches expected input
- Ensure sufficient GPU memory
- See README.md for more information

**Last Updated**: April 2026
