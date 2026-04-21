# GitHub to GPU Server - Complete Workflow

## 📋 Complete Step-by-Step Guide

### Phase 1: Push Code to GitHub (Local Machine)

#### 1.1 Verify Local Setup
```bash
cd /Users/niharshah/Desktop/SMURF
git log --oneline  # Check commit history
git status         # Should be clean
```

#### 1.2 Add GitHub Remote
```bash
git remote add origin https://github.com/Nihar1402-iit/Smurf-Deepuse-implementation.git
git branch -M main
```

#### 1.3 Push Code
```bash
git push -u origin main --force-with-lease
```

#### 1.4 Verify on GitHub
- Visit: https://github.com/Nihar1402-iit/Smurf-Deepuse-implementation
- Check all files are uploaded

---

### Phase 2: Setup on GPU Server

#### 2.1 SSH into GPU Server
```bash
ssh user@gpu-server.example.com
# or
ssh -i /path/to/key.pem ubuntu@gpu-instance-ip
```

#### 2.2 Clone Repository
```bash
cd /workspace  # or preferred location
git clone https://github.com/Nihar1402-iit/Smurf-Deepuse-implementation.git smurf-deepuse
cd smurf-deepuse
```

#### 2.3 Verify GPU
```bash
nvidia-smi
# Should show GPU(s) available
```

#### 2.4 Install Dependencies
```bash
# Make install script executable
chmod +x install.sh

# Run installation (for CUDA 11.8)
bash install.sh cu118

# Or for CUDA 12.1
bash install.sh cu121

# Activate virtual environment
source smurf_env/bin/activate
```

#### 2.5 Verify Installation
```bash
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

---

### Phase 3: Prepare Training Data

#### 3.1 Upload/Link Training Data
```bash
# Option A: Copy from local storage
scp -r /local/path/to/training/data user@gpu-server:/workspace/smurf-deepuse/data/train

# Option B: Create symlink if data already on server
ln -s /shared/training/data /workspace/smurf-deepuse/data/train
ln -s /shared/test/data /workspace/smurf-deepuse/data/test

# Option C: Specify paths directly in command
```

#### 3.2 Verify Data
```bash
# Check training data
ls -lh /workspace/smurf-deepuse/data/train/*.mat | head -5

# Check test data
ls -lh /workspace/smurf-deepuse/data/test/*.raw | head -5
```

---

### Phase 4: GPU Training

#### 4.1 Single GPU Training
```bash
cd /workspace/smurf-deepuse
python3 train_gpu.py \
  --gpu 0 \
  --batch-size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --workers 8 \
  --train-data ./data/train \
  --test-data ./data/test
```

#### 4.2 Multi-GPU Training
```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Increase batch size
python3 train_gpu.py \
  --batch-size 64 \
  --epochs 50 \
  --workers 16
```

#### 4.3 Monitor Training (In Separate SSH Session)
```bash
# Terminal 1: Watch GPU usage
ssh user@gpu-server
watch -n 1 nvidia-smi

# Terminal 2: TensorBoard
ssh user@gpu-server
cd /workspace/smurf-deepuse
tensorboard --logdir checkpoints/tensorboard --port 6006

# On local machine, tunnel to server
ssh -L 6006:localhost:6006 user@gpu-server
# Open browser: http://localhost:6006
```

#### 4.4 Check Training Progress
```bash
# SSH to server
tail -f training.log

# Or check checkpoints
ls -lh checkpoints/
```

---

### Phase 5: GPU Testing/Inference

#### 5.1 Run Inference
```bash
cd /workspace/smurf-deepuse

# Use best checkpoint
python3 test_gpu.py \
  --gpu 0 \
  --checkpoint checkpoints/best_model.pt \
  --test-data ./data/test \
  --output-dir test_results
```

#### 5.2 Verify Outputs
```bash
# Check MAT files created
ls -lh test_results/pair_*/result_*.mat

# Verify structure
python3 << 'EOF'
import scipy.io as sio
result = sio.loadmat('test_results/pair_000/result_pair_000.mat')
print("Keys:", list(result.keys()))
print("Displacement shape:", result['displacement'].shape)
print("Strain shape:", result['strain'].shape)
EOF
```

---

### Phase 6: Download Results (Optional)

#### 6.1 Copy Results to Local
```bash
# From local machine
scp -r user@gpu-server:/workspace/smurf-deepuse/test_results ./results_from_gpu

# Check results
ls -la results_from_gpu/pair_*/
```

#### 6.2 Compare with DeepUse
```bash
python3 << 'EOF'
import scipy.io as sio
import numpy as np

# Load SMURF output
smurf_result = sio.loadmat('results_from_gpu/pair_000/result_pair_000.mat')

# Load DeepUse output (if available)
deepuse_result = sio.loadmat('path/to/deepuse/result.mat')

# Compare
smurf_strain = smurf_result['strain']
deepuse_strain = deepuse_result['strain']

correlation = np.corrcoef(smurf_strain.flatten(), deepuse_strain.flatten())[0, 1]
print(f"Strain correlation: {correlation:.4f}")
EOF
```

---

## 🚀 Quick Command Reference

### First-Time Setup
```bash
# On local machine
cd /Users/niharshah/Desktop/SMURF
git remote add origin https://github.com/Nihar1402-iit/Smurf-Deepuse-implementation.git
git branch -M main
git push -u origin main

# On GPU server
git clone https://github.com/Nihar1402-iit/Smurf-Deepuse-implementation.git smurf-deepuse
cd smurf-deepuse
bash install.sh cu118
source smurf_env/bin/activate
```

### Train
```bash
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 50 \
  --train-data /path/to/train \
  --test-data /path/to/test
```

### Test
```bash
python3 test_gpu.py --gpu 0 \
  --checkpoint checkpoints/best_model.pt \
  --test-data /path/to/test
```

### Monitor
```bash
# Terminal 1
watch -n 1 nvidia-smi

# Terminal 2
tail -f training.log

# Terminal 3
tensorboard --logdir checkpoints/tensorboard --port 6006
```

---

## 📊 Expected Timeline

| Task | Time | Notes |
|------|------|-------|
| Clone repo | 1 min | |
| Install deps | 5-10 min | Depends on internet |
| GPU verify | 1 min | Check nvidia-smi |
| Data prep | 5-30 min | Depends on data size |
| **Training** | **30 min - 2 hrs** | **Depends on data/epochs** |
| Inference | 5-10 min | On 10-20 test samples |
| Download results | 5 min | Optional |
| **Total** | **~1-3 hours** | |

---

## 🔧 Troubleshooting

### Cannot Connect to GPU Server
```bash
# Check SSH key permissions
chmod 600 ~/.ssh/id_rsa

# Test connection
ssh -v user@gpu-server  # Verbose output for debugging

# If behind corporate firewall, try with different port
ssh -p 2222 user@gpu-server
```

### GPU Not Detected After Installation
```bash
# Verify CUDA
nvcc --version

# Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory During Training
```bash
# Reduce batch size
python3 train_gpu.py --batch-size 8

# Or use gradient accumulation (add to code)
```

### Training is Slow
```bash
# Increase batch size
python3 train_gpu.py --batch-size 32

# Increase workers
python3 train_gpu.py --workers 16

# Check GPU utilization
nvidia-smi -l 1
```

### Data Files Not Found
```bash
# Verify paths
ls -la /workspace/smurf-deepuse/data/train/
ls -la /workspace/smurf-deepuse/data/test/

# Create symlinks if needed
mkdir -p data
ln -s /actual/path/to/train data/train
ln -s /actual/path/to/test data/test
```

---

## 📝 File Checklist

Before pushing to GitHub, verify:
- ✅ `train_gpu.py` - Main training script
- ✅ `test_gpu.py` - Main testing script
- ✅ `smurf_core.py` - SMURF model
- ✅ `smurf_ultrasound_wrapper.py` - Output wrapper
- ✅ `lsqse.py` - Strain computation
- ✅ `data_loaders.py` - Data loading
- ✅ `requirements.txt` - Dependencies
- ✅ `install.sh` - Installation script
- ✅ `GPU_TRAINING_GUIDE.md` - GPU guide
- ✅ `README_GPU.md` - GPU README
- ✅ `.gitignore` - Git ignore rules

---

## 🎯 Success Criteria

After following this guide, you should:
1. ✅ Have code on GitHub
2. ✅ Can clone and install on GPU server
3. ✅ Can run training and see losses decreasing
4. ✅ Can run inference and get MAT file outputs
5. ✅ Outputs match DeepUse format

---

## 📞 Support

If issues arise:
1. Check `training.log` for errors
2. Monitor `nvidia-smi` for GPU status
3. Verify data paths with `ls`
4. Check internet connection for package downloads
5. Consult GPU_TRAINING_GUIDE.md for detailed help

---

**Last Updated**: April 2026
