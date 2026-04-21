# SMURF Training - Debugging & FAQ

## 🔴 Common Errors & Fixes

### Error 1: Grid Dimension Mismatch
**Error Message:**
```
RuntimeError: grid_sampler(): expected 4D input and grid with same number of dimensions, 
but got input with sizes [16, 1, 256, 256] and grid with sizes [16, 256, 256, 256, 2]
```

**Cause:** Grid coordinate expansion was incorrect

**Fix:** 
```bash
git pull origin master  # Get latest fix
python3 train_gpu.py --gpu 0 --epochs 100
```

---

### Error 2: In-place Operation During Backward Pass
**Error Message:**
```
RuntimeError: one of the variables needed for gradient computation has been modified 
by an inplace operation: [torch.cuda.FloatTensor [...]], which is output 0 of AsStridedBackward0
```

**Cause:** Loss computation was modifying tensors that need gradients

**Fix:**
- ✅ Already fixed in latest version
- Run: `git pull origin master`
- The fix uses `.clone()` where needed and avoids in-place operations

---

### Error 3: No Training Data Loaded
**Error Message:**
```
ERROR: No training data loaded!
```

**Cause:** Data directory is wrong or empty

**Fix:**
```bash
# Check data directory
ls -la /teamspace/studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/train_data/

# If empty, make sure .mat files exist
# If files exist, ensure they have 'RF' key:
python3 << 'EOF'
import scipy.io as sio
mat_data = sio.loadmat('/teamspace/studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/train_data/model2_all.mat')
print(list(mat_data.keys()))
EOF
```

---

### Error 4: CUDA Out of Memory
**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Cause:** Batch size is too large for GPU memory

**Fix:**
```bash
# Reduce batch size
python3 train_gpu.py --gpu 0 --batch-size 8 --epochs 100

# Or even smaller
python3 train_gpu.py --gpu 0 --batch-size 4 --epochs 100
```

**Memory Usage Reference:**
| Batch Size | GPU Memory (L4) | Time/Epoch |
|-----------|-----------------|-----------|
| 4 | ~8 GB | ~90s |
| 8 | ~14 GB | ~60s |
| 16 | ~22 GB | ~45s |
| 32 | >23 GB ❌ | - |

---

### Error 5: NaN in Loss
**Error Message:**
```
Loss is NaN after first batch
```

**Cause:** Learning rate too high or numerical instability

**Fix:**
```bash
# Use smaller learning rate
python3 train_gpu.py --gpu 0 --lr 5e-5 --epochs 100

# Or even smaller
python3 train_gpu.py --gpu 0 --lr 1e-5 --epochs 100
```

---

### Error 6: Data Loading is Slow
**Error Message:**
```
Data loading taking >30s per epoch
```

**Cause:** Too few workers or slow disk access

**Fix:**
```bash
# Increase workers (if CPU has cores)
python3 train_gpu.py --gpu 0 --workers 16 --epochs 100

# Or check disk I/O
iostat -x 1 5
```

---

## 🟡 Warnings & Solutions

### Warning 1: Training Loss Not Decreasing
**Symptoms:**
- Loss stays constant or increases
- After 5+ epochs, no improvement

**Causes:**
1. Learning rate too high → adjust `--lr`
2. Model not converging → train longer
3. Data issue → verify .mat file format

**Solutions:**
```bash
# Try lower learning rate
python3 train_gpu.py --gpu 0 --lr 5e-5 --epochs 200

# Or check data
python3 debug_inference.py
```

---

### Warning 2: Validation Loss Much Higher Than Training Loss
**Symptoms:**
- Training loss: 0.15
- Validation loss: 0.45

**Causes:**
1. Model is overfitting
2. Batch normalization issues
3. Not enough validation data

**Solutions:**
```bash
# Add more regularization (increase smoothness weight)
# Edit smurf_ultrasound_wrapper.py and increase:
# self.weight_smoothness = 0.2  # was 0.1
python3 train_gpu.py --gpu 0 --epochs 100
```

---

### Warning 3: GPU Not Being Used Fully
**Symptoms:**
- GPU usage: 30-40% despite high batch size
- GPU memory: 5/23 GB used

**Causes:**
1. Data loading is the bottleneck
2. CPU pinning disabled

**Solutions:**
```bash
# Increase batch size and workers
python3 train_gpu.py --gpu 0 --batch-size 32 --workers 16 --epochs 100

# Monitor with nvidia-smi
watch -n 1 nvidia-smi
```

---

## 🟢 Expected Training Behavior

### First Epoch
```
Epoch 1/100
  Batch 1/50: Loss=0.48±0.02, Photo=0.42±0.02, Smooth=0.06±0.01
  Batch 2/50: Loss=0.45±0.02, Photo=0.40±0.02, Smooth=0.05±0.01
  ...
  Train Loss: 0.42 | Val Loss: 0.41
```

### Middle Training (Epoch 30)
```
Epoch 30/100
  Batch 1/50: Loss=0.18±0.01, Photo=0.16±0.01, Smooth=0.02±0.00
  ...
  Train Loss: 0.17 | Val Loss: 0.19
```

### Late Training (Epoch 100)
```
Epoch 100/100
  Batch 1/50: Loss=0.09±0.00, Photo=0.08±0.00, Smooth=0.01±0.00
  ...
  Train Loss: 0.08 | Val Loss: 0.09
  Best model saved: 0.08
```

### Key Indicators of Good Training:
- ✅ Loss decreases steadily
- ✅ Validation loss follows training loss
- ✅ No NaN or Inf values
- ✅ GPU memory stable
- ✅ New best model saved periodically

---

## 🔧 Advanced Debugging

### Enable Full Anomaly Detection
If you suspect in-place operations, add this to `train_real_data.py`:

```python
def train(self):
    # Add at the start of train() method
    torch.autograd.set_detect_anomaly(True)
    
    # Then run training - it will pinpoint exact problem line
    for epoch in range(self.config.num_epochs):
        ...
```

### Profile GPU Memory
```bash
# In another terminal, monitor memory
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.free --format=csv,noheader'
```

### Check Data Quality
```bash
python3 << 'EOF'
import scipy.io as sio
import numpy as np

mat_file = "/teamspace/studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/train_data/model2_all.mat"
data = sio.loadmat(mat_file)
rf = data['RF']

print(f"RF shape: {rf.shape}")
print(f"RF dtype: {rf.dtype}")
print(f"RF range: [{rf.min():.2f}, {rf.max():.2f}]")
print(f"RF mean: {rf.mean():.2f}, std: {rf.std():.2f}")

# Check for NaN or Inf
print(f"NaN values: {np.isnan(rf).sum()}")
print(f"Inf values: {np.isinf(rf).sum()}")
EOF
```

### Check Model Architecture
```bash
python3 << 'EOF'
from smurf_core import SMURFModel
from smurf_ultrasound_wrapper import SMURFUltrasoundWithLosses

model = SMURFUltrasoundWithLosses(
    SMURFModel(in_channels=1, max_displacement=4, num_refinement_steps=4)
)

# Print architecture
print(model)

# Check total parameters
total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")

# Check trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")
EOF
```

---

## 📊 Performance Benchmarks

### Expected Training Time (100 epochs, batch size 16, GPU L4)
| Component | Time |
|-----------|------|
| Data loading | ~5 min |
| First epoch | ~45s |
| Typical epoch | ~40s |
| Total 100 epochs | ~65-70 min |
| Testing | ~5 min |

### Expected Final Loss Values (after 100 epochs)
| Metric | Expected Range |
|--------|-----------------|
| Training loss | 0.08-0.12 |
| Validation loss | 0.09-0.13 |
| Photometric loss | 0.07-0.11 |
| Smoothness loss | 0.01-0.02 |

---

## ✅ Verification Checklist

Before running training:
- [ ] Git is up to date: `git pull origin master`
- [ ] Data directory exists: `/teamspace/studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/train_data/`
- [ ] Data files present: `ls train_data/*.mat` returns files
- [ ] GPU available: `python3 -c "import torch; print(torch.cuda.is_available())"`
- [ ] Enough disk space for checkpoints: `df -h | grep /teamspace`

After training completes:
- [ ] `checkpoints/best_model.pt` exists
- [ ] `training_history.json` created
- [ ] Loss values reasonable (not NaN/Inf)
- [ ] Model can run inference on test data

---

## 🚀 Next Steps

1. **Pull latest fix**: `git pull origin master`
2. **Verify data**: `ls train_data/*.mat`
3. **Run training**: `python3 train_gpu.py --gpu 0 --epochs 100`
4. **Monitor**: `tail -f training_*.log`
5. **Test**: `python3 test_gpu.py --checkpoint checkpoints/best_model.pt`

Good luck! 🎯
