# Server Commands - SMURF Training Quick Reference

## 🚀 Quick Start on GPU Server

### Step 1: Pull Latest Fix
```bash
cd ~/Model_comparisons/Smurf-Deepuse-implementation
git pull origin master
```

### Step 2: Run Training (100 epochs)
```bash
chmod +x train_gpu.sh
./train_gpu.sh
```

Or use Python directly:
```bash
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100
```

### Step 3: Monitor Training
```bash
# In another terminal, watch the logs
tail -f training_*.log

# Or check checkpoint sizes
ls -lh checkpoints/
```

---

## 📊 Expected Output

### Training Progress
```
Epoch 1/100
  Batch 1/50: Loss=0.4532, Photometric=0.4201, Smoothness=0.0331
  Batch 2/50: Loss=0.4321, Photometric=0.3998, Smoothness=0.0323
  ...
  Train Loss: 0.3856 | Val Loss: 0.3721
```

### What It Saves
- `checkpoints/best_model.pt` - Best model (lowest validation loss)
- `checkpoints/epoch_*.pt` - Checkpoint every epoch
- `training_YYYYMMDD_HHMMSS.log` - Detailed logs
- `training_history.json` - Loss curves as JSON

---

## 🧪 After Training: Test/Inference

### Test on Test Data
```bash
python3 test_gpu.py --gpu 0 \
  --checkpoint checkpoints/best_model.pt \
  --test-data /teamspace/studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/test_data_deepuse \
  --output-dir test_results_v1
```

### Expected Test Outputs
```
test_results_v1/
├── displacements/
│   ├── rf0299_07_displacement.png
│   ├── rf0300_25_displacement.png
│   └── ...
├── strains/
│   ├── rf0299_07_strain.png
│   ├── rf0300_25_strain.png
│   └── ...
└── comparison.png
```

---

## 🔍 Troubleshooting

### If Training Fails with Grid Error
- ✅ Already fixed in latest version
- Just run: `git pull origin master`

### If Data Not Found
```bash
# Check training data exists
ls -la /teamspace/studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/train_data
# Should show .mat files

# Check test data exists
ls -la /teamspace/studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/test_data_deepuse
# Should show .mhd/.raw files
```

### If Out of Memory
```bash
# Reduce batch size
python3 train_gpu.py --gpu 0 --batch-size 8 --epochs 100
```

### If Training is Too Slow
```bash
# Increase workers (if CPU allows)
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100 --workers 16
```

---

## 📈 Training Parameters Explained

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--gpu` | 0 | GPU device ID (0, 1, 2, ...) |
| `--batch-size` | 16 | Batch size (higher = faster but more memory) |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate (lower = more stable, slower) |
| `--workers` | 8 | Data loading workers (higher = faster data loading) |

---

## 💾 Resuming Training

If training gets interrupted, simply run again:
```bash
python3 train_gpu.py --gpu 0 --epochs 150
```

It will automatically:
1. Load the best checkpoint so far
2. Continue training for more epochs
3. Save new checkpoints

---

## 🎯 Complete Workflow (One Command)

For a full end-to-end workflow (pull → train → test → report):
```bash
bash train_deepuse_workflow.sh 0 16 100
```

This runs:
1. Git pull latest changes
2. Installs dependencies
3. Trains for 100 epochs with batch size 16 on GPU 0
4. Tests on validation data
5. Generates performance report

---

## 📋 File Structure After Training

```
~/Model_comparisons/Smurf-Deepuse-implementation/
├── train_gpu.py              ← Main training script
├── train_gpu.sh              ← Training bash wrapper
├── test_gpu.py               ← Testing/inference script
├── train_data/               ← Training .mat files
├── test_data_deepuse/        ← Test .mhd/.raw files
├── checkpoints/              ← Model checkpoints (created)
│   ├── best_model.pt
│   ├── epoch_1.pt
│   └── ...
├── test_results_v1/          ← Test outputs (created)
│   ├── displacements/
│   ├── strains/
│   └── ...
└── training_*.log            ← Training logs (created)
```

---

## 🚨 Critical Fixes in Latest Version

### Fixed Issues:
- ✅ Grid coordinate expansion (dimension mismatch)
- ✅ In-place operation errors during backward pass
- ✅ Data loading from correct server paths
- ✅ Anomaly detection for debugging

### To Get Fixes:
```bash
git pull origin master
```

---

## 📞 Need Help?

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| `No training data loaded` | Check data path, ensure .mat files exist |
| `grid_sample(): expected 4D` | Run `git pull origin master` |
| `CUDA out of memory` | Reduce `--batch-size` to 8 or 4 |
| `Training is slow` | Increase `--workers` or use batch size 32 |
| `NaN in loss` | Reduce `--lr` to 5e-5 |

---

## ✅ Next Steps

1. **Pull latest**: `git pull origin master`
2. **Run training**: `./train_gpu.sh` or `python3 train_gpu.py --gpu 0 --epochs 100`
3. **Wait for training** (should take ~2-3 hours for 100 epochs)
4. **Test model**: `python3 test_gpu.py --checkpoint checkpoints/best_model.pt`
5. **Analyze results**: Check `test_results_v1/` for outputs

Good luck! 🚀
