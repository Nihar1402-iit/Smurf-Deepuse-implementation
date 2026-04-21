# 🚀 Quick Start Commands - SMURF Training & Testing

## 📍 Local Mac (Desktop)

### Training
```bash
cd /Users/niharshah/Desktop/SMURF

# Train with local data (using _Data_10M_Part1_)
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100

# Or with explicit paths
python3 train_gpu.py \
  --gpu 0 \
  --batch-size 16 \
  --epochs 100 \
  --train-data /Users/niharshah/Desktop/Omnistrain/_Data_10M_Part1_ \
  --test-data /Users/niharshah/Desktop/Omnistrain/our_algo/test_data_deepuse
```

### Testing/Inference
```bash
cd /Users/niharshah/Desktop/SMURF

python3 test_gpu.py \
  --gpu 0 \
  --checkpoint checkpoints/best_model.pt \
  --test-data /Users/niharshah/Desktop/Omnistrain/our_algo/test_data_deepuse \
  --output-dir test_results_v1
```

---

## 🖥️ GPU Server (Remote)

### Step 1: SSH into server
```bash
ssh your_username@server_address
cd ~/Model_comparisons/Smurf-Deepuse-implementation
```

### Step 2: Training
```bash
# Make sure train_data directory exists first!
# ls -la ./train_data

# Train with server data
python3 train_gpu.py \
  --gpu 0 \
  --batch-size 16 \
  --epochs 100 \
  --train-data ./train_data \
  --test-data ./test_data_deepuse
```

### Step 3: Testing (after training)
```bash
python3 test_gpu.py \
  --gpu 0 \
  --checkpoint checkpoints/best_model.pt \
  --test-data ./test_data_deepuse \
  --output-dir test_results_v1
```

---

## 📊 Full Workflow Examples

### Option A: Interactive Training on GPU (Recommended for First Run)
```bash
# Train for 10 epochs first to verify data loads
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 10

# Check results, then train longer
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100
```

### Option B: Background Training (Using nohup)
```bash
# Run training in background and save output to log
nohup python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100 > training.log 2>&1 &

# Monitor progress
tail -f training.log

# Check if still running
ps aux | grep train_gpu.py
```

### Option C: Background Training (Using screen/tmux)
```bash
# Start new screen session
screen -S training

# Inside screen, run training
cd ~/Model_comparisons/Smurf-Deepuse-implementation
python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100

# Detach from screen: Ctrl-A then D
# Reattach later: screen -r training
# List sessions: screen -ls
```

---

## 🔍 Troubleshooting

### Problem: "No training data loaded!"
**Solution:** Verify your data directory has .mat files
```bash
# Check what's in train_data
ls -la ./train_data/
ls -la /studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/train_data/

# Check file size (should be large)
du -sh ./train_data/
```

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size
```bash
# Instead of batch-size 16, use 8
python3 train_gpu.py --gpu 0 --batch-size 8 --epochs 100
```

### Problem: "Training runs but GPU not being used"
**Solution:** Check GPU is accessible
```bash
python3 -c "import torch; print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 📈 Monitoring Training

### During Training
```bash
# Watch loss output (if using tail)
tail -f training.log

# Check GPU usage (on server)
nvidia-smi

# Check GPU in real-time
watch -n 1 nvidia-smi
```

### After Training
```bash
# Check saved checkpoints
ls -lh checkpoints/

# View training history
python3 << 'EOF'
import json
with open('training_history.json') as f:
    history = json.load(f)
    print(f"Total epochs trained: {len(history['train_loss'])}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
EOF
```

---

## ⏱️ Expected Times

| Task | Duration | GPU |
|------|----------|-----|
| 1 epoch (100 steps) | 2-3 min | L4 (23.7 GB) |
| 10 epochs | 20-30 min | L4 |
| 50 epochs | 100-150 min | L4 |
| 100 epochs | 200-300 min | L4 |

---

## 💾 Output Files Generated

### During Training
- `checkpoints/model_epoch_*.pt` - Checkpoint every N epochs
- `checkpoints/best_model.pt` - Best model (lowest validation loss)
- `training_history.json` - Loss history for plotting
- `training.log` - Detailed training output

### After Testing
- `test_results/` - Test output directory
- `test_results/displacements/` - Displacement heatmaps
- `test_results/strains/` - Strain heatmaps
- `test_results/results.json` - Test metrics

---

## 🎯 Recommended Next Steps

1. **Verify data exists:**
   ```bash
   ls -la ./train_data/
   ```

2. **Start with short training:**
   ```bash
   python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 10
   ```

3. **Check training is working:**
   - Loss should decrease
   - GPU should be getting used
   - Checkpoints should be saved

4. **If step 3 works, run full training:**
   ```bash
   python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100
   ```

5. **After training completes, test the model:**
   ```bash
   python3 test_gpu.py --gpu 0 --checkpoint checkpoints/best_model.pt
   ```

---

## 📋 Common Command Templates

### Train with Different Settings
```bash
# Larger batch size (faster, needs more GPU memory)
python3 train_gpu.py --gpu 0 --batch-size 32 --epochs 100

# Smaller batch size (slower, less memory)
python3 train_gpu.py --gpu 0 --batch-size 8 --epochs 100

# Different learning rate
python3 train_gpu.py --gpu 0 --epochs 100 --lr 5e-5

# Multiple data loaders
python3 train_gpu.py --gpu 0 --epochs 100 --workers 8
```

### Train and Save to Custom Location
```bash
python3 train_gpu.py \
  --gpu 0 \
  --epochs 100 \
  --checkpoint-dir ./my_checkpoints \
  --train-data /path/to/data
```

---

**Good luck with training! 🚀**
