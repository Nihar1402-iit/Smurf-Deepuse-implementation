# SMURF Ultrasound - Debugging Results Issues

## Problems Identified

### 1. **Strain Boundary Artifacts**
- **Issue**: The strain computation had extreme values at image boundaries (dark red/blue)
- **Cause**: Using `reflect` padding mode in Sobel kernel created artificial gradients
- **Fix**: Changed to `replicate` padding and clamping boundary values

### 2. **Displacement May Not Be Learning**
- **Issue**: Displacement shows mostly noise without clear structure
- **Possible Causes**:
  - **Trivial Solution**: Model outputting near-zero displacement (optimization prefers low-loss state)
  - **Photometric Loss**: Ultrasound intensity patterns may not follow optical flow assumptions
  - **Untrained Model**: If checkpoint didn't save properly, model uses random weights

- **Fix**: Added displacement magnitude regularization to prevent zero-solution bias

### 3. **Strain Computation Sensitivity**
- **Issue**: Strain (which is gradient of displacement) amplifies noise
- **Fix**: Added smoothing filter; improved gradient computation

---

## What to Check Next

### On the Server:

1. **Verify checkpoint was saved correctly:**
   ```bash
   python3 -c "
   import torch
   ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu')
   if 'model_state_dict' in ckpt:
       keys = list(ckpt['model_state_dict'].keys())[:5]
       print('Checkpoint keys:', keys)
       print('First weight range:', ckpt['model_state_dict'][keys[0]].min().item(), ckpt['model_state_dict'][keys[0]].max().item())
   "
   ```

2. **Check training loss curves:**
   ```bash
   python3 -c "
   import json
   history = json.load(open('training_history.json', 'r'))
   print('Train loss (first 5 epochs):', history['train_loss'][:5])
   print('Train loss (last 5 epochs):', history['train_loss'][-5:])
   "
   ```

3. **Re-train with new fixes:**
   ```bash
   python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100
   ```
   - Increased epochs to let new regularization term guide training
   - Monitor that losses decrease and stabilize

4. **Test with debug script locally:**
   ```bash
   python3 debug_inference.py
   ```
   This will show:
   - Model weight distributions
   - Input data ranges
   - Output displacement/strain statistics
   - Visualization of results

---

## Changes Made

### `/Users/niharshah/Desktop/SMURF/lsqse.py`
- Changed padding mode from `reflect` to `replicate`
- Added boundary artifact suppression
- Clamp boundary strain values to prevent extreme values

### `/Users/niharshah/Desktop/SMURF/smurf_ultrasound_wrapper.py`
- Added displacement magnitude regularization loss
- This prevents the model from converging to trivial zero-solution
- Loss weight: 0.01 (small but consistent pressure)

---

## Expected Results After Retraining

**Before Fix:**
- Displacement: Random noise pattern
- Strain: Extreme values at boundaries, noise elsewhere
- Loss: Plateau early (model finds low-loss zero-solution)

**After Fix:**
- Displacement: Clear structures in high-motion areas (vessels)
- Strain: Smooth, continuous gradients without boundary artifacts
- Loss: Steady decrease over epochs, more expressive outputs
- Visualization: Clearer distinction between moving and stationary regions

---

## Quick Reference: Loss Components

| Loss Term | Purpose | Weight | Comment |
|-----------|---------|--------|---------|
| Photometric | Intensity conservation | 1.0 | Main driver for optical flow |
| Smoothness | Smooth displacement fields | 0.1 | Regularization |
| Strain Reg | Smooth strain maps | 0.05 | Reduces strain noise |
| **Displacement Reg** | **Non-zero outputs** | **0.01** | **NEW: Prevents trivial solution** |

---

## Next Steps

1. ✅ Apply fixes (done - pushed to GitHub)
2. 🔄 Pull fixes on server: `git pull origin master`
3. 🔄 Re-train on server: `python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100`
4. 🧪 Test inference: `python3 test_gpu.py --gpu 0 --checkpoint checkpoints/best_model.pt --test-data ...`
5. 📊 Compare results - should see much better displacement/strain maps

---

## Questions?

The debug script (`debug_inference.py`) can help investigate further issues:
- Check model weight initialization
- Verify data preprocessing
- Inspect forward pass outputs
- Generate diagnostic visualizations
