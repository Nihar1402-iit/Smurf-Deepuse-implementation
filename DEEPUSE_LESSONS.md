# DeepUse vs SMURF: Key Differences & Lessons Learned

## 1. **Data Preprocessing & Cropping**

### DeepUse Approach:
```python
# Aggressive cropping to remove imaging artifacts
img = data[idx:idx+up, 100:-300, :]  # Crop 100 pixels from left, 300 from right
```
- **Why**: Removes boundary artifacts where imaging quality is poor
- **Advantage**: Reduces noisy gradients at boundaries
- **Our fix**: We added boundary clamping, but didn't address root cause

### SMURF Current Approach:
```python
# No cropping - uses full image
I_t = I_t_batch  # Uses all 512x1000 pixels
```

**→ ACTION**: Crop to remove boundary artifacts like DeepUse

---

## 2. **Strain Computation Method**

### DeepUse Approach (Least Squares):
```python
def get_strain(disp, x_wind=143):
    # Uses local least squares fitting over patches
    # Fits linear model: strain = β₀ + β₁ * depth
    # Returns β₁ (the slope = strain)
    # Uses MUCH larger window (287 pixels!) compared to typical 5x5
    
    d = x_wind*2+1  # 287 pixel window
    depthX = torch.linspace(1, d, d)  # Coordinate vector
    XtX = depthX.T @ depthX  # Design matrix
    betas_cholesky = solve(XtX, XtY)  # Cholesky decomposition
    Uxx = betas_cholesky[0, :]  # Slope = strain
```

**Key insights:**
- Uses **large windows** (287 pixels) for robustness
- Uses **Cholesky decomposition** (numerically stable)
- Returns **slope** (linear fit) not gradient
- Much more robust to noise than finite differences

### SMURF Current Approach (Finite Differences):
```python
def _compute_strain_gradient(self, u_axial):
    # Uses Sobel kernel (3x3)
    strain = F.conv2d(u_padded, self.grad_kernel_y, padding=0)
```

**Problem:**
- Only uses 3x3 neighborhood → amplifies noise
- No smoothing or least squares robustness
- Sensitive to displacement noise

---

## 3. **Strain Consistency Regularization**

### DeepUse Approach:
```python
# Computed over sequence of frames
if len(self.strain_compensated_list) > 1:
    # Motion-compensate strain: warp(strain[t-1]) should match strain[t]
    self.loss_consistency_strain = [
        NCC(strain_warped[t-1], strain[t]) for t in range(1, len(strain))
    ]
    self.loss_consistency_strain_mean = mean(stack(consistency_strain))
    total_loss += (1 - consistency_mean) * beta
```

**Key insight:**
- Compares **warped previous strain** with **current strain**
- Enforces **temporal consistency**
- Strongly constrains strain to be physically consistent

### SMURF Current Approach:
```python
# Just computes strain gradients independently
# No temporal/spatial consistency constraint
```

**→ This is probably your missing piece!**

---

## 4. **Loss Function Design**

### DeepUse Losses:
```
total_loss = (1 - similarity) + smooth_weight * smoothness + consistency_weight * (1 - consistency)
```

**Key components:**
1. **Similarity (1 - NCC)**: Image matching via normalized cross-correlation
   - Much better for ultrasound than photometric (intensity-based)
   - Why: Ultrasound is envelope-detected, not intensity-linear
   
2. **Smoothness (GradNorm)**: Penalize displacement gradients
   
3. **Consistency (1 - NCC)**: Motion-compensated strain consistency
   - **NEW**: Enforces strain smoothness over time

### SMURF Current Losses:
```
total_loss = photometric + 0.1*smoothness + 0.05*strain_reg + 0.01*displacement_reg
```

**Issues:**
- Photometric loss assumes linear intensity relationship
- No temporal consistency
- No NCC similarity metric

---

## 5. **Output Extraction**

### DeepUse:
```python
# For each frame pair:
displacement = disp_list[t]  # [B, 2, H, W]
strain = strain_list[t]      # [B, 1, H, W] - computed via LS fitting
axial_displacement = disp_list[t][:, 1, :, :]  # Extract channel 1

# Save axial displacement (not axial alone)
data = {'displacement': axial_displacement,  # Full 2D map
        'strain': strain,                    # Computed strain
        'bmode': bmode}
```

### SMURF Current:
```python
# Single frame pair:
displacement = output['displacement']  # Only axial
strain = output['strain']              # Computed via gradient
```

**→ DeepUse returns full displacement field + properly computed strain**

---

## 6. **Window Cropping During Loss Computation**

### DeepUse:
```python
# Only compare central regions (crop 143 pixels from top/bottom)
self.loss_consistency_strain = [
    NCC(strain_comp[t-1][:,:,143:-143,:], strain[t][:,:,143:-143,:])
    for t in range(1, len(strain))
]
```

**Why:**
- Avoids boundary artifacts in loss computation
- Focuses on good-quality central region
- Prevents bad boundaries from affecting training

### SMURF Current:
- Uses entire image including bad boundaries

---

## Recommended Changes for SMURF

### Priority 1: Replace Strain Computation
```python
# Change from Sobel to least squares
def _compute_strain_lsqse(self, u_axial, window_size=287):
    # Use large window least squares fit
    # Similar to DeepUse implementation
    # Much more robust to noise
```

### Priority 2: Add NCC Loss
```python
# Replace photometric with NCC
def _compute_similarity_loss(self, I_t, I_t1_warped):
    # Use normalized cross-correlation instead of L1/L2 difference
    # Better for ultrasound
```

### Priority 3: Add Temporal Consistency
```python
# For sequences, add:
strain_compensated = warp_image(strain[t-1], disp[t])
consistency_loss = NCC(strain_compensated, strain[t])
total_loss += consistency_weight * (1 - consistency)
```

### Priority 4: Crop Boundaries in Loss
```python
# Only compute losses on central region
crop_pixels = 143
loss_region = strain[:, :, crop_pixels:-crop_pixels, :]
```

---

## Quick Reference: Why DeepUse Works Well

| Aspect | DeepUse | SMURF | Impact |
|--------|---------|-------|--------|
| **Strain** | LS fitting (287px window) | Sobel 3x3 | DeepUse: robust; SMURF: noisy |
| **Similarity** | NCC (non-parametric) | Photometric L1 | DeepUse: ultrasound-appropriate; SMURF: assumes linear intensity |
| **Consistency** | Temporal (frame-to-frame) | None | DeepUse: enforces physics; SMURF: unconstrained |
| **Boundary** | Cropped (100:-300) | Full image | DeepUse: clean; SMURF: artifactual |
| **Window Loss** | Central only (143:-143) | Full | DeepUse: ignores bad edges; SMURF: uses them |

---

## Implementation Roadmap

1. ✅ Apply boundary cropping (like DeepUse)
2. 🔄 Implement LS-based strain computation
3. 🔄 Replace photometric with NCC
4. 🔄 Add temporal consistency if processing sequences
5. 🔄 Crop boundaries in loss computation

---

## Files to Reference

- **DeepUse Strain**: `/Users/niharshah/Desktop/Omnistrain/our_algo/DeepUse/utils/__init__.py` → `get_strain()`
- **DeepUse Model**: `/Users/niharshah/Desktop/Omnistrain/our_algo/DeepUse/models/reusenet_model.py` → loss computation
- **DeepUse Dataset**: `/Users/niharshah/Desktop/Omnistrain/our_algo/DeepUse/dataset/invivo_dataset.py` → data loading with cropping

