# SMURF Ultrasound Project Structure & Setup Guide

## Project Structure

```
/Users/niharshah/Desktop/SMURF/
├── README.md                        # Main documentation
├── requirements.txt                 # Python dependencies
├── setup.py                         # Installation script
│
├── Core Models
├── smurf_core.py                    # SMURF optical flow model
├── lsqse.py                         # LSQSE strain computation module
├── smurf_ultrasound_wrapper.py      # Wrapper for ReUSENet-style output
├── utils.py                         # Preprocessing, postprocessing utilities
│
├── Training & Inference
├── train.py                         # Training script with losses
├── inference.py                     # Inference pipeline + visualization
├── test_smurf_ultrasound.py        # Comprehensive test suite
│
├── Checkpoints & Results (generated)
├── checkpoints/
│   ├── best_model.pt               # Best model from training
│   ├── checkpoint_epoch_*.pt       # Intermediate checkpoints
│   └── history.json                # Training history
│
└── Visualizations (generated)
    ├── displacement_heatmap.png
    ├── strain_heatmap.png
    ├── displacement_vectors.png
    └── strain_histogram.png
```

## Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd /Users/niharshah/Desktop/SMURF

# Install dependencies
pip install -r requirements.txt
```

**Verified Compatibility:**
- Python: 3.8+
- PyTorch: 2.0+
- CUDA: 11.8+ (optional, for GPU)

### 2. Run Tests

```bash
# Comprehensive test suite
python test_smurf_ultrasound.py

# Expected output:
# ✓ SMURF Model test
# ✓ LSQSE Module test
# ✓ Wrapper test
# ✓ Loss computation test
# ✓ Fast prediction methods test
# ✓ Preprocessing utilities test
# ✓ Postprocessing utilities test
# ✓ ReUSENet format compatibility test
# ALL TESTS PASSED!
```

### 3. Run Inference Example

```bash
# Quick inference with visualization
python inference.py

# Generates:
# - displacement_heatmap.png
# - strain_heatmap.png
# - displacement_vectors.png
# - strain_histogram.png
```

### 4. Train on Custom Data

```bash
# Start training
python train.py

# Training logs and checkpoints saved to ./checkpoints/
# Monitor training with tensorboard:
# tensorboard --logdir=./checkpoints/
```

---

## Module Reference

### 1. smurf_core.py
**RAFT-based Optical Flow Model**

Key Classes:
- `ConvBlock`: Basic Conv2d + ReLU block
- `FeatureEncoder`: Multi-scale feature extraction
- `CostVolumeLayer`: 4D cost volume construction
- `FlowHead`: Initial flow prediction
- `RecurrentFlowRefinement`: Iterative refinement
- `SMURFModel`: Full SMURF pipeline

Example:
```python
from smurf_core import SMURFModel
import torch

model = SMURFModel(in_channels=1, max_displacement=4, num_refinement_steps=4)
I_t = torch.randn(2, 1, 256, 256)
I_t1 = torch.randn(2, 1, 256, 256)

flow_predictions, final_flow = model(I_t, I_t1)
# flow_predictions: list of 5 intermediate predictions
# final_flow: [2, 2, 256, 256] - final flow at original resolution
```

### 2. lsqse.py
**Least Squares Strain Estimation**

Key Classes:
- `LSQSEModule`: Strain computation from displacement

Methods:
- `forward()`: Compute strain with optional smoothing
- `_compute_strain_gradient()`: Fast Sobel-based method
- `_compute_strain_lsqse()`: Robust least squares fitting
- `_smooth_strain()`: Gaussian/median/bilateral filtering

Example:
```python
from lsqse import LSQSEModule
import torch

lsqse = LSQSEModule(window_size=5, filter_type='gaussian')
u_axial = torch.randn(2, 1, 256, 256)

strain = lsqse(u_axial, smooth=True)
# [2, 1, 256, 256] - axial strain
```

### 3. smurf_ultrasound_wrapper.py
**Ultrasound-specific Wrapper**

Key Classes:
- `SMURFUltrasoundWrapper`: Forward pass + displacement/strain output
- `SMURFUltrasoundWithLosses`: Includes training losses

Example:
```python
from smurf_ultrasound_wrapper import SMURFUltrasoundWrapper
from smurf_core import SMURFModel

smurf = SMURFModel(in_channels=1)
wrapper = SMURFUltrasoundWrapper(smurf)

output = wrapper(I_t, I_t1)
# output["displacement"]: [B, 2, H, W] - [axial, lateral]
# output["strain"]: [B, 1, H, W] - axial strain

# Fast variants
displacement = wrapper.forward_displacement_only(I_t, I_t1)
strain = wrapper.forward_strain_only(I_t, I_t1)
```

### 4. utils.py
**Preprocessing, Postprocessing, Visualization**

Key Classes:
- `UltrasoundPreprocessor`: Normalization (minmax, zscore, log), filtering
- `DisplacementPostprocessor`: Magnitude, angle, outlier removal
- `StrainPostprocessor`: Clipping, outlier removal, statistics
- `VisualizationUtils`: Colorization, heatmaps

Example:
```python
from utils import UltrasoundPreprocessor, DisplacementPostprocessor

# Preprocessing
rf_frame = torch.randn(256, 256)
normalized = UltrasoundPreprocessor.normalize_rf(rf_frame, method='minmax')

# Postprocessing
displacement = torch.randn(2, 2, 256, 256)
magnitude = DisplacementPostprocessor.magnitude_of_displacement(displacement)
angle = DisplacementPostprocessor.angle_of_displacement(displacement)
```

### 5. inference.py
**Inference Pipeline & Visualization**

Key Classes:
- `UltrasoundInference`: Inference engine with checkpoint loading
- `UltrasoundVisualizer`: 4 visualization methods

Example:
```python
from inference import UltrasoundInference, UltrasoundVisualizer

# Create inference engine
inference = UltrasoundInference(device='cuda')

# Predict
output = inference.predict(I_t, I_t1)

# Visualize
visualizer = UltrasoundVisualizer()
fig, axes = visualizer.create_displacement_heatmap(output["displacement"], I_t)
fig.savefig("displacement.png")
```

### 6. train.py
**Training Script with Full Pipeline**

Key Classes:
- `TrainingConfig`: Hyperparameter configuration
- `UltrasoundDataset`: Dummy dataset (replace with real data)
- `UltrasoundTrainer`: Full training loop

Example:
```python
from train import TrainingConfig, UltrasoundTrainer

config = TrainingConfig()
config.batch_size = 8
config.num_epochs = 50
config.learning_rate = 1e-4

trainer = UltrasoundTrainer(config)
trainer.train()
```

---

## API Cheatsheet

### Inference (Most Common)

```python
from inference import UltrasoundInference
from smurf_ultrasound_wrapper import SMURFUltrasoundWrapper
from smurf_core import SMURFModel
import torch

# Method 1: Using inference engine
inference = UltrasoundInference()
output = inference.predict(I_t, I_t1)
# output: {"displacement": [B,2,H,W], "strain": [B,1,H,W]}

# Method 2: Using wrapper directly
smurf = SMURFModel(in_channels=1)
wrapper = SMURFUltrasoundWrapper(smurf)
output = wrapper(I_t, I_t1)

# Fast variants
displacement = wrapper.forward_displacement_only(I_t, I_t1)
strain = wrapper.forward_strain_only(I_t, I_t1)
```

### Preprocessing

```python
from utils import UltrasoundPreprocessor

# Normalize RF frames
I_t_norm = UltrasoundPreprocessor.normalize_rf(I_t, method='minmax')

# Normalize IQ frames
I_t_norm = UltrasoundPreprocessor.normalize_iq(I_t, method='zscore')

# Gaussian blur
I_t_blurred = UltrasoundPreprocessor.gaussian_blur(I_t, kernel_size=5)

# Contrast enhancement (CLAHE)
I_t_enhanced = UltrasoundPreprocessor.adaptive_histogram_equalization(I_t)
```

### Postprocessing

```python
from utils import DisplacementPostprocessor, StrainPostprocessor

# Displacement postprocessing
magnitude = DisplacementPostprocessor.magnitude_of_displacement(displacement)
angle = DisplacementPostprocessor.angle_of_displacement(displacement)
displacement_filtered = DisplacementPostprocessor.median_filter_displacement(displacement)
displacement_clean = DisplacementPostprocessor.remove_displacement_outliers(displacement)

# Strain postprocessing
strain_clipped = StrainPostprocessor.clip_strain(strain, min_val=-0.5, max_val=0.5)
strain_clean = StrainPostprocessor.remove_strain_outliers(strain)
stats = StrainPostprocessor.compute_strain_statistics(strain)
```

### Visualization

```python
from inference import UltrasoundVisualizer

vis = UltrasoundVisualizer()

# Displacement heatmap
fig, axes = vis.create_displacement_heatmap(displacement, I_t)

# Strain heatmap
fig, ax = vis.create_strain_heatmap(strain, I_t)

# Displacement vectors
fig, ax = vis.create_displacement_vectors(displacement, I_t, stride=10)

# Strain histogram
fig, ax = vis.create_strain_histogram(strain)
```

### Training

```python
from train import TrainingConfig, UltrasoundTrainer
from smurf_ultrasound_wrapper import SMURFUltrasoundWithLosses
from smurf_core import SMURFModel

config = TrainingConfig()
trainer = UltrasoundTrainer(config)
trainer.train()

# Check training history
import json
with open("checkpoints/history.json") as f:
    history = json.load(f)
```

---

## Configuration Reference

### TrainingConfig (from train.py)

```python
class TrainingConfig:
    batch_size = 8                      # Batch size
    num_epochs = 50                     # Number of training epochs
    learning_rate = 1e-4                # Initial learning rate
    weight_decay = 1e-5                 # L2 regularization
    
    # Loss weights
    weight_photometric = 1.0            # Intensity constancy
    weight_smoothness = 0.1             # Displacement smoothness
    weight_strain_reg = 0.05            # Strain smoothness
    
    # Logging
    log_interval = 10                   # Log every N epochs
    save_interval = 5                   # Save checkpoint every N epochs
    
    # Data
    frame_height = 256                  # Frame resolution
    frame_width = 256
    frame_channels = 1                  # RF (1) or IQ (2)
    num_train_samples = 1000
    num_val_samples = 100
```

### SMURFModel Parameters

```python
model = SMURFModel(
    in_channels=1,              # RF or IQ channels
    max_displacement=4,         # Max search range (pixels)
    num_refinement_steps=4      # RAFT refinement iterations
)
```

### LSQSEModule Parameters

```python
lsqse = LSQSEModule(
    window_size=5,              # Local window for fitting
    strain_window=5,            # Smoothing window
    filter_type='gaussian'      # 'gaussian', 'median', 'bilateral'
)
```

### SMURFUltrasoundWrapper Parameters

```python
wrapper = SMURFUltrasoundWrapper(
    smurf_model,
    lsqse_window_size=5,
    strain_smoothing=True,
    strain_smoothing_type='gaussian',   # 'gaussian', 'median', 'bilateral'
    return_full_output=False    # True for debugging
)
```

---

## Output Shapes & Formats

### Input
```
I_t, I_t1: [B, C, H, W]
- B: batch size
- C: channels (1 for RF, 2 for IQ)
- H, W: frame resolution (typically 256×256)
```

### Output (ReUSENet Format)
```
displacement: [B, 2, H, W]
- Channel 0: Axial displacement (strong motion expected)
- Channel 1: Lateral displacement (weak motion expected)
- Range: typically [-5, 5] pixels

strain: [B, 1, H, W]
- Channel 0: Axial strain (du/dy)
- Range: typically [-0.5, 0.5]
```

---

## Performance Tips

1. **GPU Acceleration**
   ```python
   device = torch.device("cuda")
   model = model.to(device)
   ```

2. **Batch Processing**
   ```python
   # Process multiple frames at once
   batch = torch.stack([I_t1, I_t2, I_t3, ...])
   output = model(batch[:, 0], batch[:, 1])
   ```

3. **Fast Prediction (if only displacement needed)**
   ```python
   displacement = wrapper.forward_displacement_only(I_t, I_t1)
   ```

4. **Reduce Computation**
   ```python
   model = SMURFModel(
       max_displacement=2,        # Smaller search range
       num_refinement_steps=2     # Fewer refinement steps
   )
   ```

---

## Troubleshooting

### Memory Issues
```python
# Reduce batch size or resolution
config.batch_size = 4
config.frame_height = 128
config.frame_width = 128
```

### Slow Training
```python
# Use fewer refinement steps
model = SMURFModel(num_refinement_steps=2)

# Reduce training data
config.num_train_samples = 500
```

### Poor Displacement Quality
```python
# Increase search range and refinement
model = SMURFModel(
    max_displacement=8,
    num_refinement_steps=8
)
```

### Noisy Strain Estimates
```python
# Increase smoothing
lsqse = LSQSEModule(
    window_size=7,
    strain_window=7,
    filter_type='bilateral'
)
```

---

## Citing This Work

If you use this code, please cite:

```bibtex
@article{smurf2021,
  title={SMURF: Self-Supervised Motion Understanding for Optical Flow},
  author={Stone, Austin and Maurer, Daniel and Ayvaci, Anurag and Dabiri, Amirreza and Premoze, Sergey},
  journal={arXiv preprint arXiv:2104.08278},
  year={2021}
}

@article{raft2020,
  title={RAFT: Recurrent All-Pairs Field Transforms for Optical Flow},
  author={Teed, Zachary and Deng, Jia},
  journal={ICCV},
  year={2020}
}

@article{reusenet2020,
  title={Real-time Ultrasound Elastography using Recurrent Neural Networks},
  author={Marcu, Bogdan and others},
  year={2020}
}
```

---

**Last Updated**: April 2026
