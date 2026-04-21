#!/usr/bin/env python3
"""
Debug script to diagnose inference issues
Checks model outputs, gradients, and data preprocessing
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from smurf_core import SMURFModel
from smurf_ultrasound_wrapper import SMURFUltrasoundWrapper
from data_loaders import RawUltrasoundDataset
from utils import UltrasoundPreprocessor


def debug_model_weights(model):
    """Check if model weights look reasonable (not all zeros or random)"""
    print("\n" + "="*70)
    print("DEBUG: Model Weights Analysis")
    print("="*70)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean_val = param.data.mean().item()
            std_val = param.data.std().item()
            min_val = param.data.min().item()
            max_val = param.data.max().item()
            
            print(f"\n{name}")
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}")
            print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}")
            
            # Warning flags
            if std_val < 1e-4:
                print(f"  ⚠️  WARNING: Very small std (might be uninitialized or frozen)")
            if abs(mean_val) > 10 or abs(std_val) > 10:
                print(f"  ⚠️  WARNING: Very large values (might be exploding)")


def debug_forward_pass(model, I_t, I_t1):
    """Check outputs at each stage"""
    print("\n" + "="*70)
    print("DEBUG: Forward Pass Analysis")
    print("="*70)
    
    print(f"\nInput shapes:")
    print(f"  I_t: {I_t.shape}, range=[{I_t.min():.4f}, {I_t.max():.4f}]")
    print(f"  I_t1: {I_t1.shape}, range=[{I_t1.min():.4f}, {I_t1.max():.4f}]")
    
    with torch.no_grad():
        output = model(I_t, I_t1)
    
    print(f"\nOutput shapes:")
    print(f"  displacement: {output['displacement'].shape}")
    print(f"  strain: {output['strain'].shape}")
    
    displacement = output['displacement']
    strain = output['strain']
    
    print(f"\nDisplacement stats:")
    print(f"  Range: [{displacement.min():.6f}, {displacement.max():.6f}]")
    print(f"  Mean: {displacement.mean():.6f}, Std: {displacement.std():.6f}")
    
    print(f"\nStrain stats:")
    print(f"  Range: [{strain.min():.6f}, {strain.max():.6f}]")
    print(f"  Mean: {strain.mean():.6f}, Std: {strain.std():.6f}")
    
    # Check for NaN/Inf
    if torch.isnan(displacement).any():
        print(f"  ⚠️  WARNING: NaN values in displacement!")
    if torch.isinf(displacement).any():
        print(f"  ⚠️  WARNING: Inf values in displacement!")
    
    if torch.isnan(strain).any():
        print(f"  ⚠️  WARNING: NaN values in strain!")
    if torch.isinf(strain).any():
        print(f"  ⚠️  WARNING: Inf values in strain!")
    
    return displacement, strain


def debug_data_loading(test_data_dir, num_samples=3):
    """Check data preprocessing"""
    print("\n" + "="*70)
    print("DEBUG: Data Loading Analysis")
    print("="*70)
    
    dataset = RawUltrasoundDataset(test_data_dir, frame_height=512, frame_width=1000)
    print(f"\nDataset size: {len(dataset)} frame pairs")
    
    for idx in range(min(num_samples, len(dataset))):
        I_t, I_t1 = dataset[idx]
        
        print(f"\nSample {idx}:")
        print(f"  I_t shape: {I_t.shape}, range=[{I_t.min():.4f}, {I_t.max():.4f}]")
        print(f"  I_t1 shape: {I_t1.shape}, range=[{I_t1.min():.4f}, {I_t1.max():.4f}]")
        
        # Check for issues
        if torch.isnan(I_t).any() or torch.isnan(I_t1).any():
            print(f"  ⚠️  WARNING: NaN in data!")
        if torch.isinf(I_t).any() or torch.isinf(I_t1).any():
            print(f"  ⚠️  WARNING: Inf in data!")


def visualize_outputs(displacement, strain, I_t, save_dir=None):
    """Visualize outputs for debugging"""
    print("\n" + "="*70)
    print("DEBUG: Visualization")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Input
    axes[0, 0].imshow(I_t[0, 0].cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Displacement
    disp = displacement[0, 0].cpu().numpy()
    im1 = axes[0, 1].imshow(disp, cmap='RdBu_r')
    axes[0, 1].set_title(f'Displacement (range: [{disp.min():.4f}, {disp.max():.4f}])')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Strain
    s = strain[0, 0].cpu().numpy()
    im2 = axes[1, 0].imshow(s, cmap='RdBu_r')
    axes[1, 0].set_title(f'Strain (range: [{s.min():.6f}, {s.max():.6f}])')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Histogram of strain
    axes[1, 1].hist(s.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Strain Distribution')
    axes[1, 1].set_xlabel('Strain')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / "debug_outputs.png", dpi=150, bbox_inches='tight')
        print(f"\nSaved debug visualization to {save_dir / 'debug_outputs.png'}")
    
    plt.show()


def main():
    """Run all diagnostics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create model
    print("Creating model...")
    smurf = SMURFModel(in_channels=1, max_displacement=4, num_refinement_steps=4)
    wrapper = SMURFUltrasoundWrapper(smurf).to(device)
    
    # Load checkpoint if available
    checkpoint_path = Path("checkpoints/best_model.pt")
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Fix key prefix
        state_dict_fixed = {}
        for key, value in state_dict.items():
            if key.startswith("wrapper."):
                new_key = key.replace("wrapper.", "", 1)
            else:
                new_key = key
            state_dict_fixed[new_key] = value
        
        wrapper.load_state_dict(state_dict_fixed, strict=False)
        print("Checkpoint loaded successfully\n")
    else:
        print("⚠️  WARNING: No checkpoint found! Using random weights\n")
    
    wrapper.eval()
    
    # Debug model weights
    debug_model_weights(wrapper)
    
    # Debug data loading
    test_data_dir = Path("/Users/niharshah/Desktop/Omnistrain/our_algo/test_data_deepuse")
    if test_data_dir.exists():
        debug_data_loading(test_data_dir, num_samples=2)
        
        # Load a sample and run forward pass
        dataset = RawUltrasoundDataset(str(test_data_dir), frame_height=512, frame_width=1000)
        I_t, I_t1 = dataset[0]
        I_t = I_t.unsqueeze(0).to(device)
        I_t1 = I_t1.unsqueeze(0).to(device)
        
        # Debug forward pass
        displacement, strain = debug_forward_pass(wrapper, I_t, I_t1)
        
        # Visualize
        visualize_outputs(displacement, strain, I_t, save_dir="debug_outputs")
    else:
        print(f"⚠️  Test data directory not found: {test_data_dir}")
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
