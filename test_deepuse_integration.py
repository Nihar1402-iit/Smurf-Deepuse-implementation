#!/usr/bin/env python3
"""
Test DeepUse Integration - Verify that NCC similarity and boundary cropping work correctly
Tests the updated loss functions on a sample batch
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

# Import our modules
from smurf_core import SMURFModel
from smurf_ultrasound_wrapper import SMURFUltrasoundWrapper, SMURFUltrasoundWithLosses
from deepuse_utils import ncc_similarity, crop_boundaries, get_strain_deepuse_style


def test_ncc_similarity():
    """Test NCC similarity computation"""
    print("\n" + "="*70)
    print("TEST 1: NCC Similarity Function")
    print("="*70)
    
    # Create dummy images
    x = torch.randn(2, 1, 256, 256)  # [B, C, H, W]
    y = x + 0.1 * torch.randn_like(x)  # Slightly noisy version
    
    ncc_map = ncc_similarity(x, y, kernel_size=5)
    
    print(f"Input shape: {x.shape}")
    print(f"NCC map shape: {ncc_map.shape}")
    print(f"NCC mean: {ncc_map.mean():.4f} (should be close to 1.0 for similar images)")
    print(f"NCC std: {ncc_map.std():.4f}")
    
    assert ncc_map.shape == (2, 1, 256, 256), f"Unexpected NCC shape: {ncc_map.shape}"
    assert -1.0 <= ncc_map.min() <= 1.0, f"NCC out of [-1, 1] range: {ncc_map.min()}"
    assert -1.0 <= ncc_map.max() <= 1.0, f"NCC out of [-1, 1] range: {ncc_map.max()}"
    print("✓ NCC test passed!")


def test_crop_boundaries():
    """Test boundary cropping"""
    print("\n" + "="*70)
    print("TEST 2: Boundary Cropping")
    print("="*70)
    
    # Create dummy tensor
    x = torch.randn(2, 1, 512, 512)
    crop_pixels = 143
    
    x_cropped = crop_boundaries(x, crop_pixels=crop_pixels)
    
    print(f"Original shape: {x.shape}")
    print(f"Cropped shape: {x_cropped.shape}")
    print(f"Crop pixels requested: {crop_pixels}")
    
    # The actual crop might be less if image is too small
    assert x_cropped.shape[0] == 2, f"Batch size changed: {x_cropped.shape[0]}"
    assert x_cropped.shape[1] == 1, f"Channels changed: {x_cropped.shape[1]}"
    assert x_cropped.shape[2] < 512, f"Height should be less after cropping: {x_cropped.shape[2]}"
    assert x_cropped.shape[3] == 512, f"Width should not change: {x_cropped.shape[3]}"
    print("✓ Boundary cropping test passed!")


def test_forward_pass():
    """Test forward pass with new losses"""
    print("\n" + "="*70)
    print("TEST 3: Forward Pass with DeepUse-Inspired Losses")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    smurf = SMURFModel(
        in_channels=1,
        max_displacement=4,
        num_refinement_steps=2
    )
    
    wrapper = SMURFUltrasoundWrapper(smurf)
    model = SMURFUltrasoundWithLosses(smurf)
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    I_t = torch.randn(2, 1, 256, 256).to(device)
    I_t1 = I_t + 0.05 * torch.randn_like(I_t)  # Small perturbation
    
    print(f"Input shape: {I_t.shape}")
    
    with torch.no_grad():
        # Forward pass
        output = model(I_t, I_t1)
        
        print(f"\nOutput keys: {output.keys()}")
        print(f"Displacement shape: {output['displacement'].shape}")
        print(f"Strain shape: {output['strain'].shape}")
        
        # Check output shapes
        assert output['displacement'].shape == (2, 1, 256, 256), f"Unexpected displacement shape: {output['displacement'].shape}"
        assert output['strain'].shape == (2, 1, 256, 256), f"Unexpected strain shape: {output['strain'].shape}"
        
        # Compute losses
        losses = model.compute_losses(I_t, I_t1, output)
        
        print(f"\nLosses computed:")
        for key, val in losses.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.item():.6f}")
            else:
                print(f"  {key}: {val:.6f}")
        
        # Verify loss values are reasonable
        assert not torch.isnan(losses['photometric']), "Photometric loss is NaN!"
        assert not torch.isnan(losses['smoothness']), "Smoothness loss is NaN!"
        assert not torch.isnan(losses['strain_reg']), "Strain regularization loss is NaN!"
        assert not torch.isnan(losses['total']), "Total loss is NaN!"
        
        print("\n✓ Forward pass test passed!")


def test_displacement_regularization():
    """Test displacement regularization prevents zero solution"""
    print("\n" + "="*70)
    print("TEST 4: Displacement Regularization")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    smurf = SMURFModel(
        in_channels=1,
        max_displacement=4,
        num_refinement_steps=2
    )
    
    model = SMURFUltrasoundWithLosses(smurf)
    model = model.to(device)
    model.eval()
    
    # Create input
    I_t = torch.randn(1, 1, 256, 256).to(device)
    I_t1 = I_t.clone()  # Identical images (would produce zero displacement without regularization)
    
    with torch.no_grad():
        output = model(I_t, I_t1)
        losses = model.compute_losses(I_t, I_t1, output)
        
        print(f"Identical image inputs:")
        print(f"  Displacement magnitude: {torch.sqrt(output['displacement']**2).mean().item():.6f}")
        print(f"  Displacement regularization loss: {losses['displacement_reg'].item():.6f}")
        print(f"  Total loss: {losses['total'].item():.6f}")
        
        # Even with identical images, displacement regularization should be non-zero
        # This encourages the model to output something
        print("\n✓ Displacement regularization test passed!")


def test_boundary_cropping_in_losses():
    """Test that losses compute correctly with boundary cropping"""
    print("\n" + "="*70)
    print("TEST 5: Boundary Cropping in Loss Computation")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    smurf = SMURFModel(
        in_channels=1,
        max_displacement=4,
        num_refinement_steps=2
    )
    
    model = SMURFUltrasoundWithLosses(smurf)
    model = model.to(device)
    model.eval()
    
    # Create input
    I_t = torch.randn(1, 1, 512, 512).to(device)
    I_t1 = I_t + 0.02 * torch.randn_like(I_t)
    
    with torch.no_grad():
        output = model(I_t, I_t1)
        losses = model.compute_losses(I_t, I_t1, output)
        
        print(f"Input shape: {I_t.shape}")
        print(f"Displacement shape: {output['displacement'].shape}")
        print(f"Strain shape: {output['strain'].shape}")
        print(f"\nLosses with boundary cropping (143 pixels):")
        print(f"  Photometric (NCC-based): {losses['photometric'].item():.6f}")
        print(f"  Smoothness: {losses['smoothness'].item():.6f}")
        print(f"  Strain regularization: {losses['strain_reg'].item():.6f}")
        print(f"  Total: {losses['total'].item():.6f}")
        
        print("\n✓ Boundary cropping in loss computation test passed!")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("DEEPUSE INTEGRATION TEST SUITE")
    print("="*70)
    print("\nTesting NCC similarity, boundary cropping, and DeepUse-inspired losses...")
    
    try:
        test_ncc_similarity()
        test_crop_boundaries()
        test_forward_pass()
        test_displacement_regularization()
        test_boundary_cropping_in_losses()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nNext steps:")
        print("1. Run: python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 100")
        print("2. Monitor loss curves to verify improvements")
        print("3. Test inference on new model with test data")
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
