# SMURF Ultrasound Test Suite
# Comprehensive tests for all components

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from smurf_core import SMURFModel
from lsqse import LSQSEModule
from smurf_ultrasound_wrapper import SMURFUltrasoundWrapper, SMURFUltrasoundWithLosses
from utils import UltrasoundPreprocessor, DisplacementPostprocessor, StrainPostprocessor


def test_smurf_model():
    """Test SMURF core model"""
    print("=" * 60)
    print("Testing SMURF Core Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create model
    model = SMURFModel(in_channels=1, max_displacement=4, num_refinement_steps=4).to(device)
    
    # Create dummy input
    batch_size, height, width = 2, 256, 256
    I_t = torch.randn(batch_size, 1, height, width).to(device)
    I_t1 = torch.randn(batch_size, 1, height, width).to(device)
    
    # Forward pass
    flow_predictions, final_flow = model(I_t, I_t1)
    
    # Assertions
    assert len(flow_predictions) == 5, f"Expected 5 flow predictions, got {len(flow_predictions)}"
    assert final_flow.shape == (batch_size, 2, height, width), \
        f"Expected shape {(batch_size, 2, height, width)}, got {final_flow.shape}"
    
    print(f"✓ Model created successfully")
    print(f"✓ Input shape: {I_t.shape}")
    print(f"✓ Flow predictions: {len(flow_predictions)} intermediate outputs")
    print(f"✓ Final flow shape: {final_flow.shape}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def test_lsqse_module():
    """Test LSQSE strain computation"""
    print("=" * 60)
    print("Testing LSQSE Module")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create module
    lsqse = LSQSEModule(window_size=5, strain_window=5, filter_type='gaussian').to(device)
    
    # Create dummy axial displacement
    batch_size, height, width = 2, 256, 256
    u_axial = torch.sin(torch.linspace(-np.pi, np.pi, height)).view(1, 1, -1, 1).expand(batch_size, 1, height, width).to(device)
    
    # Compute strain
    strain = lsqse(u_axial, smooth=True)
    
    # Assertions
    assert strain.shape == (batch_size, 1, height, width), \
        f"Expected shape {(batch_size, 1, height, width)}, got {strain.shape}"
    assert not torch.isnan(strain).any(), "Strain contains NaN values"
    
    print(f"✓ LSQSE module created")
    print(f"✓ Input u_axial shape: {u_axial.shape}")
    print(f"✓ Output strain shape: {strain.shape}")
    print(f"✓ Strain statistics:")
    print(f"  - Mean: {strain.mean():.6f}")
    print(f"  - Std: {strain.std():.6f}")
    print(f"  - Min: {strain.min():.6f}")
    print(f"  - Max: {strain.max():.6f}")
    print()


def test_wrapper():
    """Test SMURF Ultrasound Wrapper"""
    print("=" * 60)
    print("Testing SMURF Ultrasound Wrapper")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    smurf = SMURFModel(in_channels=1, max_displacement=4, num_refinement_steps=4)
    wrapper = SMURFUltrasoundWrapper(smurf).to(device)
    
    # Create dummy input
    batch_size, height, width = 2, 256, 256
    I_t = torch.randn(batch_size, 1, height, width).to(device)
    I_t1 = torch.randn(batch_size, 1, height, width).to(device)
    
    # Forward pass
    output = wrapper(I_t, I_t1)
    
    # Check output structure
    assert "displacement" in output, "Missing 'displacement' key"
    assert "strain" in output, "Missing 'strain' key"
    
    displacement = output["displacement"]
    strain = output["strain"]
    
    # Assertions
    assert displacement.shape == (batch_size, 2, height, width), \
        f"Expected displacement shape {(batch_size, 2, height, width)}, got {displacement.shape}"
    assert strain.shape == (batch_size, 1, height, width), \
        f"Expected strain shape {(batch_size, 1, height, width)}, got {strain.shape}"
    assert not torch.isnan(displacement).any(), "Displacement contains NaN values"
    assert not torch.isnan(strain).any(), "Strain contains NaN values"
    
    print(f"✓ Wrapper created successfully")
    print(f"✓ Displacement shape: {displacement.shape} [axial, lateral]")
    print(f"✓ Strain shape: {strain.shape}")
    print(f"✓ Displacement statistics:")
    print(f"  - Axial: mean={displacement[:, 0].mean():.6f}, std={displacement[:, 0].std():.6f}")
    print(f"  - Lateral: mean={displacement[:, 1].mean():.6f}, std={displacement[:, 1].std():.6f}")
    print(f"✓ Strain statistics:")
    print(f"  - Mean: {strain.mean():.6f}, Std: {strain.std():.6f}")
    print()


def test_wrapper_with_losses():
    """Test wrapper with loss computation"""
    print("=" * 60)
    print("Testing SMURF Ultrasound With Losses")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    smurf = SMURFModel(in_channels=1, max_displacement=4, num_refinement_steps=4)
    model = SMURFUltrasoundWithLosses(smurf).to(device)
    
    # Create dummy input
    batch_size, height, width = 2, 256, 256
    I_t = torch.randn(batch_size, 1, height, width).to(device)
    I_t1 = torch.randn(batch_size, 1, height, width).to(device)
    
    # Forward pass
    output = model(I_t, I_t1)
    
    # Compute losses
    losses = model.compute_losses(I_t, I_t1, output)
    
    # Check losses
    assert "photometric" in losses, "Missing photometric loss"
    assert "smoothness" in losses, "Missing smoothness loss"
    assert "strain_reg" in losses, "Missing strain regularization loss"
    assert "total" in losses, "Missing total loss"
    
    for key, value in losses.items():
        assert not torch.isnan(value).any() and not torch.isinf(value).any(), \
            f"{key} loss contains NaN or Inf"
        print(f"✓ {key.capitalize()} loss: {value.item():.6f}")
    
    print()


def test_fast_predictions():
    """Test fast prediction methods"""
    print("=" * 60)
    print("Testing Fast Prediction Methods")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    smurf = SMURFModel(in_channels=1, max_displacement=4, num_refinement_steps=4)
    wrapper = SMURFUltrasoundWrapper(smurf).to(device)
    
    batch_size, height, width = 2, 256, 256
    I_t = torch.randn(batch_size, 1, height, width).to(device)
    I_t1 = torch.randn(batch_size, 1, height, width).to(device)
    
    # Test displacement only
    displacement = wrapper.forward_displacement_only(I_t, I_t1)
    assert displacement.shape == (batch_size, 2, height, width)
    print(f"✓ Displacement-only prediction: {displacement.shape}")
    
    # Test strain only
    strain = wrapper.forward_strain_only(I_t, I_t1)
    assert strain.shape == (batch_size, 1, height, width)
    print(f"✓ Strain-only prediction: {strain.shape}")
    
    print()


def test_preprocessing():
    """Test preprocessing utilities"""
    print("=" * 60)
    print("Testing Preprocessing Utilities")
    print("=" * 60)
    
    # Create dummy RF frame
    rf_frame = torch.randn(256, 256)
    
    # Test minmax normalization
    norm_minmax = UltrasoundPreprocessor.normalize_rf(rf_frame, method='minmax')
    assert norm_minmax.min() >= -1 and norm_minmax.max() <= 1, "MinMax normalization failed"
    print(f"✓ MinMax normalization: range [{norm_minmax.min():.2f}, {norm_minmax.max():.2f}]")
    
    # Test zscore normalization
    norm_zscore = UltrasoundPreprocessor.normalize_rf(rf_frame, method='zscore')
    assert abs(norm_zscore.mean()) < 1e-5, "ZScore normalization failed"
    print(f"✓ ZScore normalization: mean={norm_zscore.mean():.6f}, std={norm_zscore.std():.6f}")
    
    # Test log normalization
    norm_log = UltrasoundPreprocessor.normalize_rf(rf_frame, method='log')
    assert norm_log.min() >= -1 and norm_log.max() <= 1, "Log normalization failed"
    print(f"✓ Log normalization: range [{norm_log.min():.2f}, {norm_log.max():.2f}]")
    
    # Test IQ normalization
    iq_frame = torch.randn(2, 256, 256)
    norm_iq = UltrasoundPreprocessor.normalize_iq(iq_frame, method='minmax')
    assert norm_iq.shape == (2, 256, 256)
    print(f"✓ IQ normalization: shape {norm_iq.shape}")
    
    print()


def test_postprocessing():
    """Test postprocessing utilities"""
    print("=" * 60)
    print("Testing Postprocessing Utilities")
    print("=" * 60)
    
    # Create dummy displacement
    displacement = torch.randn(2, 2, 256, 256)
    
    # Test magnitude
    magnitude = DisplacementPostprocessor.magnitude_of_displacement(displacement)
    assert magnitude.shape == (2, 1, 256, 256)
    print(f"✓ Displacement magnitude: {magnitude.shape}")
    
    # Test angle
    angle = DisplacementPostprocessor.angle_of_displacement(displacement)
    assert angle.shape == (2, 1, 256, 256)
    assert angle.min() >= -np.pi and angle.max() <= np.pi
    print(f"✓ Displacement angle: {angle.shape}, range [{angle.min():.2f}, {angle.max():.2f}]")
    
    # Test strain clipping
    strain = torch.randn(2, 1, 256, 256)
    clipped = StrainPostprocessor.clip_strain(strain, min_val=-0.5, max_val=0.5)
    assert clipped.min() >= -0.5 and clipped.max() <= 0.5
    print(f"✓ Strain clipping: range [{clipped.min():.2f}, {clipped.max():.2f}]")
    
    # Test strain statistics
    stats = StrainPostprocessor.compute_strain_statistics(strain)
    assert 'mean' in stats and 'std' in stats and 'min' in stats and 'max' in stats
    print(f"✓ Strain statistics: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    
    print()


def test_output_format():
    """Test that outputs match ReUSENet format"""
    print("=" * 60)
    print("Testing Output Format (ReUSENet Compatibility)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    smurf = SMURFModel(in_channels=1, max_displacement=4, num_refinement_steps=4)
    wrapper = SMURFUltrasoundWrapper(smurf).to(device)
    
    batch_size, height, width = 1, 256, 256
    I_t = torch.randn(batch_size, 1, height, width).to(device)
    I_t1 = torch.randn(batch_size, 1, height, width).to(device)
    
    output = wrapper(I_t, I_t1)
    
    # Check output keys
    assert set(output.keys()) == {"displacement", "strain"}, \
        f"Output keys don't match. Got {set(output.keys())}"
    print("✓ Output keys match ReUSENet format: displacement, strain")
    
    # Check displacement format
    displacement = output["displacement"]
    assert displacement.shape[1] == 2, f"Displacement should have 2 channels, got {displacement.shape[1]}"
    print(f"✓ Displacement: {displacement.shape} [B, (axial, lateral), H, W]")
    
    # Check strain format
    strain = output["strain"]
    assert strain.shape[1] == 1, f"Strain should have 1 channel, got {strain.shape[1]}"
    print(f"✓ Strain: {strain.shape} [B, 1, H, W]")
    
    # Verify no extra keys (only ReUSENet output)
    assert len(output) == 2, f"Output should have exactly 2 keys, got {len(output)}"
    print("✓ No extra outputs (flow pyramids, uncertainty, occlusion)")
    
    print()


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("█" * 60)
    print("SMURF ULTRASOUND TEST SUITE")
    print("█" * 60)
    print()
    
    try:
        test_smurf_model()
        test_lsqse_module()
        test_wrapper()
        test_wrapper_with_losses()
        test_fast_predictions()
        test_preprocessing()
        test_postprocessing()
        test_output_format()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise


if __name__ == "__main__":
    run_all_tests()
