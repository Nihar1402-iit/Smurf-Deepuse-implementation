# SMURF Ultrasound Test Inference Script
# Runs inference on test data and visualizes results
# Outputs in DeepUse format (.mat files) for compatibility

import torch
import numpy as np
from pathlib import Path
import json
import time
import scipy.io as sio
import matplotlib.pyplot as plt

from smurf_core import SMURFModel
from smurf_ultrasound_wrapper import SMURFUltrasoundWrapper
from data_loaders import RawUltrasoundDataset
from inference import UltrasoundVisualizer
from utils import UltrasoundPreprocessor


class TestConfig:
    """Test configuration"""
    def __init__(self):
        self.test_data_dir = "/Users/niharshah/Desktop/Omnistrain/our_algo/test_data_deepuse"
        self.checkpoint_path = "checkpoints/best_model.pt"
        self.output_dir = Path("test_results")
        self.frame_height = 512
        self.frame_width = 1000


def run_inference_on_test_data():
    """Run inference on all test data"""
    config = TestConfig()
    config.output_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create model
    print("Creating model...")
    smurf = SMURFModel(in_channels=1, max_displacement=4, num_refinement_steps=4)
    wrapper = SMURFUltrasoundWrapper(smurf).to(device)
    
    # Load checkpoint if exists
    if Path(config.checkpoint_path).exists():
        print(f"Loading checkpoint from {config.checkpoint_path}...")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            wrapper.load_state_dict(checkpoint["model_state_dict"])
        else:
            wrapper.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {config.checkpoint_path}")
        print("Using untrained model for demonstration\n")
    
    wrapper.eval()
    
    # Load test dataset
    print(f"Loading test dataset from {config.test_data_dir}...")
    test_dataset = RawUltrasoundDataset(
        config.test_data_dir,
        frame_height=config.frame_height,
        frame_width=config.frame_width,
        normalize=True
    )
    
    if len(test_dataset) == 0:
        print("ERROR: No test data loaded!")
        return
    
    print(f"Loaded {len(test_dataset)} test frame pairs\n")
    
    # Run inference
    print("=" * 70)
    print("RUNNING INFERENCE ON TEST DATA")
    print("=" * 70)
    print()
    
    results = []
    visualizer = UltrasoundVisualizer()
    
    start_time = time.time()
    
    with torch.no_grad():
        for idx in range(min(10, len(test_dataset))):  # Limit to first 10 for speed
            I_t, I_t1 = test_dataset[idx]
            
            # Add batch dimension
            I_t_batch = I_t.unsqueeze(0).to(device)
            I_t1_batch = I_t1.unsqueeze(0).to(device)
            
            # Forward pass
            inference_start = time.time()
            output = wrapper(I_t_batch, I_t1_batch)
            inference_time = time.time() - inference_start
            
            displacement = output["displacement"]
            strain = output["strain"]
            
            # Move to CPU for visualization
            displacement = displacement.cpu()
            strain = strain.cpu()
            I_t_vis = I_t
            
            # Compute statistics
            u_axial = displacement[0, 0]
            u_lateral = displacement[0, 1]
            
            stats = {
                "pair_idx": idx,
                "inference_time_ms": inference_time * 1000,
                "u_axial_mean": u_axial.mean().item(),
                "u_axial_std": u_axial.std().item(),
                "u_axial_min": u_axial.min().item(),
                "u_axial_max": u_axial.max().item(),
                "u_lateral_mean": u_lateral.mean().item(),
                "u_lateral_std": u_lateral.std().item(),
                "u_lateral_min": u_lateral.min().item(),
                "u_lateral_max": u_lateral.max().item(),
                "strain_mean": strain[0, 0].mean().item(),
                "strain_std": strain[0, 0].std().item(),
                "strain_min": strain[0, 0].min().item(),
                "strain_max": strain[0, 0].max().item(),
            }
            results.append(stats)
            
            # Print results
            print(f"Frame Pair {idx+1}")
            print(f"  Inference time: {stats['inference_time_ms']:.2f} ms")
            print(f"  Axial displacement:   mean={stats['u_axial_mean']:.4f}, std={stats['u_axial_std']:.4f}")
            print(f"  Lateral displacement: mean={stats['u_lateral_mean']:.4f}, std={stats['u_lateral_std']:.4f}")
            print(f"  Strain (axial):       mean={stats['strain_mean']:.6f}, std={stats['strain_std']:.6f}")
            print()
            
            # Save visualizations
            save_test_visualizations(
                config.output_dir,
                idx,
                displacement,
                strain,
                I_t_vis,
                visualizer
            )
    
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Inference complete! Total time: {total_time:.2f}s")
    print(f"Average inference time per frame pair: {(total_time / min(10, len(test_dataset))) * 1000:.2f} ms")
    print("=" * 70)
    print()
    
    # Save results
    save_results_json(config.output_dir, results)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 70)
    print(f"Average Axial Displacement:   {np.mean([r['u_axial_mean'] for r in results]):.4f}")
    print(f"Average Lateral Displacement: {np.mean([r['u_lateral_mean'] for r in results]):.4f}")
    print(f"Average Strain:               {np.mean([r['strain_mean'] for r in results]):.6f}")
    print(f"Average Inference Time:       {np.mean([r['inference_time_ms'] for r in results]):.2f} ms")
    print()


def save_test_visualizations(output_dir, idx, displacement, strain, I_t, visualizer):
    """Save visualizations for test frame pair and outputs in DeepUse format"""
    
    # Create output subdirectory
    pair_dir = output_dir / f"pair_{idx:03d}"
    pair_dir.mkdir(exist_ok=True)
    
    # Save DeepUse format (.mat file)
    try:
        mat_output = {
            'displacement': displacement[0, 0].cpu().numpy(),  # [H, W]
            'strain': strain[0, 0].cpu().numpy(),              # [H, W]
            'bmode': UltrasoundPreprocessor.normalize_rf(I_t[0] if I_t.dim() == 4 else I_t).numpy(),  # [H, W]
        }
        sio.savemat(
            str(pair_dir / f"result_pair_{idx:03d}.mat"),
            mat_output
        )
    except Exception as e:
        print(f"  Warning: Failed to save MAT file: {e}")
    
    # Displacement heatmaps
    try:
        # Create 2-channel displacement for visualization
        displacement_vis = torch.cat([displacement[:, 0:1], torch.zeros_like(displacement[:, 0:1])], dim=1)
        fig, axes = visualizer.create_displacement_heatmap(
            displacement_vis,
            I_t=I_t,
            figsize=(12, 5)
        )
        fig.savefig(pair_dir / "displacement_heatmap.png", dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"  Warning: Failed to save displacement heatmap: {e}")
    
    # Strain heatmap
    try:
        fig, ax = visualizer.create_strain_heatmap(
            strain,
            I_t=I_t,
            figsize=(8, 6)
        )
        fig.savefig(pair_dir / "strain_heatmap.png", dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"  Warning: Failed to save strain heatmap: {e}")


def save_results_json(output_dir, results):
    """Save inference results as JSON"""
    results_file = output_dir / "results.json"
    
    # Convert to JSON-serializable format
    results_json = []
    for r in results:
        results_json.append({k: float(v) if isinstance(v, (np.floating, float)) else v 
                            for k, v in r.items()})
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to {results_file}")


def plot_training_history():
    """Plot training history if available"""
    history_file = Path("checkpoints/history.json")
    
    if not history_file.exists():
        print("No training history found")
        return
    
    with open(history_file) as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Photometric loss
    axes[0, 1].plot(history['train_photometric'], label='Photometric')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Photometric Loss')
    axes[0, 1].grid(True)
    
    # Smoothness loss
    axes[1, 0].plot(history['train_smoothness'], label='Smoothness')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Smoothness Loss')
    axes[1, 0].grid(True)
    
    # Strain regularization
    axes[1, 1].plot(history['train_strain_reg'], label='Strain Reg')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Strain Regularization Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("test_results/training_history.png", dpi=150, bbox_inches='tight')
    print("Training history plot saved to test_results/training_history.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SMURF ULTRASOUND - TEST INFERENCE")
    print("=" * 70)
    print()
    
    run_inference_on_test_data()
    
    print("\nGenerating training history plot...")
    plot_training_history()
    
    print("\n" + "=" * 70)
    print("Test inference complete!")
    print("Results saved to: test_results/")
    print("=" * 70)
