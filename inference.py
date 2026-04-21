# SMURF Ultrasound Inference Script
# Demonstrates inference and visualization of displacement and strain maps

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

from smurf_core import SMURFModel
from smurf_ultrasound_wrapper import SMURFUltrasoundWrapper


class UltrasoundInference:
    """Inference pipeline for SMURF Ultrasound model"""
    
    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: Path to saved checkpoint
            device: torch device
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        smurf = SMURFModel(
            in_channels=1,  # RF frames
            max_displacement=4,
            num_refinement_steps=4
        )
        
        self.model = SMURFUltrasoundWrapper(
            smurf,
            lsqse_window_size=5,
            strain_smoothing=True,
            strain_smoothing_type='gaussian',
            return_full_output=False
        ).to(self.device)
        
        # Load checkpoint if provided
        if model_path:
            self.load_checkpoint(model_path)
        
        self.model.eval()
    
    def load_checkpoint(self, model_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, I_t, I_t1):
        """
        Predict displacement and strain
        
        Args:
            I_t: Current frame [B, C, H, W] or [C, H, W]
            I_t1: Next frame [B, C, H, W] or [C, H, W]
        
        Returns:
            output: dict with "displacement" and "strain"
        """
        # Add batch dimension if needed
        if I_t.dim() == 3:
            I_t = I_t.unsqueeze(0)
        if I_t1.dim() == 3:
            I_t1 = I_t1.unsqueeze(0)
        
        I_t = I_t.to(self.device)
        I_t1 = I_t1.to(self.device)
        
        with torch.no_grad():
            output = self.model(I_t, I_t1)
        
        # Move to CPU for processing
        output = {k: v.cpu() for k, v in output.items()}
        
        return output
    
    def predict_displacement_only(self, I_t, I_t1):
        """Fast prediction of displacement only"""
        if I_t.dim() == 3:
            I_t = I_t.unsqueeze(0)
        if I_t1.dim() == 3:
            I_t1 = I_t1.unsqueeze(0)
        
        I_t = I_t.to(self.device)
        I_t1 = I_t1.to(self.device)
        
        with torch.no_grad():
            displacement = self.model.wrapper.forward_displacement_only(I_t, I_t1)
        
        return displacement.cpu()
    
    def predict_strain_only(self, I_t, I_t1):
        """Fast prediction of strain only"""
        if I_t.dim() == 3:
            I_t = I_t.unsqueeze(0)
        if I_t1.dim() == 3:
            I_t1 = I_t1.unsqueeze(0)
        
        I_t = I_t.to(self.device)
        I_t1 = I_t1.to(self.device)
        
        with torch.no_grad():
            strain = self.model.wrapper.forward_strain_only(I_t, I_t1)
        
        return strain.cpu()


class UltrasoundVisualizer:
    """Visualization utilities for ultrasound displacement and strain"""
    
    @staticmethod
    def create_displacement_heatmap(
        displacement,
        I_t=None,
        figsize=(15, 5),
        cmap_axial='coolwarm',
        cmap_lateral='coolwarm',
        normalize=True
    ):
        """
        Create visualization of displacement field
        
        Args:
            displacement: [B, 2, H, W] or [2, H, W] - [axial, lateral]
            I_t: Optional background ultrasound image [1, H, W]
            figsize: Figure size
            cmap_axial: Colormap for axial displacement
            cmap_lateral: Colormap for lateral displacement
            normalize: Normalize displacement for visualization
        
        Returns:
            fig, axes
        """
        if displacement.dim() == 4:
            displacement = displacement[0]  # Take first batch
        
        u_axial = displacement[0].numpy()
        u_lateral = displacement[1].numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Normalize for visualization
        if normalize:
            u_axial_norm = (u_axial - u_axial.min()) / (u_axial.max() - u_axial.min() + 1e-8)
            u_lateral_norm = (u_lateral - u_lateral.min()) / (u_lateral.max() - u_lateral.min() + 1e-8)
        else:
            u_axial_norm = u_axial
            u_lateral_norm = u_lateral
        
        # Plot axial displacement
        im0 = axes[0].imshow(u_axial_norm, cmap=cmap_axial)
        axes[0].set_title("Axial Displacement (mm)")
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], label='Displacement')
        
        # Plot lateral displacement
        im1 = axes[1].imshow(u_lateral_norm, cmap=cmap_lateral)
        axes[1].set_title("Lateral Displacement (mm)")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], label='Displacement')
        
        # Plot magnitude
        magnitude = np.sqrt(u_axial ** 2 + u_lateral ** 2)
        im2 = axes[2].imshow(magnitude, cmap='viridis')
        axes[2].set_title("Total Displacement Magnitude")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], label='Magnitude')
        
        # Optional: overlay on ultrasound image
        if I_t is not None:
            if I_t.dim() == 3:
                I_t = I_t[0]  # Take first channel
            I_t_np = I_t.numpy()
            for ax in axes:
                ax.imshow(I_t_np, cmap='gray', alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def create_strain_heatmap(
        strain,
        I_t=None,
        figsize=(8, 6),
        cmap='RdBu_r',
        vmin=None,
        vmax=None,
        normalize=True
    ):
        """
        Create visualization of strain map
        
        Args:
            strain: [B, 1, H, W] or [1, H, W] or [H, W]
            I_t: Optional background ultrasound image
            figsize: Figure size
            cmap: Colormap
            vmin, vmax: Value range for colormap
            normalize: Normalize strain for visualization
        
        Returns:
            fig, ax
        """
        if strain.dim() == 4:
            strain = strain[0]  # Take first batch
        if strain.dim() == 3:
            strain = strain[0]  # Take first channel
        
        strain_np = strain.numpy()
        
        # Normalize for visualization
        if normalize:
            if vmin is None:
                vmin = np.percentile(strain_np, 2)
            if vmax is None:
                vmax = np.percentile(strain_np, 98)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(strain_np, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title("Axial Strain (LSQSE)")
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, label='Strain (du/dy)')
        
        # Overlay ultrasound image if provided
        if I_t is not None:
            if I_t.dim() == 3:
                I_t = I_t[0]
            I_t_np = I_t.numpy()
            ax.imshow(I_t_np, cmap='gray', alpha=0.2)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def create_displacement_vectors(
        displacement,
        I_t=None,
        stride=10,
        figsize=(10, 8),
        scale=1.0
    ):
        """
        Visualize displacement as vector field
        
        Args:
            displacement: [B, 2, H, W] or [2, H, W]
            I_t: Optional background image
            stride: Plot every Nth vector
            figsize: Figure size
            scale: Scale factor for arrow length
        
        Returns:
            fig, ax
        """
        if displacement.dim() == 4:
            displacement = displacement[0]
        
        u_axial = displacement[0].numpy()
        u_lateral = displacement[1].numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot background
        if I_t is not None:
            if I_t.dim() == 3:
                I_t = I_t[0]
            ax.imshow(I_t.numpy(), cmap='gray')
        
        # Create grid of vectors
        H, W = u_axial.shape
        y, x = np.mgrid[0:H:stride, 0:W:stride]
        
        # Plot vectors
        u_mag = np.sqrt(u_axial ** 2 + u_lateral ** 2)
        u_mag_sampled = u_mag[::stride, ::stride]
        
        quiver = ax.quiver(
            x, y,
            u_lateral[::stride, ::stride],  # x component
            u_axial[::stride, ::stride],     # y component
            u_mag_sampled,
            cmap='viridis',
            scale=scale,
            scale_units='inches',
            width=0.003
        )
        
        ax.set_title("Displacement Vector Field")
        plt.colorbar(quiver, ax=ax, label='Magnitude')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def create_strain_histogram(
        strain,
        figsize=(10, 6),
        bins=50
    ):
        """
        Create histogram of strain values
        
        Args:
            strain: [B, 1, H, W] or [1, H, W] or [H, W]
            figsize: Figure size
            bins: Number of bins
        
        Returns:
            fig, ax
        """
        if strain.dim() >= 3:
            strain = strain.squeeze()
        
        strain_np = strain.numpy().flatten()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(strain_np, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel("Strain Value (du/dy)")
        ax.set_ylabel("Frequency")
        ax.set_title("Strain Distribution")
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = strain_np.mean()
        std_val = strain_np.std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, label=f'Std: {std_val:.4f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)
        
        ax.legend()
        
        plt.tight_layout()
        return fig, ax


def example_inference():
    """Example inference pipeline"""
    print("=" * 60)
    print("SMURF Ultrasound Inference Example")
    print("=" * 60)
    
    # Create dummy RF frames
    print("\n1. Creating dummy ultrasound frames...")
    height, width = 256, 256
    
    # Frame at time t
    I_t = torch.randn(1, height, width) * 0.1
    
    # Frame at time t+1 (with simulated motion)
    I_t1 = torch.roll(I_t, shifts=3, dims=1)  # axial motion
    I_t1 = torch.roll(I_t1, shifts=1, dims=2)  # lateral motion
    I_t1 = I_t1 + torch.randn_like(I_t1) * 0.05
    
    # Initialize inference engine
    print("\n2. Initializing SMURF Ultrasound model...")
    inference = UltrasoundInference()
    
    # Run inference
    print("\n3. Running inference...")
    output = inference.predict(I_t, I_t1)
    
    displacement = output["displacement"]
    strain = output["strain"]
    
    print(f"   Displacement shape: {displacement.shape}")
    print(f"   Strain shape: {strain.shape}")
    
    # Print statistics
    print("\n4. Displacement Statistics:")
    print(f"   Axial (ch0): mean={displacement[0, 0].mean():.4f}, std={displacement[0, 0].std():.4f}")
    print(f"   Lateral (ch1): mean={displacement[0, 1].mean():.4f}, std={displacement[0, 1].std():.4f}")
    
    print("\n5. Strain Statistics:")
    print(f"   Mean: {strain[0, 0].mean():.6f}")
    print(f"   Std: {strain[0, 0].std():.6f}")
    print(f"   Min: {strain[0, 0].min():.6f}")
    print(f"   Max: {strain[0, 0].max():.6f}")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    
    visualizer = UltrasoundVisualizer()
    
    # Displacement heatmaps
    fig1, axes1 = visualizer.create_displacement_heatmap(
        displacement,
        I_t=I_t,
        figsize=(15, 5)
    )
    fig1.savefig("displacement_heatmap.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: displacement_heatmap.png")
    
    # Strain heatmap
    fig2, ax2 = visualizer.create_strain_heatmap(
        strain,
        I_t=I_t,
        figsize=(8, 6)
    )
    fig2.savefig("strain_heatmap.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: strain_heatmap.png")
    
    # Displacement vectors
    fig3, ax3 = visualizer.create_displacement_vectors(
        displacement,
        I_t=I_t,
        stride=15,
        figsize=(10, 8)
    )
    fig3.savefig("displacement_vectors.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: displacement_vectors.png")
    
    # Strain histogram
    fig4, ax4 = visualizer.create_strain_histogram(
        strain,
        figsize=(10, 6)
    )
    fig4.savefig("strain_histogram.png", dpi=150, bbox_inches='tight')
    print("   ✓ Saved: strain_histogram.png")
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    example_inference()
