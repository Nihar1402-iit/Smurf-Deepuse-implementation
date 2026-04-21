# SMURF Ultrasound Wrapper
# Adapts SMURF optical flow model for ultrasound imaging
# Outputs ReUSENet-style displacement and strain maps

import torch
import torch.nn as nn
from smurf_core import SMURFModel
from lsqse import LSQSEModule


class SMURFUltrasoundWrapper(nn.Module):
    """
    SMURF adapted for Ultrasound Elastography
    
    Converts SMURF optical flow predictions to ReUSENet-style outputs:
    - Dense displacement field U_t (axial + lateral)
    - Strain map S_t computed via LSQSE
    
    This wrapper:
    1. Runs SMURF optical flow estimation
    2. Extracts and reorders flow channels (lateral -> axial ordering)
    3. Computes strain from axial displacement using LSQSE
    4. Returns only the required outputs (no intermediate pyramids or uncertainty)
    
    Input:
        RF/IQ frames (normalized intensity, any number of channels)
        Expected: axial motion >> lateral motion
    
    Output:
        displacement: [B, 2, H, W] where channel 0=axial, channel 1=lateral
        strain: [B, 1, H, W] axial strain field
    
    Args:
        smurf_model: Pretrained SMURFModel instance
        lsqse_window_size: Window size for LSQSE strain computation (default: 5)
        strain_smoothing: Enable post-processing strain smoothing (default: True)
        strain_smoothing_type: Type of smoothing - 'gaussian', 'median', 'bilateral' (default: 'gaussian')
        return_full_output: If False, returns only {displacement, strain}
                           If True, also includes intermediate data
    """
    
    def __init__(
        self,
        smurf_model,
        lsqse_window_size=5,
        strain_smoothing=True,
        strain_smoothing_type='gaussian',
        return_full_output=False
    ):
        super().__init__()
        self.smurf = smurf_model
        self.lsqse = LSQSEModule(
            window_size=lsqse_window_size,
            strain_window=lsqse_window_size,
            filter_type=strain_smoothing_type
        )
        self.strain_smoothing = strain_smoothing
        self.return_full_output = return_full_output
    
    def forward(self, I_t, I_t1):
        """
        Compute displacement and strain from two consecutive ultrasound frames
        
        Args:
            I_t: Current frame [B, C, H, W] - RF/IQ data
            I_t1: Next frame [B, C, H, W] - RF/IQ data
        
        Returns:
            If return_full_output=False (default):
                {
                    "displacement": [B, 1, H, W],  # axial displacement only (DeepUse format)
                    "strain": [B, 1, H, W]          # axial strain
                }
            
            If return_full_output=True:
                {
                    "displacement": [B, 2, H, W],  # [axial, lateral]
                    "strain": [B, 1, H, W],
                    "u_lateral": [B, 1, H, W],
                    "u_axial": [B, 1, H, W],
                    "flow_predictions": [...]
                }
        """
        # Run SMURF optical flow
        flow_predictions, final_flow = self.smurf(I_t, I_t1)
        
        # Extract final flow
        # SMURF output: [B, 2, H, W]
        # Channel 0: horizontal (lateral)
        # Channel 1: vertical (axial)
        u_lateral = final_flow[:, 0:1, :, :]  # [B, 1, H, W]
        u_axial = final_flow[:, 1:2, :, :]    # [B, 1, H, W]
        
        # Compute strain from axial displacement
        strain = self.lsqse(u_axial, smooth=self.strain_smoothing)  # [B, 1, H, W]
        
        # Prepare output - match DeepUse format (axial displacement only by default)
        output = {
            "displacement": u_axial,  # [B, 1, H, W] - axial only (DeepUse compatible)
            "strain": strain           # [B, 1, H, W]
        }
        
        # Optional: Include full output for analysis/debugging
        if self.return_full_output:
            output.update({
                "displacement_full": torch.cat([u_axial, u_lateral], dim=1),  # [B, 2, H, W]
                "u_lateral": u_lateral,
                "u_axial": u_axial,
                "flow_predictions": flow_predictions,
            })
        
        return output
    
    def forward_displacement_only(self, I_t, I_t1):
        """
        Compute displacement only (faster if strain not needed)
        
        Args:
            I_t: Current frame [B, C, H, W]
            I_t1: Next frame [B, C, H, W]
        
        Returns:
            displacement: [B, 1, H, W] - axial displacement only (DeepUse format)
        """
        _, final_flow = self.smurf(I_t, I_t1)
        
        u_axial = final_flow[:, 1:2, :, :]
        
        return u_axial
    
    def forward_strain_only(self, I_t, I_t1):
        """
        Compute strain only (faster if displacement not needed)
        
        Args:
            I_t: Current frame [B, C, H, W]
            I_t1: Next frame [B, C, H, W]
        
        Returns:
            strain: [B, 1, H, W] - axial strain
        """
        _, final_flow = self.smurf(I_t, I_t1)
        
        u_axial = final_flow[:, 1:2, :, :]
        strain = self.lsqse(u_axial, smooth=self.strain_smoothing)
        
        return strain


class SMURFUltrasoundWithLosses(nn.Module):
    """
    Extended SMURF Ultrasound Wrapper with training losses
    
    Combines:
    - SMURF losses (photometric + smoothness + warping)
    - Optional strain regularization loss
    
    Args:
        smurf_model: Pretrained SMURFModel
        **kwargs: Passed to SMURFUltrasoundWrapper
    """
    
    def __init__(self, smurf_model, **kwargs):
        super().__init__()
        self.wrapper = SMURFUltrasoundWrapper(smurf_model, **kwargs)
        self.smurf = smurf_model
    
    def forward(self, I_t, I_t1):
        """Forward pass - returns predictions"""
        return self.wrapper(I_t, I_t1)
    
    def compute_losses(self, I_t, I_t1, output):
        """
        Compute training losses
        
        Args:
            I_t: Current frame [B, C, H, W]
            I_t1: Next frame [B, C, H, W]
            output: Model output from forward()
        
        Returns:
            losses: dict of individual losses
            total_loss: weighted sum of all losses
        """
        strain = output["strain"]
        
        # Get displacement components - if full output available, use it
        if "u_lateral" in output and "u_axial" in output:
            u_axial = output["u_axial"]      # [B, 1, H, W]
            u_lateral = output["u_lateral"]  # [B, 1, H, W]
        else:
            # Otherwise, only axial displacement is available
            u_axial = output["displacement"]  # [B, 1, H, W]
            u_lateral = torch.zeros_like(u_axial)  # No lateral component
        
        losses = {}
        
        # 1. Photometric loss (intensity constancy)
        # Warp I_t1 using displacement and compare to I_t
        photometric_loss = self._compute_photometric_loss(I_t, I_t1, u_lateral, u_axial)
        losses["photometric"] = photometric_loss
        
        # 2. Smoothness loss (penalize non-smooth displacement fields)
        smoothness_loss = self._compute_smoothness_loss(u_axial, u_lateral)
        losses["smoothness"] = smoothness_loss
        
        # 3. Strain regularization (optional - encourage smooth strain)
        strain_regularization = self._compute_strain_regularization(strain)
        losses["strain_reg"] = strain_regularization
        
        # Weighted combination
        total_loss = (
            1.0 * losses["photometric"] +
            0.1 * losses["smoothness"] +
            0.05 * losses["strain_reg"]
        )
        
        losses["total"] = total_loss
        
        return losses
    
    def _compute_photometric_loss(self, I_t, I_t1, u_lateral, u_axial):
        """
        Photometric (intensity constancy) loss
        
        Warps I_t1 using displacement field and compares to I_t
        """
        # Create sampling grid
        batch_size, _, height, width = I_t.shape
        device = I_t.device
        
        # Create base coordinate grids [H, W]
        y_coords = torch.arange(height, dtype=torch.float32, device=device)
        x_coords = torch.arange(width, dtype=torch.float32, device=device)
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Add batch dimension: [H, W] -> [1, H, W] -> [B, H, W]
        x_grid = x_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]
        y_grid = y_grid.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H, W]
        
        # Squeeze displacement to [B, H, W] if they're [B, 1, H, W]
        u_lat_squeezed = u_lateral.squeeze(1) if u_lateral.dim() == 4 else u_lateral  # [B, H, W]
        u_ax_squeezed = u_axial.squeeze(1) if u_axial.dim() == 4 else u_axial        # [B, H, W]
        
        # Apply displacement: u_lateral affects x, u_axial affects y
        x_grid_displaced = 2 * (x_grid + u_lat_squeezed) / (width - 1) - 1
        y_grid_displaced = 2 * (y_grid + u_ax_squeezed) / (height - 1) - 1
        
        # Stack to create sampling grid [B, H, W, 2]
        grid = torch.stack([x_grid_displaced, y_grid_displaced], dim=-1)
        
        # Warp I_t1 to I_t coordinates
        I_t1_warped = torch.nn.functional.grid_sample(
            I_t1, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        # Photometric loss: L1 or L2 difference
        photometric_loss = torch.mean(torch.abs(I_t - I_t1_warped))
        
        return photometric_loss
    
    def _compute_smoothness_loss(self, u_axial, u_lateral):
        """
        Smoothness loss - penalizes large gradients in displacement field
        
        Encourages piecewise smooth solutions
        """
        # Compute gradients
        grad_u_axial_x = torch.abs(u_axial[:, :, :, :-1] - u_axial[:, :, :, 1:])
        grad_u_axial_y = torch.abs(u_axial[:, :, :-1, :] - u_axial[:, :, 1:, :])
        
        grad_u_lateral_x = torch.abs(u_lateral[:, :, :, :-1] - u_lateral[:, :, :, 1:])
        grad_u_lateral_y = torch.abs(u_lateral[:, :, :-1, :] - u_lateral[:, :, 1:, :])
        
        # Mean absolute gradients
        smoothness_loss = (
            torch.mean(grad_u_axial_x) +
            torch.mean(grad_u_axial_y) +
            torch.mean(grad_u_lateral_x) +
            torch.mean(grad_u_lateral_y)
        )
        
        return smoothness_loss
    
    def _compute_strain_regularization(self, strain):
        """
        Strain regularization loss - encourages smooth strain fields
        
        Reduces noise in strain estimation
        """
        # Penalize large strain gradients (assume strain changes smoothly)
        grad_strain_x = torch.abs(strain[:, :, :, :-1] - strain[:, :, :, 1:])
        grad_strain_y = torch.abs(strain[:, :, :-1, :] - strain[:, :, 1:, :])
        
        strain_reg = torch.mean(grad_strain_x) + torch.mean(grad_strain_y)
        
        return strain_reg
