# LSQSE (Least Squares Strain Estimation) Module
# Computes strain from displacement field using least squares fitting
# Optimized for ultrasound imaging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSQSEModule(nn.Module):
    """
    Least Squares Strain Estimation (LSQSE)
    
    Computes axial strain from axial displacement field using
    least squares fitting over local windows.
    
    This is a core component of ReUSENet-style ultrasound analysis.
    
    Args:
        window_size: Size of local window for least squares fitting (default: 5)
        strain_window: Window size for strain smoothing (default: 5)
        filter_type: Type of smoothing filter - 'gaussian', 'median', 'bilateral' (default: 'gaussian')
    """
    
    def __init__(self, window_size=5, strain_window=5, filter_type='gaussian'):
        super().__init__()
        self.window_size = window_size
        self.strain_window = strain_window
        self.filter_type = filter_type
        
        # Precompute gradient kernel for finite difference approximation
        # Sobel-like kernels for axial gradient (dy direction)
        self.register_buffer('grad_kernel_y', torch.tensor([
            [-1.0, -2.0, -1.0],
            [ 0.0,  0.0,  0.0],
            [ 1.0,  2.0,  1.0]
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0)
        
        # Kernel for least squares fitting (polynomial approximation)
        # Optimized for local strain estimation
        self._init_lsqse_kernels()
    
    def _init_lsqse_kernels(self):
        """Initialize kernels for least squares strain computation"""
        # For window_size x window_size window, compute gradient using least squares
        w = self.window_size
        center = w // 2
        
        # Create coordinate arrays
        y_coords = torch.arange(w, dtype=torch.float32) - center
        
        # Linear least squares: find best fit y = mx + c
        # We want to fit displacement vs. position and extract slope (strain)
        self.register_buffer('lsqse_y_weights', y_coords.view(1, -1))
    
    def forward(self, u_axial, smooth=True):
        """
        Compute strain from axial displacement field
        
        Args:
            u_axial: Axial displacement [B, 1, H, W]
            smooth: Whether to apply post-smoothing (default: True)
        
        Returns:
            strain: Strain map [B, 1, H, W]
        """
        batch_size, _, height, width = u_axial.shape
        device = u_axial.device
        
        # Method 1: Direct gradient computation (fastest, suitable for real-time)
        strain = self._compute_strain_gradient(u_axial)
        
        # Method 2: Optional - Least squares fitting (more robust to noise)
        # strain = self._compute_strain_lsqse(u_axial)
        
        # Post-processing: smoothing
        if smooth:
            strain = self._smooth_strain(strain)
        
        return strain
    
    def _compute_strain_gradient(self, u_axial):
        """
        Compute strain as gradient of displacement using Sobel operator
        
        Strain = du/dy (axial strain)
        
        Uses finite difference approximation:
        strain[i,j] ≈ (u[i+1,j] - u[i-1,j]) / 2
        """
        # Pad for convolution
        u_padded = F.pad(u_axial, (1, 1, 1, 1), mode='reflect')
        
        # Apply Sobel kernel for gradient in y-direction
        strain = F.conv2d(u_padded, self.grad_kernel_y, padding=0)
        
        return strain
    
    def _compute_strain_lsqse(self, u_axial):
        """
        Compute strain using least squares fitting over local windows
        
        For each pixel, fit a linear model to displacement values in a local window:
        u(y) = strain * y + offset
        
        Returns the slope (strain) of the best-fit line.
        """
        batch_size, _, height, width = u_axial.shape
        device = u_axial.device
        w = self.window_size
        pad = w // 2
        
        # Pad input
        u_padded = F.pad(u_axial, (pad, pad, pad, pad), mode='reflect')
        
        strain = torch.zeros_like(u_axial)
        
        # Extract windows and compute strain for each
        for i in range(height):
            for j in range(width):
                # Extract window
                window = u_padded[:, :, i:i+w, j:j+w]  # [B, 1, w, w]
                
                # Flatten window values
                u_vals = window.squeeze(1).reshape(batch_size, -1)  # [B, w*w]
                
                # Create position matrix (y-coordinates)
                y_pos = torch.arange(w, dtype=torch.float32, device=device) - pad
                y_pos = y_pos.view(1, -1).expand(batch_size, -1)  # [B, w]
                y_pos = y_pos.repeat(1, w)  # [B, w*w]
                
                # Least squares: (Y'Y)^-1 Y'u
                # Compute slope (strain)
                ones = torch.ones_like(y_pos)
                A = torch.stack([y_pos, ones], dim=2)  # [B, w*w, 2]
                
                # Normal equations: A'A x = A'u
                ATA = torch.bmm(A.transpose(1, 2), A)  # [B, 2, 2]
                ATu = torch.bmm(A.transpose(1, 2), u_vals.unsqueeze(2))  # [B, 2, 1]
                
                # Solve for coefficients
                try:
                    coeffs = torch.linalg.solve(ATA, ATu)  # [B, 2, 1]
                    strain[:, :, i, j] = coeffs[:, 0, 0]  # Extract slope (strain)
                except:
                    # Fallback: use simple gradient
                    strain[:, :, i, j] = (u_padded[:, :, i+pad+1, j] - u_padded[:, :, i+pad-1, j]) / 2
        
        return strain
    
    def _smooth_strain(self, strain):
        """
        Post-process strain map with smoothing
        
        Reduces noise while preserving boundaries
        """
        if self.filter_type == 'gaussian':
            return self._gaussian_smooth(strain)
        elif self.filter_type == 'median':
            return self._median_smooth(strain)
        elif self.filter_type == 'bilateral':
            return self._bilateral_smooth(strain)
        else:
            return strain
    
    def _gaussian_smooth(self, strain):
        """Apply Gaussian smoothing"""
        kernel_size = self.strain_window
        kernel = self._create_gaussian_kernel(kernel_size, sigma=1.0, device=strain.device)
        
        # Pad and convolve
        pad = kernel_size // 2
        strain_padded = F.pad(strain, (pad, pad, pad, pad), mode='reflect')
        smoothed = F.conv2d(strain_padded, kernel, padding=0)
        
        return smoothed
    
    def _median_smooth(self, strain):
        """Apply median filtering (2D)"""
        kernel_size = self.strain_window
        pad = kernel_size // 2
        
        # Use unfold to extract patches, then apply median
        unfolded = F.unfold(
            F.pad(strain, (pad, pad, pad, pad), mode='reflect'),
            kernel_size=kernel_size
        )  # [B, kernel_size^2, num_patches]
        
        # Compute median
        smoothed = torch.median(unfolded, dim=1, keepdim=True)[0]
        
        # Fold back
        smoothed = F.fold(
            smoothed,
            output_size=strain.shape[-2:],
            kernel_size=1
        )
        
        return smoothed
    
    def _bilateral_smooth(self, strain):
        """
        Apply bilateral filtering (preserves edges)
        
        Simplified version using Gaussian kernels with intensity weighting
        """
        kernel_size = self.strain_window
        sigma_spatial = 1.0
        sigma_intensity = 0.1
        
        pad = kernel_size // 2
        strain_padded = F.pad(strain, (pad, pad, pad, pad), mode='reflect')
        
        # Create spatial Gaussian kernel
        spatial_kernel = self._create_gaussian_kernel(
            kernel_size, sigma_spatial, 
            device=strain.device
        )
        
        batch_size, _, height, width = strain.shape
        smoothed = torch.zeros_like(strain)
        
        for i in range(height):
            for j in range(width):
                # Extract window
                window = strain_padded[:, :, i:i+kernel_size, j:j+kernel_size]
                center_val = strain[:, :, i:i+1, j:j+1]
                
                # Intensity difference weight
                intensity_weight = torch.exp(
                    -((window - center_val) ** 2) / (2 * sigma_intensity ** 2)
                )
                
                # Combined weight
                combined_weight = spatial_kernel * intensity_weight
                combined_weight = combined_weight / (combined_weight.sum() + 1e-8)
                
                # Weighted average
                smoothed[:, :, i:i+1, j:j+1] = (window * combined_weight).sum(dim=(2, 3), keepdim=True)
        
        return smoothed
    
    @staticmethod
    def _create_gaussian_kernel(kernel_size, sigma, device='cpu'):
        """Create 2D Gaussian kernel"""
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        y = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, kernel_size, kernel_size)
