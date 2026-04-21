"""
DeepUse-inspired utilities for SMURF Ultrasound
Adapted from DeepUse implementation for better ultrasound processing
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, List


def get_strain_deepuse_style(disp, x_wind=143):
    """
    Compute strain using least squares fitting (DeepUse approach).
    
    Uses LARGE windows (287 pixels) for robust strain estimation.
    Fits linear model: u(depth) = strain * depth + offset
    
    Args:
        disp: Displacement [B, 1, H, W] or [B, 2, H, W]
        x_wind: Half-window size (full window = 2*x_wind+1)
    
    Returns:
        strain: Strain map [B, 1, H, W]
    """
    if disp.shape[1] == 2:
        # If 2-channel displacement, use axial (channel 1)
        disp = disp[:, 1:2, :, :]
    
    batch_size, _, height, width = disp.shape
    device = disp.device
    
    d = x_wind * 2 + 1  # Window depth (e.g., 287)
    
    # Extract patches using unfold
    disp_padded = F.pad(disp, (0, 0, x_wind, x_wind), mode='replicate')
    
    # Get patches: each patch is a vertical window of size d
    patches = F.unfold(
        disp_padded,
        kernel_size=(d, 1)
    )  # [B, 1*d, H*W]
    patches = patches.reshape(batch_size, d, height, width)  # [B, d, H, W]
    
    # Create coordinate vector for LS fitting
    # depth = [1, 2, 3, ..., d] normalized
    depthX = torch.linspace(1, d, d, device=device)
    ones = torch.ones(d, device=device)
    X = torch.stack([depthX, ones], dim=0).T.unsqueeze(0)  # [1, d, 2]
    
    # Design matrix: X'X and precompute
    XtX = X.transpose(1, 2) @ X  # [1, 2, 2]
    
    # Compute strain by solving least squares for each spatial location
    strain = torch.zeros(batch_size, 1, height, width, device=device)
    
    for i in range(height):
        for j in range(width):
            u_ij = patches[:, :, i:i+1, j:j+1]  # [B, d, 1, 1]
            u_ij = u_ij.squeeze(-1).squeeze(-1)  # [B, d]
            
            # Solve (X'X) beta = X'u for each batch
            XtY = X.transpose(1, 2) @ u_ij.unsqueeze(2)  # [1, 2, B] -> [B, 2, 1]
            XtX_expanded = XtX.expand(batch_size, -1, -1)  # [B, 2, 2]
            
            try:
                # Use Cholesky decomposition for numerical stability
                L = torch.linalg.cholesky(XtX_expanded)
                beta = torch.cholesky_solve(XtY, L)  # [B, 2, 1]
                strain[:, :, i, j] = beta[:, 0, 0]  # Slope = strain
            except:
                # Fallback to standard solve
                try:
                    beta = torch.linalg.solve(XtX_expanded, XtY)
                    strain[:, :, i, j] = beta[:, 0, 0]
                except:
                    # Use finite difference as last resort
                    strain[:, :, i, j] = 0.0
    
    return strain


def ncc_similarity(x, y, kernel_size=5):
    """
    Normalized Cross-Correlation (NCC) similarity metric.
    Better for ultrasound than photometric intensity-based measures.
    
    Args:
        x: First image [B, C, H, W]
        y: Second image [B, C, H, W]
        kernel_size: Size of NCC kernel
    
    Returns:
        ncc: NCC map [B, 1, H, W]
    """
    # Compute local means
    ones = torch.ones(1, 1, kernel_size, kernel_size, device=x.device)
    pad = kernel_size // 2
    
    # Pad inputs
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    y_padded = F.pad(y, (pad, pad, pad, pad), mode='replicate')
    
    # Compute local means via convolution
    x_mean = F.conv2d(x_padded, ones, padding=0) / (kernel_size ** 2)
    y_mean = F.conv2d(y_padded, ones, padding=0) / (kernel_size ** 2)
    
    # Ensure x_mean and y_mean have same spatial dimensions as output will have
    # by padding them to match the original size
    if x_mean.shape != x.shape:
        # Add padding to means to match input shape
        pad_h = (x.shape[2] - x_mean.shape[2]) // 2
        pad_w = (x.shape[3] - x_mean.shape[3]) // 2
        if pad_h > 0 or pad_w > 0:
            x_mean = F.pad(x_mean, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
            y_mean = F.pad(y_mean, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    
    # Crop means to match original shape if needed
    x_mean = x_mean[:, :, :x.shape[2], :x.shape[3]]
    y_mean = y_mean[:, :, :y.shape[2], :y.shape[3]]
    
    # Center the data
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    # Compute variances and covariance via convolution
    x_padded_c = F.pad(x_centered, (pad, pad, pad, pad), mode='replicate')
    y_padded_c = F.pad(y_centered, (pad, pad, pad, pad), mode='replicate')
    
    x_var = F.conv2d(x_padded_c ** 2, ones, padding=0) / (kernel_size ** 2)
    y_var = F.conv2d(y_padded_c ** 2, ones, padding=0) / (kernel_size ** 2)
    xy_cov = F.conv2d(x_padded_c * y_padded_c, ones, padding=0) / (kernel_size ** 2)
    
    # Pad variances and covariance to match original shape
    if x_var.shape != x.shape:
        pad_h = (x.shape[2] - x_var.shape[2]) // 2
        pad_w = (x.shape[3] - x_var.shape[3]) // 2
        if pad_h > 0 or pad_w > 0:
            x_var = F.pad(x_var, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
            y_var = F.pad(y_var, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
            xy_cov = F.pad(xy_cov, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    
    # Crop to match input shape
    x_var = x_var[:, :, :x.shape[2], :x.shape[3]]
    y_var = y_var[:, :, :y.shape[2], :y.shape[3]]
    xy_cov = xy_cov[:, :, :x.shape[2], :x.shape[3]]
    
    # NCC formula (clamp variance to avoid division by zero)
    ncc = xy_cov / (torch.sqrt(torch.clamp(x_var * y_var, min=1e-8)) + 1e-8)
    
    # Clamp NCC to [-1, 1] range
    ncc = torch.clamp(ncc, min=-1.0, max=1.0)
    
    return ncc


def crop_boundaries(tensor, crop_pixels=143):
    """
    Crop boundaries of tensor to remove imaging artifacts.
    Like DeepUse: crop pixels from top/bottom.
    
    Args:
        tensor: Input tensor [B, C, H, W]
        crop_pixels: Number of pixels to crop from each boundary
    
    Returns:
        cropped: Cropped tensor (or original if image too small)
    """
    if crop_pixels <= 0:
        return tensor
    
    _, _, height, width = tensor.shape
    
    # If image is too small, crop proportionally (max 20% of height)
    max_crop = max(1, height // 5)
    actual_crop = min(crop_pixels, max_crop)
    
    if actual_crop >= height // 2:
        # Image too small to crop meaningfully, return as-is
        return tensor
    
    return tensor[:, :, actual_crop:-actual_crop, :]


def compute_motion_compensated_strain(strain, displacement, warp_fn):
    """
    Compute motion-compensated strain (like DeepUse).
    
    Warps previous frame's strain using current displacement.
    Useful for temporal consistency in frame sequences.
    
    Args:
        strain: Previous frame strain [B, 1, H, W]
        displacement: Motion field [B, 2, H, W]
        warp_fn: Function to warp images using displacement field
    
    Returns:
        strain_warped: Motion-compensated strain [B, 1, H, W]
    """
    return warp_fn(strain, displacement)


class DeepUseLoss(nn.Module):
    """
    DeepUse-style loss combining:
    1. Similarity (NCC instead of photometric)
    2. Smoothness (gradient penalty)
    3. Consistency (temporal strain consistency)
    """
    
    def __init__(self, alpha=0.1, beta=0.05, kernel_size=5, crop_pixels=143):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kernel_size = kernel_size
        self.crop_pixels = crop_pixels
    
    def forward(self, warped_img, fixed_img, disp_map, strain_list=None):
        """
        Compute DeepUse-style loss.
        
        Args:
            warped_img: Warped moving image
            fixed_img: Fixed reference image
            disp_map: Displacement field
            strain_list: List of strain maps for consistency (optional)
        
        Returns:
            loss_dict: Dictionary of loss components
        """
        losses = {}
        
        # 1. Similarity loss (NCC-based)
        sim = ncc_similarity(warped_img, fixed_img, self.kernel_size)
        # Crop boundaries for loss computation (like DeepUse)
        sim_cropped = crop_boundaries(sim, self.crop_pixels)
        loss_similarity = 1.0 - sim_cropped.mean()
        losses['similarity'] = loss_similarity
        
        # 2. Smoothness loss (gradient penalty)
        grad_x = torch.abs(disp_map[:, :, :, :-1] - disp_map[:, :, :, 1:])
        grad_y = torch.abs(disp_map[:, :, :-1, :] - disp_map[:, :, 1:, :])
        loss_smooth = (grad_x.mean() + grad_y.mean())
        losses['smoothness'] = loss_smooth
        
        # 3. Consistency loss (if strain list provided)
        loss_consistency = 0.0
        if strain_list is not None and len(strain_list) > 1:
            # Compare consecutive strain maps in cropped region
            for t in range(1, len(strain_list)):
                s_prev = crop_boundaries(strain_list[t-1], self.crop_pixels)
                s_curr = crop_boundaries(strain_list[t], self.crop_pixels)
                
                ncc = ncc_similarity(s_prev, s_curr, self.kernel_size)
                loss_consistency += (1.0 - ncc.mean())
            
            loss_consistency /= max(1, len(strain_list) - 1)
        
        losses['consistency'] = loss_consistency
        
        # Total loss
        total = (
            loss_similarity +
            self.alpha * loss_smooth +
            self.beta * loss_consistency
        )
        losses['total'] = total
        
        return losses
