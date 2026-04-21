# SMURF Ultrasound Utilities
# Helper functions for data loading, preprocessing, and postprocessing

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import cv2


class UltrasoundPreprocessor:
    """Preprocessing utilities for ultrasound frames"""
    
    @staticmethod
    def normalize_rf(rf_frame, method='minmax', eps=1e-8):
        """
        Normalize RF frame for network input
        
        Args:
            rf_frame: [C, H, W] or [H, W]
            method: 'minmax' (default), 'zscore', 'log'
        
        Returns:
            normalized: [C, H, W] or [H, W] in range [-1, 1] or [0, 1]
        """
        if method == 'minmax':
            # Linear normalization to [-1, 1]
            rf_min = rf_frame.min()
            rf_max = rf_frame.max()
            if rf_max - rf_min < eps:
                return torch.zeros_like(rf_frame)
            normalized = 2 * (rf_frame - rf_min) / (rf_max - rf_min + eps) - 1
        
        elif method == 'zscore':
            # Standardization to zero mean, unit variance
            rf_mean = rf_frame.mean()
            rf_std = rf_frame.std()
            if rf_std < eps:
                return torch.zeros_like(rf_frame)
            normalized = (rf_frame - rf_mean) / (rf_std + eps)
        
        elif method == 'log':
            # Log compression (common in RF processing)
            normalized = torch.log1p(torch.abs(rf_frame)) / torch.log1p(torch.abs(rf_frame).max())
            normalized = 2 * normalized - 1
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    @staticmethod
    def normalize_iq(iq_frame, method='minmax', eps=1e-8):
        """
        Normalize IQ frame (2 channels: I and Q)
        
        Args:
            iq_frame: [2, H, W]
            method: 'minmax' (default), 'zscore'
        
        Returns:
            normalized: [2, H, W] in range [-1, 1]
        """
        if iq_frame.shape[0] != 2:
            raise ValueError("IQ frame must have 2 channels")
        
        # Process I and Q channels separately
        I_norm = UltrasoundPreprocessor.normalize_rf(iq_frame[0:1], method=method, eps=eps)
        Q_norm = UltrasoundPreprocessor.normalize_rf(iq_frame[1:2], method=method, eps=eps)
        
        return torch.cat([I_norm, Q_norm], dim=0)
    
    @staticmethod
    def gaussian_blur(frame, kernel_size=5, sigma=1.0):
        """Apply Gaussian blur for noise reduction"""
        if frame.dim() == 2:
            frame = frame.unsqueeze(0).unsqueeze(0)
        elif frame.dim() == 3:
            frame = frame.unsqueeze(1)
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
        kernel = kernel / kernel.sum()
        
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        
        blurred = F.conv2d(frame, kernel, padding=kernel_size // 2)
        
        return blurred.squeeze()
    
    @staticmethod
    def adaptive_histogram_equalization(frame, clip_limit=2.0, tile_size=8):
        """
        Adaptive Histogram Equalization (CLAHE) for contrast enhancement
        
        Args:
            frame: [H, W] numpy or torch
            clip_limit: Contrast enhancement factor
            tile_size: Size of tiles for local enhancement
        
        Returns:
            enhanced: [H, W]
        """
        if isinstance(frame, torch.Tensor):
            frame_np = frame.numpy()
        else:
            frame_np = frame
        
        # Normalize to [0, 255]
        frame_np = ((frame_np - frame_np.min()) / (frame_np.max() - frame_np.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(frame_np)
        
        # Normalize back
        enhanced = (enhanced.astype(np.float32) / 255.0) * 2 - 1
        
        if isinstance(frame, torch.Tensor):
            enhanced = torch.from_numpy(enhanced)
        
        return enhanced


class DisplacementPostprocessor:
    """Postprocessing utilities for displacement fields"""
    
    @staticmethod
    def median_filter_displacement(displacement, kernel_size=5):
        """
        Apply median filtering to displacement field
        
        Args:
            displacement: [B, 2, H, W] or [2, H, W]
            kernel_size: Size of median filter kernel
        
        Returns:
            filtered: Same shape as input
        """
        if displacement.dim() == 3:
            displacement = displacement.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, C, H, W = displacement.shape
        filtered = torch.zeros_like(displacement)
        
        for b in range(B):
            for c in range(C):
                # Convert to numpy for median filtering
                disp_np = displacement[b, c].cpu().numpy()
                filtered_np = ndimage.median_filter(disp_np, size=kernel_size)
                filtered[b, c] = torch.from_numpy(filtered_np)
        
        if squeeze_output:
            filtered = filtered.squeeze(0)
        
        return filtered
    
    @staticmethod
    def bilateral_filter_displacement(displacement, spatial_sigma=1.0, intensity_sigma=1.0):
        """
        Apply bilateral filtering to displacement field (edge-preserving)
        
        Args:
            displacement: [B, 2, H, W] or [2, H, W]
            spatial_sigma: Spatial extent of kernel
            intensity_sigma: Range of kernel
        
        Returns:
            filtered: Same shape as input
        """
        if displacement.dim() == 3:
            displacement = displacement.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, C, H, W = displacement.shape
        kernel_size = int(4 * spatial_sigma + 1)
        
        filtered = torch.zeros_like(displacement)
        
        for b in range(B):
            for c in range(C):
                disp = displacement[b:b+1, c:c+1]
                
                # Simple bilateral filtering using opencv
                disp_np = disp.squeeze().cpu().numpy()
                filtered_np = cv2.bilateralFilter(
                    disp_np.astype(np.float32),
                    d=kernel_size,
                    sigmaColor=intensity_sigma,
                    sigmaSpace=spatial_sigma
                )
                filtered[b, c] = torch.from_numpy(filtered_np)
        
        if squeeze_output:
            filtered = filtered.squeeze(0)
        
        return filtered
    
    @staticmethod
    def remove_displacement_outliers(displacement, threshold=3.0):
        """
        Remove displacement outliers using median absolute deviation
        
        Args:
            displacement: [B, 2, H, W] or [2, H, W]
            threshold: MAD multiplier (typically 2-3)
        
        Returns:
            filtered: Outliers replaced with local median
        """
        if displacement.dim() == 3:
            displacement = displacement.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, C, H, W = displacement.shape
        filtered = displacement.clone()
        
        for b in range(B):
            for c in range(C):
                disp = displacement[b, c]
                
                # Compute median absolute deviation
                median = torch.median(disp)
                mad = torch.median(torch.abs(disp - median))
                
                # Identify outliers
                outliers = torch.abs(disp - median) > threshold * mad
                
                if outliers.any():
                    # Replace outliers with local median
                    outlier_coords = torch.where(outliers)
                    for i, j in zip(outlier_coords[0], outlier_coords[1]):
                        # Extract 3x3 neighborhood
                        i_min = max(0, i - 1)
                        i_max = min(H, i + 2)
                        j_min = max(0, j - 1)
                        j_max = min(W, j + 2)
                        
                        neighborhood = disp[i_min:i_max, j_min:j_max]
                        filtered[b, c, i, j] = torch.median(neighborhood)
        
        if squeeze_output:
            filtered = filtered.squeeze(0)
        
        return filtered
    
    @staticmethod
    def magnitude_of_displacement(displacement):
        """
        Compute magnitude of displacement field
        
        Args:
            displacement: [B, 2, H, W] or [2, H, W]
        
        Returns:
            magnitude: [B, 1, H, W] or [1, H, W]
        """
        if displacement.dim() == 3:
            u_axial, u_lateral = displacement[0], displacement[1]
            return torch.sqrt(u_axial ** 2 + u_lateral ** 2).unsqueeze(0)
        else:
            return torch.sqrt(displacement[:, 0:1] ** 2 + displacement[:, 1:2] ** 2)
    
    @staticmethod
    def angle_of_displacement(displacement):
        """
        Compute angle of displacement field
        
        Args:
            displacement: [B, 2, H, W] or [2, H, W]
            Output channels: [axial, lateral]
        
        Returns:
            angle: [B, 1, H, W] or [1, H, W] in range [-π, π]
        """
        if displacement.dim() == 3:
            u_axial, u_lateral = displacement[0], displacement[1]
            angle = torch.atan2(u_axial, u_lateral)
            return angle.unsqueeze(0)
        else:
            u_axial = displacement[:, 0:1]
            u_lateral = displacement[:, 1:2]
            angle = torch.atan2(u_axial, u_lateral)
            return angle


class StrainPostprocessor:
    """Postprocessing utilities for strain fields"""
    
    @staticmethod
    def clip_strain(strain, min_val=-0.5, max_val=0.5):
        """
        Clip strain values to valid range
        
        Args:
            strain: [B, 1, H, W] or [1, H, W] or [H, W]
            min_val, max_val: Valid strain range
        
        Returns:
            clipped: Same shape as input
        """
        return torch.clamp(strain, min=min_val, max=max_val)
    
    @staticmethod
    def remove_strain_outliers(strain, threshold=3.0, method='mad'):
        """
        Remove strain outliers
        
        Args:
            strain: [B, 1, H, W] or [1, H, W] or [H, W]
            threshold: Outlier threshold
            method: 'mad' (median absolute deviation) or 'std' (standard deviation)
        
        Returns:
            filtered: Outliers replaced with local median
        """
        original_dim = strain.dim()
        
        if strain.dim() == 2:
            strain = strain.unsqueeze(0).unsqueeze(0)
        elif strain.dim() == 3:
            strain = strain.unsqueeze(1)
        
        B, C, H, W = strain.shape
        filtered = strain.clone()
        
        for b in range(B):
            for c in range(C):
                s = strain[b, c]
                
                if method == 'mad':
                    median = torch.median(s)
                    mad = torch.median(torch.abs(s - median))
                    outliers = torch.abs(s - median) > threshold * mad
                else:  # std
                    mean = s.mean()
                    std = s.std()
                    outliers = torch.abs(s - mean) > threshold * std
                
                if outliers.any():
                    outlier_coords = torch.where(outliers)
                    for i, j in zip(outlier_coords[0], outlier_coords[1]):
                        i_min = max(0, i - 1)
                        i_max = min(H, i + 2)
                        j_min = max(0, j - 1)
                        j_max = min(W, j + 2)
                        
                        neighborhood = s[i_min:i_max, j_min:j_max]
                        filtered[b, c, i, j] = torch.median(neighborhood)
        
        # Restore original dimensionality
        if original_dim == 2:
            filtered = filtered.squeeze(0).squeeze(0)
        elif original_dim == 3:
            filtered = filtered.squeeze(1)
        
        return filtered
    
    @staticmethod
    def compute_strain_statistics(strain, region_mask=None):
        """
        Compute statistics of strain field
        
        Args:
            strain: [B, 1, H, W] or [1, H, W] or [H, W]
            region_mask: Optional mask to compute stats in ROI
        
        Returns:
            stats: dict with mean, std, min, max
        """
        if strain.dim() >= 3:
            strain = strain.squeeze()
        
        strain_flat = strain.flatten()
        
        if region_mask is not None:
            mask_flat = region_mask.flatten()
            strain_flat = strain_flat[mask_flat > 0]
        
        stats = {
            'mean': strain_flat.mean().item(),
            'std': strain_flat.std().item(),
            'min': strain_flat.min().item(),
            'max': strain_flat.max().item(),
            'median': torch.median(strain_flat).item(),
            'q25': torch.quantile(strain_flat, 0.25).item(),
            'q75': torch.quantile(strain_flat, 0.75).item(),
        }
        
        return stats


class VisualizationUtils:
    """Additional visualization helpers"""
    
    @staticmethod
    def colorize_displacement(displacement, channel=0, colormap='hsv'):
        """
        Convert single displacement channel to RGB for visualization
        
        Args:
            displacement: [B, 2, H, W] or [2, H, W]
            channel: 0 (axial) or 1 (lateral)
            colormap: 'hsv', 'jet', 'viridis', etc.
        
        Returns:
            rgb: [B, 3, H, W] or [3, H, W]
        """
        if displacement.dim() == 3:
            displacement = displacement.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Extract channel
        disp_channel = displacement[:, channel:channel+1, :, :]
        
        # Normalize to [0, 1]
        disp_min = disp_channel.min()
        disp_max = disp_channel.max()
        if disp_max - disp_min < 1e-8:
            disp_norm = torch.zeros_like(disp_channel)
        else:
            disp_norm = (disp_channel - disp_min) / (disp_max - disp_min)
        
        # Apply colormap (using matplotlib)
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        cmap = cm.get_cmap(colormap)
        
        B, _, H, W = disp_norm.shape
        rgb = torch.zeros(B, 3, H, W)
        
        for b in range(B):
            d_np = disp_norm[b, 0].numpy()
            rgb_np = cmap(d_np)[:, :, :3]
            rgb[b] = torch.from_numpy(rgb_np).permute(2, 0, 1)
        
        if squeeze_output:
            rgb = rgb.squeeze(0)
        
        return rgb
