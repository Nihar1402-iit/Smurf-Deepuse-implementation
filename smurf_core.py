# SMURF Core Model (Simplified RAFT-based Optical Flow)
# This is a reference implementation of the core SMURF model
# For production, use the official weights from Google Research

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with ReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=kernel_size//2, bias=True
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.conv(x))


class FeatureEncoder(nn.Module):
    """Feature encoder - extracts features from input images"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=7, stride=1),
            ConvBlock(64, 64, kernel_size=3, stride=2),
            ConvBlock(64, 96, kernel_size=3, stride=1),
            ConvBlock(96, 96, kernel_size=3, stride=2),
            ConvBlock(96, 128, kernel_size=3, stride=1),
        )
    
    def forward(self, x):
        return self.layers(x)


class CostVolumeLayer(nn.Module):
    """Build cost volume from feature maps"""
    def __init__(self, max_displacement=4):
        super().__init__()
        self.max_displacement = max_displacement
    
    def forward(self, fmap1, fmap2):
        """
        Build 4D cost volume
        fmap1, fmap2: [B, C, H, W]
        output: [B, (2*max_disp+1)^2, H, W]
        """
        B, C, H, W = fmap1.shape
        cost_volume = []
        
        for dy in range(-self.max_displacement, self.max_displacement + 1):
            for dx in range(-self.max_displacement, self.max_displacement + 1):
                if dy == 0 and dx == 0:
                    cost = (fmap1 * fmap2).mean(dim=1, keepdim=True)
                else:
                    fmap2_shifted = torch.roll(fmap2, shifts=(dy, dx), dims=(2, 3))
                    cost = (fmap1 * fmap2_shifted).mean(dim=1, keepdim=True)
                cost_volume.append(cost)
        
        return torch.cat(cost_volume, dim=1)


class FlowHead(nn.Module):
    """Predict flow from cost volume"""
    def __init__(self, max_displacement=4):
        super().__init__()
        input_channels = (2 * max_displacement + 1) ** 2
        
        self.layers = nn.Sequential(
            ConvBlock(input_channels, 128, kernel_size=3),
            ConvBlock(128, 64, kernel_size=3),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )
    
    def forward(self, cost_volume):
        return self.layers(cost_volume)


class RecurrentFlowRefinement(nn.Module):
    """Recurrent flow refinement (inspired by RAFT)"""
    def __init__(self, max_displacement=4):
        super().__init__()
        cost_vol_channels = (2 * max_displacement + 1) ** 2  # 81 for max_disp=4
        input_channels = 2 + cost_vol_channels  # flow + cost volume
        
        self.conv1 = ConvBlock(input_channels, 128, kernel_size=3)
        self.conv2 = ConvBlock(128, 128, kernel_size=3)
        self.flow_head = nn.Conv2d(128, 2, kernel_size=3, padding=1)
    
    def forward(self, flow, cost_volume):
        x = torch.cat([flow, cost_volume], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        delta_flow = self.flow_head(x)
        return flow + delta_flow


class SMURFModel(nn.Module):
    """
    SMURF: Submeter Resolution Unsupervised Learning for Optical Flow
    
    RAFT-based unsupervised optical flow estimation.
    
    Output: flow tensor [B, 2, H, W] where:
        - channel 0: horizontal (lateral) displacement
        - channel 1: vertical (axial) displacement
    """
    
    def __init__(self, in_channels=3, max_displacement=4, num_refinement_steps=4):
        super().__init__()
        self.in_channels = in_channels
        self.max_displacement = max_displacement
        self.num_refinement_steps = num_refinement_steps
        
        self.encoder = FeatureEncoder(in_channels=in_channels)
        self.cost_volume = CostVolumeLayer(max_displacement=max_displacement)
        self.flow_head = FlowHead(max_displacement=max_displacement)
        self.refinement = RecurrentFlowRefinement(max_displacement=max_displacement)
    
    def forward(self, image1, image2):
        """
        Args:
            image1: [B, C, H, W]
            image2: [B, C, H, W]
        
        Returns:
            flow_predictions: list of flow tensors at different scales
            final flow: [B, 2, H, W]
        """
        # Extract features
        fmap1 = self.encoder(image1)
        fmap2 = self.encoder(image2)
        
        # Build cost volume
        cost_vol = self.cost_volume(fmap1, fmap2)
        
        # Initial flow prediction
        flow = self.flow_head(cost_vol)
        
        flow_predictions = [flow]
        
        # Recurrent refinement steps
        for _ in range(self.num_refinement_steps):
            flow = self.refinement(flow, cost_vol)
            flow_predictions.append(flow)
        
        # Upsample final flow to original resolution
        # Current flow is at 4x downsampled resolution
        # We need to upsample to original resolution
        current_h = flow_predictions[-1].shape[2]
        current_w = flow_predictions[-1].shape[3]
        orig_h = image1.shape[2]
        orig_w = image1.shape[3]
        
        # Calculate upsampling factor
        upsample_factor = orig_h // current_h
        
        final_flow = F.interpolate(
            flow_predictions[-1], 
            size=(orig_h, orig_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Scale flow by upsampling factor
        final_flow = final_flow * upsample_factor
        
        return flow_predictions, final_flow
