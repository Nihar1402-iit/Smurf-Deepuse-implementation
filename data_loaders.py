# Custom Data Loaders for Ultrasound Training and Testing
# Handles .mat files (training) and .raw/.mhd files (testing)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from pathlib import Path
import os


class MatUltrasoundDataset(Dataset):
    """
    MATLAB .mat file ultrasound dataset
    
    Expected format: .mat files with 'RF' key containing RF frames
    Shape: [num_models, num_frames, height, width] or [num_frames, height, width]
    """
    
    def __init__(self, data_dir, num_pairs=None, frame_skip=1, normalize=True):
        """
        Args:
            data_dir: Directory containing .mat files
            num_pairs: Limit number of consecutive frame pairs (None = all)
            frame_skip: Skip N frames between pairs (default: 1 = consecutive)
            normalize: Normalize frames to [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.frame_skip = frame_skip
        self.normalize = normalize
        
        # Load all .mat files
        self.frame_pairs = []
        self.load_mat_files(num_pairs)
        
        print(f"Loaded {len(self.frame_pairs)} frame pairs from {self.data_dir}")
    
    def load_mat_files(self, num_pairs):
        """Load frame pairs from all .mat files in directory"""
        mat_files = sorted(self.data_dir.glob("*.mat"))
        
        pair_count = 0
        for mat_file in mat_files:
            try:
                mat_data = sio.loadmat(str(mat_file))
                
                if 'RF' not in mat_data:
                    continue
                
                rf_frames = mat_data['RF']
                
                # Handle different shapes
                if rf_frames.ndim == 4:
                    # Shape: [num_models, num_frames, H, W]
                    # Flatten to sequence of frames
                    rf_frames = rf_frames.reshape(-1, rf_frames.shape[2], rf_frames.shape[3])
                
                # rf_frames now: [num_frames, H, W]
                num_frames = rf_frames.shape[0]
                
                # Create consecutive frame pairs
                for i in range(num_frames - self.frame_skip):
                    if num_pairs and pair_count >= num_pairs:
                        return
                    
                    frame_t = torch.from_numpy(rf_frames[i].astype(np.float32))
                    frame_t1 = torch.from_numpy(rf_frames[i + self.frame_skip].astype(np.float32))
                    
                    # Ensure frames are [H, W]
                    if frame_t.dim() == 2 and frame_t1.dim() == 2:
                        self.frame_pairs.append((frame_t, frame_t1))
                        pair_count += 1
            
            except Exception as e:
                print(f"Warning: Failed to load {mat_file}: {e}")
    
    def __len__(self):
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        frame_t, frame_t1 = self.frame_pairs[idx]
        
        # Normalize frames
        if self.normalize:
            frame_t = self._normalize(frame_t)
            frame_t1 = self._normalize(frame_t1)
        
        # Resize to 256x256 for consistency
        if frame_t.shape != (256, 256):
            frame_t = F.interpolate(
                frame_t.unsqueeze(0).unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            frame_t1 = F.interpolate(
                frame_t1.unsqueeze(0).unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Add channel dimension for network
        frame_t = frame_t.unsqueeze(0)  # [1, H, W]
        frame_t1 = frame_t1.unsqueeze(0)  # [1, H, W]
        
        return frame_t, frame_t1
    
    @staticmethod
    def _normalize(frame):
        """Normalize frame to [-1, 1]"""
        frame_min = frame.min()
        frame_max = frame.max()
        
        if frame_max - frame_min < 1e-8:
            return torch.zeros_like(frame)
        
        return 2 * (frame - frame_min) / (frame_max - frame_min) - 1


class RawUltrasoundDataset(Dataset):
    """
    RAW ultrasound dataset
    
    Expected format: .raw files containing RF frames in uint16 format
    Dimensions inferred from filename or directory structure
    """
    
    def __init__(self, data_dir, frame_height=512, frame_width=1000, 
                 frame_skip=1, normalize=True):
        """
        Args:
            data_dir: Directory containing .raw files
            frame_height: Height of each frame
            frame_width: Width of each frame
            frame_skip: Skip N frames between pairs
            normalize: Normalize frames to [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_skip = frame_skip
        self.normalize = normalize
        
        self.frame_pairs = []
        self.load_raw_files()
        
        print(f"Loaded {len(self.frame_pairs)} frame pairs from {self.data_dir}")
    
    def load_raw_files(self):
        """Load frame pairs from all .raw files in directory"""
        raw_files = sorted(self.data_dir.glob("*.raw"))
        
        # Also check subdirectories
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                raw_files.extend(sorted(subdir.glob("*.raw")))
        
        for raw_file in raw_files:
            try:
                raw_data = np.fromfile(str(raw_file), dtype=np.uint16)
                
                # Reshape to frames
                num_frames = len(raw_data) // (self.frame_height * self.frame_width)
                
                if num_frames < 2:
                    continue
                
                raw_data = raw_data[:num_frames * self.frame_height * self.frame_width]
                frames = raw_data.reshape(num_frames, self.frame_height, self.frame_width)
                
                # Create consecutive frame pairs
                for i in range(num_frames - self.frame_skip):
                    frame_t = torch.from_numpy(frames[i].astype(np.float32))
                    frame_t1 = torch.from_numpy(frames[i + self.frame_skip].astype(np.float32))
                    
                    self.frame_pairs.append((frame_t, frame_t1))
            
            except Exception as e:
                print(f"Warning: Failed to load {raw_file}: {e}")
    
    def __len__(self):
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        frame_t, frame_t1 = self.frame_pairs[idx]
        
        # Normalize frames
        if self.normalize:
            frame_t = self._normalize(frame_t)
            frame_t1 = self._normalize(frame_t1)
        
        # Resize to 256x256 for consistency
        if frame_t.shape != (256, 256):
            frame_t = F.interpolate(
                frame_t.unsqueeze(0).unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            frame_t1 = F.interpolate(
                frame_t1.unsqueeze(0).unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        
        # Add channel dimension
        frame_t = frame_t.unsqueeze(0)  # [1, H, W]
        frame_t1 = frame_t1.unsqueeze(0)  # [1, H, W]
        
        return frame_t, frame_t1
    
    @staticmethod
    def _normalize(frame):
        """Normalize frame to [-1, 1]"""
        frame_min = frame.min()
        frame_max = frame.max()
        
        if frame_max - frame_min < 1e-8:
            return torch.zeros_like(frame)
        
        return 2 * (frame - frame_min) / (frame_max - frame_min) - 1


def create_train_val_loaders(train_data_dir, batch_size=4, val_split=0.1):
    """
    Create training and validation dataloaders from training data
    
    Args:
        train_data_dir: Directory with .mat training files
        batch_size: Batch size
        val_split: Validation split fraction
    
    Returns:
        train_loader, val_loader
    """
    dataset = MatUltrasoundDataset(train_data_dir, normalize=True)
    
    # Split into train/val
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_test_loader(test_data_dir, batch_size=1):
    """
    Create test dataloader from test data
    
    Args:
        test_data_dir: Directory with .raw test files
        batch_size: Batch size
    
    Returns:
        test_loader
    """
    dataset = RawUltrasoundDataset(
        test_data_dir,
        frame_height=512,
        frame_width=1000,
        normalize=True
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return test_loader
