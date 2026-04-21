# SMURF Ultrasound Training Script
# Trains the SMURF model adapted for ultrasound elastography

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from smurf_core import SMURFModel
from smurf_ultrasound_wrapper import SMURFUltrasoundWithLosses


class UltrasoundDataset(Dataset):
    """
    Dummy Ultrasound Dataset for demonstration
    
    In production, this would load actual RF/IQ frames from disk.
    
    Expected format:
    - Each sample: consecutive RF/IQ frame pairs
    - Shape: [C, H, W] where C is number of channels (1 for RF, 2 for IQ)
    - Normalized intensity [-1, 1] or [0, 1]
    """
    
    def __init__(self, num_samples=100, height=256, width=256, channels=1):
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.channels = channels
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns consecutive frame pairs
        
        In production, load from disk:
        I_t = load_rf_frame(frame_index)
        I_t1 = load_rf_frame(frame_index + 1)
        """
        # Simulate RF frame: random Gaussian noise (typical RF appearance)
        I_t = torch.randn(self.channels, self.height, self.width) * 0.1
        
        # Create slightly displaced frame (simulate cardiac/tissue motion)
        I_t1 = torch.roll(I_t, shifts=2, dims=1)  # axial shift
        I_t1 = torch.roll(I_t1, shifts=1, dims=2)  # lateral shift
        I_t1 = I_t1 + torch.randn_like(I_t1) * 0.05  # add noise
        
        return I_t, I_t1


class TrainingConfig:
    """Training hyperparameters"""
    
    def __init__(self):
        self.batch_size = 8
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_workers = 4
        
        # Loss weights
        self.weight_photometric = 1.0
        self.weight_smoothness = 0.1
        self.weight_strain_reg = 0.05
        
        # Logging
        self.log_interval = 10
        self.save_interval = 5
        self.checkpoint_dir = Path("checkpoints")
        
        # Data
        self.frame_height = 256
        self.frame_width = 256
        self.frame_channels = 1  # RF frames (1 channel)
        self.num_train_samples = 1000
        self.num_val_samples = 100


class UltrasoundTrainer:
    """Trainer for SMURF Ultrasound model"""
    
    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        smurf = SMURFModel(
            in_channels=self.config.frame_channels,
            max_displacement=4,
            num_refinement_steps=4
        )
        
        self.model = SMURFUltrasoundWithLosses(smurf).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_photometric": [],
            "train_smoothness": [],
            "train_strain_reg": [],
        }
    
    def train(self):
        """Full training loop"""
        # Create datasets
        train_dataset = UltrasoundDataset(
            num_samples=self.config.num_train_samples,
            height=self.config.frame_height,
            width=self.config.frame_width,
            channels=self.config.frame_channels
        )
        
        val_dataset = UltrasoundDataset(
            num_samples=self.config.num_val_samples,
            height=self.config.frame_height,
            width=self.config.frame_width,
            channels=self.config.frame_channels
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for simple dataset
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print()
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_losses = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_photometric"].append(train_losses["photometric"])
            self.history["train_smoothness"].append(train_losses["smoothness"])
            self.history["train_strain_reg"].append(train_losses["strain_reg"])
            
            # Validate
            val_loss = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)
            
            # Logging
            if (epoch + 1) % self.config.log_interval == 0:
                print(f"Epoch {epoch+1}/{self.config.num_epochs}")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"    - Photometric: {train_losses['photometric']:.6f}")
                print(f"    - Smoothness: {train_losses['smoothness']:.6f}")
                print(f"    - Strain Reg: {train_losses['strain_reg']:.6f}")
                print(f"  Val Loss: {val_loss:.6f}")
                print()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, best=True)
            
            # Learning rate decay
            self.scheduler.step()
        
        print("Training complete!")
        self.save_history()
    
    def train_epoch(self, train_loader):
        """Single training epoch"""
        self.model.train()
        
        total_loss = 0
        total_photometric = 0
        total_smoothness = 0
        total_strain_reg = 0
        
        for batch_idx, (I_t, I_t1) in enumerate(train_loader):
            I_t = I_t.to(self.device)
            I_t1 = I_t1.to(self.device)
            
            # Forward pass
            output = self.model(I_t, I_t1)
            
            # Compute losses
            losses = self.model.compute_losses(I_t, I_t1, output)
            loss = losses["total"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_photometric += losses["photometric"].item()
            total_smoothness += losses["smoothness"].item()
            total_strain_reg += losses["strain_reg"].item()
        
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_losses = {
            "photometric": total_photometric / num_batches,
            "smoothness": total_smoothness / num_batches,
            "strain_reg": total_strain_reg / num_batches,
        }
        
        return avg_loss, avg_losses
    
    def validate(self, val_loader):
        """Validation pass"""
        self.model.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for I_t, I_t1 in val_loader:
                I_t = I_t.to(self.device)
                I_t1 = I_t1.to(self.device)
                
                output = self.model(I_t, I_t1)
                losses = self.model.compute_losses(I_t, I_t1, output)
                loss = losses["total"]
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
        }
        
        if best:
            path = self.config.checkpoint_dir / "best_model.pt"
            print(f"Saving best model to {path}")
        else:
            path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history"""
        path = self.config.checkpoint_dir / "history.json"
        
        # Convert numpy arrays to lists for JSON serialization
        history_json = {k: [float(v) for v in vals] for k, vals in self.history.items()}
        
        with open(path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"Training history saved to {path}")


def main():
    """Main training script"""
    # Create config
    config = TrainingConfig()
    
    # Create trainer
    trainer = UltrasoundTrainer(config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
