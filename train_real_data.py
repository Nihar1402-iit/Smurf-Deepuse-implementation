# SMURF Ultrasound Training Script with Real Data
# Trains on .mat files and evaluates on .raw test data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import time

from smurf_core import SMURFModel
from smurf_ultrasound_wrapper import SMURFUltrasoundWithLosses
from data_loaders import MatUltrasoundDataset, RawUltrasoundDataset


class TrainingConfig:
    """Training hyperparameters - DeepUse-inspired configuration"""
    
    def __init__(self):
        self.batch_size = 4
        self.num_epochs = 100  # Increased from 30 for better convergence
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_workers = 0
        
        # Loss weights (DeepUse-inspired)
        # Primary loss: NCC-based similarity (replaces photometric)
        self.weight_photometric = 1.0
        # Smoothness: gradient penalty on displacement field
        self.weight_smoothness = 0.1
        # Strain regularization: smooth strain field
        self.weight_strain_reg = 0.05
        # Displacement regularization: prevent trivial zero solution
        self.weight_displacement_reg = 0.01
        
        # Logging
        self.log_interval = 5
        self.save_interval = 5
        self.checkpoint_dir = Path("checkpoints")
        
        # Data paths - default to local, override with --train-data and --test-data flags
        self.train_data_dir = "/Users/niharshah/Desktop/Omnistrain/_Data_10M_Part1_"
        self.test_data_dir = "/Users/niharshah/Desktop/Omnistrain/our_algo/test_data_deepuse"
        
        # Try server paths if local paths don't exist
        if not Path(self.train_data_dir).exists():
            server_train = "/studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/train_data"
            if Path(server_train).exists():
                self.train_data_dir = server_train
        
        if not Path(self.test_data_dir).exists():
            server_test = "/studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/test_data_deepuse"
            if Path(server_test).exists():
                self.test_data_dir = server_test


class UltrasoundTrainer:
    """Trainer for SMURF Ultrasound model with real data"""
    
    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Device: {self.device}")
        print(f"Training data dir: {self.config.train_data_dir}")
        print(f"Test data dir: {self.config.test_data_dir}")
        
        # Create model
        smurf = SMURFModel(
            in_channels=1,
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
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
    
    def train(self):
        """Full training loop"""
        # Verify data directory exists
        train_data_path = Path(self.config.train_data_dir)
        if not train_data_path.exists():
            print(f"ERROR: Training data directory not found: {self.config.train_data_dir}")
            print("\nPlease provide training data directory using --train-data flag:")
            print("  python3 train_gpu.py --train-data /path/to/train_data --epochs 100")
            print("\nOn server, use:")
            print("  python3 train_gpu.py --train-data /studios/this_studio/Model_comparisons/Smurf-Deepuse-implementation/train_data --epochs 100")
            return
        
        # Create datasets
        print("Loading training dataset...")
        train_dataset = MatUltrasoundDataset(
            self.config.train_data_dir,
            normalize=True
        )
        
        if len(train_dataset) == 0:
            print("ERROR: No training data loaded!")
            print(f"\nChecking directory: {self.config.train_data_dir}")
            mat_files = list(train_data_path.glob("*.mat"))
            print(f"Found {len(mat_files)} .mat files")
            if mat_files:
                print("Files found:")
                for f in mat_files[:5]:
                    print(f"  - {f.name}")
            print("\nMake sure .mat files contain 'RF' key with ultrasound frames")
            return
        
        # Split into train/val
        val_size = max(1, int(len(train_dataset) * 0.1))
        train_size = len(train_dataset) - val_size
        
        train_dataset_split, val_dataset_split = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset_split,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset_split,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Batch size: {self.config.batch_size}")
        print()
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_losses = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_photometric"].append(train_losses["photometric"])
            self.history["train_smoothness"].append(train_losses["smoothness"])
            self.history["train_strain_reg"].append(train_losses["strain_reg"])
            
            # Validate
            val_loss = self.validate(val_loader)
            self.history["val_loss"].append(val_loss)
            
            # Logging
            if (epoch + 1) % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{self.config.num_epochs} [{elapsed:.0f}s]")
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
        
        print("=" * 60)
        print("Training complete!")
        print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
        print("=" * 60)
        print()
        
        self.save_history()
    
    def train_epoch(self, train_loader, epoch):
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
            print(f"  → Saving best model to {path}")
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
    config = TrainingConfig()
    trainer = UltrasoundTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
