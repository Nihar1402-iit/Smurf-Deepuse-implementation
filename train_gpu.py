#!/usr/bin/env python3
# SMURF Ultrasound - GPU Training Script (Python version)
# Run: python3 train_gpu.py --gpu 0 --batch-size 16 --epochs 50

import argparse
import torch
import sys
from pathlib import Path

from train_real_data import TrainingConfig, UltrasoundTrainer


def main():
    parser = argparse.ArgumentParser(description='SMURF Ultrasound GPU Training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--train-data', type=str, help='Training data directory')
    parser.add_argument('--test-data', type=str, help='Test data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Set GPU
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    
    # Print device info
    print("=" * 70)
    print("SMURF ULTRASOUND - GPU TRAINING")
    print("=" * 70)
    print()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"CUDA device: {torch.cuda.get_device_name(args.gpu)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available! Training on CPU (will be slow)")
    
    print()
    
    # Create config
    config = TrainingConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.num_workers = args.workers
    config.checkpoint_dir = Path(args.checkpoint_dir)
    
    if args.train_data:
        config.train_data_dir = args.train_data
    if args.test_data:
        config.test_data_dir = args.test_data
    
    # Print config
    print("TRAINING CONFIGURATION")
    print("-" * 70)
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of workers: {config.num_workers}")
    print(f"Train data: {config.train_data_dir}")
    print(f"Test data: {config.test_data_dir}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print()
    
    # Run training
    trainer = UltrasoundTrainer(config)
    trainer.train()
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
