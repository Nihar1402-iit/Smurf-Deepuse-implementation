#!/usr/bin/env python3
# SMURF Ultrasound - GPU Inference Script (Python version)
# Run: python3 test_gpu.py --gpu 0 --checkpoint best_model.pt

import argparse
import torch
from pathlib import Path
import sys

from test_inference import run_inference_on_test_data, plot_training_history, TestConfig


def main():
    parser = argparse.ArgumentParser(description='SMURF Ultrasound GPU Inference')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    parser.add_argument('--test-data', type=str, help='Test data directory')
    parser.add_argument('--output-dir', type=str, default='test_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Set GPU
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    
    # Print device info
    print("=" * 70)
    print("SMURF ULTRASOUND - GPU INFERENCE")
    print("=" * 70)
    print()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"CUDA device: {torch.cuda.get_device_name(args.gpu)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA not available!")
    
    print()
    
    # Update config
    config = TestConfig()
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.test_data:
        config.test_data_dir = args.test_data
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    print("INFERENCE CONFIGURATION")
    print("-" * 70)
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Test data: {config.test_data_dir}")
    print(f"Output directory: {config.output_dir}")
    print()
    
    # Run inference
    run_inference_on_test_data()
    
    # Plot training history
    print("\nGenerating training history plot...")
    plot_training_history()
    
    print()
    print("=" * 70)
    print("INFERENCE COMPLETED!")
    print("=" * 70)
    print(f"\nResults saved to: {config.output_dir}")
    print("- Displacement/strain heatmaps")
    print("- Results in DeepUse format (.mat files)")
    print("- Training history plots")


if __name__ == "__main__":
    main()
