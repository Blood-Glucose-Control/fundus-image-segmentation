#!/usr/bin/env python3
"""
Main training script for fundus image segmentation
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import UNet
from data import create_data_loaders
from training import Trainer
from utils import create_sample_dataset_structure

def parse_args():
    parser = argparse.ArgumentParser(description='Train fundus image segmentation model')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, default='dataset',
                       help='Base directory containing the dataset')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--n_channels', type=int, default=3,
                       help='Number of input channels')
    parser.add_argument('--n_classes', type=int, default=2,
                       help='Number of output classes')
    parser.add_argument('--bilinear', action='store_true',
                       help='Use bilinear upsampling')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training')
    
    # Setup arguments
    parser.add_argument('--setup_dataset', action='store_true',
                       help='Create sample dataset structure')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup dataset structure if requested
    if args.setup_dataset:
        create_sample_dataset_structure(args.data_dir)
        print(f"Dataset structure created at: {args.data_dir}")
        print("Please add your images and masks to the appropriate directories.")
        return
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if dataset directories exist
    train_images_dir = os.path.join(args.data_dir, 'train', 'images')
    train_masks_dir = os.path.join(args.data_dir, 'train', 'masks')
    val_images_dir = os.path.join(args.data_dir, 'val', 'images')
    val_masks_dir = os.path.join(args.data_dir, 'val', 'masks')
    
    for dir_path in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} does not exist.")
            print("Please run with --setup_dataset first, or provide a valid dataset directory.")
            return
    
    # Check if directories contain files
    train_images = len([f for f in os.listdir(train_images_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    val_images = len([f for f in os.listdir(val_images_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    if train_images == 0 or val_images == 0:
        print(f"Warning: Found {train_images} training images and {val_images} validation images.")
        print("Please ensure your dataset directories contain image files.")
        if train_images == 0 and val_images == 0:
            return
    
    print(f"Found {train_images} training images and {val_images} validation images.")
    
    # Create data loaders
    try:
        train_loader, val_loader = create_data_loaders(
            train_images_dir, train_masks_dir,
            val_images_dir, val_masks_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers
        )
        print(f"Data loaders created successfully.")
        print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # Create model
    model = UNet(
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        bilinear=args.bilinear
    )
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        log_dir=logs_dir
    )
    
    # Load optimizer state if resuming
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print("Starting training...")
    print(f"Training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Image size: {args.image_size}")
    print("-" * 50)
    
    # Train model
    try:
        trainer.train(num_epochs=args.epochs, save_dir=checkpoints_dir)
        
        # Save final model
        final_model_path = os.path.join(checkpoints_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'n_channels': args.n_channels,
                'n_classes': args.n_classes,
                'bilinear': args.bilinear,
                'image_size': args.image_size
            }
        }, final_model_path)
        
        print(f"Final model saved: {final_model_path}")
        
        # Plot training history
        trainer.plot_training_history(
            save_path=os.path.join(args.output_dir, 'training_history.png')
        )
        print(f"Training history saved: {os.path.join(args.output_dir, 'training_history.png')}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save current state
        interrupt_path = os.path.join(checkpoints_dir, 'interrupted_training.pth')
        torch.save({
            'epoch': len(trainer.train_losses),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'val_metrics': trainer.val_metrics
        }, interrupt_path)
        print(f"Current state saved: {interrupt_path}")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()