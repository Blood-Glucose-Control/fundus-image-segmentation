#!/usr/bin/env python3
"""
Evaluation script for fundus image segmentation model
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import UNet
from data import create_data_loaders
from evaluation import ModelEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate fundus image segmentation model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Base directory containing the test dataset')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary segmentation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction masks')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization images')
    
    return parser.parse_args()

def load_model(model_path, device):
    """Load model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model = UNet(
            n_channels=config.get('n_channels', 3),
            n_classes=config.get('n_classes', 2),
            bilinear=config.get('bilinear', True)
        )
    else:
        # Default configuration
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("Model loaded successfully")
    return model

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist.")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist.")
        return
    
    # Check for test data
    test_images_dir = os.path.join(args.data_dir, 'test', 'images')
    test_masks_dir = os.path.join(args.data_dir, 'test', 'masks')
    
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found: {test_images_dir}")
        print("Using validation data instead...")
        test_images_dir = os.path.join(args.data_dir, 'val', 'images')
        test_masks_dir = os.path.join(args.data_dir, 'val', 'masks')
    
    if not os.path.exists(test_images_dir) or not os.path.exists(test_masks_dir):
        print("Error: No test or validation data found.")
        return
    
    # Count test images
    test_images = len([f for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    if test_images == 0:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"Found {test_images} test images")
    
    # Load model
    try:
        model = load_model(args.model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create data loader
    try:
        from data import FundusDataset, get_val_transform
        from torch.utils.data import DataLoader
        
        test_dataset = FundusDataset(
            test_images_dir,
            test_masks_dir,
            transform=get_val_transform(args.image_size),
            image_size=args.image_size
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Test loader created with {len(test_loader)} batches")
        
    except Exception as e:
        print(f"Error creating data loader: {e}")
        return
    
    # Run evaluation
    print("Starting evaluation...")
    evaluator = ModelEvaluator(model, device)
    
    try:
        summary, detailed_metrics = evaluator.evaluate_model(
            test_loader,
            threshold=args.threshold,
            save_results=True,
            output_dir=args.output_dir
        )
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for metric, stats in summary.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"\n{metric.upper().replace('_', ' ')}:")
                print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Median: {stats['median']:.4f}")
        
        # Calculate overall performance score
        if all(metric in summary for metric in ['dice', 'iou', 'accuracy']):
            dice_score = summary['dice']['mean']
            iou_score = summary['iou']['mean']
            accuracy = summary['accuracy']['mean']
            
            overall_score = (dice_score + iou_score + accuracy) / 3
            print(f"\n{'OVERALL PERFORMANCE SCORE:':<30} {overall_score:.4f}")
            
            # Performance interpretation
            if overall_score >= 0.9:
                performance = "Excellent"
            elif overall_score >= 0.8:
                performance = "Very Good"
            elif overall_score >= 0.7:
                performance = "Good"
            elif overall_score >= 0.6:
                performance = "Fair"
            else:
                performance = "Needs Improvement"
            
            print(f"{'PERFORMANCE RATING:':<30} {performance}")
        
        # Percentage prediction analysis
        if 'percentage_error' in summary:
            error_stats = summary['percentage_error']
            print(f"\n{'PERCENTAGE PREDICTION ERROR:'}")
            print(f"  Mean Error: {error_stats['mean']:.2f}%")
            print(f"  Std Error: {error_stats['std']:.2f}%")
            print(f"  Max Error: {error_stats['max']:.2f}%")
            
            if error_stats['mean'] < 2.0:
                print("  Prediction Accuracy: Excellent")
            elif error_stats['mean'] < 5.0:
                print("  Prediction Accuracy: Very Good")
            elif error_stats['mean'] < 10.0:
                print("  Prediction Accuracy: Good")
            else:
                print("  Prediction Accuracy: Needs Improvement")
        
        print("\n" + "="*60)
        print(f"Detailed results saved in: {args.output_dir}")
        
        # Create summary report
        report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("Fundus Image Segmentation - Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Test Data: {test_images_dir}\n")
            f.write(f"Number of test images: {test_images}\n")
            f.write(f"Threshold: {args.threshold}\n")
            f.write(f"Image size: {args.image_size}\n")
            f.write(f"Device: {device}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 20 + "\n")
            
            for metric, stats in summary.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
                    f.write(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                    f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                    f.write(f"  Median: {stats['median']:.4f}\n")
            
            if 'overall_score' in locals():
                f.write(f"\nOVERALL PERFORMANCE SCORE: {overall_score:.4f}\n")
                f.write(f"PERFORMANCE RATING: {performance}\n")
        
        print(f"Evaluation report saved: {report_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()