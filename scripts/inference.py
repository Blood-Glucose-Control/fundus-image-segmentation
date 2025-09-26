#!/usr/bin/env python3
"""
Inference script for fundus image segmentation
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import UNet
from data import preprocess_single_image
from utils import visualize_segmentation, calculate_affected_percentage, save_prediction
from evaluation import ModelEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on fundus images')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str,
                       help='Path to input image')
    parser.add_argument('--image_dir', type=str,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for results')
    
    # Inference arguments
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary segmentation')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    
    # Output arguments
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--save_masks', action='store_true',
                       help='Save prediction masks')
    
    return parser.parse_args()

def load_model(model_path, device):
    """Load trained model from checkpoint"""
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
    model.eval()
    
    print("Model loaded successfully")
    return model

def process_single_image(model, image_path, args, device):
    """Process a single image and return results"""
    try:
        # Load and preprocess image
        image_tensor = preprocess_single_image(image_path, args.image_size)
        image_tensor = image_tensor.to(device)
        
        # Load original image for visualization
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(original_image, (args.image_size, args.image_size))
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_mask = (probs[0, 1, :, :] > args.threshold).float().cpu().numpy()
        
        # Calculate affected percentage
        percentage = calculate_affected_percentage(pred_mask, args.threshold)
        
        return {
            'original_image': original_image,
            'prediction_mask': pred_mask,
            'percentage': percentage,
            'filename': os.path.basename(image_path)
        }
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

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
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist.")
        return
    
    # Load model
    try:
        model = load_model(args.model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Collect image paths
    image_paths = []
    
    if args.image_path:
        if os.path.exists(args.image_path):
            image_paths = [args.image_path]
        else:
            print(f"Error: Image file {args.image_path} does not exist.")
            return
    
    elif args.image_dir:
        if os.path.exists(args.image_dir):
            image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
            image_paths = [
                os.path.join(args.image_dir, f)
                for f in os.listdir(args.image_dir)
                if f.lower().endswith(image_extensions)
            ]
            if not image_paths:
                print(f"No image files found in {args.image_dir}")
                return
        else:
            print(f"Error: Directory {args.image_dir} does not exist.")
            return
    
    else:
        print("Error: Please provide either --image_path or --image_dir")
        return
    
    print(f"Found {len(image_paths)} image(s) to process.")
    
    # Process images
    results = []
    for image_path in image_paths:
        print(f"Processing: {os.path.basename(image_path)}")
        
        result = process_single_image(model, image_path, args, device)
        if result:
            results.append(result)
            
            # Display result
            print(f"  Affected percentage: {result['percentage']:.2f}%")
            
            # Save visualization if requested
            if args.save_visualizations:
                vis_path = os.path.join(
                    args.output_dir, 
                    f"{os.path.splitext(result['filename'])[0]}_visualization.png"
                )
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(result['original_image'])
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Prediction mask
                axes[1].imshow(result['prediction_mask'], cmap='gray')
                axes[1].set_title(f'Prediction\n({result["percentage"]:.1f}% affected)')
                axes[1].axis('off')
                
                # Overlay
                overlay = result['original_image'].copy()
                colored_mask = np.zeros_like(overlay)
                colored_mask[result['prediction_mask'] > 0.5] = [255, 0, 0]
                combined = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
                
                axes[2].imshow(combined)
                axes[2].set_title('Overlay')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(vis_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  Visualization saved: {vis_path}")
            
            # Save mask if requested
            if args.save_masks:
                mask_path = os.path.join(
                    args.output_dir,
                    f"{os.path.splitext(result['filename'])[0]}_mask.png"
                )
                
                # Convert to 8-bit image
                mask_8bit = (result['prediction_mask'] * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask_8bit)
                
                print(f"  Mask saved: {mask_path}")
    
    # Summary
    if results:
        percentages = [r['percentage'] for r in results]
        
        print("\n" + "="*50)
        print("SUMMARY RESULTS")
        print("="*50)
        print(f"Total images processed: {len(results)}")
        print(f"Average affected percentage: {np.mean(percentages):.2f}% ± {np.std(percentages):.2f}%")
        print(f"Range: {np.min(percentages):.2f}% - {np.max(percentages):.2f}%")
        print(f"Median: {np.median(percentages):.2f}%")
        
        # Individual results
        print("\nIndividual Results:")
        print("-" * 30)
        for result in results:
            print(f"{result['filename']}: {result['percentage']:.2f}%")
        
        # Save summary to file
        summary_path = os.path.join(args.output_dir, 'inference_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Fundus Image Segmentation - Inference Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Threshold: {args.threshold}\n")
            f.write(f"Image size: {args.image_size}\n")
            f.write(f"Device: {device}\n\n")
            
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Average affected percentage: {np.mean(percentages):.2f}% ± {np.std(percentages):.2f}%\n")
            f.write(f"Range: {np.min(percentages):.2f}% - {np.max(percentages):.2f}%\n")
            f.write(f"Median: {np.median(percentages):.2f}%\n\n")
            
            f.write("Individual Results:\n")
            f.write("-" * 30 + "\n")
            for result in results:
                f.write(f"{result['filename']}: {result['percentage']:.2f}%\n")
        
        print(f"\nSummary saved: {summary_path}")
    
    else:
        print("No images were successfully processed.")

if __name__ == '__main__':
    main()