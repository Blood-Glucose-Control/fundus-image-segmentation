#!/usr/bin/env python3
"""
Quick demo script to showcase the fundus image segmentation capabilities
"""

import os
import sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import UNet
from utils import create_sample_dataset_structure, visualize_segmentation, calculate_affected_percentage

def create_demo_image(size=512):
    """Create a synthetic fundus image for demonstration"""
    # Create a basic fundus-like image
    image = np.random.randint(100, 200, (size, size, 3), dtype=np.uint8)
    
    # Add circular fundus boundary
    center = (size // 2, size // 2)
    radius = size // 3
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    circle_mask = dist_from_center <= radius
    
    # Make background black outside circle
    image[~circle_mask] = 0
    
    # Add some vessel-like structures
    for _ in range(10):
        start_point = (np.random.randint(50, size-50), np.random.randint(50, size-50))
        end_point = (np.random.randint(50, size-50), np.random.randint(50, size-50))
        cv2.line(image, start_point, end_point, (50, 50, 50), thickness=2)
    
    # Create corresponding mask with some lesions
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # Add random lesions
    num_lesions = np.random.randint(2, 6)
    for _ in range(num_lesions):
        lesion_center = (np.random.randint(100, size-100), np.random.randint(100, size-100))
        lesion_radius = np.random.randint(15, 40)
        
        Y, X = np.ogrid[:size, :size]
        lesion_dist = np.sqrt((X - lesion_center[0])**2 + (Y - lesion_center[1])**2)
        lesion_mask = lesion_dist <= lesion_radius
        
        mask[lesion_mask & circle_mask] = 255
    
    return image, mask

def demo_model_creation():
    """Demonstrate model creation and architecture"""
    print("🏗️  Creating U-Net Model")
    print("-" * 40)
    
    # Create model
    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture: U-Net")
    print(f"Input channels: 3 (RGB)")
    print(f"Output classes: 2 (background/foreground)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Model created successfully!")
    
    return model

def demo_percentage_prediction():
    """Demonstrate percentage prediction capability"""
    print("\n🔍 Percentage Prediction Demo")
    print("-" * 40)
    
    # Create demo images with different levels of pathology
    scenarios = [
        ("Healthy Retina", 0.5),
        ("Mild Pathology", 3.0),
        ("Moderate Pathology", 8.0),
        ("Severe Pathology", 15.0)
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, (scenario_name, target_percentage) in enumerate(scenarios):
        # Create demo image and mask
        image, mask = create_demo_image()
        
        # Adjust mask to achieve target percentage
        total_pixels = np.sum(mask > 0)
        target_pixels = int((target_percentage / 100) * (512 * 512))
        
        if total_pixels > target_pixels and target_pixels > 0:
            # Reduce mask
            mask_coords = np.where(mask > 0)
            remove_indices = np.random.choice(len(mask_coords[0]), 
                                            total_pixels - target_pixels, replace=False)
            mask[mask_coords[0][remove_indices], mask_coords[1][remove_indices]] = 0
        
        # Calculate actual percentage
        actual_percentage = calculate_affected_percentage(mask)
        
        # Display original image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'{scenario_name}\nOriginal Image')
        axes[0, i].axis('off')
        
        # Display mask with percentage
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Affected: {actual_percentage:.1f}%')
        axes[1, i].axis('off')
    
    plt.suptitle('Fundus Image Segmentation - Percentage Prediction Demo', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("✅ Percentage prediction demo completed!")

def demo_visualization():
    """Demonstrate visualization capabilities"""
    print("\n🎨 Visualization Demo")
    print("-" * 40)
    
    # Create a demo image and mask
    image, mask = create_demo_image()
    
    # Create a synthetic prediction (slightly different from ground truth)
    prediction = mask.copy().astype(np.float32) / 255.0
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, prediction.shape)
    prediction = np.clip(prediction + noise, 0, 1)
    
    # Apply threshold
    prediction_binary = (prediction > 0.5).astype(np.float32)
    
    # Calculate percentages
    true_percentage = calculate_affected_percentage(mask)
    pred_percentage = calculate_affected_percentage(prediction_binary * 255)
    
    print(f"Ground truth affected area: {true_percentage:.2f}%")
    print(f"Predicted affected area: {pred_percentage:.2f}%")
    print(f"Prediction error: {abs(true_percentage - pred_percentage):.2f}%")
    
    # Visualize
    visualize_segmentation(
        image, 
        mask / 255.0, 
        prediction_binary,
        title=f"Demo: GT={true_percentage:.1f}%, Pred={pred_percentage:.1f}%"
    )
    
    print("✅ Visualization demo completed!")

def demo_data_structure():
    """Demonstrate dataset structure creation"""
    print("\n📁 Dataset Structure Demo")
    print("-" * 40)
    
    # Create temporary demo structure
    demo_dir = "demo_dataset"
    create_sample_dataset_structure(demo_dir)
    
    # Show structure
    print(f"Created dataset structure at: {demo_dir}")
    
    # Clean up
    import shutil
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
        print("✅ Demo dataset structure created and cleaned up!")

def main():
    """Run the complete demo"""
    print("🚀 Fundus Image Segmentation - Interactive Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Model creation
        model = demo_model_creation()
        
        # Demo 2: Dataset structure
        demo_data_structure()
        
        # Demo 3: Visualization
        demo_visualization()
        
        # Demo 4: Percentage prediction
        demo_percentage_prediction()
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Next steps:")
        print("1. Run 'python setup.py' to install dependencies")
        print("2. Add your dataset to the dataset/ directory") 
        print("3. Train with 'python scripts/train.py --data_dir dataset'")
        print("4. Or try the Google Colab notebook for an easy start!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("\nPlease make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()