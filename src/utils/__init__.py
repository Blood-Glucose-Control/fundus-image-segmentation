"""
Utility functions for fundus image segmentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os

def visualize_segmentation(image, mask, prediction=None, title="Fundus Segmentation"):
    """
    Visualize fundus image with segmentation mask and prediction
    
    Args:
        image (np.ndarray): Original image
        mask (np.ndarray): Ground truth mask
        prediction (np.ndarray): Predicted mask (optional)
        title (str): Plot title
    """
    if prediction is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Overlay
        overlay = image.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
        # Create colored masks
        pred_colored = np.zeros_like(overlay)
        gt_colored = np.zeros_like(overlay)
        
        pred_colored[prediction > 0.5] = [255, 0, 0]  # Red for prediction
        gt_colored[mask > 0.5] = [0, 255, 0]  # Green for ground truth
        
        # Combine overlays
        combined_overlay = cv2.addWeighted(overlay, 0.7, pred_colored, 0.3, 0)
        combined_overlay = cv2.addWeighted(combined_overlay, 1.0, gt_colored, 0.3, 0)
        
        axes[3].imshow(combined_overlay)
        axes[3].set_title('Overlay (Red: Pred, Green: GT)')
        axes[3].axis('off')
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = image.copy()
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
        
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 0.5] = [255, 0, 0]  # Red for affected areas
        
        combined = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        axes[2].imshow(combined)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def calculate_affected_percentage(mask, threshold=0.5):
    """
    Calculate the percentage of affected area in the mask
    
    Args:
        mask (np.ndarray): Segmentation mask
        threshold (float): Threshold for binary classification
    
    Returns:
        float: Percentage of affected area
    """
    binary_mask = (mask > threshold).astype(np.uint8)
    total_pixels = mask.size
    affected_pixels = np.sum(binary_mask)
    percentage = (affected_pixels / total_pixels) * 100
    return percentage

def preprocess_fundus_image(image_path, target_size=(512, 512)):
    """
    Preprocess fundus image for analysis
    
    Args:
        image_path (str): Path to the image
        target_size (tuple): Target size for resizing
    
    Returns:
        np.ndarray: Preprocessed image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return image

def create_circular_mask(image_shape, center=None, radius=None):
    """
    Create a circular mask for fundus images
    
    Args:
        image_shape (tuple): Shape of the image (height, width)
        center (tuple): Center of the circle (optional)
        radius (int): Radius of the circle (optional)
    
    Returns:
        np.ndarray: Circular mask
    """
    h, w = image_shape
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.uint8)

def enhance_fundus_contrast(image):
    """
    Enhance contrast of fundus image
    
    Args:
        image (np.ndarray): Input fundus image
    
    Returns:
        np.ndarray: Enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced

def remove_fundus_noise(image):
    """
    Remove noise from fundus image
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        np.ndarray: Denoised image
    """
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    return denoised

def normalize_image(image):
    """
    Normalize image values to [0, 1] range
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        np.ndarray: Normalized image
    """
    return image.astype(np.float32) / 255.0

def tensor_to_numpy(tensor):
    """
    Convert PyTorch tensor to numpy array
    
    Args:
        tensor (torch.Tensor): Input tensor
    
    Returns:
        np.ndarray: Numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def numpy_to_tensor(array, device='cpu'):
    """
    Convert numpy array to PyTorch tensor
    
    Args:
        array (np.ndarray): Input array
        device (str): Target device
    
    Returns:
        torch.Tensor: PyTorch tensor
    """
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array).float()
        return tensor.to(device)
    return array

def save_prediction(image, mask, prediction, output_path, percentage):
    """
    Save prediction results with visualization
    
    Args:
        image (np.ndarray): Original image
        mask (np.ndarray): Ground truth mask
        prediction (np.ndarray): Predicted mask
        output_path (str): Output file path
        percentage (float): Affected percentage
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title(f'Prediction\n({percentage:.1f}% affected)')
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    
    colored_pred = np.zeros_like(overlay)
    colored_pred[prediction > 0.5] = [255, 0, 0]  # Red for prediction
    
    combined = cv2.addWeighted(overlay, 0.7, colored_pred, 0.3, 0)
    axes[3].imshow(combined)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_sample_dataset_structure(base_dir):
    """
    Create sample dataset directory structure
    
    Args:
        base_dir (str): Base directory for the dataset
    """
    dirs_to_create = [
        os.path.join(base_dir, 'train', 'images'),
        os.path.join(base_dir, 'train', 'masks'),
        os.path.join(base_dir, 'val', 'images'),
        os.path.join(base_dir, 'val', 'masks'),
        os.path.join(base_dir, 'test', 'images'),
        os.path.join(base_dir, 'test', 'masks')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create a README file with dataset structure info
    readme_content = """
# Fundus Image Segmentation Dataset Structure

This directory contains the fundus image dataset organized as follows:

```
dataset/
├── train/
│   ├── images/          # Training fundus images
│   └── masks/           # Training segmentation masks
├── val/
│   ├── images/          # Validation fundus images
│   └── masks/           # Validation segmentation masks
└── test/
    ├── images/          # Test fundus images
    └── masks/           # Test segmentation masks
```

## Dataset Guidelines:

1. **Image Format**: Images should be in PNG, JPG, or TIFF format
2. **Mask Format**: Masks should be binary images (0 for background, 255 for affected areas)
3. **Naming Convention**: Corresponding masks should have the same filename as images
4. **Image Size**: Recommended minimum size is 512x512 pixels
5. **Quality**: High-resolution retinal photographs with good contrast

## Common Datasets:

- DRIVE Dataset: https://drive.grand-challenge.org/
- STARE Dataset: http://cecas.clemson.edu/~ahoover/stare/
- CHASE-DB1: https://blogs.kingston.ac.uk/retinal/chasedb1/
- IDRiD: https://idrid.grand-challenge.org/

Place your dataset files in the appropriate directories following this structure.
"""
    
    with open(os.path.join(base_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"Dataset structure created at: {base_dir}")
    print("Please place your images and masks in the appropriate directories.")