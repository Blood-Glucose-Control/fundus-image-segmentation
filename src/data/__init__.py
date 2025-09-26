"""
Data Loading and Preprocessing Utilities for Fundus Images
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FundusDataset(Dataset):
    """
    Custom Dataset class for fundus images and their segmentation masks
    """
    
    def __init__(self, images_dir, masks_dir, transform=None, image_size=512):
        """
        Args:
            images_dir (str): Directory path containing fundus images
            masks_dir (str): Directory path containing segmentation masks
            transform (callable): Optional transform to be applied on images
            image_size (int): Target size for image resizing
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding mask
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # If no mask available, create a dummy mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Resize image and mask
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 127).astype(np.float32)
        
        return image, mask

def get_train_transform(image_size=512):
    """Get training data augmentation transforms"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transform(image_size=512):
    """Get validation data transforms"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def create_data_loaders(train_images_dir, train_masks_dir, 
                       val_images_dir, val_masks_dir,
                       batch_size=4, image_size=512, num_workers=2):
    """
    Create training and validation data loaders
    
    Args:
        train_images_dir (str): Training images directory
        train_masks_dir (str): Training masks directory
        val_images_dir (str): Validation images directory
        val_masks_dir (str): Validation masks directory
        batch_size (int): Batch size for data loaders
        image_size (int): Target image size
        num_workers (int): Number of worker processes
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Create datasets
    train_dataset = FundusDataset(
        train_images_dir, 
        train_masks_dir,
        transform=get_train_transform(image_size)
    )
    
    val_dataset = FundusDataset(
        val_images_dir,
        val_masks_dir,
        transform=get_val_transform(image_size)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def preprocess_single_image(image_path, image_size=512):
    """
    Preprocess a single fundus image for inference
    
    Args:
        image_path (str): Path to the image file
        image_size (int): Target image size
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply preprocessing
    transform = get_val_transform(image_size)
    
    # Convert to PIL Image for albumentations
    image = Image.fromarray(image)
    image = np.array(image)
    
    # Apply transforms
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor