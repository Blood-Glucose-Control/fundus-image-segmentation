"""
Training Pipeline for Fundus Image Segmentation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply softmax to predictions
        predictions = torch.softmax(predictions, dim=1)
        predictions = predictions[:, 1, :, :]  # Get foreground predictions
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions, targets):
        targets_long = targets.long()
        bce_loss = self.bce(predictions, targets_long)
        dice_loss = self.dice(predictions, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate segmentation metrics
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth masks
        threshold (float): Threshold for binary classification
    
    Returns:
        dict: Dictionary containing various metrics
    """
    with torch.no_grad():
        # Apply softmax and get predictions
        probs = torch.softmax(predictions, dim=1)
        preds = (probs[:, 1, :, :] > threshold).float()
        
        # Flatten tensors
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        # Calculate metrics
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum() - intersection
        
        # IoU (Intersection over Union)
        iou = intersection / (union + 1e-6)
        
        # Dice coefficient
        dice = (2 * intersection) / (preds_flat.sum() + targets_flat.sum() + 1e-6)
        
        # Accuracy
        correct = (preds_flat == targets_flat).sum()
        accuracy = correct / targets_flat.numel()
        
        # Sensitivity (Recall)
        true_positives = intersection
        sensitivity = true_positives / (targets_flat.sum() + 1e-6)
        
        # Specificity
        true_negatives = ((1 - preds_flat) * (1 - targets_flat)).sum()
        specificity = true_negatives / ((1 - targets_flat).sum() + 1e-6)
        
        return {
            'iou': iou.item(),
            'dice': dice.item(),
            'accuracy': accuracy.item(),
            'sensitivity': sensitivity.item(),
            'specificity': specificity.item()
        }

class Trainer:
    """
    Training class for fundus image segmentation
    """
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-3, log_dir='logs'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} - Training')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        metrics_sum = {'iou': 0, 'dice': 0, 'accuracy': 0, 'sensitivity': 0, 'specificity': 0}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} - Validation')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, masks)
                for key in metrics_sum:
                    metrics_sum[key] += metrics[key]
                
                pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: v / len(self.val_loader) for k, v in metrics_sum.items()}
        
        self.val_losses.append(avg_loss)
        self.val_metrics.append(avg_metrics)
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        for key, value in avg_metrics.items():
            self.writer.add_scalar(f'Val/{key.capitalize()}', value, epoch)
        
        return avg_loss, avg_metrics
    
    def train(self, num_epochs, save_dir='checkpoints'):
        """
        Full training loop
        
        Args:
            num_epochs (int): Number of epochs to train
            save_dir (str): Directory to save model checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved with val_loss: {val_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val IoU: {val_metrics['iou']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
            print("-" * 50)
        
        self.writer.close()
        print("Training completed!")
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # IoU and Dice
        if self.val_metrics:
            iou_values = [m['iou'] for m in self.val_metrics]
            dice_values = [m['dice'] for m in self.val_metrics]
            
            ax2.plot(epochs, iou_values, 'g-', label='IoU')
            ax2.plot(epochs, dice_values, 'm-', label='Dice')
            ax2.set_title('IoU and Dice Coefficient')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Score')
            ax2.legend()
            ax2.grid(True)
            
            # Accuracy
            accuracy_values = [m['accuracy'] for m in self.val_metrics]
            ax3.plot(epochs, accuracy_values, 'c-', label='Accuracy')
            ax3.set_title('Validation Accuracy')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy')
            ax3.legend()
            ax3.grid(True)
            
            # Sensitivity and Specificity
            sensitivity_values = [m['sensitivity'] for m in self.val_metrics]
            specificity_values = [m['specificity'] for m in self.val_metrics]
            
            ax4.plot(epochs, sensitivity_values, 'y-', label='Sensitivity')
            ax4.plot(epochs, specificity_values, 'k-', label='Specificity')
            ax4.set_title('Sensitivity and Specificity')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Score')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()