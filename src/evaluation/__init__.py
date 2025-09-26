"""
Evaluation metrics and functions for fundus image segmentation
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient
    
    Args:
        pred (torch.Tensor): Predictions
        target (torch.Tensor): Ground truth
        smooth (float): Smoothing factor
    
    Returns:
        float: Dice coefficient
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()

def iou_score(pred, target, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union) score
    
    Args:
        pred (torch.Tensor): Predictions
        target (torch.Tensor): Ground truth
        smooth (float): Smoothing factor
    
    Returns:
        float: IoU score
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def pixel_accuracy(pred, target):
    """
    Calculate pixel accuracy
    
    Args:
        pred (torch.Tensor): Predictions
        target (torch.Tensor): Ground truth
    
    Returns:
        float: Pixel accuracy
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    correct = (pred == target).sum()
    total = target.numel()
    
    return (correct.float() / total).item()

def sensitivity(pred, target):
    """
    Calculate sensitivity (recall/true positive rate)
    
    Args:
        pred (torch.Tensor): Predictions
        target (torch.Tensor): Ground truth
    
    Returns:
        float: Sensitivity
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    
    if actual_positive == 0:
        return 0.0
    
    return (true_positive.float() / actual_positive).item()

def specificity(pred, target):
    """
    Calculate specificity (true negative rate)
    
    Args:
        pred (torch.Tensor): Predictions
        target (torch.Tensor): Ground truth
    
    Returns:
        float: Specificity
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    true_negative = ((1 - pred) * (1 - target)).sum()
    actual_negative = (1 - target).sum()
    
    if actual_negative == 0:
        return 0.0
    
    return (true_negative.float() / actual_negative).item()

def hausdorff_distance(pred, target):
    """
    Calculate Hausdorff distance (simplified 2D version)
    
    Args:
        pred (torch.Tensor): Predictions
        target (torch.Tensor): Ground truth
    
    Returns:
        float: Hausdorff distance
    """
    # Convert to numpy for easier processing
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    
    # Find boundary pixels
    from scipy import ndimage
    
    pred_boundary = pred_np - ndimage.binary_erosion(pred_np)
    target_boundary = target_np - ndimage.binary_erosion(target_np)
    
    # Get coordinates of boundary pixels
    pred_coords = np.argwhere(pred_boundary)
    target_coords = np.argwhere(target_boundary)
    
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0.0
    
    # Calculate distances
    from scipy.spatial.distance import cdist
    
    distances_pred_to_target = cdist(pred_coords, target_coords)
    distances_target_to_pred = cdist(target_coords, pred_coords)
    
    hausdorff_dist = max(
        np.max(np.min(distances_pred_to_target, axis=1)),
        np.max(np.min(distances_target_to_pred, axis=1))
    )
    
    return hausdorff_dist

class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_model(self, test_loader, threshold=0.5, save_results=True, output_dir='results'):
        """
        Evaluate model on test dataset
        
        Args:
            test_loader (DataLoader): Test data loader
            threshold (float): Threshold for binary predictions
            save_results (bool): Whether to save detailed results
            output_dir (str): Directory to save results
        
        Returns:
            dict: Evaluation results
        """
        import os
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        all_metrics = {
            'dice': [],
            'iou': [],
            'accuracy': [],
            'sensitivity': [],
            'specificity': [],
            'hausdorff': [],
            'percentages_pred': [],
            'percentages_true': []
        }
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluating')
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = (probs[:, 1, :, :] > threshold).float()
                
                # Calculate metrics for each image in batch
                for i in range(images.size(0)):
                    pred_i = preds[i]
                    mask_i = masks[i]
                    
                    # Calculate metrics
                    dice = dice_coefficient(pred_i, mask_i)
                    iou = iou_score(pred_i, mask_i)
                    acc = pixel_accuracy(pred_i, mask_i)
                    sens = sensitivity(pred_i, mask_i)
                    spec = specificity(pred_i, mask_i)
                    
                    # Calculate affected percentages
                    pred_percentage = (pred_i.sum() / pred_i.numel() * 100).item()
                    true_percentage = (mask_i.sum() / mask_i.numel() * 100).item()
                    
                    # Store metrics
                    all_metrics['dice'].append(dice)
                    all_metrics['iou'].append(iou)
                    all_metrics['accuracy'].append(acc)
                    all_metrics['sensitivity'].append(sens)
                    all_metrics['specificity'].append(spec)
                    all_metrics['percentages_pred'].append(pred_percentage)
                    all_metrics['percentages_true'].append(true_percentage)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Dice': f'{dice:.4f}',
                        'IoU': f'{iou:.4f}',
                        'Acc': f'{acc:.4f}'
                    })
        
        # Calculate summary statistics
        summary = {}
        for metric, values in all_metrics.items():
            if values:  # Check if list is not empty
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        # Calculate percentage error
        if all_metrics['percentages_pred'] and all_metrics['percentages_true']:
            percentage_errors = [abs(p - t) for p, t in 
                               zip(all_metrics['percentages_pred'], all_metrics['percentages_true'])]
            summary['percentage_error'] = {
                'mean': np.mean(percentage_errors),
                'std': np.std(percentage_errors),
                'min': np.min(percentage_errors),
                'max': np.max(percentage_errors),
                'median': np.median(percentage_errors)
            }
        
        if save_results:
            self.save_evaluation_results(summary, all_metrics, output_dir)
        
        return summary, all_metrics
    
    def save_evaluation_results(self, summary, all_metrics, output_dir):
        """Save evaluation results to files"""
        import json
        import pandas as pd
        
        # Save summary statistics
        with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed metrics
        df = pd.DataFrame(all_metrics)
        df.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'), index=False)
        
        # Create evaluation plots
        self.plot_evaluation_results(summary, all_metrics, output_dir)
        
        print(f"Evaluation results saved to: {output_dir}")
    
    def plot_evaluation_results(self, summary, all_metrics, output_dir):
        """Plot evaluation results"""
        import os
        
        # Plot 1: Metric distributions
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics_to_plot = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in all_metrics and all_metrics[metric]:
                axes[i].hist(all_metrics[metric], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{metric.capitalize()} Distribution')
                axes[i].set_xlabel(metric.capitalize())
                axes[i].set_ylabel('Frequency')
                axes[i].axvline(summary[metric]['mean'], color='red', linestyle='--', 
                              label=f'Mean: {summary[metric]["mean"]:.3f}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Plot percentage comparison
        if 'percentages_pred' in all_metrics and 'percentages_true' in all_metrics:
            axes[5].scatter(all_metrics['percentages_true'], all_metrics['percentages_pred'], 
                          alpha=0.6)
            axes[5].plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
            axes[5].set_xlabel('True Percentage (%)')
            axes[5].set_ylabel('Predicted Percentage (%)')
            axes[5].set_title('Percentage Prediction Accuracy')
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Error analysis
        if 'percentage_error' in summary:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Percentage error distribution
            percentage_errors = [abs(p - t) for p, t in 
                               zip(all_metrics['percentages_pred'], all_metrics['percentages_true'])]
            ax1.hist(percentage_errors, bins=30, alpha=0.7, edgecolor='black')
            ax1.set_title('Percentage Prediction Error Distribution')
            ax1.set_xlabel('Absolute Error (%)')
            ax1.set_ylabel('Frequency')
            ax1.axvline(summary['percentage_error']['mean'], color='red', linestyle='--',
                       label=f'Mean Error: {summary["percentage_error"]["mean"]:.2f}%')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Error vs true percentage
            ax2.scatter(all_metrics['percentages_true'], percentage_errors, alpha=0.6)
            ax2.set_xlabel('True Percentage (%)')
            ax2.set_ylabel('Absolute Error (%)')
            ax2.set_title('Error vs True Percentage')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def predict_single_image(self, image_tensor, threshold=0.5, return_percentage=True):
        """
        Predict segmentation for a single image
        
        Args:
            image_tensor (torch.Tensor): Input image tensor
            threshold (float): Threshold for binary prediction
            return_percentage (bool): Whether to return affected percentage
        
        Returns:
            tuple: (prediction_mask, percentage) if return_percentage else prediction_mask
        """
        self.model.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
            probs = torch.softmax(output, dim=1)
            pred_mask = (probs[:, 1, :, :] > threshold).float()
            
            if return_percentage:
                percentage = (pred_mask.sum() / pred_mask.numel() * 100).item()
                return pred_mask.cpu().numpy(), percentage
            else:
                return pred_mask.cpu().numpy()