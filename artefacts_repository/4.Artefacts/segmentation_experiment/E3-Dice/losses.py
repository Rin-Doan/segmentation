"""
Advanced Loss Functions for Addressing Class Imbalance in Medical Segmentation

This module implements:
1. Focal Loss - Focuses on hard-to-classify examples
2. Dice Loss - Optimizes overlap directly (good for segmentation)
3. Tversky Loss - Generalization of Dice with adjustable precision/recall balance
4. Combined Loss - Hybrid approaches for best results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Focuses training on hard examples by down-weighting easy examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights (tensor of shape [num_classes]) or scalar
        gamma: Focusing parameter (default: 2.0)
               Higher gamma = more focus on hard examples
        reduction: 'mean', 'sum', or 'none'
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) - logits from model
            targets: (N, H, W) - ground truth labels
        """
        # Get cross entropy loss (no reduction yet)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p_t = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss - Directly optimizes Dice coefficient (F1 score)
    
    Especially good for segmentation tasks as it directly optimizes
    the overlap between prediction and ground truth.
    
    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        weight: Per-class weights (tensor of shape [num_classes])
        ignore_index: Class index to ignore in loss calculation
    """
    def __init__(self, smooth=1.0, weight=None, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) - logits from model
            targets: (N, H, W) - ground truth labels
        """
        num_classes = inputs.shape[1]
        
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)
        
        # Calculate Dice coefficient per class
        dice_losses = []
        for cls in range(num_classes):
            if self.ignore_index is not None and cls == self.ignore_index:
                continue
                
            input_cls = inputs[:, cls, :, :]
            target_cls = targets_one_hot[:, cls, :, :]
            
            intersection = (input_cls * target_cls).sum()
            union = input_cls.sum() + target_cls.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            
            # Convert to loss (1 - dice) then apply class weight
            dice_loss_cls = 1.0 - dice
            if self.weight is not None:
                dice_loss_cls = dice_loss_cls * self.weight[cls]
            
            dice_losses.append(dice_loss_cls)
        
        # Return weighted average of per-class losses
        if self.weight is not None:
            # Normalize by sum of weights to keep loss in reasonable range
            total_weight = sum([self.weight[cls].item() for cls in range(num_classes) 
                               if self.ignore_index is None or cls != self.ignore_index])
            dice_loss = torch.stack(dice_losses).sum() / total_weight
        else:
            dice_loss = torch.stack(dice_losses).mean()
        
        return dice_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss
    
    Allows control over false positives vs false negatives via alpha/beta.
    Useful when you want to penalize one type of error more than the other.
    
    Args:
        alpha: Weight for false positives (default: 0.5)
        beta: Weight for false negatives (default: 0.5)
               alpha=beta=0.5 is equivalent to Dice loss
               alpha>beta: penalize false positives more
               alpha<beta: penalize false negatives more
        smooth: Smoothing factor
        weight: Per-class weights
    
    Reference:
        Salehi et al. "Tversky loss function for image segmentation 
        using 3D fully convolutional deep networks" (2017)
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, weight=None):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.weight = weight
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C, H, W) - logits from model
            targets: (N, H, W) - ground truth labels
        """
        num_classes = inputs.shape[1]
        
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate Tversky index per class
        tversky_losses = []
        for cls in range(num_classes):
            input_cls = inputs[:, cls, :, :]
            target_cls = targets_one_hot[:, cls, :, :]
            
            # True Positives, False Positives, False Negatives
            tp = (input_cls * target_cls).sum()
            fp = (input_cls * (1 - target_cls)).sum()
            fn = ((1 - input_cls) * target_cls).sum()
            
            tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            
            # Convert to loss (1 - tversky) then apply class weight
            tversky_loss_cls = 1.0 - tversky
            if self.weight is not None:
                tversky_loss_cls = tversky_loss_cls * self.weight[cls]
            
            tversky_losses.append(tversky_loss_cls)
        
        # Return weighted average of per-class losses
        if self.weight is not None:
            # Normalize by sum of weights to keep loss in reasonable range
            total_weight = sum([self.weight[cls].item() for cls in range(num_classes)])
            tversky_loss = torch.stack(tversky_losses).sum() / total_weight
        else:
            tversky_loss = torch.stack(tversky_losses).mean()
        
        return tversky_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss: Weighted combination of multiple loss functions
    
    Combines the benefits of:
    - CrossEntropy/Focal: Pixel-wise classification accuracy
    - Dice: Region overlap optimization
    
    This often works better than using a single loss function.
    
    Args:
        ce_weight: Weight for cross-entropy/focal term
        dice_weight: Weight for dice term
        class_weights: Class weights for imbalance
        use_focal: Use Focal Loss instead of CE (recommended for severe imbalance)
        focal_gamma: Gamma parameter for Focal Loss
        dice_smooth: Smoothing factor for Dice Loss
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, class_weights=None, 
                 use_focal=True, focal_gamma=2.0, dice_smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        if use_focal:
            self.ce_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        self.dice_loss = DiceLoss(smooth=dice_smooth, weight=class_weights)
    
    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice


class WeightedCrossEntropyLoss(nn.Module):
    """
    CrossEntropy with automatic class weighting
    
    Automatically computes inverse frequency weights from data.
    Useful when you don't know the class distribution in advance.
    
    Args:
        num_classes: Number of classes
        ignore_index: Class to ignore
    """
    def __init__(self, num_classes, ignore_index=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.register_buffer('class_weights', torch.ones(num_classes))
    
    def update_weights(self, targets):
        """Update class weights based on current batch"""
        class_counts = torch.bincount(targets.flatten(), minlength=self.num_classes).float()
        class_counts = torch.clamp(class_counts, min=1.0)  # Avoid division by zero
        
        # Inverse frequency weighting
        total = class_counts.sum()
        self.class_weights = total / (self.num_classes * class_counts)
        
        # Normalize weights
        self.class_weights = self.class_weights / self.class_weights.sum() * self.num_classes
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.class_weights, 
                               ignore_index=self.ignore_index)


# Utility function to create loss from config
def get_loss_function(loss_type='combined', class_weights=None, **kwargs):
    """
    Factory function to create loss functions
    
    Args:
        loss_type: 'ce', 'focal', 'dice', 'tversky', 'combined'
        class_weights: Class weights tensor
        **kwargs: Additional arguments for specific loss functions
    
    Returns:
        Loss function module
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    
    elif loss_type == 'dice':
        smooth = kwargs.get('smooth', 1.0)
        return DiceLoss(smooth=smooth, weight=class_weights)
    
    elif loss_type == 'tversky':
        alpha = kwargs.get('alpha', 0.5)
        beta = kwargs.get('beta', 0.5)
        smooth = kwargs.get('smooth', 1.0)
        return TverskyLoss(alpha=alpha, beta=beta, smooth=smooth, weight=class_weights)
    
    elif loss_type == 'combined':
        ce_weight = kwargs.get('ce_weight', 0.5)
        dice_weight = kwargs.get('dice_weight', 0.5)
        use_focal = kwargs.get('use_focal', True)
        focal_gamma = kwargs.get('focal_gamma', 2.0)
        return CombinedLoss(ce_weight=ce_weight, dice_weight=dice_weight,
                           class_weights=class_weights, use_focal=use_focal,
                           focal_gamma=focal_gamma)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


