# utils/metrics.py
import torch
import numpy as np

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.float()
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_coefficient(pred, target, smooth=1e-6):
    """Calculate IoU score (Jaccard index)"""
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.float()
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def pixel_accuracy(pred, target):
    """Calculate pixel accuracy"""
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.float()
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    correct = (pred == target).sum()
    total = pred.numel()
    
    return (correct / total).item()


def evaluate_metrics(model, dataloader, device):
    """Evaluate model performance on dataset"""
    model.eval()
    dice_scores = []
    iou_scores = []
    accuracies = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Calculate metrics
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            acc = pixel_accuracy(outputs, masks)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            accuracies.append(acc)
    
    return {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'accuracy': np.mean(accuracies)
    }

def dice_coefficient_3d(pred, target, smooth=1e-6):
    """3D version of Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    # Flatten all dimensions except batch
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    
    intersection = (pred * target).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    
    return dice.mean().item()