"""Focal Tversky and contrastive alignment objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for highly imbalanced change detection datasets.
    """
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, probs, targets):
        # Probabilities should be strictly between 0 and 1 (from Sigmoid)
        t = targets.float()
        tp = (probs * t).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * ((1 - probs) * t).sum() + self.beta * (probs * (1 - t)).sum() + self.smooth)
        return (1 - tversky) ** self.gamma


class ContrastiveAlignmentLoss(nn.Module):
    """
    Contrastive Feature Alignment Loss to reduce pseudo-change sensitivity.
    - Unchanged pixels (label=0): Minimizes the L2 distance between T1 and T2 features.
    - Changed pixels (label=1): Maximizes the L2 distance up to a margin.
    """
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, f1, f2, change_label):
        # f1, f2 shape: (B, C, H, W)
        # change_label shape: (B, H, W)

        # We need to resize the change_label to match the feature spatial size
        if change_label.dim() == 3:
            change_label = change_label.unsqueeze(1)  # (B, 1, H, W)

        if change_label.shape[2:] != f1.shape[2:]:
            # Use nearest neighbor to avoid blurring the exact binary labels
            change_label = F.interpolate(change_label.float(), size=f1.shape[2:], mode='nearest')

        # Calculate L2 distance along channel dimension on normalized features
        f1_norm = F.normalize(f1, p=2, dim=1)
        f2_norm = F.normalize(f2, p=2, dim=1)
        dist = torch.norm(f1_norm - f2_norm, p=2, dim=1, keepdim=True)  # (B, 1, H, W)

        # Unchanged: minimize distance
        loss_unchanged = (1 - change_label) * dist.pow(2)

        # Changed: push distance away until it hits margin
        loss_changed = change_label * F.relu(self.margin - dist).pow(2)

        return (loss_unchanged + loss_changed).mean()
