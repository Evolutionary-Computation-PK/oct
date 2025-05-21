import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, Adam, AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from params import DISEASE_CLASSES


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Args:
        gamma: "focus" factor on challenging examples
        alpha: class weights (list or None)
    """

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        if self.alpha is not None:
            at = self.alpha[targets]
            loss = at * (1 - pt) ** self.gamma * ce
        else:
            loss = (1 - pt) ** self.gamma * ce
        return loss.mean()


def get_optimizer(name: str, params, lr: float, weight_decay: float):
    """
    Returns the optimizer by name.
    """
    name = name.lower()
    if name == 'rmsprop':
        return RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == 'adam':
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Nieznany optymalizator: {name}")


def compute_metrics(y_true, y_pred, y_prob, disease_classes=DISEASE_CLASSES):
    """Compute all required metrics."""
    # Map to binary: 1 = any disease, 0 = healthy
    y_true_binary = np.isin(y_true, disease_classes).astype(int)
    y_pred_binary = np.isin(y_pred, disease_classes).astype(int)

    disease_recall = recall_score(y_true_binary, y_pred_binary)

    metrics = {
        'disease_recall': disease_recall,
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'roc_auc': roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    }
    return metrics
