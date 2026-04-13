import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics


def compute_robust_metrics(y_true_clean, y_pred_clean, y_true_adv, y_pred_adv):
    """Compute robustness metrics"""
    # Robust accuracy
    robust_acc = accuracy_score(y_true_adv, y_pred_adv)
    
    # Attack success rate (misclassification of malicious samples)
    malicious_mask_clean = (y_true_clean == 1)
    malicious_mask_adv = (y_true_adv == 1)
    
    if malicious_mask_adv.sum() > 0:
        attack_success = ((y_pred_clean[malicious_mask_clean] == 1) & 
                          (y_pred_adv[malicious_mask_adv] == 0)).sum() / malicious_mask_adv.sum()
    else:
        attack_success = 0.0
    
    return {
        'robust_accuracy': robust_acc,
        'attack_success_rate': attack_success
    }
