"""
OpenMax-style open-set detection using per-class EVT
"""

import numpy as np
from open_set_evt import _fit_weibull_tail, weibull_tail_cdf, softmax

def compute_class_mavs(logits, labels, num_classes):
    """Compute mean activation vectors (MAVs) per class"""
    mavs = {}
    for c in range(num_classes):
        class_mask = (labels == c)
        if np.any(class_mask):
            class_logits = logits[class_mask]
            mavs[c] = np.mean(class_logits, axis=0)
        else:
            mavs[c] = np.zeros(logits.shape[1])
    return mavs

def compute_distances_to_mavs(logits, labels, mavs):
    """Compute distances from each sample to its predicted class MAV"""
    distances = []
    predictions = np.argmax(softmax(logits), axis=1)
    
    for i, (logit, pred, true_label) in enumerate(zip(logits, predictions, labels)):
        if pred in mavs:
            # Euclidean distance to predicted class MAV
            dist = np.linalg.norm(logit - mavs[pred])
            distances.append(dist)
        else:
            distances.append(float('inf'))
    
    return np.array(distances), predictions

def fit_per_class_weibulls(logits, labels, num_classes, correct_only=True):
    """Fit Weibull distributions to per-class distance tails"""
    weibull_params = {}
    mavs = compute_class_mavs(logits, labels, num_classes)
    
    for c in range(num_classes):
        class_mask = (labels == c)
        if not np.any(class_mask):
            weibull_params[c] = (1.0, 1.0)  # Default parameters
            continue
            
        class_logits = logits[class_mask]
        class_labels = labels[class_mask]
        
        if correct_only:
            # Only use correctly classified samples for fitting
            class_probs = softmax(class_logits)
            class_preds = np.argmax(class_probs, axis=1)
            correct_mask = (class_preds == class_labels)
            if not np.any(correct_mask):
                weibull_params[c] = (1.0, 1.0)
                continue
            class_logits = class_logits[correct_mask]
        
        # Compute distances to class MAV
        distances = []
        for logit in class_logits:
            dist = np.linalg.norm(logit - mavs[c])
            distances.append(dist)
        
        if len(distances) >= 10:
            k, lam = _fit_weibull_tail(distances, tail_frac=0.2)
        else:
            k, lam = 1.0, 1.0
        
        weibull_params[c] = (k, lam)
    
    return weibull_params, mavs

def openmax_unknown_scores(logits, weibull_params, mavs):
    """Compute unknown scores using OpenMax-style approach"""
    unknown_scores = []
    probs = softmax(logits)
    predictions = np.argmax(probs, axis=1)
    
    for logit, pred in zip(logits, predictions):
        if pred in mavs and pred in weibull_params:
            # Distance to predicted class MAV
            dist = np.linalg.norm(logit - mavs[pred])
            
            # Weibull tail probability = unknown mass
            k, lam = weibull_params[pred]
            unknown_mass = 1.0 - weibull_tail_cdf(dist, k, lam)
            unknown_scores.append(unknown_mass)
        else:
            unknown_scores.append(1.0)  # Maximum uncertainty
    
    return np.array(unknown_scores)

def oscr_from_openmax(yhat_known, y_true_known, unknown_scores_known, unknown_scores_unknown, alphas=None):
    """Compute OSCR curve from OpenMax unknown scores"""
    if alphas is None:
        alphas = np.linspace(0.01, 0.99, 40)
    
    fpr_u, ccr = [], []
    
    for alpha in alphas:
        # Accept if unknown score <= alpha (low unknown mass = more confident)
        accept_known = (unknown_scores_known <= alpha)
        accept_unknown = (unknown_scores_unknown <= alpha)
        
        # CCR: correct and accepted knowns / total knowns
        correct = (yhat_known == y_true_known) & accept_known
        ccr_val = float(np.mean(correct)) if len(y_true_known) > 0 else 0.0
        ccr.append(ccr_val)
        
        # FPR: accepted unknowns / total unknowns  
        fpr_val = float(np.mean(accept_unknown)) if len(unknown_scores_unknown) > 0 else 0.0
        fpr_u.append(fpr_val)
    
    return np.array(fpr_u), np.array(ccr)