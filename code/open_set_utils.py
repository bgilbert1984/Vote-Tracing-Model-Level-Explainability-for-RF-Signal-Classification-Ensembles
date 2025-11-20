
import numpy as np

def softmax(logits, axis=-1):
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)

def scores_from_logits(logits):
    # Return (s_max, entropy, energy, probs)
    p = softmax(logits)
    s_max = float(p.max())
    H = float(-(p * np.log(p + 1e-12)).sum())
    E = float(-np.log(np.exp(logits).sum()))
    return s_max, H, E, p

def oscr_curve(known_logits, known_labels, unknown_logits, thresholds=None):
    # CCR vs FPR_U across a threshold sweep of s_max
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.99, 40)
    def prep(logit_arr):
        smax = []
        yhat = []
        for l in logit_arr:
            p = softmax(l)
            smax.append(float(p.max()))
            yhat.append(int(np.argmax(p)))
        return np.array(smax), np.array(yhat)
    s_known, yhat_known = prep(known_logits)
    s_unknown, _ = prep(unknown_logits)

    fpr_u, ccr = [], []
    for t in thresholds:
        fpr = float(np.mean(s_unknown >= t)) if len(s_unknown) else 0.0
        mask_acc = (s_known >= t)
        correct = (yhat_known == known_labels) & mask_acc
        ccr.append(float(np.mean(correct)) if len(known_labels) else 0.0)
        fpr_u.append(fpr)
    return np.array(fpr_u), np.array(ccr), np.array(thresholds)

def precision_recall_unknown(known_logits, unknown_logits, thresholds=None):
    # Treat unknown as positive; score = 1 - s_max
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    def smax_arr(logit_arr):
        return np.array([softmax(l).max() for l in logit_arr])
    s_k = smax_arr(known_logits)
    s_u = smax_arr(unknown_logits)
    y_true = np.concatenate([np.zeros_like(s_k), np.ones_like(s_u)])  # 1=unknown
    s_unknown = np.concatenate([1.0 - s_k, 1.0 - s_u])

    precision, recall = [], []
    for t in thresholds:
        pred_pos = (s_unknown >= t)
        tp = float(((pred_pos) & (y_true == 1)).sum())
        fp = float(((pred_pos) & (y_true == 0)).sum())
        fn = float(((~pred_pos) & (y_true == 1)).sum())
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        precision.append(p); recall.append(r)
    # Left Riemann sum for AUPR
    aupr = 0.0
    for i in range(1, len(thresholds)):
        aupr += (recall[i] - recall[i-1]) * precision[i]
    return np.array(precision), np.array(recall), np.array(thresholds), float(aupr)

def utility_vs_threshold(known_logits, known_labels, thresholds=None):
    # Utility = accuracy_on_accepted * coverage_on_known (max-prob gate)
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.99, 40)
    accs, covs, utils = [], [], []
    p = softmax(known_logits)
    smax = p.max(axis=1)
    yhat = p.argmax(axis=1)
    for t in thresholds:
        accept = (smax >= t)
        cov = float(np.mean(accept)) if len(accept) else 0.0
        acc = float(np.mean((yhat == known_labels)[accept])) if cov > 0 else 0.0
        accs.append(acc); covs.append(cov); utils.append(acc*cov)
    return np.array(accs), np.array(covs), np.array(utils), np.array(thresholds)

def apply_open_set_policy(probs, logits, tau_p=0.60, tau_H=1.2, tau_E=None):
    """
    Apply open-set policy for unknown detection.
    
    Args:
        probs: Softmax probabilities (numpy array)
        logits: Raw logits (numpy array) 
        tau_p: Maximum probability threshold
        tau_H: Entropy threshold
        tau_E: Optional energy threshold
        
    Returns:
        (accept, metrics): (bool, dict) - whether to accept as known class
    """
    s_max = float(probs.max())
    H = float(-(probs * np.log(probs + 1e-12)).sum())
    E = None
    if tau_E is not None:
        E = float(-np.log(np.exp(logits).sum()))
    
    accept = (s_max >= tau_p) and (H <= tau_H) and (tau_E is None or E >= tau_E)
    return accept, {"s_max": s_max, "entropy": H, "energy": E}