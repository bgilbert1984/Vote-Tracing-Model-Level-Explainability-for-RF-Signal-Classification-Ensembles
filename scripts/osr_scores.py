#!/usr/bin/env python3
"""
Open-Set Rejection Scoring for RF Ensembles
Vote-Tracing Paper: Explainability from Vote Traces

Implements multiple OSR methods including our novel Energy - λ·σ approach.
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional

def energy_from_logits(logits: torch.Tensor, T: float = 1.0) -> float:
    """Compute energy score from logits: -log(sum(exp(z_i/T)))"""
    return float(-torch.logsumexp(logits / T, dim=-1).item())

def prob_disagreement(per_model_probs: Dict[str, List[float]], target_idx: int) -> float:
    """Compute standard deviation of per-model probabilities for predicted class."""
    vals = [float(v[target_idx]) for v in per_model_probs.values()]
    return float(np.std(vals, ddof=0)) if len(vals) else 0.0

def osr_energy_minus_disagreement(
    ensemble_logits: torch.Tensor,
    per_model_probs: Dict[str, List[float]],
    target_idx: int,
    lam: float = 10.2,
    T: float = 1.0
) -> Dict[str, float]:
    """
    Our novel OSR score: Energy - λ·disagreement
    
    Args:
        ensemble_logits: Averaged logits tensor [1, C]
        per_model_probs: Dict of model_name -> probability vector
        target_idx: Predicted class index
        lam: Lambda weighting parameter 
        T: Temperature for energy computation
        
    Returns:
        Dict with energy, sigma (disagreement), and final score
    """
    E = energy_from_logits(ensemble_logits, T=T)
    sigma = prob_disagreement(per_model_probs, target_idx)
    score = E - lam * sigma
    return {"energy": E, "sigma": sigma, "score": score}

# --- Mahalanobis Distance (tied covariance) ---
@dataclass
class MahalanobisModel:
    """Fitted Mahalanobis model for OOD detection."""
    mu: np.ndarray        # [C, D] class means
    inv_cov: np.ndarray   # [D, D] inverse covariance
    classes: List[str]

    def distance(self, feat: np.ndarray) -> float:
        """Compute minimum Mahalanobis distance to any class mean."""
        diffs = self.mu - feat[None, :]
        # (x - mu)^T inv_cov (x - mu) 
        left = diffs @ self.inv_cov
        d = np.einsum("ij,ij->i", left, diffs)
        return float(d.min())

def fit_mahalanobis_tied(
    feats: np.ndarray,           # [N, D]
    labels: np.ndarray,          # [N]
    classes: List[str],
    shrink: float = 0.0          # 0 = none; e.g., 0.1 diag shrink for stability
) -> MahalanobisModel:
    """
    Fit tied-covariance Mahalanobis model.
    
    Args:
        feats: Feature matrix [N, D]
        labels: Class labels [N]
        classes: List of class names
        shrink: Diagonal shrinkage factor (0-1)
        
    Returns:
        Fitted Mahalanobis model
    """
    C = len(classes)
    D = feats.shape[1]
    mu = np.stack([feats[labels==i].mean(axis=0) for i in range(C)], axis=0)
    # tied covariance
    centered = np.vstack([feats[labels==i] - mu[i] for i in range(C)])
    cov = (centered.T @ centered) / max(1, centered.shape[0]-1)
    if shrink > 0.0:
        cov = (1-shrink)*cov + shrink*np.eye(D)*np.trace(cov)/D
    inv_cov = np.linalg.pinv(cov)
    return MahalanobisModel(mu=mu, inv_cov=inv_cov, classes=classes)

# --- MOS (Maximum Logit Score via sub-sampled class heads) ---
def mos_score_from_logits(
    logits: torch.Tensor,  # [1, C]
    K: int = 50, 
    subfrac: float = 0.5,
    rng: np.random.Generator | None = None
) -> float:
    """
    MOS (Maximum Over Sub-sampled) score for OOD detection.
    
    Args:
        logits: Logit tensor [1, C]
        K: Number of random sub-samples
        subfrac: Fraction of classes to sample each time
        rng: Random number generator
        
    Returns:
        Maximum logit score over sub-sampled class heads
    """
    if rng is None: 
        rng = np.random.default_rng(123)
    C = logits.shape[-1]
    m = max(1, int(round(C*subfrac)))
    best = -1e9
    with torch.no_grad():
        for _ in range(K):
            idx = np.sort(rng.choice(C, size=m, replace=False))
            sl = logits[..., idx]
            best = max(best, float(sl.max().item()))
    return best

# --- Classic baseline scores ---
def maxprob_score(probs: torch.Tensor) -> float:
    """Simple maximum probability confidence score."""
    return float(probs.max().item())

def entropy_score(probs: torch.Tensor) -> float:
    """Negative entropy as confidence score (lower entropy = higher confidence)."""
    return float(-(probs * torch.log(probs.clamp_min(1e-9))).sum().item())

def odin_score(logits: torch.Tensor, T: float = 1000.0, epsilon: float = 0.002) -> float:
    """
    ODIN (Out-of-DIstribution detector for Neural networks) score.
    Simplified version using temperature scaling only.
    """
    return float(torch.max(F.softmax(logits/T, dim=-1)).item())

# --- Utility functions ---
def calibrate_threshold(scores: List[float], labels: List[int], target_coverage: float = 0.95) -> float:
    """
    Calibrate threshold to achieve target coverage on known samples.
    
    Args:
        scores: List of confidence scores
        labels: List of labels (0=known, 1=unknown)
        target_coverage: Target fraction of known samples to accept
        
    Returns:
        Calibrated threshold
    """
    known_scores = [s for s, l in zip(scores, labels) if l == 0]
    if not known_scores:
        return 0.0
    return float(np.quantile(known_scores, 1.0 - target_coverage))

def compute_auroc(scores: List[float], labels: List[int]) -> float:
    """
    Compute AUROC for OOD detection.
    
    Args:
        scores: Confidence scores (higher = more likely known)
        labels: Binary labels (0=known, 1=unknown)
        
    Returns:
        AUROC score
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        # Fallback: simple implementation
        pos_scores = [s for s, l in zip(scores, labels) if l == 1]  # unknown
        neg_scores = [s for s, l in zip(scores, labels) if l == 0]  # known
        
        if not pos_scores or not neg_scores:
            return 0.5
            
        total = 0
        for pos_score in pos_scores:
            for neg_score in neg_scores:
                if pos_score < neg_score:  # unknown has lower confidence than known
                    total += 1
                elif pos_score == neg_score:
                    total += 0.5
                    
        return total / (len(pos_scores) * len(neg_scores))


if __name__ == "__main__":
    # Quick test
    import torch
    
    # Test data
    logits = torch.randn(1, 8)
    probs = F.softmax(logits, dim=-1)
    per_model_probs = {
        "model_0": [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02],
        "model_1": [0.15, 0.25, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02], 
        "model_2": [0.05, 0.15, 0.35, 0.25, 0.1, 0.05, 0.03, 0.02]
    }
    target_idx = 2
    
    # Test our OSR method
    osr = osr_energy_minus_disagreement(logits, per_model_probs, target_idx)
    print("Energy - λ·σ OSR:")
    print(f"  Energy: {osr['energy']:.3f}")
    print(f"  Disagreement σ: {osr['sigma']:.3f}")
    print(f"  OSR Score: {osr['score']:.3f}")
    
    # Test other methods
    print(f"\nOther methods:")
    print(f"  MaxProb: {maxprob_score(probs):.3f}")
    print(f"  Entropy: {entropy_score(probs):.3f}") 
    print(f"  ODIN: {odin_score(logits):.3f}")
    print(f"  MOS: {mos_score_from_logits(logits):.3f}")