#!/usr/bin/env python3
"""
Mahalanobis Model Fitter for OSR
Vote-Tracing Paper: Fit tied-covariance Mahalanobis + EVT tails
"""

import json
import argparse
import math
from pathlib import Path
import numpy as np
from typing import List, Dict

def _weibull_fit_tail(dists: np.ndarray, tail_frac: float = 0.1):
    """
    Fit Weibull to the top tail of distances using SciPy if available; 
    fallback to robust MOM if SciPy is absent.
    
    Args:
        dists: Distance array
        tail_frac: Fraction of tail to fit
        
    Returns:
        Dict with shape, scale, loc, and n parameters
    """
    tail_n = max(20, int(round(len(dists) * tail_frac)))
    tail = np.sort(dists)[-tail_n:]
    
    try:
        import scipy.stats as st
        # Constrain loc>=0 for distances
        c, loc, scale = st.weibull_min.fit(tail, floc=0)
        return {
            "shape": float(c), 
            "scale": float(scale), 
            "loc": float(loc), 
            "n": int(tail_n)
        }
    except Exception:
        # Simple moment-match fallback (not perfect but stable)
        m = tail.mean()
        v = tail.var()
        k = max(0.5, (m**2)/(v+1e-9))  # pseudo-shape
        lam = max(1e-9, m / math.gamma(1 + 1/max(1e-3, k)))
        return {
            "shape": float(k), 
            "scale": float(lam), 
            "loc": 0.0, 
            "n": int(tail_n)
        }

def fit_mahalanobis(train_traces: List[Dict], args) -> Dict:
    """
    Fit Mahalanobis model from training traces.
    
    Args:
        train_traces: List of training sample dicts
        args: Command line arguments
        
    Returns:
        Fitted model dictionary
    """
    # Collect features and labels for KNOWN samples only
    feats, labels, classes = [], [], set()
    key = "ensemble_logits"
    
    # Check which logit key exists
    if key not in train_traces[0]:
        key = "logits"
    
    for r in train_traces:
        if not r.get("known", False):
            continue
        if "label_idx" not in r:
            continue  # need ground-truth for fitting
            
        feats.append(np.asarray(r[key], dtype=np.float64).reshape(-1))
        labels.append(int(r["label_idx"]))
        classes.add(int(r["label_idx"]))
    
    if not feats:
        raise ValueError("No valid training samples found")
    
    X = np.vstack(feats)  # [N, D]
    y = np.asarray(labels, dtype=np.int64)
    C = len(sorted(classes))
    D = X.shape[1]
    
    print(f"Fitting Mahalanobis: {len(X)} samples, {C} classes, {D} dims")
    
    # Per-class means
    mu = np.zeros((C, D), dtype=np.float64)
    for c in range(C):
        if np.sum(y == c) > 0:
            mu[c] = X[y == c].mean(axis=0)
    
    # Tied covariance (+ shrinkage)
    centered = np.vstack([X[y == c] - mu[c] for c in range(C) if np.sum(y == c) > 0])
    cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
    
    if args.shrink > 0:
        cov = (1 - args.shrink) * cov + args.shrink * np.eye(D) * np.trace(cov) / D
    
    inv_cov = np.linalg.pinv(cov)
    
    # EVT tail fitting per class
    evt_params = []
    for c in range(C):
        if np.sum(y == c) > 0:
            # Compute distances for this class
            X_c = X[y == c]
            diffs = X_c - mu[c]
            d2 = np.einsum("nd,dd,nd->n", diffs, inv_cov, diffs)
            distances = np.sqrt(np.maximum(0.0, d2))
            evt_params.append(_weibull_fit_tail(distances, args.tail_frac))
        else:
            evt_params.append({"shape": 1.0, "scale": 1.0, "loc": 0.0, "n": 0})
    
    model = {
        "space": "logits",
        "C": C, 
        "D": D,
        "mu": mu.tolist(),
        "inv_cov": inv_cov.tolist(),
        "evt": evt_params,
        "meta": {
            "shrink": args.shrink, 
            "tail_frac": args.tail_frac,
            "n_samples": len(X),
            "classes": sorted(list(classes))
        }
    }
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Fit Mahalanobis OSR model")
    parser.add_argument("--train-trace-json", required=True, 
                      help="Training traces JSON with known samples")
    parser.add_argument("--out", default="data/mahal_model.json",
                      help="Output model JSON file")
    parser.add_argument("--shrink", type=float, default=0.05,
                      help="Diagonal shrinkage (0..0.5)")
    parser.add_argument("--tail-frac", type=float, default=0.1,
                      help="Fraction for EVT tail fit per class")
    
    args = parser.parse_args()
    
    # Load training data
    train_data = json.loads(Path(args.train_trace_json).read_text())
    print(f"Loaded {len(train_data)} training samples")
    
    # Fit model
    model = fit_mahalanobis(train_data, args)
    
    # Save model
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(model, indent=2))
    
    print(f"✅ Mahalanobis model written to {args.out}")
    print(f"   Classes: {model['C']}, Dims: {model['D']}")
    print(f"   Shrinkage: {args.shrink}, Tail fraction: {args.tail_frac}")

if __name__ == "__main__":
    main()