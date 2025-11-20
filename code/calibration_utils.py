#!/usr/bin/env python3
"""
Calibration functions and aggregator - extracted for standalone use
"""

import json
import math
import numpy as np
from pathlib import Path

def _softmax(z, axis=-1):
    """Compute softmax probabilities"""
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def _ece_mce(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15):
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    
    Args:
        probs: Predicted probabilities, shape (N, C)
        y_true: True class labels, shape (N,)  
        n_bins: Number of confidence bins
        
    Returns:
        Tuple of (ECE, MCE, (bin_confs, bin_accs, bin_sizes))
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    ece = 0.0
    mce = 0.0
    bin_confs, bin_accs, bin_sizes = [], [], []
    
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        
        if n == 0:
            bin_confs.append(np.nan)
            bin_accs.append(np.nan) 
            bin_sizes.append(0)
            continue
            
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        gap = abs(acc - avg_conf)
        
        ece += (n / len(conf)) * gap
        mce = max(mce, gap)
        
        bin_confs.append(avg_conf)
        bin_accs.append(acc)
        bin_sizes.append(n)
    
    return float(ece), float(mce), (np.array(bin_confs), np.array(bin_accs), np.array(bin_sizes))

def _nll_loss(probs: np.ndarray, y_true: np.ndarray):
    """Compute negative log-likelihood loss"""
    eps = 1e-12
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps)))

def _prob_temperature(p: np.ndarray, T: float):
    """
    Apply temperature scaling to probabilities
    
    Args:
        p: Probabilities, shape (N, C)
        T: Temperature parameter
        
    Returns:
        Temperature-scaled probabilities
    """
    if T == 1.0:
        return p
    
    p = np.clip(p, 1e-12, 1.0)
    q = np.power(p, 1.0 / T)
    q /= q.sum(axis=1, keepdims=True)
    return q

def _utility_from_probs(P: np.ndarray, y_true: np.ndarray, tau: float):
    """
    Compute accuracy, coverage, and utility at confidence threshold tau
    
    Args:
        P: Predicted probabilities, shape (N, C)
        y_true: True labels, shape (N,)
        tau: Confidence threshold
        
    Returns:
        Tuple of (accuracy, coverage, utility)
    """
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    accept = conf >= tau
    
    coverage = float(np.mean(accept.astype(float)))
    
    if coverage > 0:
        acc = float(np.mean((pred[accept] == y_true[accept]).astype(float)))
    else:
        acc = 0.0
    
    util = acc * coverage
    return acc, coverage, util

class CalibrationAggregator:
    """
    Lightweight collector for calibration metrics
    """
    
    _y = []
    _p_uncal = []
    _p_cal = []

    @classmethod
    def add(cls, probs_uncal: np.ndarray, probs_cal: np.ndarray, y_true: int = None):
        """
        Add sample to aggregator
        
        Args:
            probs_uncal: Uncalibrated probabilities, shape (C,)
            probs_cal: Calibrated probabilities, shape (C,)  
            y_true: True class label (optional)
        """
        if y_true is None:
            return
            
        cls._y.append(int(y_true))
        cls._p_uncal.append(probs_uncal.astype(np.float64))
        cls._p_cal.append(probs_cal.astype(np.float64))

    @classmethod
    def reset(cls):
        """Reset aggregator state"""
        cls._y.clear()
        cls._p_uncal.clear()
        cls._p_cal.clear()

    @classmethod
    def finalize(cls, out_path: str, tau: float = 0.6, temperatures_per_model=None):
        """
        Compute final metrics and save to JSON
        
        Args:
            out_path: Output file path
            tau: Confidence threshold
            temperatures_per_model: List of per-model temperatures
            
        Returns:
            Dictionary of computed metrics
        """
        if not cls._y:
            return None
            
        Y = np.array(cls._y, dtype=np.int64)
        P_u = np.stack(cls._p_uncal, axis=0)
        P_c = np.stack(cls._p_cal, axis=0)

        # Compute calibration metrics
        ece_u, mce_u, bins_u = _ece_mce(P_u, Y)
        ece_c, mce_c, bins_c = _ece_mce(P_c, Y)
        
        # Compute other metrics
        nll_u = _nll_loss(P_u, Y)
        nll_c = _nll_loss(P_c, Y)
        acc_u, cov_u, util_u = _utility_from_probs(P_u, Y, tau)
        acc_c, cov_c, util_c = _utility_from_probs(P_c, Y, tau)

        # Build output payload
        payload = {
            "N_val": int(len(Y)),
            "tau_default": float(tau),
            "temperatures_per_model": list(temperatures_per_model or []),
            "uncalibrated": {
                "ECE": ece_u, "MCE": mce_u, "NLL": nll_u,
                "acc": acc_u, "coverage": cov_u, "utility": util_u,
                "bins": {
                    "mean_conf": [None if np.isnan(x) else float(x) for x in bins_u[0]],
                    "mean_acc":  [None if np.isnan(x) else float(x) for x in bins_u[1]],
                    "count":     [int(x) for x in bins_u[2]],
                },
            },
            "calibrated": {
                "ECE": ece_c, "MCE": mce_c, "NLL": nll_c,
                "acc": acc_c, "coverage": cov_c, "utility": util_c,
                "bins": {
                    "mean_conf": [None if np.isnan(x) else float(x) for x in bins_c[0]],
                    "mean_acc":  [None if np.isnan(x) else float(x) for x in bins_c[1]],
                    "count":     [int(x) for x in bins_c[2]],
                },
            },
        }
        
        # Save to file
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        
        return payload