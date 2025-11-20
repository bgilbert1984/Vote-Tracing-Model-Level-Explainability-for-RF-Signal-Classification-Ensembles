#!/usr/bin/env python3
"""
Multi-Method OSR ROC Figure Generator
Vote-Tracing Paper: Beautiful ROC curves for all OSR methods
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Set up matplotlib for publication quality
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

# Robust import: works whether run as module or as a plain script
try:
    from scripts.osr_scores import (
        energy_from_logits,
        osr_energy_minus_disagreement,
        maxprob_score,
        entropy_score,
        odin_score,
        mos_score_from_logits,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    sys.path.append(str(ROOT))           # repo root
    sys.path.append(str(HERE))           # scripts/
    from osr_scores import (
        energy_from_logits,
        osr_energy_minus_disagreement,
        maxprob_score,
        entropy_score,
        odin_score,
        mos_score_from_logits,
    )

def mahal_distance(logit_vec, model):
    """Compute Mahalanobis distance using fitted model."""
    mu = np.array(model["mu"])
    inv_cov = np.array(model["inv_cov"]) 
    diffs = mu - logit_vec[None, :]
    left = diffs @ inv_cov
    d2 = np.einsum("ij,ij->i", left, diffs)
    min_dist = float(np.sqrt(np.maximum(0.0, np.min(d2))))
    return min_dist

def evt_survival(x, shape, scale, loc=0.0):
    """Compute EVT survival probability."""
    try:
        import scipy.stats as st
        return float(st.weibull_min.sf(x, shape, loc=loc, scale=scale))
    except ImportError:
        # Crude fallback: exp(-(x/scale)^shape)
        z = max(0.0, (x - loc) / max(1e-9, scale))
        return float(np.exp(-(z ** shape)))

def compute_roc_curve(scores, labels):
    """Simple ROC curve computation."""
    # Sort by score (descending for confidence)
    paired = list(zip(scores, labels))
    paired.sort(key=lambda x: x[0], reverse=True)
    
    tpr_list, fpr_list = [0], [0]
    tp = fp = 0
    
    total_pos = sum(labels)  # unknown samples
    total_neg = len(labels) - total_pos  # known samples
    
    for score, label in paired:
        if label == 1:  # unknown (positive)
            tp += 1
        else:  # known (negative) 
            fp += 1
        
        tpr = tp / max(1, total_pos)
        fpr = fp / max(1, total_neg)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return fpr_list, tpr_list

def compute_auc(fpr, tpr):
    """Compute AUC from FPR/TPR arrays."""
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return auc

def plot_osr_rocs(traces_file: str, mahal_model_file: str = None, output_file: str = "figs/osr_rocs.pdf"):
    """
    Generate multi-method OSR ROC curves.
    
    Args:
        traces_file: JSON file with test traces (known + unknown)
        mahal_model_file: Optional Mahalanobis model file
        output_file: Output PDF file
    """
    # Load data
    rows = json.loads(Path(traces_file).read_text())
    print(f"Loaded {len(rows)} test samples")
    
    # Extract labels (0=known, 1=unknown)
    y_true = np.array([0 if r["known"] else 1 for r in rows])
    print(f"  Known: {np.sum(y_true == 0)}, Unknown: {np.sum(y_true == 1)}")
    
    # Collect scores for each method (higher = more confident/known)
    methods = {}
    
    # Energy-only
    energy_scores = []
    for r in rows:
        logits = torch.tensor(r.get("ensemble_logits", r.get("logits")), dtype=torch.float32).unsqueeze(0)
        energy_scores.append(-energy_from_logits(logits))  # flip for knownness
    methods["Energy"] = energy_scores
    
    # Our Energy - λ·σ method
    ours_scores = []
    for r in rows:
        logits = torch.tensor(r.get("ensemble_logits", r.get("logits")), dtype=torch.float32).unsqueeze(0)
        osr = osr_energy_minus_disagreement(logits, r["per_model_probs"], r["pred_idx"], lam=10.2)
        ours_scores.append(-osr["score"])  # flip for knownness
    methods["Energy - λ·σ (ours)"] = ours_scores
    
    # MaxProb + Entropy
    maxprob_entropy_scores = []
    for r in rows:
        probs = torch.tensor(r["ensemble_prob"], dtype=torch.float32)
        maxp = maxprob_score(probs)
        entropy = entropy_score(probs)
        maxprob_entropy_scores.append(maxp - 0.2 * entropy)
    methods["MaxProb+Entropy"] = maxprob_entropy_scores
    
    # ODIN
    odin_scores = []
    for r in rows:
        logits = torch.tensor(r.get("ensemble_logits", r.get("logits")), dtype=torch.float32).unsqueeze(0)
        odin_scores.append(odin_score(logits, T=1000.0))
    methods["ODIN (T=1000)"] = odin_scores
    
    # MOS
    mos_scores = []
    for r in rows:
        logits = torch.tensor(r.get("ensemble_logits", r.get("logits")), dtype=torch.float32).unsqueeze(0)
        mos_scores.append(mos_score_from_logits(logits, K=50))
    methods["MOS (K=50)"] = mos_scores
    
    # Optional Mahalanobis methods
    if mahal_model_file and Path(mahal_model_file).exists():
        model = json.loads(Path(mahal_model_file).read_text())
        print("Including Mahalanobis methods")
        
        mahal_scores = []
        mahal_evt_scores = []
        
        for r in rows:
            logit_vec = np.array(r.get("ensemble_logits", r.get("logits")), dtype=np.float64)
            dist = mahal_distance(logit_vec, model)
            
            # Basic Mahalanobis (negative distance for knownness)
            mahal_scores.append(-dist)
            
            # Mahalanobis + EVT
            pred_class = r["pred_idx"]
            if 0 <= pred_class < len(model["evt"]):
                evt_params = model["evt"][pred_class]
                survival_prob = evt_survival(dist, evt_params["shape"], 
                                           evt_params["scale"], evt_params["loc"])
                # Higher survival = more OOD, so flip for knownness
                mahal_evt_scores.append(1.0 - survival_prob)
            else:
                mahal_evt_scores.append(-dist)
        
        methods["Mahalanobis (tied cov)"] = mahal_scores
        methods["Mahalanobis + EVT"] = mahal_evt_scores
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curves
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, (name, scores) in enumerate(methods.items()):
        # Flip scores for unknown detection (higher score = more likely unknown)
        ood_scores = [-s for s in scores]
        fpr, tpr = compute_roc_curve(ood_scores, y_true.tolist())
        auc_val = compute_auc(fpr, tpr)
        
        color = colors[i % len(colors)]
        linestyle = '-' if 'ours' in name.lower() else '-'
        linewidth = 3.0 if 'ours' in name.lower() else 2.0
        
        plt.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=linewidth,
                label=f"{name} (AUC={auc_val:.3f})")
    
    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
    
    # Formatting
    plt.xlabel("FPR (Known → Unknown)")
    plt.ylabel("TPR (Unknown detected)")
    plt.title("OSR Performance – Vote Tracing Ensemble")
    plt.legend(loc="lower right", framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor='white')
    
    print(f"🖼  ROC curves written to {output_file}")
    
    # Print summary
    print("\nAUROC Summary:")
    for name, scores in methods.items():
        ood_scores = [-s for s in scores]
        fpr, tpr = compute_roc_curve(ood_scores, y_true.tolist())
        auc_val = compute_auc(fpr, tpr)
        print(f"  {name:25s}: {auc_val:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Generate OSR ROC curves")
    parser.add_argument("--trace-json", required=True,
                      help="Test traces JSON (known + unknown)")
    parser.add_argument("--mahal-model", default=None,
                      help="Optional Mahalanobis model JSON")
    parser.add_argument("--out", default="figs/osr_rocs.pdf",
                      help="Output PDF file")
    
    args = parser.parse_args()
    
    plot_osr_rocs(args.trace_json, args.mahal_model, args.out)

if __name__ == "__main__":
    main()