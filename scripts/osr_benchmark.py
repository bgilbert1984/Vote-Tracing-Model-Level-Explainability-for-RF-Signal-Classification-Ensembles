#!/usr/bin/env python3
"""
OSR Benchmark Runner for Vote-Tracing Paper
Compares multiple Open-Set Rejection methods.
"""

import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict
from scripts.osr_scores import (
    osr_energy_minus_disagreement, 
    mos_score_from_logits,
    maxprob_score,
    entropy_score,
    odin_score,
    energy_from_logits,
    compute_auroc,
    calibrate_threshold
)

def run_bench(rows: List[Dict], tau_cov: float=0.95, lam: float=10.2, T: float=1.0, mos_K: int=50):
    """
    Run OSR benchmark across multiple methods.
    
    Args:
        rows: List of dict with keys:
              "known": bool, "correct": bool, 
              "logits" or "ensemble_logits": list[float], 
              "ensemble_prob": list[float],
              "per_model_probs": {name: [floats]}, 
              "pred_idx": int
        tau_cov: Target coverage for known samples
        lam: Lambda parameter for our Energy-σ method
        T: Temperature for energy computation
        mos_K: Number of samples for MOS
        
    Returns:
        (table_dict, threshold) tuple for rendering
    """
    
    def eval_method(score_fn, extra_forwards=0):
        """Evaluate a single OSR method."""
        y_true = []  # 0=known, 1=unknown
        scores = []  # higher = more confident/known
        
        for r in rows:
            y_true.append(0 if r["known"] else 1)
            scores.append(score_fn(r))
        
        # Calibrate threshold for target coverage on known samples
        threshold = calibrate_threshold(scores, y_true, tau_cov)
        
        # Compute metrics
        auroc = compute_auroc(scores, y_true)
        
        # Acceptance/rejection rates
        accepted = [s >= threshold for s in scores]
        known_accepted = sum(1 for i, acc in enumerate(accepted) 
                           if y_true[i] == 0 and acc)
        total_known = sum(1 for y in y_true if y == 0)
        known_acc_rate = known_accepted / max(1, total_known)
        
        unknown_rejected = sum(1 for i, acc in enumerate(accepted)
                             if y_true[i] == 1 and not acc)
        total_unknown = sum(1 for y in y_true if y == 1)
        unk_reject_rate = unknown_rejected / max(1, total_unknown)
        
        # Accuracy on accepted known samples
        correct_known_accepted = sum(1 for i, acc in enumerate(accepted)
                                   if y_true[i] == 0 and acc and 
                                      rows[i].get("correct", True))
        known_accuracy = correct_known_accepted / max(1, known_accepted)
        
        return known_accuracy, unk_reject_rate, auroc, extra_forwards
    
    # Define scoring functions
    def energy_score_fn(r):
        logits = torch.tensor(r.get("ensemble_logits", r.get("logits")), 
                             dtype=torch.float32).unsqueeze(0)
        return -energy_from_logits(logits, T=T)
    
    def our_score_fn(r):
        logits = torch.tensor(r.get("ensemble_logits", r.get("logits")), 
                             dtype=torch.float32).unsqueeze(0)
        osr = osr_energy_minus_disagreement(logits, r["per_model_probs"], 
                                          r["pred_idx"], lam=lam, T=T)
        return -osr["score"]  # flip sign for "knownness"
    
    def maxprob_entropy_fn(r):
        probs = torch.tensor(r["ensemble_prob"], dtype=torch.float32)
        maxp = maxprob_score(probs) 
        entropy = entropy_score(probs)
        return maxp - 0.2 * entropy  # combined score
    
    def odin_score_fn(r):
        logits = torch.tensor(r.get("ensemble_logits", r.get("logits")), 
                             dtype=torch.float32).unsqueeze(0)
        return odin_score(logits, T=1000.0)
    
    def mos_score_fn(r):
        logits = torch.tensor(r.get("ensemble_logits", r.get("logits")), 
                             dtype=torch.float32).unsqueeze(0)
        return mos_score_from_logits(logits, K=mos_K, subfrac=0.5)
    
    # Evaluate all methods
    methods = [
        ("MaxProb+Entropy", maxprob_entropy_fn, 0),
        ("ODIN(T=1000,eps=0.002)", odin_score_fn, 1),  
        ("Energy", energy_score_fn, 0),
        ("MOS(K=50)", mos_score_fn, mos_K),
        ("Energy+Disagreement (ours)", our_score_fn, 0),
    ]
    
    results = {
        "method": [],
        "known_acc": [],
        "unk_reject": [],
        "auroc": [],
        "extra_forwards": [],
        "mem_overhead": [],
        "needs_fit": [],
    }
    
    for method_name, score_fn, extra_fwd in methods:
        known_acc, unk_rej, auroc, _ = eval_method(score_fn, extra_fwd)
        
        results["method"].append(method_name)
        results["known_acc"].append(round(100 * known_acc, 1))
        results["unk_reject"].append(round(100 * unk_rej, 1))
        results["auroc"].append(round(auroc, 3))
        results["extra_forwards"].append(extra_fwd)
        results["mem_overhead"].append("None")
        results["needs_fit"].append("No")
    
    # Compute threshold from our method for reporting
    threshold = 0.0
    try:
        scores = [our_score_fn(r) for r in rows]
        y_true = [0 if r["known"] else 1 for r in rows]
        threshold = calibrate_threshold(scores, y_true, tau_cov)
    except Exception:
        pass
    
    return results, threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OSR benchmark")
    parser.add_argument("--trace-json", required=True, 
                      help="Trace JSON with known/unknown samples")
    parser.add_argument("--out-json", required=True,
                      help="Output results JSON")
    parser.add_argument("--lambda", type=float, default=10.2,
                      help="Lambda parameter for Energy-σ method")
    parser.add_argument("--temp", type=float, default=1.0,
                      help="Temperature for energy computation")
    parser.add_argument("--coverage", type=float, default=0.95,
                      help="Target coverage for known samples")
    
    args = parser.parse_args()
    
    # Load trace data
    rows = json.loads(Path(args.trace_json).read_text())
    print(f"Loaded {len(rows)} samples")
    
    # Run benchmark
    table, threshold = run_bench(
        rows, 
        tau_cov=args.coverage,
        lam=getattr(args, 'lambda'),  # 'lambda' is keyword
        T=args.temp
    )
    
    # Save results
    output = {"table": table, "threshold": threshold, "params": vars(args)}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(output, indent=2))
    
    print(f"✅ Results written to {args.out_json}")
    print(f"Threshold (95% coverage): {threshold:.4f}")
    
    # Print quick summary
    print("\nMethod Performance Summary:")
    for i, method in enumerate(table["method"]):
        print(f"{method:30s}: {table['known_acc'][i]:5.1f}% acc, "
              f"{table['unk_reject'][i]:5.1f}% rej, "
              f"AUROC {table['auroc'][i]:.3f}")