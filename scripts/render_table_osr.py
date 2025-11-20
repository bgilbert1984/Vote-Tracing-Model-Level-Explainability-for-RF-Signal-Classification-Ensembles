#!/usr/bin/env python3
"""
OSR Table Renderer for Vote-Tracing Paper
Generates LaTeX table from OSR benchmark results.
"""

import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from sklearn.metrics import roc_auc_score

def compute_osr_metrics(scores: List[float], labels: List[bool], coverage: float = 0.95) -> Dict[str, float]:
    """
    Compute OSR metrics from scores and labels.
    
    Args:
        scores: OSR scores (higher = more confident it's known)
        labels: True for known samples, False for unknown
        coverage: Target coverage for known samples
        
    Returns:
        Dictionary with metrics
    """
    scores = np.array(scores)
    labels = np.array(labels)
    
    # AUROC (1 = known, 0 = unknown)
    auroc = roc_auc_score(labels.astype(int), scores)
    
    # Find threshold for target coverage on known samples
    known_scores = scores[labels]
    if len(known_scores) > 0:
        threshold = np.percentile(known_scores, (1 - coverage) * 100)
        
        # Known accuracy at threshold
        known_accepted = np.sum(known_scores >= threshold)
        known_acc = known_accepted / len(known_scores) * 100
        
        # Unknown rejection rate at threshold
        unknown_scores = scores[~labels]
        if len(unknown_scores) > 0:
            unknown_rejected = np.sum(unknown_scores < threshold)
            unk_reject = unknown_rejected / len(unknown_scores) * 100
        else:
            unk_reject = 0.0
    else:
        known_acc = 0.0
        unk_reject = 0.0
    
    return {
        "known_acc": known_acc,
        "unk_reject": unk_reject, 
        "auroc": auroc
    }

def render_osr_table(results_json: str, output_path: str = "tables/osr_table.tex"):
    """
    Render OSR benchmark results as LaTeX table.
    
    Args:
        results_json: Path to benchmark results JSON
        output_path: Output LaTeX file path
    """
    # Load results
    data = json.loads(Path(results_json).read_text())
    
    # Extract raw scores and labels
    labels = data["known"]  # True for known, False for unknown
    coverage = data.get("metadata", {}).get("coverage", 0.95)
    
    # Methods to include in table
    methods = [
        ("Energy", "energy"),
        ("Energy - λ·σ (ours)", "energy_minus_disagreement"),
        ("MaxProb+Entropy", "max_prob"),  # Using max_prob as proxy
        ("ODIN", "odin"),
        ("MOS", "mos"),
        ("Mahalanobis", "mahalanobis")
    ]
    
    # Table header
    content = [
        r"\begin{table}[t]",
        r"\centering", 
        r"\caption{Open-Set performance at $\approx$95\% known-class coverage (Dummy Test Data; 4-model ensemble)}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Known Acc. & Unknown Reject & AUROC & Extra Forwards & Memory & Train Fit \\",
        r"\midrule"
    ]
    
    # Compute metrics for each method
    for method_name, method_key in methods:
        if method_key in data:
            scores = data[method_key]
            metrics = compute_osr_metrics(scores, labels, coverage)
            
            # Method-specific properties
            props = {
                "Energy": {"extra": "0", "memory": "Low", "fit": "No"},
                "Energy - λ·σ (ours)": {"extra": "0", "memory": "Low", "fit": "No"},
                "MaxProb+Entropy": {"extra": "0", "memory": "Low", "fit": "No"},
                "ODIN": {"extra": "1", "memory": "Low", "fit": "No"},
                "MOS": {"extra": "0", "memory": "Low", "fit": "No"},
                "Mahalanobis": {"extra": "0", "memory": "Med", "fit": "Yes"}
            }.get(method_name, {"extra": "?", "memory": "?", "fit": "?"})
            
            # Format row
            method_tex = method_name.replace("_", r"\_")
            row = (f"{method_tex} & "
                   f"{metrics['known_acc']:.1f}\\% & "
                   f"{metrics['unk_reject']:.1f}\\% & "
                   f"{metrics['auroc']:.3f} & "
                   f"{props['extra']} & "
                   f"{props['memory']} & "
                   f"{props['fit']} \\\\")
            
            content.append(row)
    
    # Table footer
    content.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:osr-comparison}",
        r"\end{table}"
    ])
    
    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(content))
    
    print(f"🧾 OSR table written to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_table_osr.py <results.json> [output.tex]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "tables/osr_table.tex"
    
    render_osr_table(results_file, output_file)