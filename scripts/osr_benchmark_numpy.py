#!/usr/bin/env python3
"""
OSR benchmark without PyTorch - basic version for testing.
Uses NumPy implementations of ODIN and Energy scoring.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

def softmax(logits: np.ndarray, axis=-1) -> np.ndarray:
    """Compute softmax probabilities."""
    exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

def prob_disagreement(per_model_probs: Dict[str, List[float]]) -> float:
    """Compute probability disagreement between ensemble models."""
    probs_array = np.array(list(per_model_probs.values()))  # (M, C)
    mean_prob = np.mean(probs_array, axis=0)  # (C,)
    variances = np.var(probs_array, axis=0)  # (C,)
    return float(np.mean(variances))

def osr_energy_minus_disagreement(logits: List[float], per_model_probs: Dict[str, List[float]], lam: float = 10.0) -> float:
    """Energy - λ·disagreement OSR score."""
    energy = np.log(np.sum(np.exp(logits)))
    disagreement = prob_disagreement(per_model_probs)
    return float(energy - lam * disagreement)

def osr_energy(logits: List[float]) -> float:
    """Basic energy OSR score."""
    return float(np.log(np.sum(np.exp(logits))))

def osr_odin(logits: List[float], temp: float = 1000.0) -> float:
    """ODIN OSR score (temperature scaling)."""
    scaled_logits = np.array(logits) / temp
    probs = softmax(scaled_logits)
    return float(np.max(probs))

def osr_max_prob(probs: List[float]) -> float:
    """Maximum probability OSR score."""
    return float(np.max(probs))

def osr_entropy(probs: List[float]) -> float:
    """Negative entropy OSR score (higher = more confident)."""
    p = np.array(probs)
    p = p + 1e-12  # avoid log(0)
    entropy = -np.sum(p * np.log(p))
    return float(-entropy)  # negative so higher = better

def osr_mahalanobis(logits: List[float], mahal_model: Dict[str, Any]) -> float:
    """Simplified Mahalanobis distance OSR score."""
    # Note: Real implementation would use proper distance calculation
    # For now, just return energy as placeholder
    return osr_energy(logits)

def osr_mos(per_model_probs: Dict[str, List[float]]) -> float:
    """Model Output Statistics (MOS) OSR score."""
    probs_array = np.array(list(per_model_probs.values()))  # (M, C)
    max_probs = np.max(probs_array, axis=1)  # (M,)
    return float(np.mean(max_probs))

def run_osr_benchmark(traces: List[Dict], mahal_model: Dict = None, lam: float = 10.0) -> Dict[str, List[float]]:
    """Run OSR benchmark on all methods."""
    
    results = {
        "energy": [],
        "energy_minus_disagreement": [], 
        "odin": [],
        "max_prob": [],
        "entropy": [],
        "mahalanobis": [],
        "mos": [],
        "known": []  # ground truth labels
    }
    
    for trace in traces:
        logits = trace["ensemble_logits"]
        probs = trace["ensemble_prob"]
        per_model_probs = trace["per_model_probs"]
        known = trace["known"]
        
        # Compute all OSR scores
        results["energy"].append(osr_energy(logits))
        results["energy_minus_disagreement"].append(osr_energy_minus_disagreement(logits, per_model_probs, lam))
        results["odin"].append(osr_odin(logits))
        results["max_prob"].append(osr_max_prob(probs))
        results["entropy"].append(osr_entropy(probs))
        results["mahalanobis"].append(osr_mahalanobis(logits, mahal_model))
        results["mos"].append(osr_mos(per_model_probs))
        results["known"].append(known)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="OSR Benchmark (NumPy version)")
    parser.add_argument("--trace-json", required=True, help="Test traces JSON file")
    parser.add_argument("--out-json", required=True, help="Output results JSON file")
    parser.add_argument("--lambda", type=float, default=10.0, dest="lam", help="Lambda for energy-disagreement")
    parser.add_argument("--coverage", type=float, default=0.95, help="Coverage for scoring")
    args = parser.parse_args()

    # Load test traces
    with open(args.trace_json) as f:
        traces = json.load(f)
    
    # Load Mahalanobis model if available
    mahal_model = None
    mahal_path = Path("data/mahal_model.json")
    if mahal_path.exists():
        with open(mahal_path) as f:
            mahal_model = json.load(f)
    
    # Run benchmark
    print(f"Running OSR benchmark on {len(traces)} traces...")
    print(f"Lambda: {args.lam}, Coverage: {args.coverage}")
    
    results = run_osr_benchmark(traces, mahal_model, args.lam)
    
    # Add metadata
    results["metadata"] = {
        "lambda": args.lam,
        "coverage": args.coverage,
        "num_traces": len(traces),
        "num_known": sum(results["known"]),
        "num_unknown": len(results["known"]) - sum(results["known"])
    }
    
    # Save results
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ OSR benchmark complete → {args.out_json}")
    print(f"   Known: {results['metadata']['num_known']}, Unknown: {results['metadata']['num_unknown']}")

if __name__ == "__main__":
    main()