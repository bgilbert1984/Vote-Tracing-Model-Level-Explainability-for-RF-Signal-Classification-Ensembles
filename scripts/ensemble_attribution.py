#!/usr/bin/env python3
"""
Exact Shapley Value Attribution for RF Ensembles
Vote-Tracing Paper: Explainability from Vote Traces

Provides both exact permutation-based Shapley values (default, O(M!)) 
and fast symmetric marginal approximation for larger ensembles.
"""

import math
import itertools
from typing import Dict, List, Sequence

def _ensure_names(n: int, names: Sequence[str] | None) -> List[str]:
    """Ensure we have model names."""
    return list(names) if names else [f"m{i}" for i in range(n)]

def shapley_exact_from_probs(
    per_model_probs: List[Sequence[float]],
    target_idx: int,
    model_names: Sequence[str] | None = None,
) -> Dict[str, float]:
    """
    Exact Shapley for mean-ensemble f(S) = mean_{j∈S} p_j(target), with f(∅)=0.
    Cost: O(M! * M). OK for M<=9 (362,880 perms); borderline but doable at M=10.
    No network forwards here: requires *probabilities per model* (already logged).
    
    Args:
        per_model_probs: List of probability vectors, one per model
        target_idx: Index of target class
        model_names: Optional model names for output dict
        
    Returns:
        Dict mapping model names to Shapley values
    """
    p = [float(v[target_idx]) for v in per_model_probs]  # [M]
    M = len(p)
    names = _ensure_names(M, model_names)
    fact = math.factorial(M)
    contrib = [0.0] * M

    # For each permutation π, the marginal of item i at position k:
    # Δ = (sum_pre + p_i)/(k+1) - (sum_pre)/k  for k>=1; for k=0: Δ = p_i
    # Accumulate per-model contributions then average over permutations.
    for perm in itertools.permutations(range(M)):
        sum_pre = 0.0
        k = 0
        for pos, i in enumerate(perm):
            if k == 0:
                delta = p[i]
            else:
                delta = (sum_pre + p[i])/(k+1) - (sum_pre)/k
            contrib[i] += delta
            sum_pre += p[i]
            k += 1

    # Average over permutations
    contrib = [c / fact for c in contrib]

    # Optional normalization so ∑φ_i = p_ens (nice for plots)
    p_ens = sum(p)/M if M else 0.0
    s = sum(contrib)
    if abs(s) > 1e-12 and p_ens > 0:
        scale = p_ens / s
        contrib = [c*scale for c in contrib]

    return {names[i]: float(contrib[i]) for i in range(M)}


def shapley_fast_marginal(
    per_model_probs: List[Sequence[float]],
    target_idx: int,
    model_names: Sequence[str] | None = None,
) -> Dict[str, float]:
    """
    Very fast symmetric marginal (leave-one-out / add-last) approximation.
    Deterministic; ~1000x faster than MC; great for M>10.
    
    Args:
        per_model_probs: List of probability vectors, one per model
        target_idx: Index of target class  
        model_names: Optional model names for output dict
        
    Returns:
        Dict mapping model names to approximate Shapley values
    """
    p = [float(v[target_idx]) for v in per_model_probs]
    M = len(p)
    names = _ensure_names(M, model_names)
    p_ens = sum(p)/M if M else 0.0
    out = {}
    for i in range(M):
        p_wo_i = (p_ens*M - p[i])/(M-1) if M>1 else 0.0
        gain_add_last = p_ens - p_wo_i
        loss_remove = p_ens - p_wo_i
        phi = 0.5*(gain_add_last + loss_remove)
        out[names[i]] = float(phi)
    # Normalize to sum to p_ens (nice)
    s = sum(out.values())
    if abs(s) > 1e-12 and p_ens > 0:
        scale = p_ens / s
        out = {k: v*scale for k,v in out.items()}
    return out


def shapley_values(
    per_model_probs: List[Sequence[float]],
    target_idx: int,
    model_names: Sequence[str] | None = None,
    mode: str = "exact",  # default ON
    exact_max_m: int = 10
) -> Dict[str, float]:
    """
    Compute Shapley values with automatic method selection.
    
    Args:
        per_model_probs: List of probability vectors, one per model
        target_idx: Index of target class
        model_names: Optional model names for output dict  
        mode: "exact" for exact permutation method, "fast" for marginal approximation
        exact_max_m: Maximum models for exact computation
        
    Returns:
        Dict mapping model names to Shapley values
    """
    M = len(per_model_probs)
    if mode == "exact" and M <= exact_max_m:
        return shapley_exact_from_probs(per_model_probs, target_idx, model_names)
    return shapley_fast_marginal(per_model_probs, target_idx, model_names)


if __name__ == "__main__":
    # Quick test
    import numpy as np
    
    # Test with 4 models
    np.random.seed(42)
    C = 8  # classes
    M = 4  # models
    
    # Generate random probability vectors
    per_model_probs = []
    for _ in range(M):
        logits = np.random.randn(C)
        probs = np.exp(logits) / np.exp(logits).sum()
        per_model_probs.append(probs.tolist())
    
    target_idx = 3
    model_names = [f"ResNet{i}" for i in range(M)]
    
    # Compare exact vs fast
    exact = shapley_values(per_model_probs, target_idx, model_names, mode="exact")
    fast = shapley_values(per_model_probs, target_idx, model_names, mode="fast")
    
    print("Exact Shapley values:")
    for name, val in exact.items():
        print(f"  {name}: {val:.4f}")
    
    print("\nFast approximation:")
    for name, val in fast.items():
        print(f"  {name}: {val:.4f}")
        
    print(f"\nExact sum: {sum(exact.values()):.4f}")
    print(f"Fast sum: {sum(fast.values()):.4f}")
    print(f"Ensemble prob: {sum(p[target_idx] for p in per_model_probs)/M:.4f}")