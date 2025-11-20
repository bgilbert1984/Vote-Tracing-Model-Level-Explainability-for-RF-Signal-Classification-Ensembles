# -*- coding: utf-8 -*-
# File: code/ensemble_attribution.py

from typing import List, Dict, Any, Optional
import itertools
import os
import numpy as np
import torch
import torch.nn.functional as F

def _stack_probs(models: List[torch.nn.Module], iq_tensor: torch.Tensor, temperature: float) -> torch.Tensor:
    with torch.no_grad():
        probs = []
        for m in models:
            logits = m(iq_tensor)
            if temperature != 1.0:
                logits = logits / temperature
            probs.append(F.softmax(logits, dim=-1))
        return torch.cat(probs, dim=0)  # [M, C]

def shapley_exact_or_fast(
    models: List[torch.nn.Module],
    iq_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    temperature: float = 1.0,
    exact_max_members: int = 8,        # exact <= 8 by default
    mc_permutations: int = 4096,       # fallback budget for big M
) -> Dict[str, float]:
    """
    Returns per-model Shapley-like contributions φ_i for the final predicted class probability
    under a simple mean-ensemble combiner. For M<=exact_max_members, compute **exact** Shapley
    via permutation averaging. Otherwise, do a permutation-MC approximation.

    Stores no gradients; inference-only.
    """
    device = iq_tensor.device
    M = len(models)
    if M == 0:
        return {}

    # 1) Per-model probabilities [M, C]
    model_probs = _stack_probs(models, iq_tensor, temperature)  # no grad
    # Final ensemble probs [1, C]
    ensemble_prob = model_probs.mean(dim=0, keepdim=True)

    # Decide target class
    if target_class is None:
        target_class = int(torch.argmax(ensemble_prob, dim=-1).item())

    # Convenience views
    p = model_probs[:, target_class]  # [M]
    p_full = float(ensemble_prob[0, target_class].item())

    names = [getattr(m, "name", f"{m.__class__.__name__}_m{i}") for i, m in enumerate(models)]
    phi = np.zeros(M, dtype=np.float64)

    def marginal_delta(sum_prev: float, k: int, p_i: float) -> float:
        """
        Δ = f(S ∪ {i}) - f(S) with f(S)=mean probs on target class.
        For k = |S|, sum_prev = sum_{j∈S} p_j:
            if k==0: Δ = p_i
            else:    Δ = (sum_prev + p_i)/(k+1) - (sum_prev/k) = (k*p_i - sum_prev) / (k*(k+1))
        """
        if k == 0:
            return float(p_i)
        return float((k * p_i - sum_prev) / (k * (k + 1)))

    # 2) Exact Shapley for small M
    if M <= exact_max_members or os.getenv("ENABLE_EXACT_SHAPLEY", "0") == "1":
        for perm in itertools.permutations(range(M)):
            sum_prev = 0.0
            k = 0
            for idx in perm:
                d = marginal_delta(sum_prev, k, float(p[idx]))
                phi[idx] += d
                sum_prev += float(p[idx])
                k += 1
        phi /= float(np.math.factorial(M))
    else:
        # 3) MC fallback for large M
        # Sample permutations uniformly; same Δ formula
        g = torch.Generator().manual_seed(1337)
        for _ in range(mc_permutations):
            perm = torch.randperm(M, generator=g).tolist()
            sum_prev = 0.0
            k = 0
            for idx in perm:
                d = marginal_delta(sum_prev, k, float(p[idx]))
                phi[idx] += d
                sum_prev += float(p[idx])
                k += 1
        phi /= float(mc_permutations)

    # Optional normalization (purely cosmetic for plots/tables)
    # Map φ to sum roughly the final prob p_full (helps intuitive reading)
    s = float(np.sum(phi))
    if abs(s) > 1e-12:
        phi = phi / s * p_full

    return {names[i]: float(phi[i]) for i in range(M)}

def exact_ensemble_shapley_from_trace(
    ensemble_trace: List[Dict],
    target_class_idx: int
) -> Dict[str, float]:
    """
    Exact Shapley values from vote trace (zero extra inference).
    Requires trace to contain per-model target-class probabilities.
    """
    # Extract from trace
    model_entries = [e for e in ensemble_trace if "prob" in e]  # or however you log them
    if not model_entries:
        raise ValueError("No per-model probabilities in trace")
    
    per_model_p = [entry["prob"][target_class_idx] for entry in model_entries]
    model_names = [entry["model_name"] for entry in model_entries]
    
    M = len(per_model_p)
    if M == 0:
        return {}
    if M == 1:
        return {model_names[0]: float(per_model_p[0])}

    # Pre-compute factorials
    fact = [1] * (M + 1)
    for i in range(2, M + 1):
        fact[i] = fact[i - 1] * i

    phi = [0.0] * M

    # Subset enumeration
    for mask in range(1, 1 << M):  # skip empty set
        subset_size = 0
        subset_sum = 0.0
        for j in range(M):
            if mask & (1 << j):
                subset_sum += per_model_p[j]
                subset_size += 1

        v_S = subset_sum / subset_size

        weight = fact[subset_size] * fact[M - subset_size - 1] / fact[M]

        for i in range(M):
            if (mask & (1 << i)) == 0:  # i not in S
                v_union = (subset_sum + per_model_p[i]) / (subset_size + 1)
                marginal = v_union - v_S
                phi[i] += marginal * weight

    return {name: phi[i] for i, name in enumerate(model_names)}


# Optional: fast path for M ≤ 12 using itertools.permutations (even cleaner)
def exact_ensemble_shapley_permutations(per_model_p: List[float], model_names: List[str]) -> Dict[str, float]:
    import itertools
    import math
    M = len(per_model_p)
    phi = [0.0] * M
    for perm in itertools.permutations(range(M)):
        agg = 0.0
        for pos, idx in enumerate(perm):
            before = agg / pos if pos > 0 else 0.0
            agg += per_model_p[idx]
            after = agg / (pos + 1)
            phi[idx] += after - before
    phi = [p / math.factorial(M) for p in phi]
    return dict(zip(model_names, phi))


def fast_ensemble_shapley_numpy(model_probs: List[List[float]],
                               model_names: List[str],
                               target_idx: Optional[int] = None) -> Dict[str, float]:
    """
    NumPy fallback version for when models output probabilities directly.
    
    Args:
        model_probs: List of [C] probability vectors from each model
        model_names: List of model names
        target_idx: Class index for attribution (auto-detected if None)
        
    Returns:
        Dict mapping model names to attribution values
    """
    import numpy as np
    
    if not model_probs:
        return {}
    
    M = len(model_probs)
    C = len(model_probs[0])
    
    # Convert to numpy array [M, C]
    P = np.array(model_probs)
    full = P.mean(axis=0)  # [C]
    
    if target_idx is None:
        target_idx = int(np.argmax(full))
    
    p_full = float(full[target_idx])
    contrib = {}
    
    # Leave-one-out marginal contributions
    for i in range(M):
        mask = np.ones(M, dtype=bool)
        mask[i] = False
        p_without_i = float(P[mask].mean(axis=0)[target_idx])
        phi_i = 0.5 * ((p_full - p_without_i) + (p_full - p_without_i))  # symmetric LOO
        contrib[model_names[i]] = phi_i
    
    # Normalize contributions
    s = sum(contrib.values()) or 1.0
    for k in list(contrib.keys()):
        contrib[k] = contrib[k] / s * p_full
    
    return contrib


def time_attribution_overhead(models: List[torch.nn.Module],
                             iq_tensor: torch.Tensor,
                             n_trials: int = 100) -> Dict[str, float]:
    """
    Measure attribution overhead for performance validation.
    
    Returns timing statistics in milliseconds.
    """
    import time
    
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = shapley_exact_or_fast(models, iq_tensor)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # Convert to ms
    
    return {
        "mean_ms": sum(times) / len(times),
        "median_ms": sorted(times)[len(times) // 2],
        "p95_ms": sorted(times)[int(0.95 * len(times))],
        "max_ms": max(times),
        "min_ms": min(times)
    }


# Integration hook for existing ensemble classifiers
def add_fast_attribution_hook(classify_signal_method):
    """
    Decorator to add fast attribution to classify_signal() method.
    
    Usage:
        @add_fast_attribution_hook
        def classify_signal(self, signal, override_temperature=None):
            # your existing classification logic
            return prediction, confidence, probabilities
    """
    def wrapper(self, signal, override_temperature=None):
        # Call original classification method
        result = classify_signal_method(self, signal, override_temperature)
        
        # Add fast attribution if enabled
        if getattr(self, "enable_attribution", True) and hasattr(self, "models"):
            try:
                import time
                t0 = time.perf_counter()
                
                # Get IQ tensor (adapt to your signal format)
                iq_tensor = getattr(signal, "iq_tensor", None)
                if iq_tensor is None and hasattr(signal, "iq_data"):
                    # Convert iq_data to tensor if needed
                    import torch
                    iq_tensor = torch.from_numpy(signal.iq_data).unsqueeze(0).float()
                
                if iq_tensor is not None:
                    # Compute attribution using new exact method
                    shap = shapley_exact_or_fast(
                        self.models, 
                        iq_tensor, 
                        temperature=override_temperature or 1.0
                    )
                    
                    # Store results
                    if not hasattr(signal, "metadata"):
                        signal.metadata = {}
                    signal.metadata["shapley_contribution"] = shap
                    
                    # Timing overhead
                    t1 = time.perf_counter()
                    signal.metadata.setdefault("timing", {})["attribution_ms"] = (t1 - t0) * 1000
                    
            except Exception as e:
                # Graceful fallback - don't break classification
                if hasattr(signal, "metadata"):
                    signal.metadata["attribution_error"] = str(e)
        
        return result
    
    return wrapper