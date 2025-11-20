#!/usr/bin/env python3
"""
Audit Hook Patch for Ensemble ML Classifier

This patch adds vote trace recording to the classify_signal() method.
Apply this patch to your existing ensemble_ml_classifier.py to enable
explainability through vote traces.

Usage:
    1. Copy this patch file to your project
    2. Import and apply the audit hooks in your classify_signal() method
    3. Use the generated traces for Shapley attribution analysis
"""

from time import perf_counter
import math
import json
from typing import List, Dict, Any


def _softmax(x: List[float]) -> List[float]:
    """Temperature-scaled softmax function."""
    if not x:
        return []
    m = max(x)
    ex = [math.exp(v - m) for v in x]
    s = sum(ex)
    return [v / s for v in ex] if s > 0 else [0.0] * len(ex)


def add_vote_trace_hooks(classify_signal_method):
    """
    Decorator to add vote tracing to classify_signal() method.
    
    Usage:
        @add_vote_trace_hooks
        def classify_signal(self, signal, override_temperature=None):
            # your existing classification logic
            return prediction, confidence, probabilities
    """
    def wrapper(self, signal, override_temperature=None):
        # Initialize audit structure
        audit = {
            "signal_id": getattr(signal, "id", None),
            "ts": getattr(signal, "timestamp", None),
            "stages": [],
            "models": [],
            "weights": list(getattr(self, "model_weights", [1.0] * len(getattr(self, "models", [])))),
            "temperatures": list(getattr(self, "temperatures", [1.0] * len(getattr(self, "models", [])))),
        }
        
        t0 = perf_counter()
        
        # Call original classification method
        result = classify_signal_method(self, signal, override_temperature)
        
        t1 = perf_counter()
        
        # Add timing and result info
        audit["aggregate"] = {
            "lat_ms": (t1 - t0) * 1000.0,
            "result": str(result) if result else "None"
        }
        
        # Store trace event
        evt = {
            "event": "classify",
            **audit,
        }
        trace = signal.metadata.get("ensemble_trace", [])
        trace.append(evt)
        signal.metadata["ensemble_trace"] = trace
        
        return result
    
    return wrapper


def manual_audit_hooks_example():
    """
    Example of manually adding audit hooks to classify_signal().
    
    Insert this code at the beginning of your classify_signal() method:
    """
    example_code = '''
def classify_signal(self, signal, override_temperature=None):
    # --- BEGIN AUDIT HOOKS ---
    from time import perf_counter
    import math
    
    def _softmax(x): 
        m = max(x) if x else 0
        ex = [math.exp(v-m) for v in x]
        s = sum(ex)
        return [v/s for v in ex] if s > 0 else [0.0] * len(x)

    audit = {
        "signal_id": getattr(signal, "id", None),
        "ts": getattr(signal, "timestamp", None),
        "stages": [],
        "models": [],
        "weights": list(getattr(self, "model_weights", [])),
        "temperatures": list(getattr(self, "temperatures", [1.0]*len(self.models))),
    }

    t0 = perf_counter()
    per_model_logits = []
    per_model_probs  = []
    
    # MODIFY YOUR EXISTING MODEL LOOP TO ADD:
    for i, model in enumerate(self.models):
        t_i0 = perf_counter()
        logits_i = model.predict_logits(signal.iq_data)  # your existing call
        t_i1 = perf_counter()
        T = audit["temperatures"][i] if override_temperature is None else override_temperature
        prob_i = _softmax([z / T for z in logits_i])
        per_model_logits.append(list(logits_i))
        per_model_probs.append(list(prob_i))
        
        audit["models"].append({
            "name": getattr(model, "name", f"model_{i}"),
            "lat_ms": (t_i1 - t_i0) * 1000.0,
            "temp": T,
            "logits": list(logits_i),
            "probs": list(prob_i),
            "top1": int(max(range(len(prob_i)), key=lambda j: prob_i[j])),
            "pmax": float(max(prob_i)),
            "entropy": float(-sum(p*math.log(max(p,1e-9)) for p in prob_i)),
        })

    # YOUR EXISTING WEIGHTED AGGREGATION CODE HERE
    # Example:
    num_classes = len(per_model_probs[0])
    w = audit["weights"]
    agg = [0.0]*num_classes
    for i in range(len(self.models)):
        for c in range(num_classes):
            agg[c] += w[i]*per_model_probs[i][c]
    s = sum(agg)
    agg = [v/s for v in agg] if s > 0 else [0.0]*num_classes

    t1 = perf_counter()
    audit["aggregate"] = {
        "lat_ms": (t1 - t0) * 1000.0,
        "probs": list(agg),
        "top1": int(max(range(num_classes), key=lambda j: agg[j])),
        "pmax": float(max(agg)),
        "margin": float(sorted(agg, reverse=True)[0] - sorted(agg, reverse=True)[1]),
        "entropy": float(-sum(p*math.log(max(p,1e-9)) for p in agg)),
    }

    # Optional OSR hooks
    if hasattr(self, "osr_gate"):
        try:
            gate_result = self.osr_gate.inspect(agg)
            audit["osr"] = gate_result
        except:
            pass

    # Store in metadata
    evt = {
        "event": "classify",
        "per_model_logits": per_model_logits,
        "per_model_probs": per_model_probs,
        **audit,
    }
    trace = signal.metadata.get("ensemble_trace", [])
    trace.append(evt)
    signal.metadata["ensemble_trace"] = trace
    # --- END AUDIT HOOKS ---
    
    # YOUR EXISTING RETURN STATEMENT
    return final_prediction, final_confidence, final_probabilities
'''
    
    return example_code


def patch_ensemble_classifier_file(filepath: str, backup: bool = True):
    """
    Automatically patch an existing ensemble_ml_classifier.py file.
    
    Args:
        filepath: Path to the ensemble_ml_classifier.py file
        backup: Whether to create a .backup file
    """
    import os
    import shutil
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Create backup
    if backup:
        shutil.copy2(filepath, filepath + ".backup")
        print(f"Created backup: {filepath}.backup")
    
    # Read original file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Insert audit hooks import at the top
    import_patch = """
# Vote tracing audit hooks
from time import perf_counter
import math
"""
    
    # Find the first import and insert after it
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            lines.insert(i + 1, import_patch)
            break
    
    # Write patched file
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Patched {filepath} - add manual audit hooks to classify_signal() method")
    print("See manual_audit_hooks_example() for the code to insert")


if __name__ == "__main__":
    print("Ensemble ML Classifier Audit Hook Patch")
    print("=" * 50)
    print()
    print("Usage Options:")
    print("1. Decorator approach:")
    print("   @add_vote_trace_hooks")
    print("   def classify_signal(self, ...):")
    print()
    print("2. Manual insertion:")
    print("   Copy code from manual_audit_hooks_example()")
    print()
    print("3. File patching:")
    print("   patch_ensemble_classifier_file('path/to/file.py')")
    print()
    print("Manual insertion code:")
    print("-" * 30)
    print(manual_audit_hooks_example())