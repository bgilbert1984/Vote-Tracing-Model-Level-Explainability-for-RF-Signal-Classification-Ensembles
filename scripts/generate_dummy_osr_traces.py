#!/usr/bin/env python3
"""
Generate dummy OSR traces for testing the OSR pipeline.
This creates synthetic known/unknown traces with proper structure.
"""

import json
import numpy as np
from pathlib import Path

def generate_dummy_osr_traces():
    """Generate realistic dummy OSR traces for testing."""
    np.random.seed(42)
    
    # Parameters
    n_known_train = 100
    n_known_test = 50
    n_unknown_test = 30
    n_classes = 8
    n_models = 4
    
    def make_trace(known=True, correct=None, label_idx=None):
        """Generate a single trace entry."""
        # Generate ensemble logits
        if known:
            # Known samples: reasonable logits with clear winner
            true_class = label_idx if label_idx is not None else np.random.randint(n_classes)
            base_logits = np.random.randn(n_classes) * 0.5
            base_logits[true_class] += 2.0  # boost true class
        else:
            # Unknown samples: more uniform/confused logits
            base_logits = np.random.randn(n_classes) * 0.8
        
        ensemble_logits = base_logits.tolist()
        
        # Ensemble probabilities
        ensemble_probs = np.exp(base_logits)
        ensemble_probs = (ensemble_probs / ensemble_probs.sum()).tolist()
        pred_idx = int(np.argmax(ensemble_probs))
        
        # Per-model probabilities (with some variation)
        per_model_probs = {}
        for m in range(n_models):
            # Add noise to ensemble logits for each model
            model_logits = base_logits + np.random.randn(n_classes) * 0.3
            model_probs = np.exp(model_logits)
            model_probs = model_probs / model_probs.sum()
            per_model_probs[f"model_{m}"] = model_probs.tolist()
        
        # Determine correctness for known samples
        if known and correct is None:
            correct = (pred_idx == (label_idx if label_idx is not None else pred_idx))
        
        trace = {
            "known": known,
            "ensemble_logits": ensemble_logits,
            "logits": ensemble_logits,  # alias for compatibility
            "ensemble_prob": ensemble_probs,
            "per_model_probs": per_model_probs,
            "pred_idx": pred_idx,
            "correct": correct if known else None
        }
        
        if known and label_idx is not None:
            trace["label_idx"] = label_idx
            
        return trace
    
    # Generate training traces (known only, with labels)
    train_traces = []
    for i in range(n_known_train):
        label_idx = i % n_classes  # cycle through classes
        train_traces.append(make_trace(known=True, label_idx=label_idx))
    
    # Generate test traces (mixed known/unknown)
    test_traces = []
    
    # Known test samples
    for i in range(n_known_test):
        label_idx = i % n_classes
        correct = np.random.random() > 0.15  # 85% accuracy
        test_traces.append(make_trace(known=True, correct=correct, label_idx=label_idx))
    
    # Unknown test samples
    for i in range(n_unknown_test):
        test_traces.append(make_trace(known=False))
    
    # Shuffle test traces
    np.random.shuffle(test_traces)
    
    return train_traces, test_traces

def main():
    """Generate and save dummy OSR traces."""
    print("🧪 Generating dummy OSR traces...")
    
    train_traces, test_traces = generate_dummy_osr_traces()
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Save training traces
    with open("data/osr_traces_train.json", "w") as f:
        json.dump(train_traces, f, indent=2)
    
    # Save test traces  
    with open("data/osr_traces.json", "w") as f:
        json.dump(test_traces, f, indent=2)
    
    print(f"✅ Generated {len(train_traces)} training traces → data/osr_traces_train.json")
    print(f"✅ Generated {len(test_traces)} test traces → data/osr_traces.json")
    print(f"   Known test: {sum(1 for t in test_traces if t['known'])}")
    print(f"   Unknown test: {sum(1 for t in test_traces if not t['known'])}")

if __name__ == "__main__":
    main()