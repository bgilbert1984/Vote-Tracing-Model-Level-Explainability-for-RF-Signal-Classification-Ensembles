#!/usr/bin/env python3
"""
Vote Trace Evaluation Runner

Runs dataset evaluation with vote trace recording and Shapley attribution analysis.
"""
import os
import json
import importlib
import argparse
from pathlib import Path
import sys

# Add the code directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

from explainability_utils import shapley_vote_contrib


def load_dataset(fn_spec):
    """Load dataset function from module:function specification."""
    mod, fn = fn_spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def load_classifier(cls_spec):
    """Load classifier class from module:class specification.""" 
    mod, cls = cls_spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def create_dummy_signal(signal_id, label_idx=None):
    """Create a dummy RF signal for testing."""
    import numpy as np
    
    # Create mock signal structure
    class DummySignal:
        def __init__(self, sid, label=None):
            self.id = sid
            self.timestamp = 1700000000.0 + int(sid.split('_')[1]) if '_' in sid else 1700000000.0
            self.iq_data = np.random.normal(0, 1, (2, 1024)) + 1j * np.random.normal(0, 1, (2, 1024))
            self.metadata = {"label_idx": label} if label is not None else {}
    
    return DummySignal(signal_id, label_idx)


def create_dummy_classifier():
    """Create a dummy classifier for testing."""
    class DummyModel:
        def __init__(self, name, bias=0.0):
            self.name = name
            self.bias = bias
        
        def predict_logits(self, iq_data):
            import numpy as np
            # Generate dummy logits with some bias
            base_logits = np.random.normal(0, 1, 5)  # 5 classes
            base_logits[0] += self.bias  # bias toward class 0
            return base_logits.tolist()
    
    class DummyClassifier:
        def __init__(self):
            self.models = [
                DummyModel("SpectralCNN", bias=0.5),
                DummyModel("SignalLSTM", bias=0.2), 
                DummyModel("TemporalCNN", bias=-0.2),
                DummyModel("ResNetRF", bias=0.1),
                DummyModel("SignalTransformer", bias=0.0),
            ]
            self.model_weights = [0.25, 0.20, 0.20, 0.20, 0.15]
            self.temperatures = [1.0, 1.2, 0.8, 1.0, 1.1]
        
        def classify_signal(self, signal, override_temperature=None):
            """Simulate ensemble classification with vote tracing."""
            from time import perf_counter
            import math
            
            def _softmax(x): 
                m = max(x)
                ex = [math.exp(v - m) for v in x]
                s = sum(ex)
                return [v / s for v in ex]
            
            # Create audit trace
            audit = {
                "signal_id": getattr(signal, "id", None),
                "ts": getattr(signal, "timestamp", None),
                "stages": [],
                "models": [],
                "weights": list(self.model_weights),
                "temperatures": list(self.temperatures),
            }
            
            t0 = perf_counter()
            per_model_logits = []
            per_model_probs = []
            
            # Process each model
            for i, m in enumerate(self.models):
                t_i0 = perf_counter()
                logits_i = m.predict_logits(signal.iq_data)
                t_i1 = perf_counter()
                T = audit["temperatures"][i] if override_temperature is None else override_temperature
                prob_i = _softmax([z / T for z in logits_i])
                per_model_logits.append(list(logits_i))
                per_model_probs.append(list(prob_i))
                audit["models"].append({
                    "name": getattr(m, "name", f"model_{i}"),
                    "lat_ms": (t_i1 - t_i0) * 1000.0,
                    "temp": T,
                    "logits": list(logits_i),
                    "probs": list(prob_i),
                    "top1": int(max(range(len(prob_i)), key=lambda j: prob_i[j])),
                    "pmax": float(max(prob_i)),
                    "entropy": float(-sum(p * math.log(max(p, 1e-9)) for p in prob_i)),
                })
            
            # Weighted aggregation
            num_classes = len(per_model_probs[0])
            w = audit["weights"]
            agg = [0.0] * num_classes
            for i in range(len(self.models)):
                for c in range(num_classes):
                    agg[c] += w[i] * per_model_probs[i][c]
            s = sum(agg)
            agg = [v / s for v in agg]
            
            t1 = perf_counter()
            audit["aggregate"] = {
                "lat_ms": (t1 - t0) * 1000.0,
                "probs": list(agg),
                "top1": int(max(range(num_classes), key=lambda j: agg[j])),
                "pmax": float(max(agg)),
                "margin": float(sorted(agg, reverse=True)[0] - sorted(agg, reverse=True)[1]),
                "entropy": float(-sum(p * math.log(max(p, 1e-9)) for p in agg)),
            }
            
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
            
            # Return classification result
            return audit["aggregate"]["top1"], audit["aggregate"]["pmax"], agg
    
    return DummyClassifier()


def create_dummy_dataset_iterator():
    """Create a dummy dataset iterator for testing."""
    def dummy_dataset():
        for i in range(100):
            yield create_dummy_signal(f"signal_{i:03d}", label_idx=i % 5)
    return dummy_dataset


def main():
    """Main evaluation function."""
    ap = argparse.ArgumentParser(description="Run vote trace evaluation")
    ap.add_argument("--dataset", default=os.getenv("DATASET_FUNC", "dummy"), 
                    help="Dataset function spec (module:function) or 'dummy' for test data")
    ap.add_argument("--classifier", default=os.getenv("CLASSIFIER_SPEC", "dummy"),
                    help="Classifier spec (module:class) or 'dummy' for test classifier")
    ap.add_argument("--outdir", default="paper_Explainability_from_Vote_Traces/data",
                    help="Output directory for data files")
    ap.add_argument("--max", type=int, default=500,
                    help="Maximum number of signals to process")
    ap.add_argument("--shapley", type=int, default=256,
                    help="Number of Shapley samples")
    args = ap.parse_args()
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_jsonl = outdir / "vote_traces.jsonl"
    
    # Load dataset and classifier
    if args.dataset == "dummy":
        dataset_iter = create_dummy_dataset_iterator()()
        print("Using dummy dataset for testing")
    else:
        dataset_iter = load_dataset(args.dataset)()
        print(f"Loaded dataset: {args.dataset}")
    
    if args.classifier == "dummy":
        clf = create_dummy_classifier()
        print("Using dummy classifier for testing")
    else:
        clf = load_classifier(args.classifier)()
        print(f"Loaded classifier: {args.classifier}")
    
    # Process signals
    processed = 0
    with open(out_jsonl, "w") as f:
        for k, signal in enumerate(dataset_iter):
            if k >= args.max:
                break
                
            try:
                # Classify signal (fills ensemble_trace metadata)
                _ = clf.classify_signal(signal)
                
                # Get the trace event
                if "ensemble_trace" not in signal.metadata or not signal.metadata["ensemble_trace"]:
                    print(f"Warning: No trace data for signal {k}")
                    continue
                    
                evt = signal.metadata["ensemble_trace"][-1]
                top1 = evt["aggregate"]["top1"]
                
                # Compute Shapley contributions
                shap = shapley_vote_contrib(
                    evt["per_model_probs"], 
                    evt["weights"], 
                    top1, 
                    nsamples=args.shapley
                )
                
                # Create output record
                evt_out = {
                    "id": getattr(signal, "id", f"s{k}"),
                    "true": signal.metadata.get("label_idx", None),
                    "pred": top1,
                    "pmax": evt["aggregate"]["pmax"],
                    "weights": evt["weights"],
                    "temperatures": evt["temperatures"],
                    "per_model_pmax": [max(p) for p in evt["per_model_probs"]],
                    "shapley_top1": shap,
                    "trace": evt,  # full snapshot
                }
                
                f.write(json.dumps(evt_out) + "\n")
                processed += 1
                
                if (processed + 1) % 100 == 0:
                    print(f"Processed {processed + 1} signals...")
                    
            except Exception as e:
                print(f"Error processing signal {k}: {e}")
                continue
    
    print(f"✅ Wrote {processed} vote traces to {out_jsonl}")


if __name__ == "__main__":
    main()