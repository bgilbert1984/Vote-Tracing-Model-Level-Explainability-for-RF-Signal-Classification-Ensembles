#!/usr/bin/env python3
import json, argparse, numpy as np
from collections import defaultdict

def load_vote_trace_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    # Handle different JSON structures  
    if isinstance(data, dict) and "signals" in data:
        rows = data["signals"]
    elif isinstance(data, list):
        rows = data
    else:
        rows = [data]
    return rows

def aggregate_mean_phi(rows):
    acc = defaultdict(list)
    for r in rows:
        contrib = r.get("shapley_contribution", {})
        for k, v in contrib.items():
            acc[k].append(v)
    return {k: float(np.mean(vs)) for k, vs in acc.items()}

def simulate_prune(rows, victim):
    # victim is model_name to zero out
    ok = 0; total = 0
    for r in rows:
        pmp = r.get("per_model_probs", {})
        if victim not in pmp:
            continue
        # recompute ensemble vote w/o victim (mean over remaining)
        probs = []
        for k, vec in pmp.items():
            if k == victim: continue
            probs.append(vec)
        if not probs: 
            continue
        p = np.mean(np.array(probs), axis=0)
        pred = int(np.argmax(p))
        true = r.get("true_idx", r.get("_true_idx", pred))
        if true is not None:
            true = int(true)
            ok += int(pred == true)
        total += 1
    return ok/total if total>0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="JSON with per-sample vote traces + shapley_contribution")
    ap.add_argument("--baseline_acc", type=float, required=True)
    ap.add_argument("--out", default="data/xai_pruning_summary.json")
    args = ap.parse_args()

    rows = load_vote_trace_json(args.traces)
    mean_phi = aggregate_mean_phi(rows)
    
    if not mean_phi:
        print("No Shapley contributions found in traces")
        return
        
    ranked = sorted(mean_phi.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) < 2:
        print("Need at least 2 models for pruning analysis")
        return
        
    top, bottom = ranked[0][0], ranked[-1][0]

    acc_top_removed = simulate_prune(rows, top)
    acc_bottom_removed = simulate_prune(rows, bottom)

    out = {
        "baseline_acc": args.baseline_acc,
        "remove_top1": {"model": top, "acc": acc_top_removed, "delta": acc_top_removed - args.baseline_acc},
        "remove_bottom1": {"model": bottom, "acc": acc_bottom_removed, "delta": acc_bottom_removed - args.baseline_acc},
        "mean_contributions": dict(ranked)
    }
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as f: 
        json.dump(out, f, indent=2)
    print(f"Saved pruning analysis to {args.out}")

if __name__ == "__main__":
    main()