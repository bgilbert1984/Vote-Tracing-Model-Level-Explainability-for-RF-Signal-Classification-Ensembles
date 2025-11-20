#!/usr/bin/env python3
"""
Vote Trace Figure Generation

Creates visualizations from vote trace data including timelines and Shapley contributions.
"""
import json
import argparse
from pathlib import Path
import sys

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, figures will not be generated")


def load_jsonl(path):
    """Load JSONL data from file."""
    for line in open(path, "r"):
        if line.strip():
            yield json.loads(line)


def fig_vote_timeline(sample, save_path):
    """Plot per-model p(target) vs model index; also ensemble pmax."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping figure generation (matplotlib not available)")
        return
    
    evt = sample["trace"]
    tgt = sample["pred"]
    per = [p[tgt] for p in evt["per_model_probs"]]
    ens = evt["aggregate"]["probs"][tgt]
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(per)), per, marker="o", label="per-model p(target)", linewidth=2)
    plt.axhline(ens, linestyle="--", color="red", label="ensemble p(target)", linewidth=2)
    plt.xlabel("Model Index")
    plt.ylabel("Probability")
    plt.title(f"Vote Trace • id={sample['id']} • pred={tgt}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def fig_shapley_bar(sample, save_path):
    """Plot Shapley contributions as bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping figure generation (matplotlib not available)")
        return
    
    contrib = sample["shapley_top1"]
    model_names = [f"M{i}" for i in range(len(contrib))]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(contrib)), contrib, alpha=0.7)
    
    # Color bars by contribution (positive = green, negative = red)
    for i, (bar, c) in enumerate(zip(bars, contrib)):
        if c > 0:
            bar.set_color('green')
        elif c < 0:
            bar.set_color('red')
        else:
            bar.set_color('gray')
    
    plt.xlabel("Model Index")
    plt.ylabel("Δ prob(target)")
    plt.title(f"Shapley-like Contribution • id={sample['id']}")
    plt.xticks(range(len(contrib)), model_names)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def fig_agreement_matrix(samples, save_path):
    """Plot model agreement heatmap across samples."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping figure generation (matplotlib not available)")
        return
    
    if not samples:
        return
    
    # Get number of models from first sample
    num_models = len(samples[0]["shapley_top1"])
    agreement_matrix = np.zeros((num_models, num_models))
    
    # Calculate pairwise agreement
    for sample in samples:
        evt = sample["trace"]
        predictions = []
        for model_probs in evt["per_model_probs"]:
            pred_class = max(range(len(model_probs)), key=lambda i: model_probs[i])
            predictions.append(pred_class)
        
        for i in range(num_models):
            for j in range(num_models):
                if predictions[i] == predictions[j]:
                    agreement_matrix[i, j] += 1
    
    # Normalize by number of samples
    agreement_matrix /= len(samples)
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(agreement_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, label='Agreement Rate')
    plt.xlabel('Model Index')
    plt.ylabel('Model Index')
    plt.title(f'Model Agreement Matrix (N={len(samples)} samples)')
    
    # Add text annotations
    for i in range(num_models):
        for j in range(num_models):
            plt.text(j, i, f'{agreement_matrix[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if agreement_matrix[i, j] < 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def fig_contribution_distribution(samples, save_path):
    """Plot distribution of Shapley contributions across dataset."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping figure generation (matplotlib not available)")
        return
        
    if not samples:
        return
    
    num_models = len(samples[0]["shapley_top1"])
    contributions = [[] for _ in range(num_models)]
    
    # Collect contributions for each model
    for sample in samples:
        for i, contrib in enumerate(sample["shapley_top1"]):
            contributions[i].append(contrib)
    
    plt.figure(figsize=(10, 6))
    
    # Box plot of contributions
    plt.subplot(1, 2, 1)
    box_data = [contributions[i] for i in range(num_models)]
    plt.boxplot(box_data, labels=[f"M{i}" for i in range(num_models)])
    plt.ylabel("Shapley Contribution")
    plt.title("Distribution by Model")
    plt.grid(True, alpha=0.3)
    
    # Mean contributions
    plt.subplot(1, 2, 2)
    means = [np.mean(contributions[i]) for i in range(num_models)]
    stds = [np.std(contributions[i]) for i in range(num_models)]
    
    bars = plt.bar(range(num_models), means, yerr=stds, capsize=5, alpha=0.7)
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if mean > 0:
            bar.set_color('green')
        elif mean < 0:
            bar.set_color('red')
        else:
            bar.set_color('gray')
    
    plt.xlabel("Model Index")
    plt.ylabel("Mean Δ prob(target)")
    plt.title("Mean Contribution ± Std")
    plt.xticks(range(num_models), [f"M{i}" for i in range(num_models)])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """Main figure generation function."""
    ap = argparse.ArgumentParser(description="Generate vote trace figures")
    ap.add_argument("--data", default="paper_Explainability_from_Vote_Traces/data/vote_traces.jsonl",
                    help="Input JSONL file with vote traces")
    ap.add_argument("--outdir", default="paper_Explainability_from_Vote_Traces/figs",
                    help="Output directory for figures")
    ap.add_argument("--examples", type=int, default=6,
                    help="Number of example timeline/Shapley plots to generate")
    args = ap.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available - install with: pip install matplotlib")
        return
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load all samples
    samples = list(load_jsonl(args.data))
    if not samples:
        print(f"No samples found in {args.data}")
        return
    
    print(f"Loaded {len(samples)} samples from {args.data}")
    
    # Generate example plots for first N samples
    for i, sample in enumerate(samples[:args.examples]):
        fig_vote_timeline(sample, outdir / f"vote_timeline_{i}.pdf")
        fig_shapley_bar(sample, outdir / f"vote_shapley_{i}.pdf")
    
    # Generate aggregate analysis figures
    if len(samples) > 1:
        # Overall mean Shapley contributions
        if samples:
            num_models = max(len(s["shapley_top1"]) for s in samples)
            mean_contrib = np.zeros(num_models)
            count = 0
            
            for s in samples:
                if len(s["shapley_top1"]) == num_models:
                    mean_contrib += np.array(s["shapley_top1"])
                    count += 1
            
            if count > 0:
                mean_contrib /= count
                
                plt.figure(figsize=(8, 5))
                bars = plt.bar(range(num_models), mean_contrib, alpha=0.7)
                
                # Color by contribution sign
                for i, (bar, contrib) in enumerate(zip(bars, mean_contrib)):
                    if contrib > 0:
                        bar.set_color('green')
                    elif contrib < 0:
                        bar.set_color('red')
                    else:
                        bar.set_color('gray')
                
                plt.xlabel("Model Index")
                plt.ylabel("Mean Δ prob(target)")
                plt.title(f"Mean Shapley-like Contribution (N={count} samples)")
                plt.xticks(range(num_models), [f"M{i}" for i in range(num_models)])
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(outdir / "vote_shapley_mean.pdf")
                plt.close()
        
        # Agreement matrix
        fig_agreement_matrix(samples, outdir / "model_agreement_matrix.pdf")
        
        # Contribution distributions
        fig_contribution_distribution(samples, outdir / "contribution_distribution.pdf")
    
    print(f"✅ Generated figures in {outdir}")


if __name__ == "__main__":
    main()