#!/usr/bin/env python3
"""
Enhanced dummy data generator with SNR stratification for testing.
"""
import json
import random
import math
import numpy as np
from pathlib import Path


def generate_enhanced_dummy_data(n_samples=100, output_path="data/vote_traces_enhanced.json"):
    """Generate dummy vote trace data with realistic SNR distribution."""
    
    model_names = ["SpectralCNN", "TemporalCNN", "ResNetRF", "SignalLSTM", "SignalTransformer"]
    modulation_classes = ["BPSK", "QPSK", "8PSK", "16QAM", "32QAM", "64QAM", "GMSK", "OFDM"]
    
    # SNR distribution: more samples at medium SNR, fewer at extremes
    snr_weights = {
        -15: 5, -10: 10, -5: 15, 0: 20, 5: 20, 10: 15, 15: 10, 20: 5
    }
    snr_values = []
    for snr, weight in snr_weights.items():
        snr_values.extend([snr] * weight)
    
    signals = []
    
    for i in range(n_samples):
        # Select SNR from weighted distribution
        snr_db = random.choice(snr_values) + random.uniform(-1, 1)  # Add some noise
        
        # Generate Shapley contributions with SNR-dependent patterns
        contributions = {}
        if snr_db < -5:
            # Low SNR: TemporalCNN and LSTM perform better (temporal processing)
            contributions = {
                "SpectralCNN": random.uniform(0.01, 0.05),
                "TemporalCNN": random.uniform(0.08, 0.15),
                "ResNetRF": random.uniform(0.02, 0.06),
                "SignalLSTM": random.uniform(0.06, 0.12),
                "SignalTransformer": random.uniform(-0.02, 0.04)
            }
        elif snr_db > 10:
            # High SNR: Spectral methods dominate
            contributions = {
                "SpectralCNN": random.uniform(0.10, 0.18),
                "TemporalCNN": random.uniform(0.02, 0.08),
                "ResNetRF": random.uniform(0.08, 0.14),
                "SignalLSTM": random.uniform(0.01, 0.05),
                "SignalTransformer": random.uniform(0.05, 0.11)
            }
        else:
            # Medium SNR: Mixed performance
            contributions = {
                "SpectralCNN": random.uniform(0.05, 0.12),
                "TemporalCNN": random.uniform(0.04, 0.10),
                "ResNetRF": random.uniform(0.06, 0.11),
                "SignalLSTM": random.uniform(0.03, 0.08),
                "SignalTransformer": random.uniform(0.04, 0.09)
            }
        
        # Normalize contributions to sum to reasonable total
        total = sum(contributions.values())
        target_total = random.uniform(0.3, 0.8)
        for model in contributions:
            contributions[model] = contributions[model] / total * target_total
        
        # Generate per-model probabilities for visualizations
        num_classes = 8  # BPSK, QPSK, 8PSK, 16QAM, 32QAM, 64QAM, GMSK, OFDM
        modulation_idx = random.randint(0, num_classes - 1)
        per_model_probs = {}
        
        for model in model_names:
            # Generate realistic probability distribution
            probs = np.random.dirichlet([2.0] * num_classes)
            # Bias toward the true modulation
            probs[modulation_idx] *= random.uniform(1.5, 3.0)
            probs = probs / np.sum(probs)  # renormalize
            per_model_probs[model] = probs.tolist()
        
        # Compute ensemble probabilities
        ensemble_probs = np.mean([per_model_probs[m] for m in model_names], axis=0)
        pred_idx = int(np.argmax(ensemble_probs))
        true_idx = modulation_idx
        
        signal = {
            "id": f"signal_{i:04d}",
            "snr_db": round(snr_db, 2),
            "modulation": modulation_classes[modulation_idx],
            "shapley_contribution": contributions,
            "per_model_probs": per_model_probs,
            "ensemble_final_prob": ensemble_probs.tolist(),
            "pred_idx": pred_idx,
            "true_idx": true_idx,
            "correct": pred_idx == true_idx,
            "metadata": {
                "snr_db": round(snr_db, 2),
                "ensemble_confidence": float(ensemble_probs[pred_idx]),
                "processing_time_ms": random.uniform(1.2, 4.8)
            }
        }
        
        signals.append(signal)
    
    # Write to JSON file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({"signals": signals}, f, indent=2)
    
    print(f"Generated {n_samples} enhanced dummy signals with SNR distribution")
    
    # Print distribution summary
    snr_dist = {}
    for s in signals:
        snr_bin = int(s["snr_db"] // 5) * 5  # 5dB bins for summary
        snr_dist[snr_bin] = snr_dist.get(snr_bin, 0) + 1
    
    print("SNR distribution:")
    for snr_bin in sorted(snr_dist.keys()):
        print(f"  {snr_bin:+3d} dB: {snr_dist[snr_bin]:3d} samples")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced dummy vote trace data")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    output_file = f"{args.output_dir}/ensemble_contributions_enhanced.json"
    generate_enhanced_dummy_data(args.num_samples, output_file)