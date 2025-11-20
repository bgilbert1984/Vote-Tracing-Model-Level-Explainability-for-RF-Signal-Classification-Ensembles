#!/usr/bin/env python3
"""
Apply calibration patch to ensemble_ml_classifier.py

This script demonstrates how to integrate temperature scaling into the existing
ensemble classifier with minimal modifications.
"""

import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add the code directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    from calibration_utils import CalibrationAggregator, _softmax, _prob_temperature
    from ensemble_ml_classifier import EnsembleMLClassifier
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure ensemble_ml_classifier.py and calibration_utils.py are in the code directory")
    sys.exit(1)

class CalibratedEnsembleMLClassifier(EnsembleMLClassifier):
    """
    Enhanced ensemble classifier with temperature calibration
    
    This extends the existing EnsembleMLClassifier with calibration capabilities
    while preserving all existing functionality.
    """
    
    def __init__(self, config):
        """Initialize calibrated ensemble classifier"""
        super().__init__(config)
        
        # Parse calibration config
        if isinstance(config, dict):
            calib_cfg = config.get("calibration", {})
        else:
            calib_cfg = getattr(config, "calibration", {})
            
        # Calibration settings
        self._calibration_enabled = bool(calib_cfg.get("enabled", False))
        self._calibration_collect = bool(calib_cfg.get("collect_metrics", False))
        self._calibration_tau = float(calib_cfg.get("tau", 0.6))
        self._calibration_override_T = calib_cfg.get("override_temperature", None)
        
        # Per-model temperatures (default to 1.0)
        n_models = len(getattr(self, "ensemble_models", []) or [])
        temps = calib_cfg.get("temperatures", [1.0] * n_models)
        if len(temps) != n_models:
            temps = (temps + [1.0] * n_models)[:n_models]
        self._calibration_T = [float(t) for t in temps]
        
        print(f"Calibration enabled: {self._calibration_enabled}")
        print(f"Calibration temperatures: {self._calibration_T}")
        print(f"Calibration tau: {self._calibration_tau}")

    def classify_signal(self, signal):
        """
        Enhanced classify_signal with calibration support
        
        This method extends the parent method with temperature scaling
        while preserving all existing logic and fallbacks.
        """
        
        # If calibration is disabled or no ensemble models, use parent method
        if not self._calibration_enabled or not getattr(self, "ensemble_models", None):
            return super().classify_signal(signal)
        
        # Get hierarchical classification as fallback
        try:
            hier_classification, hier_confidence, hier_probabilities = super().classify_signal(signal)
        except Exception as e:
            print(f"Hierarchical classification failed: {e}")
            hier_classification = "Unknown"
            hier_confidence = 0.0
            hier_probabilities = {"Unknown": 1.0}
        
        # Extract IQ data
        iq_data = signal.iq_data
        if len(iq_data) < 32:
            print("Signal too short for ensemble classification")
            return hier_classification, hier_confidence, hier_probabilities
        
        # Collect per-model probabilities
        per_model_probs_uncal = []
        per_model_probs_cal = []
        weights = []
        
        override_T = self._calibration_override_T
        
        for mi, (model_name, model) in enumerate(self.ensemble_models.items()):
            try:
                # Create model input based on type
                if model_name in ["spectral_cnn", "resnet_rf"]:
                    model_input = self._create_spectral_input(iq_data)
                elif model_name in ["signal_lstm", "temporal_cnn"]:
                    model_input = self._create_temporal_input(iq_data)
                elif model_name == "signal_transformer":
                    model_input = self._create_transformer_input(iq_data)
                else:
                    model_input = self._create_spectral_input(iq_data)
                
                # Forward pass
                with torch.no_grad():
                    model.eval()
                    model_input = model_input.to(self.device)
                    outputs = model(model_input)
                    
                    # Convert to probabilities
                    if hasattr(outputs, 'detach'):
                        logits = outputs.detach().cpu().numpy()
                    else:
                        logits = np.asarray(outputs, dtype=np.float64)
                    
                    # Handle different output shapes
                    if len(logits.shape) > 1:
                        logits = logits[0]  # Take first batch element
                    
                    # Compute uncalibrated probabilities
                    p_uncal = _softmax(logits[None, :], axis=1)[0]
                    
                    # Apply temperature scaling
                    T_model = float(override_T) if override_T is not None else float(self._calibration_T[mi])
                    p_cal = _softmax((logits / max(T_model, 1e-6))[None, :], axis=1)[0]
                    
                    per_model_probs_uncal.append(p_uncal)
                    per_model_probs_cal.append(p_cal)
                    weights.append(1.0)  # Equal weighting for now
                    
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                continue
        
        # If no ensemble predictions, fall back to hierarchical
        if not per_model_probs_uncal:
            return hier_classification, hier_confidence, hier_probabilities
        
        # Aggregate probabilities
        W = np.array(weights, dtype=np.float64)
        W = W / (W.sum() if W.sum() != 0 else 1.0)
        
        P_uncal = np.average(np.stack(per_model_probs_uncal, axis=0), axis=0, weights=W)
        P_cal = np.average(np.stack(per_model_probs_cal, axis=0), axis=0, weights=W)
        
        # Select calibrated or uncalibrated probabilities
        use_P = P_cal if self._calibration_enabled else P_uncal
        
        # Get prediction
        pred_idx = int(np.argmax(use_P))
        conf = float(use_P[pred_idx])
        
        # Apply confidence threshold
        tau = self._calibration_tau
        accept = bool(conf >= tau)
        
        # Map to class name
        if pred_idx < len(getattr(self, 'rev_class_mapping', [])):
            classification = self.rev_class_mapping[pred_idx]
        else:
            classification = "Unknown"
        
        if not accept:
            classification = "Abstain"
            
        # Create probabilities dict
        final_probabilities = {}
        class_names = getattr(self, 'rev_class_mapping', [f"Class_{i}" for i in range(len(use_P))])
        
        for i, p in enumerate(use_P):
            if i < len(class_names):
                final_probabilities[class_names[i]] = float(p)
        
        # Store metadata
        signal.metadata.update({
            "calibration_enabled": self._calibration_enabled,
            "calibration_temperatures": list(self._calibration_T),
            "calibration_tau": tau,
            "ensemble_probs_uncal": P_uncal.tolist(),
            "ensemble_probs_cal": P_cal.tolist(), 
            "final_probabilities": use_P.tolist(),
            "final_confidence": conf,
            "final_pred_idx": pred_idx,
            "accepted": accept
        })
        
        # Collect calibration metrics if enabled
        if self._calibration_collect:
            y_true = None
            if hasattr(signal, "label_idx"):
                y_true = int(signal.label_idx)
            elif hasattr(signal, "label"):
                y_true = int(signal.label) 
            elif isinstance(signal.metadata.get("label_idx"), (int, np.integer)):
                y_true = int(signal.metadata["label_idx"])
                
            if y_true is not None:
                CalibrationAggregator.add(P_uncal, P_cal, y_true)
        
        return classification, conf, final_probabilities

def example_usage():
    """Example of how to use the calibrated ensemble classifier"""
    
    # Configuration with calibration settings
    config = {
        "ensemble_enabled": True,
        "voting_method": "weighted",
        "calibration": {
            "enabled": True,
            "collect_metrics": True,
            "tau": 0.6,
            "temperatures": [1.2, 0.9, 1.1, 1.0],  # Per-model temperatures
            "override_temperature": None  # Set to float to override all models
        }
    }
    
    # Create calibrated classifier
    try:
        classifier = CalibratedEnsembleMLClassifier(config)
        print("Calibrated ensemble classifier created successfully")
        
        # Example signal classification would go here
        # result = classifier.classify_signal(some_rf_signal)
        
        # After processing a dataset, finalize calibration metrics
        output_path = "/tmp/calibration_metrics.json"
        CalibrationAggregator.finalize(
            output_path, 
            tau=config["calibration"]["tau"],
            temperatures_per_model=config["calibration"]["temperatures"]
        )
        print(f"Calibration metrics saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating classifier: {e}")

if __name__ == "__main__":
    example_usage()