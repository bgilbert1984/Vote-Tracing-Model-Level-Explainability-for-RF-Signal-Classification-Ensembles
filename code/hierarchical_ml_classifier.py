#!/usr/bin/env python3
"""
Hierarchical ML Classifier Integration

This module integrates the hierarchical classifier with the SignalIntelligence system.
It extends the existing ML classifier to support hierarchical classification using specialized models.
"""

import os
import sys
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import from project modules
from SignalIntelligence.core import RFSignal
from SignalIntelligence.ml_classifier import MLClassifier, SpectralCNN, ModelNotLoadedError

# Initialize logger
logger = logging.getLogger("Hierarchical-ML-Classifier")

class HierarchicalMLClassifier(MLClassifier):
    """
    Hierarchical ML Classifier that extends the base MLClassifier
    
    This classifier first uses a general model for initial classification,
    and then applies specialized models based on the signal type for more
    accurate classification.
    """
    
    def __init__(self, config):
        """
        Initialize the hierarchical classifier
        
        Args:
            config: Configuration dictionary with hierarchical classifier settings
        """
        # Create MLClassifierConfig from dict if needed
        if isinstance(config, dict):
            from SignalIntelligence.ml_classifier import MLClassifierConfig
            classifier_config = MLClassifierConfig(
                model_path=config.get("model_path", "models/simple"),
                model_type=config.get("model_type", "spectral_cnn"),
                feature_extraction=config.get("feature_extraction", True),
                gpu_enabled=config.get("gpu_enabled", True),
                confidence_threshold=config.get("confidence_threshold", 0.7),
                batch_size=config.get("batch_size", 32),
                signal_classes=config.get("signal_classes", [
                    "FM Radio", "NOAA Weather", "Amateur Radio", 
                    "GSM", "ADS-B", "WiFi", "Bluetooth", 
                    "GPS", "LTE", "LoRa", "Unknown"
                ])
            )
        else:
            classifier_config = config
            
        # Initialize base classifier
        super().__init__(classifier_config)
        
        # Hierarchical classifier specific settings
        self.hierarchical_enabled = config.get("hierarchical_enabled", True) if isinstance(config, dict) else config.hierarchical_enabled
        self.specialized_models_path = config.get("specialized_models_path", "models") if isinstance(config, dict) else config.specialized_models_path
        self.confidence_threshold = config.get("confidence_threshold", 0.4) if isinstance(config, dict) else config.confidence_threshold
        
        # Initialize specialized models and classes
        self.specialized_models = {}
        self.specialized_classes = {}
        
        # Load specialized models if hierarchical classification is enabled
        if self.hierarchical_enabled:
            self._load_specialized_models()
    
    def _load_specialized_models(self):
        """Load all available specialized models"""
        logger.info(f"Loading specialized models from {self.specialized_models_path}")
        
        # Create specialized models directory if it doesn't exist
        if not os.path.exists(self.specialized_models_path):
            os.makedirs(self.specialized_models_path)
            logger.warning(f"Created specialized models directory: {self.specialized_models_path}")
            return
        
        # Look for specialized model directories
        for item in os.listdir(self.specialized_models_path):
            item_path = os.path.join(self.specialized_models_path, item)
            
            # Skip non-directories
            if not os.path.isdir(item_path):
                continue
            
            # Check if this is a specialized model
            model_file = os.path.join(item_path, "spectral_cnn.pt")
            metadata_file = os.path.join(item_path, "model_metadata.json")
            class_file = os.path.join(item_path, "classes.json")
            
            if not os.path.exists(model_file):
                continue
            
            try:
                # Load classes from file or use defaults based on model name
                classes = []
                found_classes = False
                
                if os.path.exists(class_file):
                    with open(class_file, 'r') as f:
                        import json
                        classes = json.load(f)
                        found_classes = True
                elif os.path.exists(metadata_file):
                    # Try to get classes from metadata
                    with open(metadata_file, 'r') as f:
                        import json
                        metadata = json.load(f)
                        if "classes" in metadata:
                            classes = metadata["classes"]
                            found_classes = True
                
                if not found_classes:
                    # Determine class names based on model name
                    if "vhf_amateur" in item:
                        classes = ["VHF Amateur"]
                    elif "fm_radio" in item:
                        classes = ["FM Radio"]
                    elif "amateur_radio" in item:
                        classes = ["Amateur Radio"]
                    elif "noaa_weather" in item:
                        classes = ["NOAA Weather"]
                    elif "marine_vhf" in item:
                        classes = ["Marine VHF"]
                    else:
                        logger.warning(f"Could not determine classes for {item}, skipping")
                        continue
                
                # Load model
                try:
                    # Initialize model with number of classes
                    from SignalIntelligence.ml_classifier import SpectralCNN
                    model = SpectralCNN(num_classes=len(classes))
                    
                    # Load model weights
                    device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu_enabled else "cpu")
                    model.load_state_dict(torch.load(model_file, map_location=device))
                    model.to(device)
                    model.eval()
                    
                    # Store model and its classes
                    self.specialized_models[item] = model
                    self.specialized_classes[item] = classes
                    
                    logger.info(f"Loaded specialized model '{item}' with classes: {classes}")
                except Exception as e:
                    logger.error(f"Failed to load specialized model '{item}': {str(e)}")
            except Exception as e:
                logger.error(f"Error loading specialized model '{item}': {str(e)}")
    
    def classify_signal(self, signal: RFSignal) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a signal using hierarchical classification
        
        Args:
            signal: RFSignal object containing signal data
            
        Returns:
            Tuple containing (classification, confidence, probabilities)
        """
        # If hierarchical classification is disabled or no specialized models, use base classifier
        if not self.hierarchical_enabled or not self.specialized_models:
            return super().classify_signal(signal)
            
        # First classification with base model (flat)
        import time
        t0 = time.perf_counter()
        try:
            classification, confidence, probabilities = super().classify_signal(signal)
        except Exception as e:
            logger.error(f"Base classifier failed: {str(e)}")
            raise
        t_base = (time.perf_counter() - t0) * 1e3  # ms

        # Snapshot the flat/baseline decision so we can compare later
        base_pred = classification
        base_conf = confidence
        base_probs = dict(probabilities) if isinstance(probabilities, dict) else {}
        used_specialized = False
            
        # If confidence is high enough, try specialized model
        if confidence >= self.confidence_threshold:
            # Find matching specialized model
            for model_name, classes in self.specialized_classes.items():
                if classification in classes:
                    specialized_model = self.specialized_models.get(model_name)
                    
                    if specialized_model is not None:
                        try:
                            # Create input for specialized model
                            spectral_input = self._create_spectral_input(signal.iq_data)
                            
                            # Run specialized model
                            import time
                            t1 = time.perf_counter()
                            with torch.no_grad():
                                specialized_model.eval()
                                outputs = specialized_model(spectral_input.to(self.device))
                                
                                # Get probabilities
                                import torch.nn.functional as F
                                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                                
                                # Get prediction
                                pred_idx = np.argmax(probs)
                                specialized_confidence = float(probs[pred_idx])
                                
                                # Only use if confidence is higher
                                if specialized_confidence > confidence:
                                    # Get class name
                                    if pred_idx < len(self.specialized_classes[model_name]):
                                        specialized_class = self.specialized_classes[model_name][pred_idx]
                                        
                                        # Update results
                                        classification = specialized_class
                                        confidence = specialized_confidence
                                        probabilities[specialized_class] = specialized_confidence
                                        used_specialized = True
                            t_spec = (time.perf_counter() - t1) * 1e3  # ms
                            signal.metadata["lat_base_ms"] = t_base
                            signal.metadata["lat_spec_ms"] = t_spec
                            signal.metadata["lat_total_ms"] = t_base + t_spec if used_specialized else t_base
                            signal.metadata["base_pred"] = base_pred
                            signal.metadata["base_conf"] = base_conf
                            signal.metadata["specialized_pred"] = classification if used_specialized else None
                            signal.metadata["used_specialized"] = used_specialized
                            
                            # Add metadata when specialized model is used
                            if used_specialized:
                                signal.metadata["specialized_model"] = model_name
                                signal.metadata["specialized_confidence"] = specialized_confidence
                                logger.info(f"Used specialized model '{model_name}' with confidence {specialized_confidence:.2f}")
                        except Exception as e:
                            logger.error(f"Error using specialized model '{model_name}': {str(e)}")
                    
                    # We found a matching model, no need to check others
                    break
        
        # If no specialized model used, still populate latency + baseline meta
        if "lat_base_ms" not in signal.metadata:
            signal.metadata["lat_base_ms"] = t_base
            signal.metadata["lat_spec_ms"] = 0.0
            signal.metadata["lat_total_ms"] = t_base
            signal.metadata["base_pred"] = base_pred
            signal.metadata["base_conf"] = base_conf
            signal.metadata["specialized_pred"] = None
            signal.metadata["used_specialized"] = False
        return classification, confidence, probabilities
