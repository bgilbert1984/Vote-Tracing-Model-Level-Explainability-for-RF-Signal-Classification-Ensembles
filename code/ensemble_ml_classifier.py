#!/usr/bin/env python3
"""
Ensemble ML Classifier Integration

This module implements ensemble methods for machine learning classification of RF signals.
It combines multiple models and voting schemes to achieve higher accuracy than single models.
"""

import os
import sys
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from collections import Counter

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import from project modules
from SignalIntelligence.core import RFSignal
from SignalIntelligence.hierarchical_ml_classifier import HierarchicalMLClassifier
from open_set_utils import apply_open_set_policy, softmax
from SignalIntelligence.ml_classifier import MLClassifier, ModelNotLoadedError
# Import fixed models with better compatibility
from SignalIntelligence.fixed_ml_models import (
    SpectralCNN, SignalLSTM, TemporalCNN, ResNetRF, SignalTransformer, create_model
)

# Import scikit-learn for traditional ML models if available
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, some ensemble methods will be disabled")

# Initialize logger
logger = logging.getLogger("Ensemble-ML-Classifier")

class EnsembleMLClassifier(HierarchicalMLClassifier):
    """
    Ensemble ML Classifier that extends the Hierarchical Classifier
    
    This classifier uses multiple models and voting schemes to improve classification accuracy.
    It incorporates:
    1. Multiple architectures (CNN, LSTM, traditional ML)
    2. Weighted voting based on model confidence
    3. Specialized models for specific signal types
    4. Optional feature fusion for improved accuracy
    """
    
    def __init__(self, config):
        """
        Initialize the ensemble classifier
        
        Args:
            config: Configuration dictionary with ensemble classifier settings
        """
        # Call parent class init to set up hierarchical classifier
        super().__init__(config)
        
        # Ensemble-specific settings
        if isinstance(config, dict):
            self.ensemble_enabled = config.get("ensemble_enabled", True)
            self.voting_method = config.get("voting_method", "weighted")  # Options: majority, weighted, stacked
            self.ensemble_models_path = config.get("ensemble_models_path", self.config.model_path)
            self.ensemble_threshold = config.get("ensemble_threshold", 0.5)
            self.use_traditional_ml = config.get("use_traditional_ml", SKLEARN_AVAILABLE)
            self.feature_fusion = config.get("feature_fusion", False)
        else:
            self.ensemble_enabled = getattr(config, "ensemble_enabled", True)
            self.voting_method = getattr(config, "voting_method", "weighted") 
            self.ensemble_models_path = getattr(config, "ensemble_models_path", self.config.model_path)
            self.ensemble_threshold = getattr(config, "ensemble_threshold", 0.5)
            self.use_traditional_ml = getattr(config, "use_traditional_ml", SKLEARN_AVAILABLE)
            self.feature_fusion = getattr(config, "feature_fusion", False)
        
        # Initialize ensemble models and traditional ML models
        self.ensemble_models = {}
        self.trad_ml_models = {}
        
        # Load ensemble models if enabled
        if self.ensemble_enabled:
            self._load_ensemble_models()
            
            # Load traditional ML models if sklearn is available
            if self.use_traditional_ml and SKLEARN_AVAILABLE:
                self._load_traditional_ml_models()
    
    def _load_ensemble_models(self):
        """Load all available ensemble models with different architectures using the fixed model classes"""
        logger.info(f"Loading ensemble models from {self.ensemble_models_path}")
        
        # Import json for metadata reading
        import json
        
        # Model mapping - all models are now directly imported from fixed_ml_models
        model_types = {
            "spectral_cnn": SpectralCNN, 
            "signal_lstm": SignalLSTM, 
            "temporal_cnn": TemporalCNN,
            "resnet_rf": ResNetRF,
            "signal_transformer": SignalTransformer
        }
        
        # Load each available model type
        for model_key, model_class in model_types.items():
            model_file = os.path.join(self.ensemble_models_path, f"{model_key}.pt")
            
            if not os.path.exists(model_file):
                logger.info(f"Model file {model_file} not found, skipping {model_key}")
                continue
                
            try:
                # Try to load model metadata to get proper number of classes
                metadata_file = os.path.join(self.ensemble_models_path, f"{model_key}_metadata.json")
                num_classes = len(self.config.signal_classes)
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            if 'classes' in metadata:
                                # Use the number of classes from the metadata
                                num_classes = len(metadata['classes'])
                                # Save the class mapping for later use
                                class_mapping = {idx: cls for idx, cls in enumerate(metadata['classes'])}
                                logger.info(f"Using {num_classes} classes for {model_key} based on metadata")
                    except Exception as meta_error:
                        logger.warning(f"Error reading metadata for {model_key}: {str(meta_error)}")
                        class_mapping = None
                else:
                    logger.info(f"No metadata file for {model_key}, using default class mapping")
                    class_mapping = None
                
                # Initialize model with number of classes
                model = model_class(num_classes=num_classes)
                
                # Store class mapping if available
                if class_mapping:
                    model.class_mapping = class_mapping
                
                # Use our custom load method that handles mismatches
                if hasattr(model, 'load_from_checkpoint'):
                    logger.info(f"Using custom loader for {model_key}")
                    load_success = model.load_from_checkpoint(model_file, device=self.device)
                    if not load_success:
                        logger.warning(f"Custom loading failed for {model_key}, continuing with random weights")
                else:
                    # Fall back to regular loading
                    try:
                        model.load_state_dict(torch.load(model_file, map_location=self.device))
                        logger.info(f"Successfully loaded state dict for {model_key}")
                    except Exception as load_error:
                        logger.warning(f"Couldn't load state dict for {model_key}: {str(load_error)}")
                        logger.info(f"Initializing {model_key} with random weights for testing")
                
                model.to(self.device)
                model.eval()
                
                # Store model
                self.ensemble_models[model_key] = model
                logger.info(f"Loaded ensemble model: {model_key}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_key} model: {str(e)}")
    
    def _load_traditional_ml_models(self):
        """Load traditional machine learning models from pickle files"""
        # Traditional ML model types
        ml_model_types = ["random_forest", "svm", "gbm", "knn"]
        
        for model_type in ml_model_types:
            model_file = os.path.join(self.ensemble_models_path, f"{model_type}.pkl")
            
            if os.path.exists(model_file):
                try:
                    # Load model from pickle file
                    import pickle
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Store model
                    self.trad_ml_models[model_type] = model
                    logger.info(f"Loaded traditional ML model: {model_type}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {model_type} model: {str(e)}")
    
    def _extract_features(self, iq_data: np.ndarray) -> Dict[str, float]:
        """
        Extract features for traditional ML models
        
        Args:
            iq_data: Complex IQ samples
            
        Returns:
            Dictionary of extracted features
        """
        # If feature extraction is disabled, return empty dict
        if not hasattr(self, 'feature_extraction') or not self.feature_extraction:
            return {}
            
        features = {}
        
        # Time domain features
        amplitude = np.abs(iq_data)
        phase = np.angle(iq_data)
        
        # Basic statistical features
        features["mean_amp"] = float(np.mean(amplitude))
        features["std_amp"] = float(np.std(amplitude))
        features["max_amp"] = float(np.max(amplitude))
        features["mean_phase"] = float(np.mean(phase))
        features["std_phase"] = float(np.std(phase))
        
        # Frequency domain features
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq_data)))
        
        # Normalize spectrum
        spectrum = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
        
        # Spectral features
        features["spectral_kurtosis"] = float(np.mean((spectrum - np.mean(spectrum))**4) / 
                                      (np.std(spectrum)**4) if np.std(spectrum) > 0 else 0)
        features["spectral_skewness"] = float(np.mean((spectrum - np.mean(spectrum))**3) / 
                                       (np.std(spectrum)**3) if np.std(spectrum) > 0 else 0)
        
        # Peak features
        peaks = spectrum[spectrum > np.percentile(spectrum, 95)]
        features["num_peaks"] = float(len(peaks))
        features["peak_mean"] = float(np.mean(peaks) if len(peaks) > 0 else 0)
        
        # Modulation features
        # AM detection
        am_envelope = np.abs(iq_data)
        features["am_mod_index"] = float((np.max(am_envelope) - np.min(am_envelope)) / 
                                  (np.max(am_envelope) + np.min(am_envelope)) if np.max(am_envelope) > 0 else 0)
        
        # FM detection
        fm_demod = np.diff(np.unwrap(np.angle(iq_data)))
        features["fm_deviation"] = float(np.std(fm_demod))
        
        return features
    
    def _create_spectral_input(self, iq_data: np.ndarray) -> torch.Tensor:
        """
        Create spectral input for CNN model
        
        Args:
            iq_data: Complex IQ samples
            
        Returns:
            Tensor for CNN input with proper shape
        """
        try:
            # Calculate spectrum
            spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq_data)))
            
            # If spectrum is too large or too small, resize it to 256 points (standard size)
            target_size = 256
            if len(spectrum) != target_size:
                # Resample to target size
                # For larger spectrums, downsample
                if len(spectrum) > target_size:
                    indices = np.linspace(0, len(spectrum)-1, target_size, dtype=int)
                    spectrum = spectrum[indices]
                # For smaller spectrums, interpolate
                else:
                    x_original = np.linspace(0, 1, len(spectrum))
                    x_new = np.linspace(0, 1, target_size)
                    spectrum = np.interp(x_new, x_original, spectrum)
            
            # Normalize
            if np.max(spectrum) > 0:
                spectrum = spectrum / np.max(spectrum)
            
            # Convert to PyTorch tensor - reshape to match model's expected input
            # Shape: [batch_size, channels, sequence_length]
            tensor = torch.FloatTensor(spectrum).view(1, 1, target_size)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error creating spectral input: {str(e)}")
            # Return a default tensor with the right shape in case of error
            return torch.zeros(1, 1, 256)
    
    def _classify_with_traditional_ml(self, features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Classify using traditional ML models
        
        Args:
            features: Dictionary of features
            
        Returns:
            Dictionary of model_name -> {class_name: probability}
        """
        if not self.trad_ml_models or not features:
            return {}
            
        # Convert features to numpy array for sklearn
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names]).reshape(1, -1)
        
        # Normalize features
        scaler = self.trad_ml_models.get("scaler")
        if scaler:
            feature_vector = scaler.transform(feature_vector)
            
        results = {}
        
        # Get predictions from each model
        for model_name, model in self.trad_ml_models.items():
            if model_name == "scaler":
                continue
                
            try:
                # Get predictions
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(feature_vector)[0]
                    class_probas = {self.rev_class_mapping[i]: float(p) for i, p in enumerate(probas)}
                    results[model_name] = class_probas
                else:
                    # For models without probability output, use a binary result
                    prediction = model.predict(feature_vector)[0]
                    class_name = self.rev_class_mapping.get(prediction, "Unknown")
                    # Set 1.0 probability for predicted class
                    class_probas = {class_name: 1.0}
                    results[model_name] = class_probas
            except Exception as e:
                logger.error(f"Error using {model_name} model: {str(e)}")
                
        return results
    
    def classify_signal(self, signal: RFSignal) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a signal using ensemble methods
        
        Args:
            signal: RFSignal object containing signal data
            
        Returns:
            Tuple containing (classification, confidence, probabilities)
        """
        # If ensemble is disabled, use hierarchical classification
        if not self.ensemble_enabled or (not self.ensemble_models and not self.trad_ml_models):
            return super().classify_signal(signal)
        
        # Get hierarchical classification result as a starting point
        try:
            hier_classification, hier_confidence, hier_probabilities = super().classify_signal(signal)
        except Exception as e:
            logger.error(f"Hierarchical classification failed: {str(e)}")
            # If hierarchical fails, try frequency-based classification as fallback
            try:
                from SignalIntelligence.core import SignalProcessor
                # Create signal processor with minimal config
                signal_processor = SignalProcessor({"use_cuda": False})
                freq_classification, freq_confidence = signal_processor.classify_signal(signal)
                hier_classification = freq_classification
                hier_confidence = freq_confidence
                hier_probabilities = {freq_classification: freq_confidence}
                logger.info(f"Using frequency-based classification as fallback: {freq_classification} ({freq_confidence:.2f})")
            except Exception as fallback_error:
                logger.error(f"Fallback classification also failed: {str(fallback_error)}")
                hier_classification = "Unknown"
                hier_confidence = 0.0
                hier_probabilities = {"Unknown": 1.0}
        
        # Store all model predictions for ensemble voting
        all_predictions = {}
        all_probabilities = {}
        
        # Add hierarchical result
        all_predictions["hierarchical"] = (hier_classification, hier_confidence)
        all_probabilities["hierarchical"] = hier_probabilities
        
        # Add ensemble models predictions
        iq_data = signal.iq_data
        
        # Skip if IQ data is too short
        if len(iq_data) < 32:
            logger.warning("Signal IQ data too short for classification")
            return hier_classification, hier_confidence, hier_probabilities
        
        # Get predictions from deep learning ensemble models
        for model_name, model in self.ensemble_models.items():
            try:
                # Create input based on model type
                if model_name == "spectral_cnn" or model_name == "resnet_rf":
                    # Spectral input (frequency domain)
                    model_input = self._create_spectral_input(iq_data)
                elif model_name == "signal_lstm" or model_name == "temporal_cnn":
                    # Temporal input (time domain)
                    model_input = self._create_temporal_input(iq_data)
                elif model_name == "signal_transformer":
                    # Combined input for transformer
                    model_input = self._create_transformer_input(iq_data)
                else:
                    # Default to spectral input
                    model_input = self._create_spectral_input(iq_data)
                
                # Get model predictions
                with torch.no_grad():
                    model.eval()
                    try:
                        # Move input to device
                        model_input = model_input.to(self.device)
                        
                        # Run the model
                        outputs = model(model_input)
                        
                        # Convert to probabilities
                        import torch.nn.functional as F
                        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                        
                        # Get predicted class and confidence
                        pred_idx = np.argmax(probs)
                        confidence = float(probs[pred_idx])
                        
                        # Convert to class name - handle different class indices
                        if hasattr(model, 'class_mapping') and model.class_mapping:
                            # If the model has its own class mapping, use it
                            classification = model.class_mapping.get(pred_idx, "Unknown")
                        else:
                            # Use our mapping, but ensure the index is valid
                            classification = self.rev_class_mapping.get(pred_idx, "Unknown") if pred_idx < len(self.rev_class_mapping) else "Unknown"
                        
                        # Create probabilities dictionary, handling different class counts
                        probabilities = {}
                        
                        # Map probabilities to class names
                        if hasattr(model, 'class_mapping') and model.class_mapping:
                            # If model has its own mapping, use it
                            for i, p in enumerate(probs):
                                if i in model.class_mapping:
                                    class_name = model.class_mapping[i]
                                    probabilities[class_name] = float(p)
                        else:
                            # Use our mapping
                            for i, p in enumerate(probs):
                                if i < len(self.rev_class_mapping):
                                    class_name = self.rev_class_mapping[i]
                                    probabilities[class_name] = float(p)
                        
                        # Store predictions
                        all_predictions[model_name] = (classification, confidence)
                        all_probabilities[model_name] = probabilities
                        
                        logger.info(f"{model_name} prediction: {classification} ({confidence:.4f})")
                    
                    except RuntimeError as tensor_error:
                        logger.error(f"Runtime error using {model_name} model: {str(tensor_error)}")
                        # Don't add this model's predictions if it fails
                        
            except Exception as e:
                logger.error(f"Error using {model_name} model: {str(e)}")
        
        # Get predictions from traditional ML models
        if self.use_traditional_ml and SKLEARN_AVAILABLE and self.trad_ml_models:
            # Extract features for traditional ML
            features = self._extract_features(iq_data)
            
            # Get traditional ML predictions
            trad_ml_results = self._classify_with_traditional_ml(features)
            
            # Add traditional ML results to all predictions
            for model_name, class_probas in trad_ml_results.items():
                # Get predicted class (highest probability)
                if class_probas:
                    pred_class = max(class_probas.items(), key=lambda x: x[1])[0]
                    confidence = class_probas[pred_class]
                    
                    # Store predictions
                    all_predictions[f"trad_{model_name}"] = (pred_class, confidence)
                    all_probabilities[f"trad_{model_name}"] = class_probas
        
        # Perform ensemble voting
        if all_predictions:
            if self.voting_method == "majority":
                # Simple majority voting (each model gets one vote)
                votes = [pred[0] for model, pred in all_predictions.items()]
                vote_counter = Counter(votes)
                
                # Get the class with the most votes
                if vote_counter:
                    final_class = vote_counter.most_common(1)[0][0]
                    
                    # Calculate confidence based on vote proportion
                    final_confidence = vote_counter[final_class] / len(votes)
                    
                    # Calculate aggregated probabilities
                    final_probabilities = hier_probabilities.copy()  # Start with hierarchical probabilities
                    
                    # Update with average probabilities from all models
                    for class_name in final_probabilities.keys():
                        prob_sum = sum(probs.get(class_name, 0.0) for probs in all_probabilities.values())
                        final_probabilities[class_name] = prob_sum / len(all_probabilities)
                else:
                    # Fallback to hierarchical or frequency-based
                    final_class = hier_classification
                    final_confidence = hier_confidence
                    final_probabilities = hier_probabilities
                
            elif self.voting_method == "weighted":
                # Confidence-weighted voting (higher confidence gets more weight)
                weighted_votes = {}
                
                # Calculate weighted votes for each class
                for model_name, (pred_class, confidence) in all_predictions.items():
                    if confidence > 0:  # Only count predictions with non-zero confidence
                        weighted_votes[pred_class] = weighted_votes.get(pred_class, 0) + confidence
                
                # Get the class with the highest weighted vote
                if weighted_votes:
                    final_class = max(weighted_votes.items(), key=lambda x: x[1])[0]
                    
                    # Calculate confidence as normalized weight
                    total_weight = sum(weighted_votes.values())
                    final_confidence = weighted_votes[final_class] / total_weight if total_weight > 0 else 0.0
                    
                    # Calculate weighted probabilities
                    final_probabilities = {}
                    
                    # For each class, calculate weighted average probability
                    all_classes = set()
                    for probs in all_probabilities.values():
                        all_classes.update(probs.keys())
                    
                    for class_name in all_classes:
                        # Calculate weighted average probability
                        weighted_prob = 0.0
                        total_model_weight = 0.0
                        
                        for model_name, (pred_class, confidence) in all_predictions.items():
                            model_probs = all_probabilities[model_name]
                            if class_name in model_probs:
                                weighted_prob += model_probs[class_name] * confidence
                                total_model_weight += confidence
                        
                        # Store weighted probability
                        if total_model_weight > 0:
                            final_probabilities[class_name] = weighted_prob / total_model_weight
                        else:
                            final_probabilities[class_name] = 0.0
                else:
                    # Fallback to hierarchical results if no weighted votes
                    final_class = hier_classification
                    final_confidence = hier_confidence
                    final_probabilities = hier_probabilities
            
            elif self.voting_method == "stacked":
                # Stacked ensemble (use meta-model to combine predictions)
                # This is more complex and requires a trained meta-model
                # For now, we'll fall back to weighted voting
                logger.warning("Stacked ensemble not implemented yet, falling back to weighted voting")
                return self.classify_signal(signal)  # Recursively call with weighted voting
            
            else:
                # Unknown voting method, use hierarchical results
                logger.warning(f"Unknown voting method: {self.voting_method}, using hierarchical results")
                final_class = hier_classification
                final_confidence = hier_confidence
                final_probabilities = hier_probabilities
            
            # Add metadata for debugging
            signal.metadata["ensemble_predictions"] = {model: pred[0] for model, pred in all_predictions.items()}
            signal.metadata["ensemble_confidences"] = {model: pred[1] for model, pred in all_predictions.items()}
            signal.metadata["ensemble_method"] = self.voting_method
            
            # Apply open-set policy for unknown detection
            final_class, final_confidence, final_probabilities = self._apply_open_set_detection(
                final_class, final_confidence, final_probabilities, signal
            )
            
            # Return ensemble prediction
            return final_class, final_confidence, final_probabilities
        else:
            # No ensemble predictions available, return hierarchical results
            hier_class, hier_confidence, hier_probabilities = self._apply_open_set_detection(
                hier_classification, hier_confidence, hier_probabilities, signal
            )
            return hier_class, hier_confidence, hier_probabilities
            
    def _create_temporal_input(self, iq_data: np.ndarray) -> torch.Tensor:
        """
        Create temporal input for LSTM/RNN models with error handling
        
        Args:
            iq_data: Complex IQ samples
            
        Returns:
            Tensor of shape [1, seq_len, 2] for LSTM input
        """
        try:
            # Define target sequence length
            target_seq_len = 128
            
            # Convert complex IQ data to real/imag components
            iq_real = np.real(iq_data)
            iq_imag = np.imag(iq_data)
            
            # Ensure we have valid data
            if np.isnan(iq_real).any() or np.isnan(iq_imag).any():
                logger.warning("NaN values detected in IQ data, replacing with zeros")
                iq_real = np.nan_to_num(iq_real)
                iq_imag = np.nan_to_num(iq_imag)
            
            # Combine into channel dimension
            iq_combined = np.stack([iq_real, iq_imag], axis=1)
            
            # Resize to target sequence length
            if len(iq_combined) > target_seq_len:
                # Use evenly spaced samples for downsampling
                indices = np.linspace(0, len(iq_combined)-1, target_seq_len, dtype=int)
                iq_combined = iq_combined[indices]
            else:
                # Pad with zeros for shorter sequences
                padded = np.zeros((target_seq_len, 2))
                padded[:len(iq_combined)] = iq_combined
                iq_combined = padded
            
            # Convert to tensor with proper shape: [batch_size, seq_len, features]
            tensor = torch.FloatTensor(iq_combined).unsqueeze(0)
            
            return tensor
        
        except Exception as e:
            logger.error(f"Error creating temporal input: {str(e)}")
            # Return a default zero tensor with correct shape
            return torch.zeros(1, 128, 2)
    
    def _create_transformer_input(self, iq_data: np.ndarray) -> torch.Tensor:
        """
        Create input for transformer models with error handling
        
        Args:
            iq_data: Complex IQ samples
            
        Returns:
            Tensor for transformer input of shape [batch, seq_len, features]
        """
        try:
            # Define parameters
            batch_size = 1
            seq_len = 128
            
            # Get spectral and temporal features
            spectral = self._create_spectral_input(iq_data).squeeze(0)  # Remove batch dimension
            temporal = self._create_temporal_input(iq_data).squeeze(0)  # Remove batch dimension
            
            # Resize temporal if needed
            if temporal.shape[0] > seq_len:
                temporal = temporal[:seq_len]
            elif temporal.shape[0] < seq_len:
                # Pad with zeros
                pad = torch.zeros(seq_len - temporal.shape[0], temporal.shape[1], device=temporal.device)
                temporal = torch.cat([temporal, pad], dim=0)
            
            # Reshape spectral to same sequence length - it's originally [channels, features]
            if spectral.shape[0] == 1:  # If it has a channel dimension
                spectral_features = spectral.reshape(1, -1)  # Keep channel dim, flatten rest
                spectral_repeated = spectral_features.repeat(seq_len, 1)  # Repeat for each time step
            else:
                # Already flattened
                spectral_features = spectral.reshape(1, -1)  # Add feature dimension
                spectral_repeated = spectral_features.repeat(seq_len, 1)  # Repeat for each time step
            
            # Combine features along the feature dimension
            combined = torch.cat([temporal, spectral_repeated], dim=1)
            
            # Add batch dimension
            combined = combined.unsqueeze(0)  # [1, seq_len, features]
            
            return combined
            
        except Exception as e:
            logger.error(f"Error creating transformer input: {str(e)}")
            # Return a default tensor with the right shape
            # Most transformer models expect [batch, seq_len, features]
            # Use 2 (temporal) + 256 (spectral) features
            return torch.zeros(1, 128, 258)

    def _apply_open_set_detection(self, classification: str, confidence: float, 
                                  probabilities: Dict[str, float], signal) -> Tuple[str, float, Dict[str, float]]:
        """
        Apply open-set policy to detect unknown signals.
        
        Args:
            classification: Predicted class name
            confidence: Classification confidence
            probabilities: Class probabilities
            signal: RFSignal object
            
        Returns:
            (classification, confidence, probabilities) with possible "Unknown" mapping
        """
        try:
            # Convert probabilities to numpy arrays for open-set analysis
            class_names = list(probabilities.keys())
            probs_array = np.array([probabilities[name] for name in class_names])
            
            # Create dummy logits from probabilities (for energy calculation)
            # This is an approximation: logits = log(probs)
            logits_array = np.log(probs_array + 1e-12)
            
            # Get open-set thresholds from signal metadata or use defaults
            thresholds = getattr(signal, 'open_set_thresholds', {})
            tau_p = thresholds.get("tau_p", 0.60)  # Max probability threshold
            tau_H = thresholds.get("tau_H", 1.2)   # Entropy threshold  
            tau_E = thresholds.get("tau_E", None)  # Optional energy threshold
            
            # Apply open-set policy
            accept, metrics = apply_open_set_policy(probs_array, logits_array, tau_p, tau_H, tau_E)
            
            # Add open-set metrics to signal metadata
            if hasattr(signal, 'metadata'):
                signal.metadata["open_set_metrics"] = metrics
                signal.metadata["open_set_accept"] = accept
                signal.metadata["open_set_thresholds"] = {"tau_p": tau_p, "tau_H": tau_H, "tau_E": tau_E}
            
            # If rejected by open-set policy, return Unknown
            if not accept:
                logger.info(f"Open-set policy rejected signal: s_max={metrics['s_max']:.3f}, "
                           f"entropy={metrics['entropy']:.3f}, energy={metrics.get('energy', 'N/A')}")
                
                # Create Unknown probabilities
                unknown_probabilities = probabilities.copy()
                unknown_probabilities["Unknown"] = 1.0 - max(probs_array)  # Uncertainty score
                
                return "Unknown", metrics['s_max'], unknown_probabilities
            
            else:
                # Accepted by open-set policy, return original prediction
                logger.debug(f"Open-set policy accepted signal: {classification} "
                            f"(s_max={metrics['s_max']:.3f}, entropy={metrics['entropy']:.3f})")
                return classification, confidence, probabilities
                
        except Exception as e:
            logger.error(f"Error in open-set detection: {str(e)}")
            # If open-set processing fails, return original prediction
            return classification, confidence, probabilities