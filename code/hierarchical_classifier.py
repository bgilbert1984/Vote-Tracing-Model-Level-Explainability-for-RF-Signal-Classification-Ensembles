#!/usr/bin/env python3
"""
Hierarchical ML Signal Classifier

This script implements a hierarchical classifier that uses the general model
for initial classification and then specialized models for more accurate 
signal-type specific classification.
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Hierarchical-Classifier")

# Add the parent directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import Signal Intelligence classes
from SignalIntelligence.core import RFSignal, NumpyJSONEncoder
from SignalIntelligence.ml_classifier import SpectralCNN

class HierarchicalSignalClassifier:
    """Hierarchical Signal Classifier with general and specialized models"""
    
    def __init__(self, models_path=None):
        """Initialize the hierarchical classifier"""
        self.models_path = models_path or os.path.join(parent_dir, "models")
        
        # Initialize models
        self.general_model = None
        # Hardcoded classes for testing
        self.general_classes = ["VHF Amateur", "FM Radio", "Amateur Radio", "NOAA Weather", "Marine VHF"]
        self.specialized_models = {}
        self.specialized_classes = {}
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        # Load general model
        general_model_path = os.path.join(self.models_path, "simple")
        self._load_general_model(general_model_path)
        
        # Load specialized models
        self._load_specialized_models()
        
        logger.info(f"Loaded general model with {len(self.general_classes)} classes")
        logger.info(f"Loaded {len(self.specialized_models)} specialized models")
    
    def _load_general_model(self, model_path):
        """Load the general classification model"""
        model_file = os.path.join(model_path, "spectral_cnn.pt")
        metadata_file = os.path.join(model_path, "model_metadata.json")
        class_file = os.path.join(model_path, "classes.json")
        
        try:
            # Load classes
            found_classes = False
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    self.general_classes = json.load(f)
                    found_classes = True
            elif os.path.exists(metadata_file):
                # Try to get classes from metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if 'classes' in metadata:
                        self.general_classes = metadata.get('classes', [])
                        found_classes = True
            
            # Use hardcoded classes if not found in files
            if not found_classes:
                logger.warning("No classes found in files for general model, using hardcoded classes")
                self.general_classes = ["VHF Amateur", "FM Radio", "Amateur Radio", "NOAA Weather", "Marine VHF"]
            
            # Initialize model with correct number of classes
            self.general_model = SpectralCNN(num_classes=len(self.general_classes))
            
            # Load weights
            if os.path.exists(model_file):
                self.general_model.load_state_dict(torch.load(model_file))
                self.general_model.eval()
                logger.info(f"Loaded general model from {model_file}")
                logger.info(f"General model classes: {self.general_classes}")
            else:
                logger.error(f"Model file not found: {model_file}")
                self.general_model = None
        except Exception as e:
            logger.error(f"Failed to load general model: {str(e)}")
            self.general_model = None
    
    def _load_specialized_models(self):
        """Load all available specialized models"""
        # Look for specialized model directories
        for item in os.listdir(self.models_path):
            item_path = os.path.join(self.models_path, item)
            
            # Skip non-directories and the general model
            if not os.path.isdir(item_path) or item == "simple":
                continue
            
            # Check if this is a specialized model
            model_file = os.path.join(item_path, "spectral_cnn.pt")
            metadata_file = os.path.join(item_path, "model_metadata.json")
            class_file = os.path.join(item_path, "classes.json")
            
            if not os.path.exists(model_file):
                continue
            
            try:
                # Load classes
                classes = []
                found_classes = False
                if os.path.exists(class_file):
                    with open(class_file, 'r') as f:
                        classes = json.load(f)
                        found_classes = True
                elif os.path.exists(metadata_file):
                    # Try to get classes from metadata
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        if 'classes' in metadata:
                            classes = metadata.get('classes', [])
                            found_classes = True
                
                # Use hardcoded classes based on model name
                if not found_classes:
                    if 'vhf_amateur' in item:
                        classes = ["VHF Amateur"]
                    elif 'fm_radio' in item:
                        classes = ["FM Radio"]
                    elif 'amateur_radio' in item:
                        classes = ["Amateur Radio"]
                    elif 'noaa_weather' in item:
                        classes = ["NOAA Weather"]
                    elif 'marine_vhf' in item:
                        classes = ["Marine VHF"]
                    
                if not classes:
                    logger.warning(f"No classes found for model '{item}', skipping")
                    continue
                
                # Skip models with more than 3 classes (not specialized enough)
                if len(classes) > 3:
                    continue
                
                # Initialize model with correct number of classes
                model = SpectralCNN(num_classes=len(classes))
                
                # Load weights
                model.load_state_dict(torch.load(model_file))
                model.eval()
                
                # Add to specialized models
                self.specialized_models[item] = model
                
                # Hardcoded classes for testing
                if 'vhf_amateur' in item:
                    self.specialized_classes[item] = ["VHF Amateur"]
                elif 'fm_radio' in item:
                    self.specialized_classes[item] = ["FM Radio"]
                elif 'amateur_radio' in item:
                    self.specialized_classes[item] = ["Amateur Radio"]
                elif 'noaa_weather' in item:
                    self.specialized_classes[item] = ["NOAA Weather"]
                elif 'marine_vhf' in item:
                    self.specialized_classes[item] = ["Marine VHF"]
                else:
                    self.specialized_classes[item] = classes
                
                logger.info(f"Loaded specialized model '{item}' with classes: {self.specialized_classes[item]}")
            except Exception as e:
                logger.error(f"Failed to load specialized model '{item}': {str(e)}")
    
    def process_signal(self, signal):
        """Process a signal using hierarchical classification"""
        import time
        if self.general_model is None or not self.general_classes:
            logger.error("General model not loaded or no classes available")
            return "Unknown", 0.0, {}
        
        # Create spectral features
        spectral_input = self._create_spectral_input(signal.iq_data).unsqueeze(0)  # Add batch dimension
        
        # Step 1: General classification
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.general_model(spectral_input)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            if len(probs) == 0:  # Handle empty tensor case
                return "Unknown", 0.0, {}
                
            confidence, predicted_idx = torch.max(probs, 0)
            
            # Get classification and confidence
            predicted_idx_item = predicted_idx.item()
            if predicted_idx_item < len(self.general_classes):
                classification = self.general_classes[predicted_idx_item]
            else:
                classification = "Unknown"
                
            confidence_value = float(confidence.item())
            
            # Convert to probabilities dictionary
            probabilities = {cls: float(probs[i].item()) for i, cls in enumerate(self.general_classes)}
        
        # Step 2: Specialized classification if confidence is high enough
        used_specialized = False
        if confidence_value >= 0.4 and self.specialized_models:
            # Find a specialized model that matches this classification
            primary_type = classification.split(' ')[0].lower()
            specialized_model_key = None
            
            # Check for exact match
            for key in self.specialized_models.keys():
                if classification.lower().replace(' ', '_') in key:
                    specialized_model_key = key
                    break
            
            # Check for primary type match
            if specialized_model_key is None:
                for key in self.specialized_models.keys():
                    if primary_type in key:
                        specialized_model_key = key
                        break
            
            # Use specialized model if found
            if specialized_model_key is not None:
                specialized_model = self.specialized_models[specialized_model_key]
                specialized_classes = self.specialized_classes[specialized_model_key]
                
                logger.debug(f"Using specialized model '{specialized_model_key}' for {classification}")
                
                # Get specialized classification
                with torch.no_grad():
                    outputs = specialized_model(spectral_input)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                    if len(probs) > 0:  # Make sure tensor is not empty
                        spec_confidence, spec_predicted_idx = torch.max(probs, 0)
                        
                        # Get classification and confidence
                        spec_predicted_idx_item = spec_predicted_idx.item()
                        if spec_predicted_idx_item < len(specialized_classes):
                            spec_classification = specialized_classes[spec_predicted_idx_item]
                            spec_confidence_value = float(spec_confidence.item())
                            
                            # Only use specialized result if confidence is higher
                            if spec_confidence_value > confidence_value:
                                classification = spec_classification
                                confidence_value = spec_confidence_value
                                logger.debug(f"Specialized model improved classification to {classification} with confidence {confidence_value:.2f}")
                                used_specialized = True
                            
                            # Update probabilities with specialized model results
                            for i, cls in enumerate(specialized_classes):
                                if cls in probabilities:
                                    # Blend probabilities, favoring the more confident model
                                    spec_prob = float(probs[i].item())
                                    gen_prob = probabilities[cls]
                                    probabilities[cls] = max(gen_prob, spec_prob)
        
        # Latency breadcrumbs
        total_ms = (time.perf_counter() - t0) * 1e3
        try:
            md = getattr(signal, "metadata", {})
            md["base_pred"] = md.get("base_pred", None) or "N/A"
            md["lat_total_ms"] = total_ms
            md["used_specialized"] = used_specialized
            signal.metadata = md
        except Exception:
            pass
        return classification, confidence_value, probabilities
    
    def _create_spectral_input(self, iq_data):
        """Create spectral input features from IQ data"""
        # Get FFT of IQ data
        fft_data = np.fft.fftshift(np.fft.fft(iq_data))
        # Convert to power spectrum (magnitude)
        power_spectrum = np.abs(fft_data)
        # Convert to dB scale
        spectrum_db = 20 * np.log10(power_spectrum + 1e-10)
        # Normalize
        spectrum_db_norm = (spectrum_db - np.min(spectrum_db)) / (np.max(spectrum_db) - np.min(spectrum_db) + 1e-10)
        # Reshape for CNN input (1 channel)
        return torch.tensor(spectrum_db_norm, dtype=torch.float32).unsqueeze(0)  # [channel, length]

def load_signals_from_file(filepath):
    """Load signals from a signal file and convert to RFSignal objects"""
    logger.info(f"Loading signals from {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract metadata and results/signals
    metadata = data.get('metadata', {})
    signal_data_list = data.get('signals', [])
    
    # If file contains results instead of signals, convert to signal format
    if not signal_data_list and 'results' in data:
        signal_data_list = data.get('results', [])
        # Convert result format to signal format
        for i, result in enumerate(signal_data_list):
            if 'id' not in result:
                result['id'] = f"signal_{i}"
            if 'classification' not in result and 'predicted_classification' in result:
                result['classification'] = result['predicted_classification']
    
    logger.info(f"Found {len(signal_data_list)} signals in file")
    
    # Convert to RFSignal objects
    signals = []
    for i, signal_data in enumerate(signal_data_list):
        # Create synthetic IQ data for testing
        iq_len = 1024
        freq_offset = signal_data.get('frequency', 0) - metadata.get('center_frequency', 0)
        freq_norm = freq_offset / metadata.get('sample_rate', 2.4e6)
        t = np.arange(iq_len) / iq_len
        
        # Create a basic signal model with some noise
        noise = np.random.normal(0, 0.1, iq_len) + 1j * np.random.normal(0, 0.1, iq_len)
        phase = 2 * np.pi * freq_norm * np.arange(iq_len)
        iq_data = np.exp(1j * phase) + noise
            
        # Create RFSignal object
        rf_signal = RFSignal(
            id=signal_data.get('id', f"test_{i}"),
            timestamp=signal_data.get('timestamp', datetime.now().timestamp()),
            frequency=signal_data.get('frequency', 0),
            bandwidth=signal_data.get('bandwidth', 0),
            power=signal_data.get('power', signal_data.get('max_power', 0)),
            iq_data=iq_data,
            source=signal_data.get('source', 'file'),
            classification=signal_data.get('classification', signal_data.get('predicted_classification', 'Unknown'))
        )
        
        signals.append(rf_signal)
    
    # Get classification distribution
    classifications = {}
    for signal in signals:
        if signal.classification not in classifications:
            classifications[signal.classification] = 0
        classifications[signal.classification] += 1
    
    logger.info("Signal classifications in input file:")
    for cls, count in sorted(classifications.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cls}: {count} signals")
    
    return signals, metadata

def classify_signals(signals, classifier):
    """Classify signals using the hierarchical classifier"""
    results = []
    
    for i, signal in enumerate(signals):
        if (i+1) % 10 == 0:
            logger.info(f"Processed {i+1}/{len(signals)} signals")
        
        # Classify signal
        classification, confidence, probabilities = classifier.process_signal(signal)
        
        # Create result
        result = {
            "id": signal.id,
            "frequency": float(signal.frequency),
            "bandwidth": float(signal.bandwidth),
            "power": float(signal.power),
            "original_classification": signal.classification,
            "predicted_classification": classification,
            "confidence": confidence,
            "probabilities": probabilities
        }
        
        results.append(result)
    
    return results

def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hierarchical ML Signal Classifier")
    
    parser.add_argument("--input", type=str, required=True, help="Input signal file")
    parser.add_argument("--output", type=str, default=None, help="Output classified file")
    parser.add_argument("--models-path", type=str, default=os.path.join(parent_dir, "models"), 
                       help="Path to model directory")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Load signals
    signals, metadata = load_signals_from_file(args.input)
    if not signals:
        logger.error("Failed to load signals from input file")
        return 1
    
    # Initialize hierarchical classifier
    classifier = HierarchicalSignalClassifier(args.models_path)
    
    # Classify signals
    results = classify_signals(signals, classifier)
    
    # Create output file path
    if args.output:
        output_file = args.output
    else:
        input_basename = os.path.basename(args.input)
        input_name, ext = os.path.splitext(input_basename)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        output_file = os.path.join(os.path.dirname(args.input), f"{input_name}_hierarchical_{timestamp}{ext}")
    
    # Save classified data
    logger.info(f"Saving classified data to {output_file}")
    try:
        # Create output data
        output_data = {
            "metadata": {
                **metadata,
                "classification_timestamp": datetime.now().isoformat(),
                "classifier": "hierarchical",
                "models_path": args.models_path
            },
            "results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Successfully saved classified data with {len(results)} signals")
        
        # Print classification summary
        classifications = {}
        for result in results:
            cls = result["predicted_classification"]
            if cls not in classifications:
                classifications[cls] = 0
            classifications[cls] += 1
        
        logger.info("Classification summary:")
        for cls, count in sorted(classifications.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {cls}: {count} signals ({count/len(results)*100:.1f}%)")
    
    except Exception as e:
        logger.error(f"Error saving classified data: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
