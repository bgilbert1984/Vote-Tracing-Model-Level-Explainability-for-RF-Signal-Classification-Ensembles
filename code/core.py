import logging
import numpy as np
import threading
import time
import json
import requests
import os
from math import isclose
from queue import Queue
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("SignalIntelligence")

# Custom JSON encoder for numpy types
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that properly handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@dataclass
class GeoPosition:
    """Geographic position data"""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "lat": float(self.latitude),
            "lon": float(self.longitude),
            "alt": float(self.altitude) if self.altitude is not None else None,
            "accuracy": float(self.accuracy) if self.accuracy is not None else None,
            "timestamp": float(self.timestamp)
        }

@dataclass
class RFSignal:
    """RF Signal data structure"""
    id: str
    timestamp: float
    frequency: float
    bandwidth: float
    power: float
    iq_data: np.ndarray
    source: str
    classification: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    geo_position: Optional[GeoPosition] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (excludes IQ data)"""
        result = {
            "id": self.id,
            "timestamp": float(self.timestamp),
            "frequency": float(self.frequency),
            "frequency_mhz": float(self.frequency / 1e6),
            "bandwidth": float(self.bandwidth),
            "bandwidth_khz": float(self.bandwidth / 1e3),
            "power": float(self.power),
            "source": self.source,
            "classification": self.classification,
            "confidence": float(self.confidence),
            "metadata": self.metadata
        }
        
        # Add geo position if available
        if self.geo_position:
            result["geo_position"] = self.geo_position.to_dict()
            
        return result

class SignalIntelligenceSystem:
    """Main Signal Intelligence System class"""
    def __init__(self, config, comm_network):
        self.config = config
        self.comm_network = comm_network
        self.running = False
        self.signal_queue = Queue()
        self.processed_signals = []
        self.noise_floor = -120  # Default noise floor in dBm
        self.geo_visualization_url = None
        self.geo_areas_of_operation = []
        
        # ATL/TWPA design-aware processing
        self.atl_design = None
        self.recent_freqs_hz = []  # small FIFO for mixing checks
        
        # === SIMULATION FRAMEWORK === 
        # Support for simulation-driven validation of both papers
        self.simulation_mode = config.get("simulation", {}).get("enabled", False)
        self.scenario_generator = None
        self.metrics_buffer = []  # Unified metrics collection
        
        # Initialize core components
        self._initialize_components()
        
        # Setup geo visualization integration if available
        self._setup_geo_integration()
        
        # Load ATL/TWPA design configuration
        self._load_atl_design()
        
        # === SETUP SIMULATION MODE ===
        if self.simulation_mode:
            self._setup_simulation()
        
    def _initialize_components(self):
        """Initialize signal intelligence components"""
        # This will be implemented in subclasses or extended
        pass
    
    def _setup_geo_integration(self):
        """Setup integration with geographic visualization"""
        try:
            # Check for geo visualization configuration
            geo_config_path = "config/geo_visualization.json"
            if os.path.exists(geo_config_path):
                with open(geo_config_path, "r") as f:
                    geo_config = json.load(f)
                    
                # Extract server information
                server_config = geo_config.get("server", {})
                host = server_config.get("host", "localhost")
                port = server_config.get("port", 5050)
                
                # Set geo visualization URL
                self.geo_visualization_url = f"http://{host}:{port}/api/signals"
                
                # Load areas of operation
                ao_config = geo_config.get("areas_of_operation", {})
                ao_presets = ao_config.get("presets", [])
                
                if ao_presets:
                    self.geo_areas_of_operation = ao_presets
                    logger.info(f"Loaded {len(ao_presets)} Areas of Operation for geo visualization")
                
                logger.info(f"Geographic visualization integration configured at {self.geo_visualization_url}")
        except Exception as e:
            logger.warning(f"Failed to configure geo visualization integration: {e}")
    
    def _load_atl_design(self):
        """Load optional ATL/TWPA design facts from Arxiv 2510.24753v1."""
        try:
            path = "config/atl_design.json"
            if os.path.exists(path):
                with open(path, "r") as f:
                    d = json.load(f)
                # normalize expected keys
                self.atl_design = {
                    "pump_hz": float(d.get("pump_hz")) if d.get("pump_hz") else None,
                    "rpm_notch_hz": float(d.get("rpm_notch_hz")) if d.get("rpm_notch_hz") else None,
                    "rpm_pole_hz": float(d.get("rpm_pole_hz")) if d.get("rpm_pole_hz") else None,
                    "stopbands": [
                        {"center_hz": float(sb["center_hz"]), "width_hz": float(sb["width_hz"])}
                        for sb in d.get("stopbands", [])
                    ],
                    "mixing_mode": d.get("mixing_mode", "4WM")  # or "3WM"
                }
                logger.info("ATL/TWPA design loaded from config/atl_design.json")
        except Exception as e:
            logger.warning(f"Failed to load ATL design: {e}")

    def _setup_simulation(self):
        """Initialize simulation from config/simulation_scenarios.json"""
        try:
            # Import simulation module directly
            import sys
            from pathlib import Path
            
            # Add code directory to path
            code_dir = Path(__file__).parent
            if str(code_dir) not in sys.path:
                sys.path.insert(0, str(code_dir))
                
            from simulation import create_scenario_generator
            
            path = "config/simulation_scenarios.json"
            if not os.path.exists(path):
                logger.info("No simulation config found. Skipping.")
                return
            
            self.scenario_generator = create_scenario_generator(path)
            
            # Get simulation config
            sim_config = self.config.get("simulation", {})
            inject_rate_hz = sim_config.get("inject_rate_hz", 10)
            
            logger.info(f"Simulation mode enabled: {len(self.scenario_generator.scenarios)} scenarios loaded")
            
            # Start injection thread
            sim_thread = threading.Thread(
                target=self._simulation_injection_loop, 
                args=(inject_rate_hz,),
                daemon=True
            )
            sim_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            
    def _simulation_injection_loop(self, inject_rate_hz: float = 10):
        """Background thread for RF signal injection"""
        logger.info("Starting simulation injection loop")
        self.scenario_generator.injection_loop(self.signal_queue, inject_rate_hz)

    def start_scenario(self, scenario_name: str) -> bool:
        """Start a simulation scenario by name"""
        if not self.simulation_mode or not self.scenario_generator:
            logger.error("Simulation mode not enabled")
            return False
            
        return self.scenario_generator.start_scenario(scenario_name)
        
    def stop_simulation(self):
        """Stop simulation and save metrics"""
        if self.scenario_generator:
            self.scenario_generator.stop()
            self._flush_metrics()
            logger.info("Simulation stopped")

    def _log_metric(self, study: str, data: dict):
        """Log metrics data for analysis"""
        data.update({"timestamp": time.time()})
        self.metrics_buffer.append({"study": study, "data": data})
        
        # Auto-flush if buffer gets large
        if len(self.metrics_buffer) > 10000:
            self._flush_metrics()

    def _flush_metrics(self):
        """Flush metrics buffer to disk"""
        if not self.metrics_buffer:
            return
            
        # Create logs directory if needed
        os.makedirs("logs", exist_ok=True)
        
        # Write to timestamped file
        filename = f"logs/metrics_{int(time.time())}.jsonl"
        with open(filename, "w") as f:
            for m in self.metrics_buffer:
                f.write(json.dumps(m, cls=NumpyJSONEncoder) + "\n")
        
        logger.info(f"Flushed {len(self.metrics_buffer)} metrics to {filename}")
        self.metrics_buffer = []

    def _wrap_signal(self, signal_data) -> RFSignal:
        """Convert signal_data to RFSignal if needed"""
        if hasattr(signal_data, 'iq_data') and hasattr(signal_data, 'center_freq_hz'):
            # Handle simulation.RFSignal objects
            return RFSignal(
                id=f"sim_{int(time.time()*1000)}",
                timestamp=signal_data.timestamp,
                frequency=signal_data.center_freq_hz,
                bandwidth=signal_data.sample_rate_hz,  # Use sample rate as bandwidth approximation
                power=-30,  # Default power for simulation
                iq_data=signal_data.iq_data,
                source="simulation",
                metadata=signal_data.metadata
            )
        elif isinstance(signal_data, RFSignal):
            return signal_data
        elif isinstance(signal_data, dict):
            # Convert dict to RFSignal
            return RFSignal(
                id=signal_data.get("id", f"sig_{int(time.time()*1000)}"),
                timestamp=signal_data.get("timestamp", time.time()),
                frequency=signal_data.get("center_freq_hz", signal_data.get("frequency", 100e6)),
                bandwidth=signal_data.get("bandwidth_hz", signal_data.get("bandwidth", 1e6)),
                power=signal_data.get("power", -50),
                iq_data=signal_data.get("iq_data", np.array([1+1j, -1-1j])),
                source=signal_data.get("source", "unknown"),
                metadata=signal_data.get("metadata", {})
            )
        else:
            raise ValueError(f"Unsupported signal_data type: {type(signal_data)}")

    def _run_resampling_ablation(self, signal: RFSignal):
        """Paper 1: Resampling study - test different FFT sizes and sequence lengths"""
        if not self.config.get("resampling_study", {}).get("enabled", False):
            return
            
        fft_sizes = [64, 128, 256, 512, 1024]
        seq_lens = [32, 64, 96, 128, 192, 256]
        
        # Reference PSD computation
        ref_psd = np.abs(np.fft.fft(signal.iq_data, n=1024))**2
        
        for fft_size in fft_sizes:
            for seq_len in seq_lens:
                try:
                    # Resample the signal
                    resampled = self._resample_signal(signal.iq_data, seq_len)
                    
                    # Compute PSD with specified FFT size
                    if len(resampled) < fft_size:
                        resampled = np.pad(resampled, (0, fft_size - len(resampled)))
                    elif len(resampled) > fft_size:
                        resampled = resampled[:fft_size]
                        
                    test_psd = np.abs(np.fft.fft(resampled, n=fft_size))**2
                    
                    # Resize PSDs to same length for comparison
                    if len(test_psd) != len(ref_psd):
                        min_len = min(len(test_psd), len(ref_psd))
                        test_psd = test_psd[:min_len]
                        ref_psd_norm = ref_psd[:min_len]
                    else:
                        ref_psd_norm = ref_psd
                    
                    # Compute KL divergence
                    kl_divergence = self._compute_kl_divergence(ref_psd_norm, test_psd)
                    
                    self._log_metric("resampling", {
                        "fft_size": fft_size,
                        "seq_len": seq_len, 
                        "kl_divergence": kl_divergence,
                        "true_modulation": signal.metadata.get("true_modulation", "unknown"),
                        "snr_db": signal.metadata.get("snr_db", 0),
                        "signal_id": signal.id
                    })
                    
                except Exception as e:
                    logger.warning(f"Resampling ablation error (FFT={fft_size}, seq={seq_len}): {e}")

    def _resample_signal(self, iq_data: np.ndarray, target_length: int) -> np.ndarray:
        """Resample IQ data to target length"""
        current_length = len(iq_data)
        if current_length == target_length:
            return iq_data
        elif current_length > target_length:
            # Downsample
            indices = np.linspace(0, current_length - 1, target_length, dtype=int)
            return iq_data[indices]
        else:
            # Upsample with interpolation
            try:
                from scipy import interpolate
                x_old = np.arange(current_length)
                x_new = np.linspace(0, current_length - 1, target_length)
                
                # Interpolate real and imaginary parts separately
                real_interp = interpolate.interp1d(x_old, iq_data.real, kind='linear')
                imag_interp = interpolate.interp1d(x_old, iq_data.imag, kind='linear')
                
                return real_interp(x_new) + 1j * imag_interp(x_new)
            except ImportError:
                # Fallback: simple linear interpolation without scipy
                indices = np.linspace(0, current_length - 1, target_length)
                return np.interp(indices, np.arange(current_length), iq_data)

    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between two PSDs"""
        # Normalize to probabilities
        p_norm = p / (np.sum(p) + 1e-12)
        q_norm = q / (np.sum(q) + 1e-12)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        p_norm = p_norm + epsilon
        q_norm = q_norm + epsilon
        
        # Compute KL divergence
        return np.sum(p_norm * np.log(p_norm / q_norm))

    def _run_calibration_sweep(self, signal: RFSignal):
        """Paper 2: Confidence calibration study"""
        if not self.config.get("calibration_study", {}).get("enabled", False):
            return
            
        # Simulate ensemble logits (in real system, these come from classifier)
        logits = self._simulate_ensemble_logits(signal)
        
        temps = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        tau = 0.6  # Confidence threshold
        
        for T in temps:
            try:
                # Apply temperature scaling
                prob_cal = self._apply_temperature_scaling(logits, T)
                max_prob = np.max(prob_cal)
                pred_class = np.argmax(prob_cal) if max_prob >= tau else -1  # -1 = abstain
                
                self._log_metric("calibration", {
                    "temperature": T,
                    "max_probability": float(max_prob),
                    "predicted_class": int(pred_class),
                    "abstained": pred_class == -1,
                    "true_modulation": signal.metadata.get("true_modulation", "unknown"),
                    "true_class": signal.metadata.get("true_class", -1),
                    "snr_db": signal.metadata.get("snr_db", 0),
                    "signal_id": signal.id,
                    "tau": tau
                })
                
            except Exception as e:
                logger.warning(f"Calibration sweep error (T={T}): {e}")

    def _simulate_ensemble_logits(self, signal: RFSignal) -> np.ndarray:
        """Simulate ensemble classifier logits for calibration study"""
        # Simplified: create realistic logits based on signal properties
        num_classes = 6  # BPSK, QPSK, 8PSK, 16QAM, 64QAM, FM
        
        # Base logits with some randomness
        logits = np.random.randn(num_classes) * 2.0
        
        # Make prediction more confident for higher SNR
        snr = signal.metadata.get("snr_db", 0)
        confidence_boost = min(snr / 20.0, 1.0)  # Cap at 1.0
        
        # Boost the "correct" class if we know it
        true_mod = signal.metadata.get("true_modulation", "unknown")
        mod_map = {"BPSK": 0, "QPSK": 1, "8PSK": 2, "16QAM": 3, "64QAM": 4, "FM": 5}
        
        if true_mod in mod_map:
            true_idx = mod_map[true_mod]
            logits[true_idx] += 3.0 * confidence_boost
            signal.metadata["true_class"] = true_idx
        
        return logits

    def _apply_temperature_scaling(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to logits"""
        scaled_logits = logits / temperature
        # Apply softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)

    def _label_atl_band(self, f_hz: float, tol_hz: float = 0.01e9):
        """Return band label based on design facts from ATL synthesis."""
        if not self.atl_design:
            return "unknown", {}
        d = self.atl_design
        info = {}

        # near rpm notch / pole (phase-matching feature for 4WM)
        if d.get("rpm_notch_hz") and abs(f_hz - d["rpm_notch_hz"]) <= tol_hz:
            info["near_rpm_notch"] = True
            return "near_notch", info
        if d.get("rpm_pole_hz") and abs(f_hz - d["rpm_pole_hz"]) <= tol_hz:
            info["near_rpm_pole"] = True
            # still a passband point, but important
            return "passband", info

        # stopbands (e.g., wide 3fp gap)
        for sb in d.get("stopbands", []):
            if abs(f_hz - sb["center_hz"]) <= sb["width_hz"] / 2:
                info["stopband_center_hz"] = sb["center_hz"]
                return "stopband", info

        return "passband", info

    def _mixing_relations(self, f_hz: float, ppm: float = 150.0):
        """Compute candidate mixing partners to annotate metadata."""
        if not self.atl_design or not self.atl_design.get("pump_hz"):
            return {}
        fp = self.atl_design["pump_hz"]
        tol = fp * ppm * 1e-6  # parts-per-million window

        rel = {"near_3fp": False, "idlers": []}

        # third harmonic guard (esp. for KTWPA)
        if abs(f_hz - 3.0 * fp) <= tol:
            rel["near_3fp"] = True

        # 4WM: idlers around (2fp - fs) and (2fp + fs)
        if self.atl_design.get("mixing_mode", "4WM").upper() == "4WM":
            for fs in self.recent_freqs_hz[-64:]:
                i1 = abs((2.0 * fp) - fs)  # 2fp - fs
                i2 = abs((2.0 * fp) + fs)  # rarely in band, still annotate
                if abs(f_hz - i1) <= tol:
                    rel["idlers"].append({"mode": "4WM", "expr": "2fp - fs", "fs_hz": fs, "idler_hz": i1})
                if abs(f_hz - i2) <= tol:
                    rel["idlers"].append({"mode": "4WM", "expr": "2fp + fs", "fs_hz": fs, "idler_hz": i2})

        # 3WM: idlers around (fp - fs) and (fp + fs)
        else:
            for fs in self.recent_freqs_hz[-64:]:
                i1 = abs(fp - fs)  # usual forward idler
                i2 = abs(fp + fs)
                if abs(f_hz - i1) <= tol:
                    rel["idlers"].append({"mode": "3WM", "expr": "fp - fs", "fs_hz": fs, "idler_hz": i1})
                if abs(f_hz - i2) <= tol:
                    rel["idlers"].append({"mode": "3WM", "expr": "fp + fs", "fs_hz": fs, "idler_hz": i2})

        return rel

    def annotate_signal_with_atl(self, signal: "RFSignal"):
        """Attach ATL/TWPA labels to signal.metadata (no-op if no design)."""
        try:
            # keep a small memory for mixing checks
            self.recent_freqs_hz.append(float(signal.frequency))
            if len(self.recent_freqs_hz) > 512:
                self.recent_freqs_hz = self.recent_freqs_hz[-256:]

            band_label, band_info = self._label_atl_band(signal.frequency)
            mix_info = self._mixing_relations(signal.frequency)

            signal.metadata.setdefault("atl", {})
            signal.metadata["atl"].update({
                "band_label": band_label,
                **band_info,
                **mix_info
            })
            
            # Log important ATL events
            if band_info.get("near_rpm_notch") or mix_info.get("near_3fp") or band_label == "stopband":
                logger.info(f"ATL event detected - Signal {signal.id}: {band_label}, near_3fp: {mix_info.get('near_3fp', False)}")
            
        except Exception as e:
            logger.debug(f"ATL annotate failed: {e}")
    
    def process_atl_alerts(self, signal: "RFSignal"):
        """Process ATL-related alerts and update classifications as needed."""
        if not self.atl_design or "atl" not in signal.metadata:
            return
            
        atl_data = signal.metadata["atl"]
        
        # Check for important ATL events that warrant classification updates
        alert_conditions = []
        
        if atl_data.get("near_3fp"):
            alert_conditions.append("near_3fp_harmonic")
            
        if atl_data.get("band_label") == "stopband":
            alert_conditions.append("in_designed_stopband")
            
        if atl_data.get("near_rpm_notch"):
            alert_conditions.append("near_phase_matching_notch")
            
        if atl_data.get("idlers"):
            alert_conditions.append(f"parametric_mixing_detected({len(atl_data['idlers'])})")
        
        # Update classification if any alert conditions are met
        if alert_conditions:
            new_classification = f"ATL_Event: {', '.join(alert_conditions)}"
            self.update_signal_classification(
                signal.id, 
                new_classification, 
                0.85,  # High confidence for design-based detection
                update_info={"atl": atl_data, "alert_conditions": alert_conditions}
            )
            
    def send_signal_to_geo_visualization(self, signal: RFSignal) -> bool:
        """Send signal to geographic visualization system if available"""
        if not self.geo_visualization_url:
            return False
            
        # Skip signals without geo position
        if not signal.geo_position:
            return False
            
        try:
            # Convert signal to dict
            signal_dict = signal.to_dict()
            
            # Send to geo visualization
            response = requests.post(
                f"{self.geo_visualization_url}/add",
                json=signal_dict,
                headers={"Content-Type": "application/json"},
                timeout=1
            )
            
            if response.status_code == 200:
                logger.debug(f"Sent signal {signal.id} to geo visualization")
                return True
            else:
                logger.warning(f"Failed to send signal to geo visualization: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Error sending signal to geo visualization: {e}")
            return False
    
    def get_signals(self, start_time=None, end_time=None, min_frequency=None, max_frequency=None, signal_id=None):
        """
        Get all processed signals with optional filtering
        
        Args:
            start_time (float, optional): Filter signals after this timestamp
            end_time (float, optional): Filter signals before this timestamp
            min_frequency (float, optional): Filter signals above this frequency (Hz)
            max_frequency (float, optional): Filter signals below this frequency (Hz)
            signal_id (str, optional): Get a specific signal by ID
            
        Returns:
            list: List of signal dictionaries
        """
        # Apply filters
        filtered_signals = self.processed_signals.copy()
        
        # Filter by ID if provided
        if signal_id:
            filtered_signals = [s for s in filtered_signals if s.id == signal_id]
            
        # Apply time filters
        if start_time is not None:
            filtered_signals = [s for s in filtered_signals if s.timestamp >= start_time]
            
        if end_time is not None:
            filtered_signals = [s for s in filtered_signals if s.timestamp <= end_time]
            
        # Apply frequency filters
        if min_frequency is not None:
            filtered_signals = [s for s in filtered_signals if s.frequency >= min_frequency]
            
        if max_frequency is not None:
            filtered_signals = [s for s in filtered_signals if s.frequency <= max_frequency]
        
        # Convert RFSignal objects to dictionaries for JSON serialization
        signals_list = []
        for signal in filtered_signals:
            signals_list.append(signal.to_dict())
            
        return signals_list
        
    def update_signal_classification(self, signal_id, classification, confidence, update_info=None):
        """
        Update the classification of a signal
        
        Args:
            signal_id (str): ID of the signal to update
            classification (str): New classification
            confidence (float): New confidence value
            update_info (dict, optional): Additional information about the update
            
        Returns:
            bool: True if the signal was updated, False otherwise
        """
        # Find the signal
        for signal in self.processed_signals:
            if signal.id == signal_id:
                # Store old values
                old_classification = signal.classification
                old_confidence = signal.confidence
                
                # Update classification
                signal.classification = classification
                signal.confidence = confidence
                
                # Add to classification history
                if "classification_history" not in signal.metadata:
                    signal.metadata["classification_history"] = []
                    
                history_entry = {
                    "timestamp": time.time(),
                    "old_classification": old_classification,
                    "new_classification": classification,
                    "old_confidence": old_confidence,
                    "new_confidence": confidence
                }
                
                # Add update info if provided
                if update_info:
                    history_entry.update(update_info)
                    
                signal.metadata["classification_history"].append(history_entry)
                
                # Re-annotate with ATL if this is an ATL-related update
                if update_info and "atl" in update_info:
                    self.annotate_signal_with_atl(signal)
                
                # Log the update
                logger.info(f"Updated signal {signal_id} classification: {old_classification} → {classification} (confidence: {confidence:.2f})")
                
                # Publish update to communication network if available
                if hasattr(self, "comm_network") and self.comm_network:
                    self.comm_network.publish(
                        "signal_classification_updated",
                        {
                            "signal_id": signal_id,
                            "old_classification": old_classification,
                            "new_classification": classification,
                            "old_confidence": old_confidence,
                            "new_confidence": confidence,
                            "update_info": update_info
                        },
                        sender="signal_intelligence"
                    )
                
                return True
                
        # Signal not found
        logger.warning(f"Cannot update classification: Signal {signal_id} not found")
        return False
    
    def get_rf_environment(self):
        """
        Get the current RF environment information
        
        Returns:
            dict: RF environment data
        """
        # Get current time
        current_time = time.time()
        
        # Find active signals (last 60 seconds)
        active_signals = [s for s in self.processed_signals if current_time - s.timestamp < 60]
        
        # Group signals by frequency bands
        frequency_bands = {}
        for signal in active_signals:
            # Convert to MHz for readability
            freq_mhz = signal.frequency / 1_000_000
            band_key = int(freq_mhz / 100) * 100  # Group by 100 MHz bands
            
            if band_key not in frequency_bands:
                frequency_bands[band_key] = {
                    "center_frequency_mhz": band_key + 50,  # Center of the band
                    "bandwidth_mhz": 100,
                    "signals": [],
                    "power_values": [],
                    "classifications": set()
                }
                
            frequency_bands[band_key]["signals"].append(signal)
            frequency_bands[band_key]["power_values"].append(signal.power)
            
            if signal.classification and signal.classification != "Unknown":
                frequency_bands[band_key]["classifications"].add(signal.classification)
        
        # Format frequency bands for output
        formatted_bands = []
        for band_key, band_data in frequency_bands.items():
            band_dict = {
                "center_frequency_mhz": band_data["center_frequency_mhz"],
                "bandwidth_mhz": band_data["bandwidth_mhz"],
                "power_dbm": np.mean(band_data["power_values"]) if band_data["power_values"] else -120,
                "signal_count": len(band_data["signals"]),
                "protocols_detected": list(band_data["classifications"])
            }
            
            # Add ATL band labeling if design is present
            if self.atl_design:
                center = band_data["center_frequency_mhz"] * 1e6
                label, _ = self._label_atl_band(center, tol_hz=0.02e9)
                band_dict["atl_band_label"] = label
            
            formatted_bands.append(band_dict)
            
        # Calculate spectrum statistics
        spectrum_data = {
            "timestamp": current_time,
            "min_frequency_mhz": min([b["center_frequency_mhz"] - b["bandwidth_mhz"]/2 for b in formatted_bands]) if formatted_bands else 0,
            "max_frequency_mhz": max([b["center_frequency_mhz"] + b["bandwidth_mhz"]/2 for b in formatted_bands]) if formatted_bands else 0,
            "avg_noise_floor_dbm": self.noise_floor,
            "frequency_bands": formatted_bands
        }
        
        return spectrum_data
    
    def start(self):
        """Start Signal Intelligence System"""
        logger.info("Starting Signal Intelligence System")
        self.running = True
        
        # Start signal processing thread
        processing_thread = threading.Thread(target=self._signal_processing_loop)
        processing_thread.daemon = True
        processing_thread.start()
    
    def _signal_processing_loop(self):
        """Main signal processing loop"""
        while self.running:
            try:
                # Process signals from queue
                if not self.signal_queue.empty():
                    signal_data = self.signal_queue.get(timeout=1)
                    self.process_signal(signal_data)
                    self.signal_queue.task_done()
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in signal processing: {e}")
                time.sleep(1)
    
    def process_signal(self, signal_data):
        """Enhanced signal processing with Paper 1 & 2 validation hooks"""
        try:
            # Convert to standard RFSignal format
            signal = self._wrap_signal(signal_data)
            
            # === PAPER 1: Resampling Study ===
            if "resampling_study" in self.config:
                self._run_resampling_ablation(signal)
            
            # === PAPER 2: Calibration Study ===  
            if "calibration_study" in self.config:
                self._run_calibration_sweep(signal)
            
            # === ATL/TWPA Physics Annotation ===
            self.annotate_signal_with_atl(signal)
            self.process_atl_alerts(signal)
            
            # Store processed signal
            self.processed_signals.append(signal)
            
            # Log basic processing metric
            if self.simulation_mode:
                self._log_metric("processing", {
                    "signal_id": signal.id,
                    "frequency_mhz": signal.frequency / 1e6,
                    "bandwidth_khz": signal.bandwidth / 1e3,
                    "power_dbm": signal.power,
                    "atl_band": signal.metadata.get("atl", {}).get("band_label", "unknown")
                })
                
            return signal
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return None
    
    def shutdown(self):
        """Shutdown the system"""
        logger.info("Shutting down Signal Intelligence System")
        self.running = False
