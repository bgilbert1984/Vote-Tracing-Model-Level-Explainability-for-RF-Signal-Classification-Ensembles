# simulation.py
"""
Advanced RF Scenario Generation for Simulation-Driven Intelligence

This module creates realistic RF environments for validating:
- Resampling effects (Paper 1) 
- Confidence calibration (Paper 2)
- ATL/TWPA parametric mixing detection

Features:
- Physics-based parametric mixing (4WM from arXiv:2510.24753v1)
- Realistic duty cycles and burst timing
- Multi-emitter scenarios with interference
- Deterministic, reproducible RF battlefield simulation
- Tight integration with both published papers

Author: bgilbert1984
Date: November 2025
"""

import os
import json
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Generator, Optional, Any, Tuple
from queue import Queue
from scipy import signal
import logging

logger = logging.getLogger(__name__)

@dataclass
class GeoPosition:
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": float(self.latitude),
            "lon": float(self.longitude), 
            "alt": float(self.altitude) if self.altitude else None,
            "accuracy": float(self.accuracy) if self.accuracy else None,
            "timestamp": float(self.timestamp)
        }

@dataclass
class RFSignal:
    iq_data: np.ndarray
    sample_rate_hz: float
    center_freq_hz: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    geo_position: Optional[GeoPosition] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iq_data": self.iq_data.tolist() if isinstance(self.iq_data, np.ndarray) else self.iq_data,
            "sample_rate_hz": float(self.sample_rate_hz),
            "center_freq_hz": float(self.center_freq_hz),
            "timestamp": float(self.timestamp),
            "metadata": self.metadata,
            "geo_position": self.geo_position.to_dict() if self.geo_position else None,
            "processing_metadata": self.processing_metadata
        }

@dataclass
class EmitterSpec:
    """Specification for a simulated RF emitter"""
    modulation: str
    frequency_hz: float
    bandwidth_hz: float
    samples: int
    snr_range: Tuple[float, float]
    duty_cycle: float
    inject_mixing: bool = False
    burst_duration_ms: float = 100.0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EmitterSpec':
        return cls(
            modulation=data["modulation"],
            frequency_hz=data["frequency_hz"],
            bandwidth_hz=data["bandwidth_hz"],
            samples=data.get("samples", 1024),
            snr_range=tuple(data.get("snr_range", [0, 20])),
            duty_cycle=data.get("duty_cycle", 0.1),
            inject_mixing=data.get("inject_mixing", False),
            burst_duration_ms=data.get("burst_duration_ms", 100.0)
        )

@dataclass
class ScenarioSpec:
    """Complete RF scenario specification"""
    name: str
    duration_s: float
    emitters: List[EmitterSpec]
    sample_rate_hz: float = 100e6
    noise_floor_dbm: float = -90.0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScenarioSpec':
        return cls(
            name=data["name"],
            duration_s=data["duration_s"],
            emitters=[EmitterSpec.from_dict(e) for e in data["emitters"]],
            sample_rate_hz=data.get("sample_rate_hz", 100e6),
            noise_floor_dbm=data.get("noise_floor_dbm", -90.0)
        )

class ParametricMixer:
    """ATL/TWPA parametric mixing simulation from arXiv:2510.24753v1"""
    
    def __init__(self, pump_hz: float = 8.4e9):
        self.pump_hz = pump_hz
        self.rpm_notch_hz = pump_hz + 10e6  # RPM notch  
        self.rpm_pole_hz = pump_hz - 10e6   # RPM pole
        
        # Stopbands from ATL design paper
        self.stopbands = [
            {"center_hz": 25.2e9, "width_hz": 2e9},
            {"center_hz": 16.8e9, "width_hz": 1e9}  # Additional stopband
        ]
        
    def generate_idlers(self, signal_hz: float) -> List[float]:
        """Generate 4WM idler frequencies"""
        # 4-wave mixing: f_idler = 2*f_pump ± f_signal
        idler1 = 2 * self.pump_hz - signal_hz
        idler2 = 2 * self.pump_hz + signal_hz
        
        # Filter to realistic bands (1-30 GHz)
        idlers = []
        for f in [idler1, idler2]:
            if 1e9 <= f <= 30e9:
                idlers.append(f)
                
        return idlers
        
    def is_in_stopband(self, freq_hz: float) -> bool:
        """Check if frequency falls in ATL stopband"""
        for sb in self.stopbands:
            center = sb["center_hz"]
            width = sb["width_hz"]
            if abs(freq_hz - center) <= width / 2:
                return True
        return False

class ModulationLibrary:
    """Generate realistic modulated signals"""
    
    @staticmethod
    def bpsk(N: int, fc: float = 0, fs: float = 1) -> np.ndarray:
        """Binary Phase Shift Keying"""
        bits = np.random.randint(0, 2, N // 8)
        symbols = 2 * bits - 1  # Map to ±1
        upsampled = np.repeat(symbols, 8)[:N]
        
        # Add pulse shaping (root raised cosine approximation)
        h = signal.windows.hann(16)
        shaped = np.convolve(upsampled, h, mode='same')
        
        # Complex modulation
        t = np.arange(N) / fs
        carrier = np.exp(1j * 2 * np.pi * fc * t)
        return shaped * carrier
        
    @staticmethod
    def qam16(N: int, fc: float = 0, fs: float = 1) -> np.ndarray:
        """16-QAM modulation"""
        # 16-QAM constellation
        constellation = np.array([
            -3-3j, -3-1j, -3+3j, -3+1j,
            -1-3j, -1-1j, -1+3j, -1+1j,
             3-3j,  3-1j,  3+3j,  3+1j,
             1-3j,  1-1j,  1+3j,  1+1j
        ]) / np.sqrt(10)
        
        symbols = np.random.choice(constellation, N // 4)
        upsampled = np.repeat(symbols, 4)[:N]
        
        # Pulse shaping
        h = signal.windows.hann(8)
        shaped = np.convolve(upsampled, h, mode='same')
        
        # Carrier
        t = np.arange(N) / fs
        carrier = np.exp(1j * 2 * np.pi * fc * t)
        return shaped * carrier
        
    @staticmethod
    def fm(N: int, fc: float = 0, fs: float = 1, mod_idx: float = 5) -> np.ndarray:
        """Frequency Modulation"""
        # Audio signal (music/voice simulation)
        t = np.arange(N) / fs
        audio = (np.sin(2 * np.pi * 1000 * t) + 
                0.3 * np.sin(2 * np.pi * 3000 * t) +
                0.1 * np.sin(2 * np.pi * 5000 * t))
        
        # FM modulation
        phase = np.cumsum(2 * np.pi * (fc + mod_idx * audio) / fs)
        return np.exp(1j * phase)
        
    @staticmethod
    def generate(modulation: str, N: int, fc: float = 0, fs: float = 1) -> np.ndarray:
        """Generate modulated signal by type"""
        mod_map = {
            "BPSK": ModulationLibrary.bpsk,
            "16QAM": ModulationLibrary.qam16, 
            "FM": ModulationLibrary.fm,
            "CW": lambda N, fc, fs: np.exp(1j * 2 * np.pi * fc * np.arange(N) / fs)
        }
        
        if modulation not in mod_map:
            logger.warning(f"Unknown modulation {modulation}, using BPSK")
            modulation = "BPSK"
            
        return mod_map[modulation](N, fc, fs)

class RFScenarioGenerator:
    """Generate realistic RF scenarios with physics-based mixing"""
    
    def __init__(self, scenario_configs: List[Dict]):
        self.scenarios = [ScenarioSpec.from_dict(cfg) for cfg in scenario_configs]
        self.mixer = ParametricMixer()
        self.current_scenario = None
        self.start_time = None
        self._stop_event = threading.Event()
        self._signal_counter = 0
        
        # Reproducible randomness
        np.random.seed(42)
        
    def start_scenario(self, name: str) -> bool:
        """Start a named scenario"""
        for scenario in self.scenarios:
            if scenario.name == name:
                self.current_scenario = scenario
                self.start_time = time.time()
                self._signal_counter = 0
                logger.info(f"Started scenario: {name} ({scenario.duration_s}s, {len(scenario.emitters)} emitters)")
                return True
        
        logger.error(f"Scenario '{name}' not found")
        return False
        
    def stop(self):
        """Stop scenario generation"""
        self._stop_event.set()
        
    def generate_burst(self, emitter: EmitterSpec) -> RFSignal:
        """Generate a single RF burst with metadata"""
        N = emitter.samples
        fs = self.current_scenario.sample_rate_hz
        
        # Base modulated signal
        iq_data = ModulationLibrary.generate(
            emitter.modulation, N, 
            fc=emitter.frequency_hz, fs=fs
        )
        
        # Add AWGN
        snr_db = np.random.uniform(*emitter.snr_range)
        noise_power = 10**((self.current_scenario.noise_floor_dbm - snr_db) / 10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(N) + 1j * np.random.randn(N))
        iq_data += noise
        
        # ATL/TWPA parametric mixing injection
        idler_frequencies = []
        if emitter.inject_mixing:
            idlers = self.mixer.generate_idlers(emitter.frequency_hz)
            for idler_freq in idlers:
                # Inject idler tone at 10% amplitude with phase noise
                t = np.arange(N) / fs
                phase_noise = 0.1 * np.random.randn(N)
                idler_tone = 0.1 * np.exp(1j * (2 * np.pi * idler_freq * t + phase_noise))
                iq_data += idler_tone
                idler_frequencies.append(idler_freq)
                logger.debug(f"Injected idler at {idler_freq/1e9:.2f} GHz")
        
        # Enhanced metadata for validation
        metadata = {
            "timestamp": time.time(),
            "signal_id": f"sim_{self._signal_counter}",
            "true_modulation": emitter.modulation,
            "frequency_hz": emitter.frequency_hz,
            "bandwidth_hz": emitter.bandwidth_hz,
            "snr_db": snr_db,
            "has_mixing": emitter.inject_mixing,
            "scenario_name": self.current_scenario.name if self.current_scenario else "unknown",
            "burst_duration_ms": emitter.burst_duration_ms,
            "duty_cycle": emitter.duty_cycle
        }
        
        # ATL/TWPA annotations
        if emitter.inject_mixing:
            metadata["idler_frequencies"] = idler_frequencies
            metadata["atl_band_label"] = "mixing_detected"
            metadata["pump_frequency_hz"] = self.mixer.pump_hz
        else:
            metadata["atl_band_label"] = "normal"
            
        # Check if in ATL stopband
        if self.mixer.is_in_stopband(emitter.frequency_hz):
            metadata["atl_band_label"] = "stopband"
            
        self._signal_counter += 1
        
        return RFSignal(
            iq_data=iq_data,
            sample_rate_hz=fs,
            center_freq_hz=emitter.frequency_hz,
            timestamp=metadata["timestamp"],
            metadata=metadata
        )
        
    def should_emit(self, emitter: EmitterSpec) -> bool:
        """Determine if emitter should transmit (duty cycle logic)"""
        if not self.start_time:
            return False
            
        # Realistic duty cycle with burst timing
        current_time = time.time() - self.start_time
        burst_period_s = emitter.burst_duration_ms / 1000
        cycle_period_s = burst_period_s / emitter.duty_cycle
        
        phase = (current_time % cycle_period_s) / cycle_period_s
        return phase < emitter.duty_cycle
        
    def is_scenario_complete(self) -> bool:
        """Check if current scenario has completed"""
        if not self.current_scenario or not self.start_time:
            return False
            
        elapsed = time.time() - self.start_time
        return elapsed >= self.current_scenario.duration_s
        
    def get_active_emitters(self) -> List[EmitterSpec]:
        """Get list of currently transmitting emitters"""
        if not self.current_scenario:
            return []
            
        return [e for e in self.current_scenario.emitters if self.should_emit(e)]
        
    def injection_loop(self, signal_queue: Queue, inject_rate_hz: float = 10):
        """Continuous injection loop for simulation thread"""
        logger.info(f"Starting RF injection at {inject_rate_hz} Hz")
        period = 1.0 / inject_rate_hz
        
        while not self._stop_event.is_set():
            try:
                if not self.current_scenario:
                    time.sleep(period)
                    continue
                    
                if self.is_scenario_complete():
                    logger.info(f"Scenario '{self.current_scenario.name}' completed - {self._signal_counter} signals generated")
                    self.current_scenario = None
                    time.sleep(period)
                    continue
                
                # Generate bursts from active emitters
                active_emitters = self.get_active_emitters()
                for emitter in active_emitters:
                    burst = self.generate_burst(emitter)
                    signal_queue.put(burst)
                    
                time.sleep(period)
                
            except Exception as e:
                logger.error(f"Injection loop error: {e}")
                time.sleep(period)
                
        logger.info("RF injection stopped")

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)

# Factory function for easy import
def create_scenario_generator(config_path: str) -> RFScenarioGenerator:
    """Create scenario generator from JSON config"""
    with open(config_path, 'r') as f:
        scenarios = json.load(f)
    return RFScenarioGenerator(scenarios)
