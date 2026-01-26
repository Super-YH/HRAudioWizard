"""
High-Frequency Prediction V3 (HFP V3) module.

Advanced STFT-based high-frequency restoration algorithm with:
- Cepstral analysis for formant envelope extraction
- High-order LPC for accurate formant representation (minimum order 16)
- Harmonic tracking with temporal context
- AR prediction for harmonic extrapolation
- Transient detection and handling
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
import scipy.signal
import scipy.interpolate
import scipy.linalg
import librosa
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HFPv3Config:
    """Configuration for HFP v3 processing."""
    n_fft: int = 1024
    hop_length: int = 512
    sr: int = 44100
    min_f0: float = 50.0
    max_f0: float = 2000.0
    
    # LPC settings - minimum order 16 for formant representation
    lpc_order: int = 32
    
    # Temporal context
    temporal_context_frames: int = 6
    lpc_smoothing_alpha: float = 0.6
    
    # Harmonic tracking
    max_harmonics: int = 128
    harmonic_search_ratio: float = 0.25
    
    # Synthesis balance
    harmonic_weight: float = 0.85
    noise_weight: float = 0.85
    
    # Transient handling
    transient_threshold: float = 1.0
    
    # Cepstral settings
    cepstral_lifter_quefrency: int = 80
    
    # High frequency rolloff
    hf_rolloff_start_hz: float = 20000.0
    hf_rolloff_rate_db_per_octave: float = 80.0


@dataclass
class HarmonicTrack:
    """Represents a tracked harmonic across frames."""
    frequency_bins: List[float] = field(default_factory=list)
    amplitudes: List[float] = field(default_factory=list)
    phases: List[float] = field(default_factory=list)
    
    def get_velocity(self) -> float:
        if len(self.frequency_bins) < 2:
            return 0.0
        return self.frequency_bins[-1] - self.frequency_bins[-2]
    
    def predict_next(self) -> float:
        if len(self.frequency_bins) < 2:
            return self.frequency_bins[-1] if self.frequency_bins else 0.0
        if len(self.frequency_bins) >= 3:
            x = np.array([0, 1, 2])
            y = np.array(self.frequency_bins[-3:])
            coeffs = np.polyfit(x, y, 2)
            return np.polyval(coeffs, 3)
        return 2 * self.frequency_bins[-1] - self.frequency_bins[-2]


@dataclass
class FrameAnalysis:
    """Complete analysis results for a single frame."""
    f0: float = 0.0
    f0_confidence: float = 0.0
    harmonic_amplitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    harmonic_phases: np.ndarray = field(default_factory=lambda: np.array([]))
    cepstral_envelope: np.ndarray = field(default_factory=lambda: np.array([]))
    lpc_coeffs: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    lpc_residual_variance: float = 1.0
    spectral_tilt: float = 0.0
    texture_variance: float = 1.0
    harmonic_bandwidth: float = 1.0
    harmonic_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    is_transient: bool = False
    is_voiced: bool = False


class CepstralAnalyzer:
    """Performs cepstral analysis for formant envelope extraction."""
    
    def __init__(self, config: HFPv3Config):
        self.config = config
    
    def extract_envelope(self, magnitude: np.ndarray) -> np.ndarray:
        log_mag = np.log(magnitude + 1e-10)
        cepstrum = np.fft.irfft(log_mag)
        lifter_cutoff = self.config.cepstral_lifter_quefrency
        liftered = np.zeros_like(cepstrum)
        liftered[0] = cepstrum[0]
        n_low = min(lifter_cutoff, len(cepstrum) // 2)
        liftered[1:n_low] = cepstrum[1:n_low]
        if len(cepstrum) > 2 * n_low:
            liftered[-n_low+1:] = cepstrum[-n_low+1:]
        envelope = np.exp(np.fft.rfft(liftered, n=len(magnitude)*2-1).real[:len(magnitude)])
        return np.maximum(envelope, 1e-10)


class HighOrderLPC:
    """High-order LPC analysis for accurate formant representation."""
    
    def __init__(self, config: HFPv3Config):
        self.config = config
        self.min_order = 16
    
    def analyze(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(signal) < self.min_order + 1:
            return np.array([1.0]), 1.0
        order = max(self.min_order, min(self.config.lpc_order, len(signal) - 1))
        try:
            auto = np.correlate(signal, signal, mode='full')
            auto = auto[len(auto)//2:]
            coeffs, residual_var = self._levinson_durbin(auto, order)
            if not self._is_stable(coeffs):
                for reduced_order in range(order - 1, self.min_order - 1, -1):
                    coeffs, residual_var = self._levinson_durbin(auto, reduced_order)
                    if self._is_stable(coeffs):
                        break
                else:
                    return np.array([1.0]), np.var(signal)
            return coeffs, residual_var
        except Exception as e:
            logger.warning(f"LPC analysis failed: {e}")
            return np.array([1.0]), np.var(signal)
    
    def _levinson_durbin(self, autocorr: np.ndarray, order: int) -> Tuple[np.ndarray, float]:
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = autocorr[0]
        if e <= 0:
            return np.array([1.0]), 1.0
        for i in range(1, order + 1):
            lambda_val = 0.0
            for j in range(1, i):
                lambda_val += a[j] * autocorr[i - j]
            lambda_val = (autocorr[i] - lambda_val) / (e + 1e-10)
            for j in range(1, (i + 1) // 2 + 1):
                temp = a[j] - lambda_val * a[i - j]
                a[i - j] -= lambda_val * a[j]
                a[j] = temp
            a[i] = lambda_val
            e *= (1 - lambda_val ** 2)
            if e <= 0:
                break
        return a, max(e, 1e-10)
    
    def _is_stable(self, coeffs: np.ndarray) -> bool:
        if len(coeffs) < 2:
            return True
        try:
            roots = np.roots(coeffs)
            return np.all(np.abs(roots) < 0.999)
        except:
            return False
    
    def predict_envelope(self, history: np.ndarray, n_predict: int,
                         coeffs: np.ndarray, residual_var: float) -> np.ndarray:
        if len(coeffs) < 2:
            return np.zeros(n_predict)
        order = len(coeffs) - 1
        hist = np.zeros(order)
        if len(history) >= order:
            hist = history[-order:].copy()
        else:
            hist[-len(history):] = history
        predictions = []
        for _ in range(n_predict):
            pred = 0.0
            for k in range(1, order + 1):
                pred -= coeffs[k] * hist[-k]
            innovation = np.random.normal(0, np.sqrt(residual_var) * 0.5)
            val = pred + innovation
            predictions.append(val)
            hist = np.roll(hist, -1)
            hist[-1] = val
        return np.array(predictions)


class HarmonicTracker:
    """Tracks harmonic trajectories across frames."""
    
    def __init__(self, config: HFPv3Config):
        self.config = config
        self.tracks: Dict[int, HarmonicTrack] = {}
        self.n_frames_history = config.temporal_context_frames
    
    def update(self, f0_bins: float, harmonic_amplitudes: np.ndarray,
               harmonic_phases: Optional[np.ndarray] = None):
        if f0_bins <= 1.0 or len(harmonic_amplitudes) == 0:
            return
        for h_idx, amp in enumerate(harmonic_amplitudes):
            h_num = h_idx + 1
            freq_bin = f0_bins * h_num
            if h_num not in self.tracks:
                self.tracks[h_num] = HarmonicTrack()
            track = self.tracks[h_num]
            track.frequency_bins.append(freq_bin)
            track.amplitudes.append(amp)
            if harmonic_phases is not None and h_idx < len(harmonic_phases):
                track.phases.append(harmonic_phases[h_idx])
            if len(track.frequency_bins) > self.n_frames_history:
                track.frequency_bins.pop(0)
                track.amplitudes.pop(0)
                if track.phases:
                    track.phases.pop(0)
    
    def get_harmonic_velocities(self) -> np.ndarray:
        max_h = max(self.tracks.keys()) if self.tracks else 0
        velocities = np.zeros(max_h)
        for h_num, track in self.tracks.items():
            velocities[h_num - 1] = track.get_velocity()
        return velocities
    
    def clear(self):
        self.tracks.clear()


class TemporalLPCSmoother:
    """Smooths LPC coefficients across frames."""
    
    def __init__(self, config: HFPv3Config):
        self.config = config
        self.lpc_history: deque = deque(maxlen=config.temporal_context_frames)
        self.smoothed_coeffs: Optional[np.ndarray] = None
    
    def update(self, coeffs: np.ndarray) -> np.ndarray:
        self.lpc_history.append(coeffs.copy())
        if len(self.lpc_history) == 1:
            self.smoothed_coeffs = coeffs.copy()
            return self.smoothed_coeffs
        alpha = self.config.lpc_smoothing_alpha
        if self.smoothed_coeffs is None or len(self.smoothed_coeffs) != len(coeffs):
            self.smoothed_coeffs = coeffs.copy()
        else:
            if len(coeffs) == len(self.smoothed_coeffs):
                self.smoothed_coeffs = alpha * coeffs + (1 - alpha) * self.smoothed_coeffs
            else:
                self.smoothed_coeffs = coeffs.copy()
        return self.smoothed_coeffs
    
    def clear(self):
        self.lpc_history.clear()
        self.smoothed_coeffs = None


class F0Detector:
    """Robust F0 detection combining multiple methods."""
    
    def __init__(self, config: HFPv3Config):
        self.config = config
        self.f0_history: deque = deque(maxlen=config.temporal_context_frames)
    
    def detect(self, signal: np.ndarray, magnitude: np.ndarray) -> Tuple[float, float]:
        f0_auto, conf_auto = self._autocorr_f0(signal)
        f0_hps, conf_hps = self._hps_f0(magnitude)
        if conf_auto > conf_hps and conf_auto > 0.4:
            f0 = f0_auto
            confidence = conf_auto
        elif conf_hps > 0.3:
            f0 = f0_hps
            confidence = conf_hps
        else:
            f0 = 0.0
            confidence = 0.0
        if f0 > 0:
            f0 = self._temporal_smooth(f0, confidence)
        return f0, confidence
    
    def _autocorr_f0(self, signal: np.ndarray) -> Tuple[float, float]:
        sr = self.config.sr
        min_lag = int(sr / self.config.max_f0)
        max_lag = int(sr / self.config.min_f0)
        max_lag = min(max_lag, len(signal) // 2)
        if max_lag <= min_lag:
            return 0.0, 0.0
        sig = signal - np.mean(signal)
        autocorr = np.correlate(sig, sig, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        search_region = autocorr[min_lag:max_lag]
        if len(search_region) < 3:
            return 0.0, 0.0
        peaks, _ = scipy.signal.find_peaks(search_region, height=0.25)
        if len(peaks) == 0:
            return 0.0, 0.0
        best_peak = peaks[0] + min_lag
        confidence = autocorr[best_peak]
        if 0 < best_peak < len(autocorr) - 1:
            alpha, beta, gamma = autocorr[best_peak-1:best_peak+2]
            denom = alpha - 2*beta + gamma
            if abs(denom) > 1e-10:
                p = 0.5 * (alpha - gamma) / denom
                best_peak = best_peak + p
        f0_hz = sr / best_peak
        f0_bins = f0_hz * self.config.n_fft / sr
        return f0_bins, confidence
    
    def _hps_f0(self, magnitude: np.ndarray) -> Tuple[float, float]:
        hps = magnitude.copy()
        for factor in [2, 3, 4, 5]:
            decimated = scipy.signal.decimate(magnitude, factor, zero_phase=True)
            hps[:len(decimated)] *= decimated
        freq_per_bin = self.config.sr / self.config.n_fft
        min_bin = max(1, int(self.config.min_f0 / freq_per_bin))
        max_bin = min(int(self.config.max_f0 / freq_per_bin), len(hps) - 1)
        hps[:min_bin] = 0
        hps[max_bin:] = 0
        peak_idx = np.argmax(hps)
        if peak_idx == 0:
            return 0.0, 0.0
        local_mean = np.mean(hps[max(0, peak_idx-10):peak_idx+10])
        confidence = min(1.0, hps[peak_idx] / (local_mean + 1e-10) / 10)
        if 0 < peak_idx < len(hps) - 1:
            alpha, beta, gamma = hps[peak_idx-1:peak_idx+2]
            denom = alpha - 2*beta + gamma
            if abs(denom) > 1e-10:
                p = 0.5 * (alpha - gamma) / denom
                peak_idx = peak_idx + p
        return peak_idx, confidence
    
    def _temporal_smooth(self, f0: float, confidence: float) -> float:
        self.f0_history.append((f0, confidence))
        if len(self.f0_history) < 2:
            return f0
        total_weight = 0.0
        weighted_sum = 0.0
        for h_f0, h_conf in self.f0_history:
            if h_f0 > 0:
                weight = h_conf
                weighted_sum += h_f0 * weight
                total_weight += weight
        if total_weight > 0:
            return weighted_sum / total_weight
        return f0
    
    def clear(self):
        self.f0_history.clear()


class TransientDetector:
    """Detects transient frames based on spectral flux."""
    
    def __init__(self, config: HFPv3Config):
        self.config = config
        self.prev_magnitude: Optional[np.ndarray] = None
    
    def is_transient(self, magnitude: np.ndarray) -> bool:
        if self.prev_magnitude is None:
            self.prev_magnitude = magnitude.copy()
            return False
        if len(self.prev_magnitude) != len(magnitude):
            self.prev_magnitude = magnitude.copy()
            return False
        diff = magnitude - self.prev_magnitude
        flux = np.sum(np.maximum(0, diff) ** 2)
        expected_var = np.var(magnitude) + 1e-10
        is_trans = flux > self.config.transient_threshold * expected_var * len(magnitude)
        self.prev_magnitude = magnitude.copy()
        return is_trans
    
    def clear(self):
        self.prev_magnitude = None


class HighFrequencyRestorerV3:
    """
    HFP V3: Main processor combining all analysis and synthesis components.
    
    Compatible wrapper class for integration with HRAudioWizard UI.
    """
    
    def __init__(self, config: Optional[HFPv3Config] = None):
        self.config = config or HFPv3Config()
        self.cepstral_analyzer = CepstralAnalyzer(self.config)
        self.lpc_analyzer = HighOrderLPC(self.config)
        self.harmonic_tracker = HarmonicTracker(self.config)
        self.lpc_smoother = TemporalLPCSmoother(self.config)
        self.f0_detector = F0Detector(self.config)
        self.transient_detector = TransientDetector(self.config)
        self.noise_level_history: deque = deque(maxlen=5)
        self.smoothed_noise_level: float = 0.0
        self.progress_callback = None
    
    def progress_updated_emit(self, value, msg):
        """Emit progress update."""
        if self.progress_callback:
            self.progress_callback(value, msg)
    
    def run_hfp_v3(self, dat: np.ndarray, sr: int, lowpass: int = -1,
                   compressed_mode: bool = False) -> np.ndarray:
        """
        HFP V3: STFT-based High-Frequency Prediction
        
        Parameters:
        -----------
        dat : ndarray
            Stereo audio data (samples, 2)
        sr : int
            Sample rate
        lowpass : int
            Lowpass frequency in Hz (-1 for auto-detect)
        compressed_mode : bool
            Apply compressed audio fixes
        
        Returns:
        --------
        output : ndarray
            Restored stereo audio
        """
        self.config.sr = sr
        
        # Convert to mid/side
        mid = (dat[:, 0] + dat[:, 1]) / 2
        side = (dat[:, 0] - dat[:, 1]) / 2
        sample_length = len(mid)
        
        self.progress_updated_emit(5, "HFP V3: Computing STFT...")
        
        # STFT
        mid_stft = librosa.stft(mid, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
        side_stft = librosa.stft(side, n_fft=self.config.n_fft, hop_length=self.config.hop_length)
        
        # Determine cutoff bin
        freq_per_bin = sr / self.config.n_fft
        if lowpass != -1:
            cutoff_bin = int(lowpass / freq_per_bin)
        else:
            # Auto-detect
            magnitude = np.abs(mid_stft)
            avg_mag = np.mean(magnitude, axis=1)
            threshold = np.max(avg_mag) * 0.001
            cutoff_candidates = np.where(avg_mag < threshold)[0]
            if len(cutoff_candidates) > 0:
                cutoff_bin = max(64, cutoff_candidates[0])
            else:
                cutoff_bin = mid_stft.shape[0] // 2
        
        cutoff_bin = min(cutoff_bin, mid_stft.shape[0] - 1)
        
        self.progress_updated_emit(10, f"HFP V3: Cutoff at bin {cutoff_bin} ({cutoff_bin * freq_per_bin:.0f} Hz)")
        
        # Process mid and side
        self.progress_updated_emit(15, "HFP V3: Processing mid channel...")
        mid_restored = self._process_channel(mid_stft, cutoff_bin)
        
        self.progress_updated_emit(55, "HFP V3: Processing side channel...")
        side_restored = self._process_channel(side_stft, cutoff_bin)
        
        self.progress_updated_emit(95, "HFP V3: Inverse STFT...")
        
        # Reconstruct with original phase below cutoff, random phase above
        mid_phase = np.angle(mid_stft)
        side_phase = np.angle(side_stft)
        
        # Generate random phase for HF region
        mid_phase[cutoff_bin:, :] = np.random.uniform(-np.pi, np.pi, mid_phase[cutoff_bin:, :].shape)
        side_phase[cutoff_bin:, :] = np.random.uniform(-np.pi, np.pi, side_phase[cutoff_bin:, :].shape)
        
        mid_reconstructed = mid_restored * np.exp(1j * mid_phase)
        side_reconstructed = side_restored * np.exp(1j * side_phase)
        
        # ISTFT
        iffted_mid = librosa.istft(mid_reconstructed, hop_length=self.config.hop_length, length=sample_length)
        iffted_side = librosa.istft(side_reconstructed, hop_length=self.config.hop_length, length=sample_length)
        
        # Convert back to stereo
        output = np.stack([iffted_mid + iffted_side, iffted_mid - iffted_side], axis=1)
        
        self.progress_updated_emit(100, "HFP V3: Complete")
        
        return output
    
    def _process_channel(self, stft_matrix: np.ndarray, cutoff_bin: int) -> np.ndarray:
        """Process a single channel's spectrogram."""
        n_bins, n_frames = stft_matrix.shape
        magnitude = np.abs(stft_matrix)
        
        # HPSS separation
        H, P = librosa.decompose.hpss(magnitude, margin=2.5)
        
        # Reset state
        self._reset_state()
        
        # Analyze frames
        analyses: List[FrameAnalysis] = []
        for i in range(n_frames):
            analysis = self._analyze_frame(H[:, i], P[:, i], cutoff_bin)
            analyses.append(analysis)
        
        # Synthesize
        restored = np.zeros_like(magnitude)
        boundary_width = 32
        
        for i in range(n_frames):
            frame_mag = magnitude[:, i].copy()
            
            # Get boundary energy
            boundary_start = max(0, cutoff_bin - boundary_width)
            boundary_energy = np.mean(frame_mag[boundary_start:cutoff_bin] ** 2)
            boundary_level = np.sqrt(boundary_energy) if boundary_energy > 0 else 1e-10
            
            # Synthesize HF
            hf_content = self._synthesize_high_frequency(
                analyses[i], frame_mag, cutoff_bin, n_bins
            )
            
            # Energy matching
            hf_region = hf_content[cutoff_bin:]
            if len(hf_region) > 0:
                hf_energy = np.mean(hf_region ** 2)
                if hf_energy > 1e-20:
                    current_level = np.sqrt(hf_energy)
                    scale = boundary_level / (current_level + 1e-10)
                    scale = np.clip(scale, 0.1, 5.0)
                    hf_content[cutoff_bin:] *= scale
            
            # Crossfade - start BEFORE cutoff to ensure smooth transition
            # This prevents the gap when original content drops off sharply
            fade_len = min(64, n_bins - cutoff_bin)
            pre_fade = min(48, cutoff_bin)  # Start 48 bins before cutoff (~2kHz)
            if fade_len > 0:
                # Extended crossfade region
                total_fade = pre_fade + fade_len
                t = np.linspace(0, 1, total_fade)
                alpha = 0.5 * (1 - np.cos(np.pi * t))
                
                start = cutoff_bin - pre_fade
                end = cutoff_bin + fade_len
                
                # Blend: use more of original at start, more of synthesized at end
                frame_mag[start:end] = frame_mag[start:end] * (1 - alpha) + hf_content[start:end] * alpha
                if end < n_bins:
                    frame_mag[end:] = hf_content[end:]
            
            restored[:, i] = frame_mag
        
        # Temporal smoothing
        for bin_idx in range(cutoff_bin, n_bins):
            restored[bin_idx, :] = scipy.signal.medfilt(restored[bin_idx, :], kernel_size=3)
        
        return restored
    
    def _analyze_frame(self, harm_mag: np.ndarray, perc_mag: np.ndarray,
                       cutoff_bin: int) -> FrameAnalysis:
        """Analyze a single frame."""
        analysis = FrameAnalysis()
        analysis.is_transient = self.transient_detector.is_transient(harm_mag)
        
        f0_bins, confidence = self.f0_detector.detect(np.fft.irfft(harm_mag), harm_mag)
        analysis.f0 = f0_bins
        analysis.f0_confidence = confidence
        analysis.is_voiced = confidence > 0.3 and f0_bins > 1.0
        
        if not analysis.is_voiced:
            return analysis
        
        analysis_region = harm_mag[:cutoff_bin] if cutoff_bin > 0 else harm_mag
        analysis.cepstral_envelope = self.cepstral_analyzer.extract_envelope(analysis_region)
        analysis.harmonic_amplitudes = self._extract_harmonic_peaks(analysis_region, analysis.f0)
        
        if len(analysis.harmonic_amplitudes) < 3:
            return analysis
        
        log_amps = librosa.amplitude_to_db(analysis.harmonic_amplitudes + 1e-10)
        x = np.arange(len(log_amps))
        coeffs = np.polyfit(x, log_amps, 1)
        analysis.spectral_tilt = coeffs[0]
        
        trend = x * analysis.spectral_tilt + coeffs[1]
        flat_envelope = log_amps - trend
        analysis.texture_variance = max(0.01, np.std(flat_envelope))
        
        lpc_coeffs, residual_var = self.lpc_analyzer.analyze(flat_envelope.astype(np.float64))
        smoothed_coeffs = self.lpc_smoother.update(lpc_coeffs)
        analysis.lpc_coeffs = smoothed_coeffs
        analysis.lpc_residual_variance = residual_var
        
        self.harmonic_tracker.update(analysis.f0, analysis.harmonic_amplitudes)
        analysis.harmonic_velocities = self.harmonic_tracker.get_harmonic_velocities()
        
        return analysis
    
    def _extract_harmonic_peaks(self, magnitude: np.ndarray, f0_bins: float) -> np.ndarray:
        """Extract harmonic amplitudes."""
        if f0_bins <= 1.0:
            return np.array([])
        
        n_harmonics = int(len(magnitude) / f0_bins)
        search_width = max(2, int(f0_bins * self.config.harmonic_search_ratio))
        noise_floor = scipy.signal.medfilt(magnitude, kernel_size=min(31, len(magnitude) // 4 * 2 + 1))
        contrast = magnitude / (noise_floor + 1e-10)
        
        amplitudes = []
        for h in range(1, n_harmonics + 1):
            center = f0_bins * h
            if center >= len(magnitude) - 1:
                break
            start = max(0, int(center) - search_width)
            end = min(len(magnitude), int(center) + search_width + 1)
            if end <= start + 2:
                continue
            window_contrast = contrast[start:end]
            window_mag = magnitude[start:end]
            peak_idx_local = np.argmax(window_contrast)
            prominence = window_contrast[peak_idx_local]
            if prominence > 1.05:
                if 0 < peak_idx_local < len(window_mag) - 1:
                    alpha = window_mag[peak_idx_local - 1]
                    beta = window_mag[peak_idx_local]
                    gamma = window_mag[peak_idx_local + 1]
                    denom = alpha - 2 * beta + gamma
                    if abs(denom) > 1e-10:
                        p = 0.5 * (alpha - gamma) / denom
                        amp = beta - 0.25 * (alpha - gamma) * p
                    else:
                        amp = beta
                else:
                    amp = window_mag[peak_idx_local]
                amplitudes.append(amp)
            else:
                center_frac = center - int(center)
                center_int = int(center)
                if center_int < len(magnitude) - 1:
                    interp_val = magnitude[center_int] * (1 - center_frac) + magnitude[center_int + 1] * center_frac
                    amplitudes.append(interp_val)
                else:
                    amplitudes.append(magnitude[center_int])
        return np.array(amplitudes)
    
    def _synthesize_high_frequency(self, analysis: FrameAnalysis, low_freq_mag: np.ndarray,
                                    cutoff_bin: int, total_bins: int) -> np.ndarray:
        """Synthesize high frequency content."""
        hf_content = np.zeros(total_bins)
        
        # Extend synthesis region slightly below cutoff for smooth blending
        synth_start = max(0, cutoff_bin - 48)  # Start 48 bins before cutoff
        
        # Base noise - extend below cutoff
        noise_hf = self._synthesize_noise_hf(low_freq_mag, synth_start, total_bins,
                                              analysis.cepstral_envelope if analysis.is_voiced else None)
        hf_content = noise_hf.copy()
        
        if analysis.is_transient:
            transient_hf = self._synthesize_transient_hf(low_freq_mag, synth_start, total_bins)
            hf_content = np.maximum(hf_content, transient_hf * 1.5)
        elif analysis.is_voiced and len(analysis.harmonic_amplitudes) >= 2:
            harmonic_hf = self._synthesize_harmonic_hf(analysis, low_freq_mag, synth_start, total_bins)
            hf_content = hf_content + self.config.harmonic_weight * harmonic_hf
        
        hf_content = self._apply_hf_rolloff(hf_content, cutoff_bin, total_bins)
        return hf_content
    
    def _apply_hf_rolloff(self, hf_content: np.ndarray, cutoff_bin: int, total_bins: int) -> np.ndarray:
        """Apply gradual rolloff above specified frequency."""
        freq_per_bin = self.config.sr / self.config.n_fft
        rolloff_start_bin = int(self.config.hf_rolloff_start_hz / freq_per_bin)
        if rolloff_start_bin >= total_bins:
            return hf_content
        for bin_idx in range(max(cutoff_bin, rolloff_start_bin), total_bins):
            freq = bin_idx * freq_per_bin
            if freq > self.config.hf_rolloff_start_hz:
                octaves = np.log2(freq / self.config.hf_rolloff_start_hz)
                attenuation_db = self.config.hf_rolloff_rate_db_per_octave * octaves
                gain = librosa.db_to_amplitude(-attenuation_db)
                hf_content[bin_idx] *= gain
        return hf_content
    
    def _synthesize_harmonic_hf(self, analysis: FrameAnalysis, low_freq_mag: np.ndarray,
                                 cutoff_bin: int, total_bins: int) -> np.ndarray:
        """Synthesize harmonic high frequencies using LPC prediction."""
        hf = np.zeros(total_bins)
        f0 = analysis.f0
        if f0 <= 1.0:
            return hf
        
        n_existing = len(analysis.harmonic_amplitudes)
        n_target = int(total_bins / f0)
        n_new = n_target - n_existing
        if n_new <= 0:
            return hf
        
        log_amps = librosa.amplitude_to_db(analysis.harmonic_amplitudes + 1e-10)
        trend = np.arange(len(log_amps)) * analysis.spectral_tilt
        flat_history = log_amps - trend - (log_amps[0] - trend[0])
        
        predicted_flat = self.lpc_analyzer.predict_envelope(
            flat_history, n_new, analysis.lpc_coeffs, analysis.lpc_residual_variance
        )
        
        effective_tilt = np.clip(analysis.spectral_tilt, -0.25, -0.01)
        new_indices = np.arange(n_new) + n_existing
        new_trend = new_indices * effective_tilt
        
        last_known_db = log_amps[-1]
        predicted_db = predicted_flat + new_trend + last_known_db - new_trend[0]
        predicted_db = np.clip(predicted_db, -80, last_known_db)
        
        for i in range(1, len(predicted_db)):
            if predicted_db[i] > predicted_db[i-1]:
                predicted_db[i] = predicted_db[i-1]
        
        predicted_amps = librosa.db_to_amplitude(predicted_db)
        fluctuation = np.random.normal(0, analysis.texture_variance * 0.3, len(predicted_amps))
        fluctuation_factor = librosa.db_to_amplitude(fluctuation)
        predicted_amps *= fluctuation_factor
        
        sigma = max(0.5, analysis.harmonic_bandwidth if analysis.harmonic_bandwidth > 0.5 else f0 * 0.1)
        for i, amp in enumerate(predicted_amps):
            h_num = n_existing + i + 1
            freq_jitter = np.random.normal(0, 0.1)
            center_bin = int(round(f0 * h_num + freq_jitter))
            if center_bin >= total_bins:
                break
            window = int(3 * sigma)
            for j in range(-window, window + 1):
                bin_idx = center_bin + j
                if cutoff_bin <= bin_idx < total_bins:
                    weight = np.exp(-0.5 * (j / sigma) ** 2)
                    hf[bin_idx] += amp * weight
        return hf
    
    def _synthesize_transient_hf(self, low_freq_mag: np.ndarray, cutoff_bin: int, total_bins: int) -> np.ndarray:
        """Synthesize HF for transient frames."""
        hf = np.zeros(total_bins)
        if cutoff_bin <= 0 or cutoff_bin >= total_bins:
            return hf
        boundary_region = low_freq_mag[max(0, cutoff_bin-16):cutoff_bin]
        boundary_level = np.mean(boundary_region) if len(boundary_region) > 0 else 1e-6
        overall_avg = np.mean(low_freq_mag[:cutoff_bin]) if cutoff_bin > 0 else 1e-4
        boundary_level = max(boundary_level, overall_avg * 0.5, 1e-4)
        n_hf = total_bins - cutoff_bin
        noise = np.random.rand(n_hf)
        decay = np.exp(np.linspace(0, -0.7, n_hf))
        hf[cutoff_bin:] = noise * boundary_level * decay
        return hf
    
    def _synthesize_noise_hf(self, low_freq_mag: np.ndarray, cutoff_bin: int, total_bins: int,
                              cepstral_envelope: Optional[np.ndarray] = None) -> np.ndarray:
        """Synthesize noise-based HF content."""
        hf = np.zeros(total_bins)
        if cutoff_bin <= 0 or cutoff_bin >= total_bins:
            return hf
        
        n_boundary = min(16, cutoff_bin)
        boundary_region = low_freq_mag[cutoff_bin - n_boundary:cutoff_bin]
        boundary_level = np.mean(boundary_region) if len(boundary_region) > 0 else 1e-4
        overall_avg = np.mean(low_freq_mag[:cutoff_bin]) if cutoff_bin > 0 else 1e-3
        min_noise_level = max(overall_avg * 0.2, 1e-3)
        boundary_level = max(boundary_level, min_noise_level)
        
        self.noise_level_history.append(boundary_level)
        if len(self.noise_level_history) > 1:
            alpha = 0.2
            self.smoothed_noise_level = alpha * boundary_level + (1 - alpha) * self.smoothed_noise_level
        else:
            self.smoothed_noise_level = boundary_level
        
        max_recent = max(self.noise_level_history)
        boundary_level = max(self.smoothed_noise_level, max_recent * 0.7)
        
        n_estimate = min(64, cutoff_bin)
        if n_estimate > 4:
            segment = low_freq_mag[cutoff_bin - n_estimate:cutoff_bin]
            log_seg = np.log(np.maximum(segment, 1e-10))
            x = np.arange(n_estimate)
            try:
                tilt = np.polyfit(x, log_seg, 1)[0]
            except:
                tilt = -0.005
        else:
            tilt = -0.005
        
        if abs(tilt) < 0.005:
            tilt = np.clip(tilt, -0.005, 0.0)
        else:
            tilt = np.clip(tilt, -0.05, -0.001)
        
        n_hf = total_bins - cutoff_bin
        noise = np.random.rand(n_hf)
        x_hf = np.arange(n_hf)
        envelope = boundary_level * np.exp(tilt * x_hf)
        
        cutoff_hz = cutoff_bin * (self.config.sr / self.config.n_fft)
        low_cutoff_boost = np.clip(2.0 - cutoff_hz / 8000.0, 1.0, 1.0)
        
        hf[cutoff_bin:] = noise * envelope * low_cutoff_boost
        hf[cutoff_bin:] = np.minimum(hf[cutoff_bin:], boundary_level)
        
        return hf
    
    def _reset_state(self):
        """Reset all stateful components."""
        self.harmonic_tracker.clear()
        self.lpc_smoother.clear()
        self.f0_detector.clear()
        self.transient_detector.clear()
        self.noise_level_history.clear()
        self.smoothed_noise_level = 0.0
