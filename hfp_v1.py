"""
High-Frequency Prediction V1 (HFP V1) module.

STFT-based high-frequency restoration algorithm with:
- HPSS (Harmonic/Percussive Source Separation)
- Overtone-based harmonic extrapolation
- Griffin-Lim phase reconstruction
"""

import numpy as np
import scipy.signal
import librosa

# Import with fallback for both direct and package execution
try:
    from .spectra_utils import connect_spectra_smooth, flatten_spectrum, griffin_lim_stft
except ImportError:
    from spectra_utils import connect_spectra_smooth, flatten_spectrum, griffin_lim_stft


class OVERTONE_V1:
    """Data structure for overtone information in V1 algorithm."""
    def __init__(self):
        self.width = 2
        self.amplitude = 0
        self.base_freq = 0
        self.slope = []
        self.loop = 0
        self.power = None


class HighFrequencyRestorerV1:
    """
    HFP V1: STFT-based High-Frequency Restoration
    
    Uses HPSS for harmonic/percussive separation and
    overtone extrapolation for high-frequency synthesis.
    """
    
    def __init__(self):
        self.progress_callback = None
    
    def progress_updated_emit(self, value, msg):
        """Emit progress update (compatibility wrapper)."""
        if self.progress_callback:
            self.progress_callback(value, msg)
        # Also support signal-style call
        pass
    
    def _remove_harmonics(self, peaks, threshold=6):
        """
        Remove harmonic peaks to keep only fundamental frequencies.
        
        Parameters:
        -----------
        peaks : array_like
            Peak indices
        threshold : int
            Frequency bin tolerance for harmonic detection
        
        Returns:
        --------
        filtered : ndarray
            Filtered peak indices (fundamentals only)
        """
        filtered = []
        peaks = np.sort(peaks)
        fft_size = 3072  # Default FFT size
        
        for idx, f in enumerate(peaks):
            if any(abs(f - f0) < threshold for f0 in filtered):
                continue
            max_k = int(fft_size / (2 * f)) if f > 0 else 1
            is_harmonic = False
            for k in range(2, max_k + 1):
                if any(abs(f * k - other) < threshold for other in peaks):
                    is_harmonic = True
                    break
            if not is_harmonic:
                filtered.append(f)
        
        return np.array(filtered)
    
    def _process_peaks(self, peaks, sample, rebuild, fft_size):
        """
        Process peaks and reconstruct harmonics.
        
        Parameters:
        -----------
        peaks : array_like
            Peak indices
        sample : ndarray
            Source spectrum
        rebuild : ndarray
            Target spectrum to fill
        fft_size : int
            FFT size
        
        Returns:
        --------
        rebuild : ndarray
            Reconstructed spectrum
        """
        for peak in peaks:
            ot = OVERTONE_V1()
            ot.base_freq = peak // 4
            
            if ot.base_freq == 0:
                continue
            
            ot.loop = (fft_size // 2) // ot.base_freq
            
            # Extract harmonic amplitude series
            harmonics = np.array([
                sample[ot.base_freq * l]
                for l in range(1, ot.loop)
                if ot.base_freq * l < len(sample)
            ])
            
            if len(harmonics) == 0 or harmonics[0] == 0:
                continue
            
            # Create envelope with Gaussian convolution
            template = scipy.signal.windows.gaussian(len(harmonics), std=len(harmonics) / 1.3)
            slope = scipy.signal.fftconvolve(harmonics / 12, template, mode='same')
            ot.slope = slope[:ot.loop * 12]  # Predict future harmonics
            
            # Determine peak width
            ot.width = 2
            for k in range(2, 3):
                if peak - k // 2 >= 0 and peak + k // 2 < len(sample):
                    if abs(abs(sample[peak - k // 2]) - abs(sample[peak + k // 2])) < 4:
                        ot.width = k
                        break
            
            # Extract power around peak
            start_power = max(peak - ot.width // 2, 0)
            end_power = min(peak + ot.width // 2, len(sample))
            ot.power = sample[start_power:end_power]
            
            # Synthesize harmonics
            for k in range(1, ot.loop * 2 + 1):
                start = int(ot.base_freq * k - ot.width // 2)
                end = int(ot.base_freq * k + ot.width // 2)
                if start < 0 or end > len(rebuild):
                    continue
                if k < len(ot.slope):
                    rebuild[start:end] = ot.power * abs(ot.slope[k - 1])
        
        return rebuild
    
    def _temporal_smoothing(self, spec, filter_size=5):
        """
        Apply temporal smoothing across frames.
        
        Parameters:
        -----------
        spec : ndarray
            Spectrogram (bins x frames)
        filter_size : int
            Base filter size
        
        Returns:
        --------
        smoothed : ndarray
            Temporally smoothed spectrogram
        """
        num_bins = spec.shape[0]
        smoothed = np.zeros_like(spec)
        
        for b in range(num_bins):
            # Higher frequencies get larger smoothing window
            cur_filter_size = int(filter_size + (b / num_bins) * filter_size)
            cur_filter_size = max(1, cur_filter_size)
            kernel = np.ones(cur_filter_size) / cur_filter_size
            smoothed[b, :] = scipy.signal.fftconvolve(spec[b, :], kernel, mode='same')
        
        return smoothed
    
    def run_hfp_v1(self, dat, sr, lowpass, compressed_mode=False, use_hpss=True,
                   temporal_filter_size=5, griffin_lim_iter=2):
        """
        HFP V1: STFT-based High-Frequency Prediction
        
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
        use_hpss : bool
            Use HPSS for harmonic/percussive separation
        temporal_filter_size : int
            Temporal smoothing filter size
        griffin_lim_iter : int
            Griffin-Lim iterations
        
        Returns:
        --------
        output : ndarray
            Restored stereo audio
        """
        # STFT parameters
        fft_size = 3072
        hop_length = fft_size // 2
        
        # Convert to mid/side
        mid = (dat[:, 0] + dat[:, 1]) / 2
        side = (dat[:, 0] - dat[:, 1]) / 2
        sample_length = len(mid)
        
        self.progress_updated_emit(5, "HFP V1: Computing STFT...")
        
        # STFT
        mid_ffted = librosa.stft(mid, n_fft=fft_size, hop_length=hop_length)
        side_ffted = librosa.stft(side, n_fft=fft_size, hop_length=hop_length)
        mid_mag, mid_phs = librosa.magphase(mid_ffted)
        side_mag, side_phs = librosa.magphase(side_ffted)
        
        # Lowpass boundary detection
        if lowpass != -1:
            lowpass_fft = int((fft_size // 2 + 1) * (lowpass / (sr / 2)))
        else:
            indices = np.where(mid_mag < 0.000001)[0]
            if len(indices) > 0:
                for i in range(len(indices)):
                    if indices[0] < 64:
                        continue
                    else:
                        lowpass_fft = int(indices[i]) - 1
                        break
            else:
                lowpass_fft = fft_size // 2
            lowpass_fft = max(0, min(lowpass_fft, fft_size // 2))
        
        self.progress_updated_emit(10, f"HFP V1: Lowpass at bin {lowpass_fft}")
        
        # HPSS
        if use_hpss:
            mid_hm, mid_pc = librosa.decompose.hpss(mid_mag, kernel_size=31)
            side_hm, side_pc = librosa.decompose.hpss(side_mag, kernel_size=31)
        else:
            mid_hm, mid_pc = mid_mag.copy(), np.zeros_like(mid_mag)
            side_hm, side_pc = side_mag.copy(), np.zeros_like(side_mag)
        
        # Frame-by-frame reconstruction
        num_frames = mid_hm.shape[1]
        
        for i in range(num_frames):
            if i % 20 == 0:
                progress = 15 + int(60 * i / num_frames)
                self.progress_updated_emit(progress, f"HFP V1: Processing frame {i}/{num_frames}")
            
            # Clear high frequencies in original
            mid_ffted[lowpass_fft:, i] = 0
            side_ffted[lowpass_fft:, i] = 0
            
            # Copy and clear components
            sample_mid_hm = mid_hm[:, i].copy()
            sample_mid_pc = mid_pc[:, i].copy()
            sample_side_hm = side_hm[:, i].copy()
            sample_side_pc = side_pc[:, i].copy()
            sample_mid_hm[lowpass_fft:] = 0
            sample_mid_pc[lowpass_fft:] = 0
            sample_side_hm[lowpass_fft:] = 0
            sample_side_pc[lowpass_fft:] = 0
            
            # Initialize rebuild arrays
            rebuild_mid = np.zeros_like(sample_mid_hm)
            rebuild_noise_mid = np.zeros_like(sample_mid_pc)
            rebuild_side = np.zeros_like(sample_mid_hm)
            rebuild_noise_side = np.zeros_like(sample_mid_pc)
            
            # Peak detection
            db_mid_hm = librosa.amplitude_to_db(sample_mid_hm, ref=np.max)
            peaks_mid_hm = scipy.signal.find_peaks(db_mid_hm, distance=4)[0]
            peaks_mid_hm = peaks_mid_hm[peaks_mid_hm > lowpass_fft // 2]
            
            db_mid_pc = librosa.amplitude_to_db(sample_mid_pc, ref=np.max)
            peaks_mid_pc = scipy.signal.find_peaks(db_mid_pc, distance=4)[0]
            peaks_mid_pc = peaks_mid_pc[peaks_mid_pc > lowpass_fft // 2]
            
            db_side_hm = librosa.amplitude_to_db(sample_side_hm, ref=np.max)
            peaks_side_hm = scipy.signal.find_peaks(db_side_hm, distance=4)[0]
            peaks_side_hm = peaks_side_hm[peaks_side_hm > lowpass_fft // 2]
            
            db_side_pc = librosa.amplitude_to_db(sample_side_pc, ref=np.max)
            peaks_side_pc = scipy.signal.find_peaks(db_side_pc, distance=4)[0]
            peaks_side_pc = peaks_side_pc[peaks_side_pc > lowpass_fft // 2]
            
            # Remove harmonics
            peaks_mid_hm = self._remove_harmonics(peaks_mid_hm)
            peaks_mid_pc = self._remove_harmonics(peaks_mid_pc)
            peaks_side_hm = self._remove_harmonics(peaks_side_hm)
            peaks_side_pc = self._remove_harmonics(peaks_side_pc)
            
            # Process peaks
            rebuild_mid = self._process_peaks(peaks_mid_hm, sample_mid_hm, rebuild_mid, fft_size)
            rebuild_noise_mid = self._process_peaks(peaks_mid_pc, sample_mid_pc, rebuild_noise_mid, fft_size)
            rebuild_side = self._process_peaks(peaks_side_hm, sample_side_hm, rebuild_side, fft_size)
            rebuild_noise_side = self._process_peaks(peaks_side_pc, sample_side_pc, rebuild_noise_side, fft_size)
            
            # Apply to output (only above lowpass)
            mid_hm[:, i] = np.concatenate([np.zeros(lowpass_fft), rebuild_mid[lowpass_fft:]])
            mid_pc[:, i] = np.concatenate([np.zeros(lowpass_fft), rebuild_noise_mid[lowpass_fft:]])
            side_hm[:, i] = np.concatenate([np.zeros(lowpass_fft), rebuild_side[lowpass_fft:]])
            side_pc[:, i] = np.concatenate([np.zeros(lowpass_fft), rebuild_noise_side[lowpass_fft:]])
            
            # Frequency smoothing with random modulation
            mid_hm[:, i] = flatten_spectrum(mid_hm[:, i], window_size=3) * np.random.uniform(0.15125, 1., mid_pc[:, i].shape)
            mid_pc[:, i] = flatten_spectrum(mid_pc[:, i], window_size=40) * np.random.uniform(0.15125, 1., mid_pc[:, i].shape)
            side_hm[:, i] = flatten_spectrum(side_hm[:, i], window_size=5) * np.random.uniform(0.15125, 1., mid_pc[:, i].shape)
            side_pc[:, i] = flatten_spectrum(side_pc[:, i], window_size=40) * np.random.uniform(0.15125, 1., mid_pc[:, i].shape)
        
        self.progress_updated_emit(80, "HFP V1: Temporal smoothing...")
        
        # Temporal smoothing
        mid_hm = self._temporal_smoothing(mid_hm, temporal_filter_size)
        side_hm = self._temporal_smoothing(side_hm, temporal_filter_size)
        
        # Phase reconstruction
        self.progress_updated_emit(85, "HFP V1: Phase reconstruction...")
        
        mid_reconstructed_mag = mid_hm + mid_pc
        side_reconstructed_mag = side_hm + side_pc
        
        mid_phs_reconstructed = griffin_lim_stft(mid_reconstructed_mag, n_iter=griffin_lim_iter,
                                                  hop_length=hop_length, n_fft=fft_size)
        side_phs_reconstructed = griffin_lim_stft(side_reconstructed_mag, n_iter=griffin_lim_iter,
                                                   hop_length=hop_length, n_fft=fft_size)
        
        # Replace phase above lowpass
        mid_phs[lowpass_fft:] = mid_phs_reconstructed[lowpass_fft:]
        side_phs[lowpass_fft:] = side_phs_reconstructed[lowpass_fft:]
        
        rebuilt_mid = mid_reconstructed_mag * np.exp(1j * np.angle(mid_phs))
        rebuilt_side = side_reconstructed_mag * np.exp(1j * np.angle(side_phs))
        
        # Connect spectra smoothly - use magnitude for connection, then apply phase
        fade = np.linspace(1, 0.25, rebuilt_mid.shape[0] - lowpass_fft) ** 3
        for i in range(mid_ffted.shape[1]):
            # Get magnitude of low frequencies
            low_mid_mag = np.abs(mid_ffted[:lowpass_fft, i])
            low_side_mag = np.abs(side_ffted[:lowpass_fft, i])
            
            # Get magnitude of rebuilt high frequencies
            high_mid_mag = np.abs(rebuilt_mid[lowpass_fft:, i]) * fade
            high_side_mag = np.abs(rebuilt_side[lowpass_fft:, i]) * fade
            
            # Connect magnitudes (real arrays)
            connected_mid_mag = np.concatenate([low_mid_mag, high_mid_mag])
            connected_side_mag = np.concatenate([low_side_mag, high_side_mag])
            
            # Get phase: use original for low, reconstructed for high
            low_mid_phase = np.angle(mid_ffted[:lowpass_fft, i])
            low_side_phase = np.angle(side_ffted[:lowpass_fft, i])
            high_mid_phase = np.angle(rebuilt_mid[lowpass_fft:, i])
            high_side_phase = np.angle(rebuilt_side[lowpass_fft:, i])
            
            connected_mid_phase = np.concatenate([low_mid_phase, high_mid_phase])
            connected_side_phase = np.concatenate([low_side_phase, high_side_phase])
            
            # Reconstruct complex spectrum
            mid_ffted[:, i] = connected_mid_mag * np.exp(1j * connected_mid_phase)
            side_ffted[:, i] = connected_side_mag * np.exp(1j * connected_side_phase)
        
        self.progress_updated_emit(95, "HFP V1: Inverse STFT...")
        
        # Inverse STFT
        iffted_mid = librosa.istft(mid_ffted, hop_length=hop_length)
        iffted_side = librosa.istft(side_ffted, hop_length=hop_length)
        
        # Length adjustment
        if len(iffted_mid) > sample_length:
            iffted_mid = iffted_mid[:sample_length]
            iffted_side = iffted_side[:sample_length]
        elif len(iffted_mid) < sample_length:
            padding = sample_length - len(iffted_mid)
            iffted_mid = np.pad(iffted_mid, (0, padding))
            iffted_side = np.pad(iffted_side, (0, padding))
        
        # Convert back to stereo
        output = np.array([iffted_mid + iffted_side, iffted_mid - iffted_side]).T
        
        self.progress_updated_emit(100, "HFP V1: Complete")
        
        return output

