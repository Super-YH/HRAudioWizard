"""
Spectral utility functions and data structures.

Contains OVERTONE dataclass and spectral connection utilities
for harmonic extrapolation.
"""

import numpy as np
import scipy.signal


class OVERTONE:
    """
    Data structure for overtone/harmonic information.
    
    Attributes:
    -----------
    base_level : float
        Base level of the overtone
    width : int
        Width in frequency bins
    fft_idx : int
        FFT frame index
    ch : int
        Channel (0=mid, 1=side)
    tone_idx : int
        Tone/peak index in spectrum
    """
    base_level = 0
    width = 1
    fft_idx = 0
    ch = 0
    tone_idx = 0


def connect_spectra_smooth(signal1, signal2, overlap_size=32, is_harm=True):
    """
    Smoothly connect two spectral signals with envelope adaptation.
    
    Uses Hilbert transform for envelope extraction and adaptive
    level matching at the connection point.
    
    Parameters:
    -----------
    signal1 : ndarray
        First signal (lower frequencies)
    signal2 : ndarray
        Second signal (higher frequencies to append)
    overlap_size : int
        Number of samples for overlap matching
    is_harm : bool
        Whether this is harmonic content (affects processing)
    
    Returns:
    --------
    result : ndarray
        Connected signal
    """
    if overlap_size == 0:
        return np.concatenate([signal1, signal2])
    
    env = scipy.signal.hilbert(signal1)
    new_env = np.hstack([
        scipy.signal.resample(env[len(env)//4:len(env)//2], len(env)),
        scipy.signal.resample(env[len(env)//4::len(env)//2][::-1], len(env)),
        scipy.signal.resample(env[len(env)//4::len(env)//2], len(env)*2), 
        scipy.signal.resample(env[len(env)//4::len(env)//2][::-1], len(env)*2),
        scipy.signal.resample(env[len(env)//4::len(env)//2], len(env)*4), 
        scipy.signal.resample(env[len(env)//4::len(env)//2][::-1], len(env)*4)
    ])
    new_env /= np.max(abs(new_env))
    new_env **= 0.1
    
    overlap_size = min(len(signal1), len(signal2), overlap_size)
    signal2_envelope_adapted = abs((new_env[:len(signal2)].real)) * signal2
    level_diff = (np.mean(signal1[-overlap_size:])+1e-5) / (np.mean(signal2_envelope_adapted[:overlap_size])+1e-5)
    signal2_adjusted = signal2_envelope_adapted * level_diff
    
    if np.mean(abs(signal1[-overlap_size:])) < 1e-5:
        signal2_adjusted = np.zeros(len(signal2_adjusted))
    
    result = np.concatenate([signal1, signal2_adjusted])
    return result


def flatten_spectrum(spectrum, window_size=5):
    """
    Apply moving average smoothing to spectrum.
    
    Parameters:
    -----------
    spectrum : ndarray
        Input spectrum
    window_size : int
        Smoothing window size
    
    Returns:
    --------
    smoothed : ndarray
        Smoothed spectrum
    """
    if window_size <= 1:
        return spectrum
    
    kernel = np.ones(window_size) / window_size
    smoothed = scipy.signal.convolve(spectrum, kernel, mode='same')
    return smoothed


def griffin_lim_stft(magnitude, n_iter=32, hop_length=None, n_fft=2048):
    """
    Griffin-Lim algorithm for STFT phase reconstruction.
    
    Parameters:
    -----------
    magnitude : ndarray
        Magnitude spectrogram (bins x frames)
    n_iter : int
        Number of iterations
    hop_length : int, optional
        Hop length (default: n_fft // 4)
    n_fft : int
        FFT size
    
    Returns:
    --------
    phase : ndarray
        Reconstructed phase spectrogram
    """
    import librosa
    
    if hop_length is None:
        hop_length = n_fft // 4
    
    # Initialize with random phase
    phase = np.exp(2j * np.pi * np.random.random(magnitude.shape))
    complex_spec = magnitude * phase
    
    for _ in range(n_iter):
        # Inverse STFT
        audio = librosa.istft(complex_spec, hop_length=hop_length)
        # Forward STFT
        complex_spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        # Preserve magnitude, update phase
        phase = np.exp(1j * np.angle(complex_spec))
        complex_spec = magnitude * phase
    
    return phase


def proc_for_compressed(mid_ffted, side_ffted, threshold_db=-48):
    """
    Apply processing for compressed audio sources.
    
    Helps restore spectral balance in previously compressed audio.
    
    Parameters:
    -----------
    mid_ffted : ndarray
        Mid channel STFT
    side_ffted : ndarray
        Side channel STFT
    threshold_db : float
        Threshold in dB for processing
    
    Returns:
    --------
    mid_ffted, side_ffted : tuple
        Processed STFT arrays
    """
    # Currently a passthrough - can be extended for specific compression artifacts
    return mid_ffted, side_ffted

