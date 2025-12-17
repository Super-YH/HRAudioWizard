"""
Numba JIT-compiled utility functions for performance-critical operations.

These functions are accelerated using Numba's JIT compiler for
significant performance improvements in inner loops.
"""

import numpy as np
from numba import jit, prange


@jit(nopython=True, cache=True, fastmath=True)
def _numba_tukey_window(window_length, alpha):
    """
    Generate a Tukey (tapered cosine) window.
    
    Parameters:
    -----------
    window_length : int
        Length of the window
    alpha : float
        Shape parameter (0 = rectangular, 1 = Hann)
    
    Returns:
    --------
    window : ndarray
        Tukey window
    """
    if window_length == 0:
        return np.array([1.0])
    x = np.linspace(0, 1, window_length)
    window = np.ones(x.shape)
    if alpha <= 0:
        return window
    elif alpha >= 1:
        return 0.5 * (1 - np.cos(2 * np.pi * x))

    boundary = alpha / 2.0
    for i in range(window_length):
        if x[i] < boundary:
            window[i] = 0.5 * (1 - np.cos(2 * np.pi * x[i] / alpha))
        elif x[i] > (1 - boundary):
            window[i] = 0.5 * (1 - np.cos(2 * np.pi * (x[i] - 1 + alpha) / alpha))
    return window


@jit(nopython=True, cache=True, fastmath=True)
def _numba_apply_pre_echo_suppression(y_rec_padded, hop_length, transient_frames_indices):
    """
    Apply pre-echo suppression using Tukey windowing at transient frames.
    
    Parameters:
    -----------
    y_rec_padded : ndarray
        Padded audio signal
    hop_length : int
        Hop size in samples
    transient_frames_indices : ndarray
        Indices of transient frames
    
    Returns:
    --------
    y_processed : ndarray
        Processed audio with suppressed pre-echoes
    """
    y_processed = y_rec_padded.copy()
    for frame_idx in transient_frames_indices:
        pre_echo_start_sample = (frame_idx - 1) * hop_length
        pre_echo_end_sample = frame_idx * hop_length

        if pre_echo_start_sample >= 0 and pre_echo_end_sample <= len(y_processed):
            window_len = pre_echo_end_sample - pre_echo_start_sample
            if window_len > 0:
                window = _numba_tukey_window(window_len, alpha=0.5)
                y_processed[pre_echo_start_sample:pre_echo_end_sample] *= window
    return y_processed


@jit(nopython=True, cache=True, fastmath=True)
def _numba_apply_compressed_fix_loop(mid_spec, side_spec, threshold_db=-24, high_freq_bin_start=256):
    """
    Apply compression fix to spectral data.
    
    Currently a passthrough (returns input unchanged).
    Reserved for future implementation.
    
    Parameters:
    -----------
    mid_spec : ndarray
        Mid-channel spectrum
    side_spec : ndarray
        Side-channel spectrum
    threshold_db : float
        Threshold in dB (unused)
    high_freq_bin_start : int
        Starting bin for high frequency processing (unused)
    
    Returns:
    --------
    mid_spec, side_spec : tuple of ndarray
        Processed spectra (currently unchanged)
    """
    return mid_spec, side_spec


@jit(nopython=True, cache=True, fastmath=True)
def _numba_process_channel_dither(channel_data, chunk_size, b, quantization_step, lpc_order):
    """
    Apply noise-shaped dithering to audio channel.
    
    Parameters:
    -----------
    channel_data : ndarray
        Input audio samples
    chunk_size : int
        Processing chunk size
    b : ndarray
        Noise shaping filter coefficients
    quantization_step : float
        Quantization step size
    lpc_order : int
        LPC filter order
    
    Returns:
    --------
    output : ndarray
        Dithered and quantized audio
    """
    num_samples = len(channel_data)
    output = np.zeros(num_samples)
    error_hist = np.zeros(lpc_order + 1)
    
    for n in range(num_samples):
        error_shaped = np.dot(b[1:], error_hist[:lpc_order])
        
        target_sample = channel_data[n] + error_shaped
        quantized_val = np.round(target_sample / quantization_step) * quantization_step
        
        output[n] = quantized_val
        
        current_error = quantized_val - target_sample
        for j in range(lpc_order, 0, -1):
            error_hist[j] = error_hist[j-1]
        error_hist[0] = current_error
        
    return output


@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _fast_sign_with_magnitude(re_mdct_spec, magnitudes):
    """
    Extract phase and apply to magnitudes (parallelized).
    
    Parameters:
    -----------
    re_mdct_spec : ndarray
        MDCT spectrum with phase information (shape: [bins, frames])
    magnitudes : ndarray
        Target magnitudes (shape: [bins, frames])
    
    Returns:
    --------
    result : ndarray
        Magnitudes with phase from re_mdct_spec
    """
    num_bins, num_frames = re_mdct_spec.shape
    result = np.empty_like(re_mdct_spec)
    
    for j in prange(num_frames):
        for i in range(num_bins):
            phase_val = re_mdct_spec[i, j]
            if phase_val > 0:
                result[i, j] = magnitudes[i, j]
            elif phase_val < 0:
                result[i, j] = -magnitudes[i, j]
            else:
                result[i, j] = magnitudes[i, j]
    
    return result


@jit(nopython=True, cache=True, fastmath=True)
def _apply_momentum_update(current_spec, prev_spec, prev_momentum, momentum_coef):
    """
    Apply momentum update for accelerated convergence.
    
    Parameters:
    -----------
    current_spec : ndarray
        Current spectrum
    prev_spec : ndarray
        Previous iteration spectrum
    prev_momentum : ndarray
        Previous momentum values
    momentum_coef : float
        Momentum coefficient (0.95-0.99 recommended)
    
    Returns:
    --------
    result : ndarray
        Updated spectrum
    new_momentum : ndarray
        Updated momentum values
    """
    num_bins, num_frames = current_spec.shape
    new_momentum = np.empty_like(current_spec)
    result = np.empty_like(current_spec)
    
    for j in range(num_frames):
        for i in range(num_bins):
            # モーメンタム計算: v_new = β * v_old + (x_current - x_prev)
            delta = current_spec[i, j] - prev_spec[i, j]
            new_momentum[i, j] = momentum_coef * prev_momentum[i, j] + delta
            
            # 新しいスペクトル: x_new = x_current + v_new
            result[i, j] = current_spec[i, j] + new_momentum[i, j]
    
    return result, new_momentum
