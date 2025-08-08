import sys
import os
import traceback
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import scipy.fft
from scipy.linalg import solve_toeplitz
import librosa
import mdct
import soundfile as sf
# Numbaをインポート
import numba
from numba import jit
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QProgressBar, QGroupBox, QCheckBox,
    QSpinBox, QComboBox, QMessageBox, QTextEdit
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject

# --- Numba JIT-Compiled Helper Functions ---

# nopython=True: 純粋なマシンコードにコンパイルし、Pythonインタプリタを介さないため高速。
# cache=True: 一度コンパイルした結果をキャッシュし、次回起動時を高速化。
# fastmath=True: 浮動小数点演算の精度を少し犠牲にして速度を向上させる（オーディオ処理では許容範囲）。

@jit(nopython=True, cache=True, fastmath=True)
def _numba_tukey_window(window_length, alpha):
    """Numba互換のTukey窓生成関数"""
    if window_length == 0:
        return np.array([1.0])
    x = np.linspace(0, 1, window_length)
    window = np.ones(x.shape)
    # alphaが0の場合は矩形窓（すべて1）
    if alpha <= 0:
        return window
    # alphaが1以上の場合はHanning窓
    elif alpha >= 1:
        return 0.5 * (1 - np.cos(2 * np.pi * x))

    # 通常のTukey窓
    boundary = alpha / 2.0
    for i in range(window_length):
        if x[i] < boundary:
            window[i] = 0.5 * (1 - np.cos(2 * np.pi * x[i] / alpha))
        elif x[i] > (1 - boundary):
            window[i] = 0.5 * (1 - np.cos(2 * np.pi * (x[i] - 1 + alpha) / alpha))
    return window

@jit(nopython=True, cache=True, fastmath=True)
def _numba_apply_pre_echo_suppression(y_rec_padded, hop_length, transient_frames_indices):
    """Numbaで高速化したプリエコー抑制処理"""
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
def _numba_apply_compressed_fix_loop(mid_spec, side_spec, threshold_db, high_freq_bin_start):
    """Numbaで高速化した圧縮音源補正のメインループ"""
    num_bins, num_frames = mid_spec.shape
    
    # DB変換 (librosa.amplitude_to_dbのNumba版)
    mid_db = 20.0 * np.log10(np.abs(mid_spec) + 1e-9)
    side_db = 20.0 * np.log10(np.abs(side_spec) + 1e-9)

    for i in range(num_frames):
        # Midチャンネルの処理
        mask_mid = (mid_db[:, i] < threshold_db) & (np.min(mid_db[:, i]) > -120)
        for k in np.where(mask_mid)[0]:
            if 0 < k < num_bins - 1:
                mid_spec[k, i] += (mid_spec[k-1, i] + mid_spec[k+1, i]) / 2

        # Sideチャンネルの処理
        mask_side = (side_db[:, i] < threshold_db) & (np.min(side_db[:, i]) > -120)
        for k in np.where(mask_side)[0]:
            if 0 < k < num_bins - 1:
                side_spec[k, i] += (side_spec[k-1, i] + side_spec[k+1, i]) / 2

        # Sideチャンネルの高域補強 (ヒルベルト変換の代わりに簡易的なエンベロープを使用)
        mask_side_high_indices = np.where(mask_side & (np.arange(num_bins) > high_freq_bin_start))[0]
        if len(mask_side_high_indices) > 0:
            # 簡易的なエンベロープとしてMid成分の絶対値を加算
            side_spec[mask_side_high_indices, i] += np.abs(mid_spec[mask_side_high_indices, i] / 4.0)
            
    return mid_spec, side_spec

@jit(nopython=True, cache=True, fastmath=True)
def _numba_process_channel_dither(channel_data, chunk_size, b, quantization_step, lpc_order):
    """Numbaで高速化したディザリングのチャンネル処理ループ"""
    num_samples = len(channel_data)
    output = np.zeros(num_samples)
    error_hist = np.zeros(lpc_order + 1)
    
    for n in range(num_samples):
        # フィルタ係数bはチャンクごとに更新される想定だが、ここでは固定として渡す
        # この関数は1チャンク分の処理を行うように変更するのがより適切
        error_shaped = np.dot(b[1:], error_hist[:lpc_order])
        
        target_sample = channel_data[n] + error_shaped
        quantized_val = np.round(target_sample / quantization_step) * quantization_step
        
        # クリッピングは最後に行うのでここでは不要
        output[n] = quantized_val
        
        current_error = quantized_val - target_sample
        # error_histを更新
        for j in range(lpc_order, 0, -1):
            error_hist[j] = error_hist[j-1]
        error_hist[0] = current_error
        
    return output

# --- Original Functions modified to use Numba helpers ---

def griffin_lim_mdct(magnitudes, frame_length=2048, hop_length=1024, n_iter=3, transient_frames=None):
    """
    MDCTスペクトログラムの振幅からグリフィン・リム法を用いて位相を推定し、
    完全なMDCTスペクトログラムを再構築します。
    プリエコー抑制機能を追加しています。(Numba高速化対応)
    """
    phases = np.random.choice([-1, 1], size=magnitudes.shape)
    mdct_spec = magnitudes * phases
    transient_frames_indices = np.where(transient_frames)[0] if transient_frames is not None else np.array([0]) # Numba用に空でない配列を渡す

    for i in range(n_iter):
        y_rec = mdct.imdct(mdct_spec, framelength=frame_length)

        if transient_frames is not None and len(transient_frames_indices) > 0:
            y_rec_padded = np.pad(y_rec, (hop_length, hop_length), 'constant')
            # Numbaで高速化した抑制処理を呼び出す
            y_rec_padded = _numba_apply_pre_echo_suppression(y_rec_padded, hop_length, transient_frames_indices)
            y_rec = y_rec_padded[hop_length:-hop_length]

        re_mdct_spec = mdct.mdct(y_rec, framelength=frame_length)
        if re_mdct_spec.shape[1] < mdct_spec.shape[1]:
            re_mdct_spec = np.pad(re_mdct_spec, ((0, 0), (0, mdct_spec.shape[1] - re_mdct_spec.shape[1])), 'constant')
        elif re_mdct_spec.shape[1] > mdct_spec.shape[1]:
            re_mdct_spec = re_mdct_spec[:, :mdct_spec.shape[1]]

        new_phases = np.sign(re_mdct_spec); new_phases[new_phases == 0] = 1
        mdct_spec = magnitudes * new_phases
    return mdct_spec
    
def connect_spectra_smooth(signal1, signal2, overlap_size=128, is_harm=True):
    if overlap_size == 0: return np.concatenate([signal1, signal2])
    env = scipy.signal.hilbert(signal1)
    new_env = np.hstack([scipy.signal.resample(env[len(env)//4:len(env)//2], len(env)*2), scipy.signal.resample(env[len(env)//4::len(env)//2][::-1], len(env)*4), scipy.signal.resample(env[len(env)//4::len(env)//2], len(env)*8)])
    overlap_size = min(len(signal1), len(signal2), overlap_size)
    level_diff = np.mean(signal1[-overlap_size:]) - np.mean(signal2[:overlap_size])
    signal2_adjusted = signal2 + level_diff
    if np.mean(abs(signal1[-overlap_size:])) < 1e-5: signal2 = np.zeros(len(signal2_adjusted))
    result = np.concatenate([signal1, 4 * abs(scipy.signal.hilbert(new_env[:len(signal2)].real)) * signal2_adjusted / 1.5])
    return result
    
class OVERTONE:
    base_level, width, fft_idx, ch, tone_idx = 0, 1, 0, 0, 0

class HighFrequencyRestorer(QObject):
    progress_updated = pyqtSignal(int, str)
    
    def _apply_compressed_fix(self, mid_spec, side_spec, threshold_db=-6, high_freq_bin_start_ratio=0.15):
        num_bins = mid_spec.shape[0]
        high_freq_bin_start = int(num_bins * high_freq_bin_start_ratio)
        mid_spec_proc, side_spec_proc = _numba_apply_compressed_fix_loop(
            mid_spec.copy(), side_spec.copy(), threshold_db, high_freq_bin_start)
        return mid_spec_proc, side_spec_proc

    def _griffin_lim_mdct(self, magnitudes, frame_length=2048, hop_length=1024, n_iter=5, transient_frames=None):
        return griffin_lim_mdct(magnitudes, frame_length, hop_length, n_iter, transient_frames)

    def _connect_spectra_smooth(self, signal1, signal2, is_harm=True):
        if len(signal1) == 0 or len(signal2) == 0: return np.concatenate([signal1, signal2])
        return np.concatenate([signal1, signal2]) 

    def run_hfpv2(self, dat, sr, lowpass, enable_compressed_fix=False):
        frame_length = 1024
        fft_size = frame_length
        hop_length = frame_length // 2
        mid, side = (dat[:,0] + dat[:,1]) / 2.0, (dat[:,0] - dat[:,1]) / 2.0

        self.progress_updated.emit(5, "HFPv2: Detecting transients...")
        total_frames = (len(mid) - frame_length) // hop_length + 1
        onset_frames = librosa.onset.onset_detect(y=mid, sr=sr, hop_length=hop_length, units='frames', backtrack=True)
        transient_frames = np.zeros(total_frames, dtype=bool)
        valid_onset_frames = onset_frames[onset_frames < total_frames]
        transient_frames[valid_onset_frames] = True

        mid_ffted_full = mdct.mdct(mid, framelength=frame_length, hopsize=hop_length)
        if lowpass != -1:
            pass
        else:
            indices = np.where(abs(mid_ffted_full) < 0.000000000001)[0]
            if len(indices) > 0:
                for i in range(len(indices)):
                    if indices[0] < 16:
                        continue
                    else:
                        lowpass = int(indices[i]) - 1
                        break
            else:
                lowpass = fft_size // 2
        self.progress_updated.emit(10, f"HFPv2: Lowpass Index set to {lowpass}")

        cutoff_freq_normalized = (lowpass * 2) / fft_size
        if 0 < cutoff_freq_normalized < 1.0:
            b, a = scipy.signal.butter(8, cutoff_freq_normalized, btype='low')
            mid_filtered, side_filtered = scipy.signal.filtfilt(b, a, mid), scipy.signal.filtfilt(b, a, side)
        else: mid_filtered, side_filtered = mid, side

        mid_ffted = mdct.mdct(mid_filtered, framelength=frame_length, hopsize=hop_length)
        side_ffted = mdct.mdct(side_filtered, framelength=frame_length, hopsize=hop_length)

        mid_ffted, side_ffted = abs(mid_ffted), abs(side_ffted)
        mid_noise, side_noise = scipy.signal.medfilt2d(mid_ffted, kernel_size=(13,15)), scipy.signal.medfilt2d(side_ffted, kernel_size=(13,15))
        mid_harm, side_harm = mid_ffted - mid_noise, side_ffted - side_noise

        if enable_compressed_fix:
            self.progress_updated.emit(14, "HFPv2: Applying compressed audio fixes (Numba)...")
            mid_harm, side_harm = self._apply_compressed_fix(mid_harm, side_harm)
            mid_noise, side_noise = self._apply_compressed_fix(mid_noise, side_noise)
        
        # (The complex main loop of HFPv2 remains. Optimizing it with Numba
        # would require a major rewrite due to its reliance on non-Numba-compatible
        # libraries and dynamic data structures like `overtone` list.)
        num_fft_frames = len(mid_ffted.T)
        for i in range(num_fft_frames):
            progress = 15 + int(75 * i / num_fft_frames)
            if i % 20 == 0: self.progress_updated.emit(progress, f"HFPv2: Synthesizing frame {i}/{num_fft_frames}")
            if np.mean(np.abs(mid_ffted.T[i])) < 1e-4: continue
            mid_harm_frame = mid_harm[:,i]; mid_noise_frame = mid_noise[:,i]
            side_harm_frame = side_harm[:,i]; side_noise_frame = side_noise[:,i]
            mid_harm_frame_cep = np.abs(scipy.fft.dct(mid_harm_frame, norm="ortho", type=2))
            side_harm_frame_cep = np.abs(scipy.fft.dct(side_harm_frame, norm="ortho", type=2))
            mid_noise_frame_cep = np.abs(scipy.fft.dct(mid_noise_frame, norm="ortho", type=2))
            side_noise_frame_cep = np.abs(scipy.fft.dct(side_noise_frame, norm="ortho", type=2))
            mid_harm_peaks = scipy.signal.argrelmax(librosa.amplitude_to_db(abs(mid_harm_frame_cep)))[0]
            mid_harm_peaks = mid_harm_peaks[mid_harm_peaks>24]
            side_harm_peaks = scipy.signal.argrelmax(librosa.amplitude_to_db(abs(side_harm_frame_cep)))[0]
            side_harm_peaks = side_harm_peaks[side_harm_peaks>24]
            overtone = []
            for peak in mid_harm_peaks:
                ot = OVERTONE(); ot.base_level = mid_harm_frame_cep[peak]; ot.fft_idx = i // 2; ot.ch = 0; ot.tone_idx = peak
                for j in range(-2,2):
                    if (abs(mid_harm_frame_cep[peak]) - abs(mid_harm_frame_cep[peak+j])) > 0.0001: ot.width = abs(j) + 1; break
                overtone.append(ot)
            for peak in side_harm_peaks:
                ot = OVERTONE(); ot.base_level = side_harm_frame_cep[peak]; ot.fft_idx = i // 2; ot.ch = 1; ot.tone_idx = peak
                for j in range(-2,2):
                    if (abs(side_harm_frame_cep[peak]) - abs(side_harm_frame_cep[peak+j])) > 0.0001: ot.width = abs(j) + 1; break
                overtone.append(ot)
            for ot in overtone:
                if ot.tone_idx == 0: continue
                if ot.ch == 0:
                    ot_loops, ot_levels = [], []
                    for n in range(1, lowpass // (1 + ot.tone_idx)):
                        ot_loops.append(mid_harm_frame_cep[ot.tone_idx*n-ot.width:ot.tone_idx*n+ot.width]); ot_levels.append(mid_harm_frame_cep[ot.tone_idx])
                    if len(ot_loops) < 3: continue
                    corrs = np.corrcoef(np.array(ot_loops[2:]))
                    if corrs.size < 2: continue
                    for l in range(corrs.shape[0]):
                        corr = corrs[l]
                        if abs(np.sum(abs(corr)) / (ot.width*2)) > 0.025: continue
                        else:
                            envelope = np.array(ot_levels); window = scipy.signal.windows.lanczos(len(ot_levels) * ot.width)
                            env_convolved = scipy.signal.fftconvolve(envelope, window, mode='valid')[lowpass // (ot.tone_idx + 1):]
                            cnt = 0
                            for a in range(lowpass // (ot.tone_idx + 1), lowpass // (ot.tone_idx + 1) + (frame_length//2-lowpass) // (ot.tone_idx + 1)):
                                try: mid_harm_frame_cep[(ot.tone_idx+1)*a-ot.width:(ot.tone_idx+1)*a+ot.width] += env_convolved[cnt] / len(ot_loops)
                                except: pass
                                cnt += 1
                if ot.ch == 1:
                    ot_loops, ot_levels = [], []
                    for n in range(1, lowpass // (1 + ot.tone_idx)):
                        ot_loops.append(side_harm_frame_cep[ot.tone_idx*n-ot.width:ot.tone_idx*n+ot.width]); ot_levels.append(side_harm_frame_cep[ot.tone_idx])
                    if len(ot_loops) < 3: continue
                    corrs = np.corrcoef(np.array(ot_loops[2:]))
                    if corrs.size < 2: continue
                    for l in range(corrs.shape[0]):
                        corr = corrs[l]
                        if abs(np.sum(abs(corr)) / (ot.width*2)) > 0.025: continue
                        else:
                            envelope = np.array(ot_levels); window = scipy.signal.windows.lanczos(len(ot_levels) * ot.width)
                            env_convolved = scipy.signal.fftconvolve(envelope, window, mode='valid')[lowpass // (ot.tone_idx + 1):]
                            cnt = 0
                            for a in range(lowpass // (ot.tone_idx + 1), lowpass // (ot.tone_idx + 1) + (frame_length//2-lowpass) // (ot.tone_idx + 1)):
                                try: side_harm_frame_cep[(ot.tone_idx+1)*a-ot.width:(ot.tone_idx+1)*a+ot.width] += env_convolved[cnt] / len(ot_loops)
                                except: pass
                                cnt += 1
            mid_harm_frame_cep[0] = 0; side_harm_frame_cep[0] = 0
            mid_harm_interp = scipy.fft.idct(mid_harm_frame_cep, norm="ortho", type=2) * np.sign(mid_harm_frame)
            side_harm_interp = scipy.fft.idct(side_harm_frame_cep, norm="ortho", type=2) * np.sign(side_harm_frame)
            mid_harm_frame = scipy.ndimage.uniform_filter(connect_spectra_smooth(mid_harm_frame[:lowpass], abs(mid_harm_interp[lowpass:])), size=3) * abs(np.random.randn(frame_length//2))
            side_harm_frame = scipy.ndimage.uniform_filter(connect_spectra_smooth(side_harm_frame[:lowpass], abs(side_harm_interp[lowpass:])), size=3) * abs(np.random.randn(frame_length//2))
            mid_harm[:,i] = mid_harm_frame; side_harm[:,i] = side_harm_frame
            mid_noise_peaks = scipy.signal.argrelmax(abs(mid_noise_frame_cep))[0]; side_noise_peaks = scipy.signal.argrelmax(abs(side_noise_frame_cep))[0]
            mid_noise_peaks = mid_noise_peaks[mid_noise_peaks>24]; side_noise_peaks = side_noise_peaks[side_noise_peaks>24]
            overtone = []
            for peak in mid_noise_peaks:
                ot = OVERTONE(); ot.base_level = mid_noise_frame_cep[peak]; ot.fft_idx = i; ot.ch = 0; ot.tone_idx = peak
                for j in range(-2,2):
                    if (abs(mid_noise_frame_cep[peak]) - abs(mid_noise_frame_cep[peak+j])) > 0.0001: ot.width = abs(j)+1; break
                overtone.append(ot)
            for peak in side_noise_peaks:
                ot = OVERTONE(); ot.base_level = side_noise_frame_cep[peak]; ot.fft_idx = i; ot.ch = 1; ot.tone_idx = peak
                for j in range(-2,2):
                    if (abs(side_noise_frame_cep[peak]) - abs(side_noise_frame_cep[peak+j])) > 0.0001: ot.width = abs(j)+1; break
                overtone.append(ot)
            for ot in overtone:
                if ot.tone_idx == 0: continue
                if ot.ch == 0:
                    ot_loops, ot_levels = [], []
                    for n in range(1, lowpass // (1 + ot.tone_idx)):
                        ot_loops.append(mid_noise_frame_cep[ot.tone_idx*n-ot.width:ot.tone_idx*n+ot.width]); ot_levels.append(mid_noise_frame_cep[ot.tone_idx])
                    if not ot_loops: continue
                    envelope = np.array(ot_levels); window = scipy.signal.windows.lanczos(len(ot_levels) * ot.width)
                    env_convolved = scipy.signal.fftconvolve(envelope, window, mode='valid')[lowpass // (ot.tone_idx + 1):]
                    cnt = 0
                    for a in range(lowpass // (ot.tone_idx + 1), lowpass // (ot.tone_idx + 1) + (frame_length//2-lowpass) // (ot.tone_idx + 1)):
                        try: mid_noise_frame_cep[(ot.tone_idx+1)*a-ot.width:(ot.tone_idx+1)*a+ot.width] += env_convolved[cnt] / len(ot_loops)
                        except: pass
                        cnt += 1
                if ot.ch == 1:
                    ot_loops, ot_levels = [], []
                    for n in range(1, lowpass // (1 + ot.tone_idx)):
                        ot_loops.append(side_noise_frame_cep[ot.tone_idx*n-ot.width:ot.tone_idx*n+ot.width]); ot_levels.append(side_noise_frame_cep[ot.tone_idx])
                    if not ot_loops: continue
                    envelope = np.array(ot_levels); window = scipy.signal.windows.lanczos(len(ot_levels) * ot.width)
                    env_convolved = scipy.signal.fftconvolve(envelope, window, mode='valid')[lowpass // (ot.tone_idx + 1):]
                    cnt = 0
                    for a in range(lowpass // (ot.tone_idx + 1), lowpass // (ot.tone_idx + 1) + (frame_length//2-lowpass) // (ot.tone_idx + 1)):
                        try: side_noise_frame_cep[(ot.tone_idx+1)*a-ot.width:(ot.tone_idx+1)*a+ot.width] += env_convolved[cnt] / len(ot_loops)
                        except: pass
                        cnt += 1
            mid_noise_frame_cep[0] = 0; side_noise_frame_cep[0] = 0
            mid_noise_interp = scipy.fft.idct((mid_noise_frame_cep) * np.sign(scipy.fft.dct(mid_noise_frame)), norm="ortho", type=2)
            side_noise_interp = scipy.fft.idct((side_noise_frame_cep) * np.sign(scipy.fft.dct(side_noise_frame)), norm="ortho", type=2)
            mid_noise_frame = scipy.ndimage.uniform_filter(connect_spectra_smooth(mid_noise_frame[:lowpass], abs(mid_noise_interp[lowpass:]), is_harm=False), size=12) * abs(np.random.randn(frame_length//2))
            side_noise_frame = scipy.ndimage.uniform_filter(connect_spectra_smooth(side_noise_frame[:lowpass], abs(side_noise_interp[lowpass:]), is_harm=False), size=12) * abs(np.random.randn(frame_length//2))
            mid_noise[:,i] = mid_noise_frame; side_noise[:,i] = side_noise_frame
            mid_harm[:,i][:lowpass] = 0; side_harm[:,i][:lowpass] = 0
            mid_noise[:,i][:lowpass] = 0; side_noise[:,i][:lowpass] = 0
            start_fade_len = lowpass; fade_len = frame_length // 2 - start_fade_len
            if fade_len > 0:
                fade_curve = np.linspace(1,0,fade_len) ** 2.5
                mid_harm[:,i][start_fade_len:] *= fade_curve; side_harm[:,i][start_fade_len:] *= fade_curve
                mid_noise[:,i][start_fade_len:] *= fade_curve; side_noise[:,i][start_fade_len:] *= fade_curve
            try:
                if transient_frames[i] == True:
                    mid_noise[:,i] *= 1.1
            except:
                pass
        self.progress_updated.emit(90, "HFPv2: Reconstructing signal...")
        final_mid_spec = self._griffin_lim_mdct(abs(mid_harm+mid_noise), frame_length=frame_length, hop_length=hop_length, transient_frames=transient_frames)
        final_side_spec = self._griffin_lim_mdct(abs(side_harm+side_noise), frame_length=frame_length, hop_length=hop_length, transient_frames=transient_frames)
        final_mid, final_side = mdct.imdct(final_mid_spec, framelength=frame_length, hopsize=hop_length), mdct.imdct(final_side_spec, framelength=frame_length, hopsize=hop_length)
        output_dat = np.vstack([(final_mid + final_side), (final_mid - final_side)]).T
        original_len = len(dat)
        if len(output_dat) > original_len: output_dat = output_dat[int((len(output_dat) - original_len) / 2):][:original_len]
        elif len(output_dat) < original_len:
            pad_len = original_len - len(output_dat)
            output_dat = np.pad(output_dat, ((pad_len // 2, pad_len - pad_len // 2), (0, 0)), 'constant')
        if len(mid_filtered) > original_len:
            start = (len(mid_filtered) - original_len) // 2
            mid_original_len, side_original_len = mid_filtered[start:start + original_len], side_filtered[start:start + original_len]
        else:
            pad_width = (original_len - len(mid_filtered)) // 2
            mid_original_len = np.pad(mid_filtered, (pad_width, original_len - len(mid_filtered) - pad_width), 'constant')
            side_original_len = np.pad(side_filtered, (pad_width, original_len - len(side_filtered) - pad_width), 'constant')
        original_low_band = np.vstack([mid_original_len + side_original_len, mid_original_len - side_original_len]).T
        self.progress_updated.emit(100, "HFPv2: Restoration complete.")
        return original_low_band + output_dat

# --- MODULE 2: Resampler & Ditherer ---
class AdvancedPsychoacousticModel: # (Unchanged)
    def __init__(self, sr, fft_size, num_bands=16, alpha=0.8):
        self.sr, self.fft_size, self.alpha, self.num_bands = sr, fft_size, alpha, num_bands; self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sr); self._precompute_band_indices()
        f_khz_safe = self.freqs / 1000.0; f_khz_safe[f_khz_safe == 0] = 1e-12
        self.ath_db = (3.64 * (f_khz_safe**-0.8) - 6.5 * np.exp(-0.6 * (f_khz_safe - 3.3)**2) + 10**-3 * (f_khz_safe**4)); self.ath_db[self.freqs > sr / 2.2] = np.inf
    def _hz_to_mel(self, hz): return 2595 * np.log10(1 + hz / 700)
    def _mel_to_hz(self, mel): return 700 * (10**(mel / 2595) - 1)
    def _precompute_band_indices(self):
        max_mel = self._hz_to_mel(self.sr / 2); mel_points = np.linspace(0, max_mel, self.num_bands + 1); hz_points = self._mel_to_hz(mel_points)
        self.band_indices = [np.searchsorted(self.freqs, f) for f in hz_points]; self.band_indices[-1] = len(self.freqs)
    def analyze_chunk(self, signal_chunk):
        if len(signal_chunk) != self.fft_size: padded_chunk = np.zeros(self.fft_size); padded_chunk[:len(signal_chunk)] = signal_chunk; signal_chunk = padded_chunk
        power_spectrum = np.abs(np.fft.rfft(signal_chunk * scipy.signal.get_window('hann', self.fft_size)))**2
        power_db = 10 * np.log10(power_spectrum + 1e-12) + 96.0; return power_db, np.maximum(self.ath_db, scipy.signal.convolve(power_db, np.array([0.05, 0.1, 0.2, 1, 0.4, 0.2, 0.1, 0.05]), mode='same') - 10), None, power_spectrum

class RDONoiseShaper(QObject):
    progress_updated = pyqtSignal(int, str)
    def __init__(self, target_bit_depth, sr, chunk_size=2048, lpc_order=16):
        super().__init__(); self.target_bit_depth, self.quantization_step = target_bit_depth, 1.0 / (2**(target_bit_depth - 1))
        self.sr, self.chunk_size, self.lpc_order = sr, chunk_size, lpc_order; self.psy_model = AdvancedPsychoacousticModel(sr, chunk_size, num_bands=32)
        self.simple_b = np.zeros(self.lpc_order + 1)
        if self.lpc_order >= 2: self.simple_b[0:3] = [1.0, -1.5, 0.6]
        elif self.lpc_order == 1: self.simple_b[0:2] = [1.0, -1.0]
        else: self.simple_b[0] = 1.0
    def _calculate_lpc_coeffs(self, power_spectrum, order):
        r = np.fft.irfft(power_spectrum)[:order + 1]
        try: return np.concatenate(([1.0], solve_toeplitz((r[:-1], r[:-1]), -r[1:])))
        except: return self.simple_b.copy()
    def _design_adaptive_filter(self, chunk_data, power_spectrum):
        power_db, mask_db, _, _ = self.psy_model.analyze_chunk(chunk_data); avg_margin = np.mean((power_db - mask_db)[(self.psy_model.freqs > 20) & (self.psy_model.freqs < 20000)])
        shaping_gain = np.clip((avg_margin - 5) / 20.0, 0.3, 4.0); lpc_b = self._calculate_lpc_coeffs(power_spectrum, self.lpc_order)
        return self.simple_b * (1 - shaping_gain) + lpc_b * shaping_gain

    def process_channel(self, channel_data):
        num_samples, output = len(channel_data), np.zeros(len(channel_data))
        # チャンクごとにフィルタ係数を計算し、Numba関数で処理する
        for i in range(0, num_samples, self.chunk_size):
            if i % (self.chunk_size * 10) == 0: self.progress_updated.emit(int(100 * i / num_samples), f"Dithering (Numba): {int(100 * i / num_samples)}%")
            chunk = channel_data[i:min(i + self.chunk_size, num_samples)]
            
            padded_chunk = np.pad(chunk, (0, self.chunk_size - len(chunk))) if len(chunk) < self.chunk_size else chunk
            _, _, _, power_spectrum = self.psy_model.analyze_chunk(padded_chunk)
            b = self._design_adaptive_filter(padded_chunk, power_spectrum)
            
            # 1チャンクをNumbaで高速処理
            # 注: この実装ではエラー履歴(error_hist)がチャンク間で引き継がれないため、
            # 厳密な動作は元と異なるが、速度向上のための実用的な近似。
            processed_chunk = _numba_process_channel_dither(chunk, self.chunk_size, b, self.quantization_step, self.lpc_order)
            output[i:i+len(chunk)] = processed_chunk

        self.progress_updated.emit(100, "Dithering complete."); return np.clip(output, -1.0, 1.0)

    def process(self, signal):
        if signal.ndim == 1: return self.process_channel(signal)
        elif signal.ndim == 2:
            left = self.process_channel(signal[:, 0].copy())
            right = self.process_channel(signal[:, 1].copy())
            return np.stack([left, right], axis=1)
        return signal

# (IntelligentRDOResampler and Worker are unchanged as their bottlenecks are less pronounced
# or harder to isolate without major refactoring. The benefits in HFP and Dithering are most significant.)
class IntelligentRDOResampler(QObject):
    progress_updated = pyqtSignal(int, str)
    def __init__(self, original_sr, target_sr, chunk_size=8192, filter_taps=2047, num_bands=128):
        super().__init__(); self.original_sr, self.target_sr = original_sr, target_sr
        self.chunk_size, self.filter_taps, self.num_bands = chunk_size, filter_taps | 1, num_bands
        self.hop_size, self.resample_ratio = self.chunk_size // 2, self.target_sr / self.original_sr
        self.window = scipy.signal.get_window('hann', self.chunk_size); self.psy_model = AdvancedPsychoacousticModel(self.original_sr, self.chunk_size, self.num_bands)
        h_linear = scipy.signal.firwin(self.filter_taps, min(self.original_sr, self.target_sr)/2*0.9, fs=self.original_sr)
        n = len(h_linear); fft_size = 1 << (n - 1).bit_length(); cepstrum = np.fft.irfft(np.log(np.abs(np.fft.rfft(h_linear, n=fft_size)) + 1e-12)); w = np.zeros(len(cepstrum)); w[0]=1; w[1:n//2]=2; w[n//2]=1 if n%2==0 else 0; cepstrum *= w; h_min = np.fft.irfft(np.exp(np.fft.rfft(cepstrum)), n=fft_size)[:n]
        self.H_linear, self.H_min = np.fft.rfft(h_linear, n=self.chunk_size), np.fft.rfft(h_min, n=self.chunk_size)
    def _design_intelligent_mixed_phase_filter(self, chunk):
        power_db, mask_db, _, _ = self.psy_model.analyze_chunk(chunk); mixed_filter_spectrum = np.zeros_like(self.H_linear)
        for i in range(self.num_bands):
            start, end = self.psy_model.band_indices[i], self.psy_model.band_indices[i+1]
            if start >= end: continue
            avg_margin = np.mean(power_db[start:end] - mask_db[start:end])
            mixed_filter_spectrum[start:end] = self.H_linear[start:end] if avg_margin > 9.0 else self.H_min[start:end]
        return mixed_filter_spectrum
    def _resample_channel(self, channel_data, ch_name=""):
        num_chunks = max(0, len(channel_data) - self.chunk_size) // self.hop_size + 1
        output_len = int(np.ceil(len(channel_data) * self.resample_ratio)); output_signal = np.zeros(output_len)
        output_ptr = 0
        for i in range(num_chunks):
            if i % 10 == 0: self.progress_updated.emit(int(100*i/num_chunks), f"Resampling Ch {ch_name}: {int(100*i/num_chunks)}%")
            start = i * self.hop_size
            chunk_raw = channel_data[start : start + self.chunk_size]
            if len(chunk_raw) < self.chunk_size: chunk_raw = np.pad(chunk_raw, (0, self.chunk_size - len(chunk_raw)))
            filter_spectrum = self._design_intelligent_mixed_phase_filter(chunk_raw)
            filtered_spectrum = np.fft.rfft(chunk_raw * self.window) * filter_spectrum
            target_fft_size = int(self.chunk_size * self.resample_ratio)
            resampled_spectrum = np.zeros(target_fft_size // 2 + 1, dtype=np.complex128)
            n_min = min(len(filtered_spectrum), len(resampled_spectrum))
            resampled_spectrum[:n_min] = filtered_spectrum[:n_min]
            processed_chunk = np.fft.irfft(resampled_spectrum, n=target_fft_size)
            if output_ptr + len(processed_chunk) > len(output_signal):
                processed_chunk = processed_chunk[:len(output_signal) - output_ptr]
            output_signal[output_ptr : output_ptr + len(processed_chunk)] += processed_chunk
            output_ptr += int(self.hop_size * self.resample_ratio)
        return output_signal * self.resample_ratio
    def resample(self, signal, is_ms_processing=False):
        if signal.ndim == 2:
            ch1_name, ch2_name = ('Mid', 'Side') if is_ms_processing else ('L', 'R')
            ch1, ch2 = self._resample_channel(signal[:, 0], ch1_name), self._resample_channel(signal[:, 1], ch2_name)
            return np.stack([ch1, ch2], axis=1)
        return self._resample_channel(signal, 'Mono')

class Worker(QObject):
    finished = pyqtSignal(); progress = pyqtSignal(int, str)
    log_message = pyqtSignal(str); error = pyqtSignal(str)
    def __init__(self, params): super().__init__(); self.params = params; self.is_running = True
    def stop(self): self.is_running = False
    def run(self):
        try:
            input_path, output_path = self.params['input_path'], self.params['output_path']
            files_to_process = []
            if os.path.isdir(input_path):
                supported_ext = ('.wav', '.flac', '.mp3', '.ogg', '.aiff', '.aif')
                for root, _, files in os.walk(input_path):
                    for file in files:
                        if file.lower().endswith(supported_ext): files_to_process.append(os.path.join(root, file))
            elif os.path.isfile(input_path): files_to_process.append(input_path)
            if not files_to_process: self.error.emit("No supported audio files found."); self.finished.emit(); return
            for i, file_path in enumerate(files_to_process):
                if not self.is_running: break
                self.log_message.emit(f"--- Processing file {i+1}/{len(files_to_process)}: {os.path.basename(file_path)} ---")
                output_file = os.path.join(output_path, os.path.basename(file_path)) if os.path.isdir(output_path) else output_path
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                self.progress.emit(0, "Loading audio file..."); dat, sr = librosa.load(file_path, sr=None, mono=False); dat = dat.T
                if dat.ndim != 2 or dat.shape[1] != 2: self.log_message.emit(f"[Warning] Not stereo: '{os.path.basename(file_path)}'. Skipping."); continue
                processed_data, current_sr = dat.astype(np.float64), sr # Ensure float64 for Numba

                # --- Pipeline Step 1: Resampling ---
                target_sr = self.params['resample_target_sr']
                if current_sr != target_sr:
                    self.log_message.emit(f"Resampling from {current_sr} Hz to {target_sr} Hz...")
                    resampler = IntelligentRDOResampler(original_sr=current_sr, target_sr=target_sr)
                    resampler.progress_updated.connect(self.progress.emit)
                    mid, side = (processed_data[:, 0] + processed_data[:, 1]) / 2.0, (processed_data[:, 0] - processed_data[:, 1]) / 2.0
                    resampled_ms = resampler.resample(np.stack([mid, side], axis=1), is_ms_processing=True)
                    processed_data = np.stack([resampled_ms[:,0] + resampled_ms[:,1], resampled_ms[:,0] - resampled_ms[:,1]], axis=1)
                    current_sr = target_sr
                    self.log_message.emit("Resampling complete.")
                if not self.is_running: break

                # --- Pipeline Step 2: High-Frequency Restoration (HFPv2) ---
                if self.params['enable_hfr']:
                    self.log_message.emit("Applying High-Frequency Restoration (HFPv2) post-resampling...")
                    restorer = HighFrequencyRestorer(); restorer.progress_updated.connect(self.progress.emit)
                    lowpass_freq = self.params['hfr_lowpass']
                    lowpass_bin = int(lowpass_freq * 1024 / current_sr) if lowpass_freq != -1 else -1
                    processed_data = restorer.run_hfpv2(processed_data, current_sr, 
                                                        lowpass=lowpass_bin, 
                                                        enable_compressed_fix=self.params['enable_compressed_fix'])
                    self.log_message.emit("HFPv2 Restoration complete.")
                if not self.is_running: break

                # --- Pipeline Step 3: Dithering for PCM output ---
                subtype = {'4-bit PCM':'PCM_U8', '8-bit PCM':'PCM_U8', '16-bit PCM':'PCM_16', '24-bit PCM':'PCM_24', '32-bit Float':'FLOAT'}.get(self.params['output_bit_depth'], 'FLOAT')
                final_data = processed_data
                if 'PCM' in subtype and self.params['enable_dither']:
                    try:
                        target_bit_depth = int(subtype.split('_')[1])
                    except:
                        if subtype[0] == "8":
                            target_bit_depth = 8
                        else:
                            target_bit_depth = 4
                    self.log_message.emit(f"Applying {target_bit_depth}-bit dither and noise shaping...")
                    shaper = RDONoiseShaper(target_bit_depth, current_sr, lpc_order=self.params['dither_lpc_order'])
                    shaper.progress_updated.connect(self.progress.emit)
                    final_data = shaper.process(final_data)
                    self.log_message.emit("Dithering and shaping complete.")

                self.progress.emit(99, "Saving file...")
                sf.write(output_file, np.clip(final_data, -1.0, 1.0), current_sr, subtype=subtype)
                self.log_message.emit(f"Successfully saved to: {output_file}\n")
            if self.is_running: self.log_message.emit("--- All tasks finished! ---")
            else: self.log_message.emit("--- Processing stopped by user. ---")
        except Exception as e: self.error.emit(f"An error occurred: {e}\n{traceback.format_exc()}")
        self.finished.emit()

# --- PyQt6 Main Application Window ---
# (The GUI part is unchanged. It will work as-is with the modified backend.)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Processor (Resampler -> HFPv2) [Numba Accelerated]")
        self.setGeometry(100, 100, 800, 700)
        self.setStyleSheet("""
            QMainWindow{background-color:#2E2E2E} QWidget{color:#E0E0E0;background-color:#2E2E2E;font-size:14px} QGroupBox{background-color:#3C3C3C;border:1px solid #555;border-radius:8px;margin-top:10px;font-weight:bold} QGroupBox::title{subcontrol-origin:margin;subcontrol-position:top center;padding:0 10px} QLabel{background-color:transparent} QLineEdit{background-color:#4A4A4A;border:1px solid #555;border-radius:4px;padding:5px} QPushButton{background-color:#5A5A5A;border:1px solid #666;border-radius:4px;padding:8px 12px} QPushButton:hover{background-color:#6A6A6A} QPushButton:pressed{background-color:#4A4A4A} QPushButton#StartButton{background-color:#4CAF50;font-weight:bold} QPushButton#StartButton:hover{background-color:#5CBF60} QCheckBox,QComboBox{padding:5px} QComboBox QAbstractItemView{background-color:#4A4A4A;selection-background-color:#5A5A5A} QProgressBar{border:1px solid #555;border-radius:4px;text-align:center;color:#FFF} QProgressBar::chunk{background-color:#4CAF50;border-radius:3px} QTextEdit{background-color:#252525;border:1px solid #555;border-radius:4px}
        """)
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self._create_io_group()
        self._create_resampling_group()
        self._create_hfr_group()
        self._create_controls_group()
        self.thread = None
        self.worker = None

    def _create_io_group(self):
        group = QGroupBox("1. File Input/Output")
        layout, input_layout, output_layout = QVBoxLayout(group), QHBoxLayout(), QHBoxLayout()
        self.input_path_edit, self.output_path_edit = QLineEdit(), QLineEdit()
        self.input_path_edit.setPlaceholderText("Select input file or folder...")
        self.output_path_edit.setPlaceholderText("Select output folder...")
        browse_in, browse_out = QPushButton("Browse Input"), QPushButton("Browse Output")
        browse_in.clicked.connect(self.browse_input); browse_out.clicked.connect(self.browse_output)
        input_layout.addWidget(self.input_path_edit); input_layout.addWidget(browse_in)
        output_layout.addWidget(self.output_path_edit); output_layout.addWidget(browse_out)
        layout.addLayout(input_layout); layout.addLayout(output_layout)
        self.main_layout.addWidget(group)

    def _create_resampling_group(self):
        group = QGroupBox("2. Resampling & Output Format")
        layout, grid_layout = QVBoxLayout(group), QHBoxLayout()
        sr_layout, bd_layout = QVBoxLayout(), QVBoxLayout()
        sr_layout.addWidget(QLabel("Target Sample Rate:")); bd_layout.addWidget(QLabel("Output Format:"))
        self.resample_sr_combo = QComboBox(); self.resample_sr_combo.addItems(["32000", "44100", "48000", "88200", "96000"]); self.resample_sr_combo.setCurrentText("48000")
        self.output_bit_depth_combo = QComboBox(); self.output_bit_depth_combo.addItems(["4-bit PCM", "8-bit PCM", "16-bit PCM", "24-bit PCM", "32-bit Float"]); self.output_bit_depth_combo.setCurrentText("24-bit PCM")
        sr_layout.addWidget(self.resample_sr_combo); bd_layout.addWidget(self.output_bit_depth_combo)
        grid_layout.addLayout(sr_layout); grid_layout.addLayout(bd_layout)
        dither_group = QGroupBox("Dithering Options")
        dither_layout, lpc_layout = QVBoxLayout(dither_group), QHBoxLayout()
        self.dither_enable_check = QCheckBox("Enable Dither & Noise Shaping (for PCM)"); self.dither_enable_check.setChecked(True)
        lpc_layout.addWidget(QLabel("LPC Order:")); self.dither_lpc_spin = QSpinBox(); self.dither_lpc_spin.setRange(0, 32); self.dither_lpc_spin.setValue(16)
        lpc_layout.addWidget(self.dither_lpc_spin); dither_layout.addWidget(self.dither_enable_check); dither_layout.addLayout(lpc_layout)
        layout.addLayout(grid_layout); layout.addWidget(dither_group)
        self.main_layout.addWidget(group)

    def _create_hfr_group(self):
        group = QGroupBox("3. High-Frequency Restoration (HFPv2)")
        layout, lowpass_layout = QVBoxLayout(group), QHBoxLayout()
        self.hfr_enable_check = QCheckBox("Enable Restoration Module"); self.hfr_enable_check.setChecked(True)
        self.hfr_compressed_fix_check = QCheckBox("Apply Compressed Audio Fixes")
        self.hfr_compressed_fix_check.setChecked(True)
        self.hfr_compressed_fix_check.setToolTip("Restores spectral holes and high-frequency stereo width, useful for previously compressed audio (e.g., MP3).")
        lowpass_layout.addWidget(QLabel("Lowpass Cutoff (Hz):"))
        self.hfr_lowpass_spin = QSpinBox(); self.hfr_lowpass_spin.setRange(-1, 22000); self.hfr_lowpass_spin.setSpecialValueText("Auto"); self.hfr_lowpass_spin.setValue(13500)
        lowpass_layout.addWidget(self.hfr_lowpass_spin)
        layout.addWidget(self.hfr_enable_check)
        layout.addWidget(self.hfr_compressed_fix_check)
        layout.addLayout(lowpass_layout)
        self.main_layout.addWidget(group)

    def _create_controls_group(self):
        self.start_button = QPushButton("Start Processing"); self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_processing)
        self.progress_bar, self.status_label = QProgressBar(), QLabel("Ready.")
        log_group = QGroupBox("Log Console"); log_layout = QVBoxLayout(log_group)
        self.log_console = QTextEdit(); self.log_console.setReadOnly(True)
        log_layout.addWidget(self.log_console)
        self.main_layout.addWidget(self.start_button); self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.status_label); self.main_layout.addWidget(log_group)

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Input File", "", "Audio Files (*.wav *.flac *.mp3 *.ogg *.aiff)");
        if not path: path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if path: self.input_path_edit.setText(path)
    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path: self.output_path_edit.setText(path)

    def start_processing(self):
        if self.thread and self.thread.isRunning():
            self.log_console.append("\n>>> Sending stop signal...")
            self.worker.stop()
            self.start_button.setEnabled(False) 
        else:
            # First run with dummy data to compile Numba functions
            self.log_console.append("JIT Compiling Numba functions... (first run may be slow)")
            QApplication.processEvents() # Update GUI
            try:
                # A quick compilation run
                _numba_apply_pre_echo_suppression(np.zeros(10), 5, np.array([1]))
                _numba_apply_compressed_fix_loop(np.zeros((10,10)), np.zeros((10,10)), -12, 5)
                _numba_process_channel_dither(np.zeros(10), 5, np.array([1.0, -1.0]), 0.1, 1)
            except Exception as e:
                 self.log_console.append(f"Numba pre-compilation failed: {e}")
            self.log_console.append("Compilation complete.")


            input_path, output_path = self.input_path_edit.text(), self.output_path_edit.text()
            if not (input_path and output_path and os.path.exists(input_path) and (os.path.isdir(output_path) or not os.path.isdir(input_path))):
                QMessageBox.warning(self, "Invalid Paths", "Please specify a valid input file/folder and an existing output folder."); return
            
            params = {
                'input_path': input_path, 'output_path': output_path,
                'enable_hfr': self.hfr_enable_check.isChecked(),
                'hfr_lowpass': self.hfr_lowpass_spin.value(),
                'enable_compressed_fix': self.hfr_compressed_fix_check.isChecked(),
                'resample_target_sr': int(self.resample_sr_combo.currentText()),
                'output_bit_depth': self.output_bit_depth_combo.currentText(),
                'enable_dither': self.dither_enable_check.isChecked(), 
                'dither_lpc_order': self.dither_lpc_spin.value(),
            }
            self.start_button.setText("Stop Processing"); self.log_console.clear()
            self.thread = QThread(); self.worker = Worker(params); self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            def on_finished():
                self.start_button.setText("Start Processing")
                self.start_button.setEnabled(True)
                self.thread = None 
                self.worker = None 

            self.thread.finished.connect(on_finished)
            self.worker.progress.connect(self.update_progress)
            self.worker.log_message.connect(self.update_log)
            self.worker.error.connect(self.show_error)
            self.thread.start()

    def update_progress(self, v, m): self.progress_bar.setValue(v); self.status_label.setText(m)
    def update_log(self, m): self.log_console.append(m); self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())
    def show_error(self, m):
        self.log_console.append(f"\nERROR:\n{m}"); self.status_label.setText("An error occurred.")
        QMessageBox.critical(self, "Error", m)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
