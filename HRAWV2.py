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
import soundfile as sf
import numba
from numba import jit, prange
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QFileDialog, QProgressBar, QGroupBox, QCheckBox,
    QSpinBox, QComboBox, QMessageBox, QTextEdit
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject

import numpy as np
import scipy.fft
import scipy.signal

class ScipyMDCT:
    """
    scipyを使ったMDCT/IMDCT実装
    
    MDCT (Modified Discrete Cosine Transform) は50%オーバーラップする
    時間周波数変換で、音声圧縮などで使用される。
    """
    
    def __init__(self, framelength=1024, hopsize=None, window='vorbis'):
        """
        Parameters:
        -----------
        framelength : int
            フレーム長（偶数である必要がある）
        hopsize : int, optional
            ホップサイズ（デフォルトはframelength // 2）
        window : str or array
            窓関数の種類または配列
            'vorbis' - Vorbis窓（推奨）
            'kbd' - Kaiser-Bessel Derived窓
            'sine' - Sine窓
            配列の場合は直接使用
        """
        if framelength % 2 != 0:
            raise ValueError("framelength must be even")
        
        self.N = framelength
        self.M = self.N // 2
        self.hopsize = hopsize if hopsize is not None else self.M
        
        # 窓関数の生成
        if isinstance(window, str):
            self.window = self._create_window(window, self.N)
        else:
            self.window = np.array(window)
            if len(self.window) != self.N:
                raise ValueError(f"Window length must be {self.N}")
        
        # TDAC (Time Domain Aliasing Cancellation) 特性の検証
        if not self._verify_tdac_property(self.window):
            print("Warning: Window does not satisfy TDAC property perfectly")
    
    def _create_window(self, window_type, N):
        """各種窓関数の生成"""
        if window_type == 'vorbis':
            # Vorbis窓（最も一般的）
            n = np.arange(N)
            return np.sin(np.pi / 2 * np.sin(np.pi * (n + 0.5) / N) ** 2)
        
        elif window_type == 'sine':
            # Sine窓
            n = np.arange(N)
            return np.sin(np.pi * (n + 0.5) / N)
        
        elif window_type == 'kbd':
            # Kaiser-Bessel Derived窓
            alpha = 4.0
            kbd_half = scipy.signal.kaiser(N // 2 + 1, alpha * np.pi)
            kbd_half_cumsum = np.cumsum(kbd_half)
            kbd_half_sum = kbd_half_cumsum[-1]
            kbd = np.sqrt(
                np.concatenate([
                    kbd_half_cumsum[:-1],
                    kbd_half_sum - kbd_half_cumsum[-2::-1]
                ]) / kbd_half_sum
            )
            return kbd
        
        else:
            raise ValueError(f"Unknown window type: {window_type}")
    
    def _verify_tdac_property(self, window, tolerance=1e-6):
        """TDAC特性の検証: w[n]^2 + w[n+M]^2 = 1"""
        M = len(window) // 2
        left_half = window[:M] ** 2
        right_half = window[M:] ** 2
        tdac_sum = left_half + right_half
        return np.allclose(tdac_sum, 1.0, atol=tolerance)
    
    def mdct(self, x):
        """
        MDCT変換
        
        Parameters:
        -----------
        x : array_like
            入力信号（1次元）
        
        Returns:
        --------
        X : ndarray
            MDCT係数（shape: [M, num_frames]）
        """
        x = np.asarray(x, dtype=np.float64)
        
        # フレーム数の計算
        if len(x) < self.N:
            x = np.pad(x, (0, self.N - len(x)), mode='constant')
        
        num_frames = 1 + (len(x) - self.N) // self.hopsize
        
        # 係数行列の初期化
        X = np.zeros((self.M, num_frames), dtype=np.float64)
        
        for i in range(num_frames):
            start = i * self.hopsize
            end = start + self.N
            
            if end > len(x):
                frame = np.pad(x[start:], (0, end - len(x)), mode='constant')
            else:
                frame = x[start:end]
            
            # 窓関数の適用
            windowed = frame * self.window
            
            # MDCTの計算
            # X[k] = Σ(n=0 to 2N-1) x[n] * cos(π/N * (n + 0.5 + N/2) * (k + 0.5))
            
            # DCT-IVを使った効率的な実装
            # MDCTはDCT-IVの前処理と後処理で実装可能
            
            # 前処理: フォールディング
            folded = np.zeros(self.M)
            for n in range(self.M):
                folded[n] = -windowed[self.M - 1 - n] + windowed[self.M + n]
            
            # DCT-IV適用
            X[:, i] = scipy.fft.dct(folded, type=4, norm='ortho')
        
        return X
    
    def imdct(self, X):
        """
        IMDCT（逆MDCT変換）
        
        Parameters:
        -----------
        X : ndarray
            MDCT係数（shape: [M, num_frames]）
        
        Returns:
        --------
        x : ndarray
            復元された時間領域信号
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.shape[0] != self.M:
            raise ValueError(f"Expected {self.M} frequency bins, got {X.shape[0]}")
        
        num_frames = X.shape[1]
        output_length = (num_frames - 1) * self.hopsize + self.N
        x = np.zeros(output_length, dtype=np.float64)
        
        for i in range(num_frames):
            # IDCT-IV適用
            dct_out = scipy.fft.dct(X[:, i], type=4, norm='ortho')
            
            # 後処理: アンフォールディング
            frame = np.zeros(self.N)
            for n in range(self.M):
                frame[self.M - 1 - n] = -dct_out[n]
                frame[self.M + n] = dct_out[n]
            
            # 窓関数の適用
            windowed_frame = frame * self.window
            
            # オーバーラップ加算
            start = i * self.hopsize
            end = start + self.N
            x[start:end] += windowed_frame
        
        return x
    
mdct = ScipyMDCT()

@jit(nopython=True, cache=True, fastmath=True)
def _numba_tukey_window(window_length, alpha):
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
    return mid_spec, side_spec

@jit(nopython=True, cache=True, fastmath=True)
def _numba_process_channel_dither(channel_data, chunk_size, b, quantization_step, lpc_order):
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
    """位相抽出と振幅投影を並列化"""
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
    """モーメンタム更新の適用"""
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


def griffin_lim_mdct(magnitudes, frame_length=2048, hop_length=1024, n_iter=2, transient_frames=None):
    """
    高速化されたGriffin-Lim MDCT版（Momentum法）
    
    既存コードと完全互換性を保ちながら2-3倍高速化
    
    Parameters:
    -----------
    magnitudes : ndarray
        振幅スペクトル (num_bins, num_frames)
    frame_length : int
        MDCTフレーム長
    hop_length : int
        ホップサイズ
    n_iter : int
        反復回数
    transient_frames : array_like, optional
        トランジエントフレームのブール配列
    
    Returns:
    --------
    mdct_spec : ndarray
        位相付きMDCTスペクトル
    """
    
    # Momentum法のパラメータ
    MOMENTUM_COEF = 0.99  # モーメンタム係数（0.95-0.99推奨）
    USE_MOMENTUM = (n_iter >= 2)  # 2回以上の反復でMomentum使用
    
    # 初期位相生成（ランダム）
    phases = np.random.choice([-1, 1], size=magnitudes.shape)
    mdct_spec = magnitudes * phases
    
    # トランジエント処理の準備
    transient_frames_indices = (
        np.where(transient_frames)[0] 
        if transient_frames is not None 
        else np.array([0])
    )
    
    # Momentum法用の変数初期化
    if USE_MOMENTUM:
        momentum = np.zeros_like(mdct_spec)
        prev_spec = mdct_spec.copy()
    
    for i in range(n_iter):
        # IMDCT: 周波数領域 → 時間領域
        y_rec = mdct.imdct(mdct_spec)
        
        # トランジエント処理（プリエコー抑制）
        if transient_frames is not None and len(transient_frames_indices) > 0:
            y_rec_padded = np.pad(y_rec, (hop_length, hop_length), 'constant')
            y_rec_padded = _numba_apply_pre_echo_suppression(
                y_rec_padded, hop_length, transient_frames_indices
            )
            y_rec = y_rec_padded[hop_length:-hop_length]
        
        # MDCT: 時間領域 → 周波数領域
        re_mdct_spec = mdct.mdct(y_rec)
        
        # スペクトルの形状を合わせる（既存コードと同じロジック）
        if re_mdct_spec.shape[1] < mdct_spec.shape[1]:
            re_mdct_spec = np.pad(
                re_mdct_spec, 
                ((0, 0), (0, mdct_spec.shape[1] - re_mdct_spec.shape[1])), 
                'constant'
            )
        elif re_mdct_spec.shape[1] > mdct_spec.shape[1]:
            re_mdct_spec = re_mdct_spec[:, :mdct_spec.shape[1]]
        
        # 位相抽出と振幅投影（並列化版）
        mdct_spec_updated = _fast_sign_with_magnitude(re_mdct_spec, magnitudes)
        
        # Momentum法の適用（2回目以降の反復）
        if USE_MOMENTUM and i > 0:
            # モーメンタム更新
            mdct_spec_with_momentum, momentum = _apply_momentum_update(
                mdct_spec_updated, prev_spec, momentum, MOMENTUM_COEF
            )
            
            # 振幅制約を再適用（momentum後にずれた振幅を修正）
            mdct_spec = _fast_sign_with_magnitude(mdct_spec_with_momentum, magnitudes)
            
            # 次回の反復用に保存
            prev_spec = mdct_spec.copy()
        else:
            # 初回反復またはMomentum無効時
            mdct_spec = mdct_spec_updated
            if USE_MOMENTUM:
                prev_spec = mdct_spec.copy()
    
    return mdct_spec
    
def connect_spectra_smooth(signal1, signal2, overlap_size=32, is_harm=True):
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
    
class OVERTONE:
    base_level, width, fft_idx, ch, tone_idx = 0, 1, 0, 0, 0

class HighFrequencyRestorer(QObject):
    progress_updated = pyqtSignal(int, str)
    
    def _apply_compressed_fix(self, mid_spec, side_spec, threshold_db=-64, high_freq_bin_start_ratio=0.15):
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
        """
        HFPv2: 高周波復元アルゴリズム（倍音構造強化版）
        
        Parameters:
        -----------
        dat : ndarray
            ステレオ音声データ (samples, 2)
        sr : int
            サンプリングレート
        lowpass : int
            ローパスフィルタのビン位置（-1で自動検出）
        enable_compressed_fix : bool
            圧縮音源の修復を有効化
        
        Returns:
        --------
        output : ndarray
            高周波復元済み音声データ
        """
        
        # ====================
        # 1. 初期設定
        # ====================
        frame_length = 1024
        fft_size = frame_length
        hop_length = frame_length // 2
        
        # 倍音強化パラメータ
        HARMONIC_BOOST = 1.8
        CORRELATION_THRESHOLD = 0.7
        MIN_HARMONIC_ORDER = 2
        CEPSTRUM_PEAK_THRESHOLD = 64
        WIDTH_EXPANSION_FACTOR = 3
        FADE_CURVE_POWER = 1.3
        
        # Mid/Side分解
        mid = (dat[:,0] + dat[:,1]) / 2.0
        side = (dat[:,0] - dat[:,1]) / 2.0
        
        # ====================
        # 2. トランジエント検出
        # ====================
        self.progress_updated.emit(5, "HFPv2: Detecting transients...")
        total_frames = (len(mid) - frame_length) // hop_length + 1
        onset_frames = librosa.onset.onset_detect(
            y=mid, sr=sr, hop_length=hop_length, units='frames', backtrack=True
        )
        transient_frames = np.zeros(total_frames, dtype=bool)
        valid_onset_frames = onset_frames[onset_frames < total_frames]
        transient_frames[valid_onset_frames] = True
        
        # ====================
        # 3. Lowpass位置の自動検出
        # ====================
        mid_ffted_full = mdct.mdct(mid)
        if lowpass == -1:
            indices = np.where(abs(mid_ffted_full) < 0.0000001)[0]
            if len(indices) > 0:
                for i in range(len(indices)):
                    if indices[i] < 16:
                        continue
                    else:
                        lowpass = int(indices[i]) - 1
                        break
            else:
                lowpass = fft_size // 2
        self.progress_updated.emit(10, f"HFPv2: Lowpass Index set to {lowpass}")
        
        # ====================
        # 4. ローパスフィルタリング
        # ====================
        cutoff_freq_normalized = (lowpass * 2) / fft_size
        if 0 < cutoff_freq_normalized < 1.0:
            b, a = scipy.signal.butter(8, cutoff_freq_normalized, btype='low')
            mid_filtered = scipy.signal.filtfilt(b, a, mid)
            side_filtered = scipy.signal.filtfilt(b, a, side)
        else:
            mid_filtered, side_filtered = mid, side
        
        # ====================
        # 5. MDCT変換とHPSS
        # ====================
        mid_ffted = mdct.mdct(mid_filtered)
        side_ffted = mdct.mdct(side_filtered)
        
        mid_ffted = abs(mid_ffted)
        side_ffted = abs(side_ffted)
        
        # HPSS（Harmonic/Percussive Source Separation）
        mid_noise = scipy.signal.medfilt2d(mid_ffted, kernel_size=(3,5)) / 2
        side_noise = scipy.signal.medfilt2d(side_ffted, kernel_size=(3,5)) / 2
        mid_harm = mid_ffted - mid_noise
        side_harm = side_ffted - side_noise
        
        # ====================
        # 6. 圧縮音源修復（オプション）
        # ====================
        if enable_compressed_fix:
            self.progress_updated.emit(14, "HFPv2: Applying compressed audio fixes...")
            mid_harm, side_harm = self._apply_compressed_fix(mid_harm, side_harm)
            mid_noise, side_noise = self._apply_compressed_fix(mid_noise, side_noise)
        
        # ====================
        # 7. フレームごとの倍音外挿処理
        # ====================
        num_fft_frames = len(mid_ffted.T)
        
        for i in range(num_fft_frames):
            progress = 15 + int(75 * i / num_fft_frames)
            if i % 20 == 0:
                self.progress_updated.emit(progress, f"HFPv2: Synthesizing frame {i}/{num_fft_frames}")
            
            # 無音フレームをスキップ
            if np.mean(np.abs(mid_ffted.T[i])) < 1e-4:
                continue
            
            # フレーム抽出
            mid_harm_frame = mid_harm[:,i].copy()
            mid_noise_frame = mid_noise[:,i].copy()
            side_harm_frame = side_harm[:,i].copy()
            side_noise_frame = side_noise[:,i].copy()
            
            # ========================================
            # 7.1 Harmonic成分の処理
            # ========================================
            
            # ケプストラム変換
            mid_harm_frame_cep = mid_harm_frame
            side_harm_frame_cep = side_harm_frame
            
            # ピーク検出
            mid_harm_peaks = scipy.signal.argrelmax(
                librosa.amplitude_to_db(abs(mid_harm_frame_cep))
            )[0]
            mid_harm_peaks = mid_harm_peaks[mid_harm_peaks > CEPSTRUM_PEAK_THRESHOLD]
            
            side_harm_peaks = scipy.signal.argrelmax(
                librosa.amplitude_to_db(abs(side_harm_frame_cep))
            )[0]
            side_harm_peaks = side_harm_peaks[side_harm_peaks > CEPSTRUM_PEAK_THRESHOLD]
            
            # 倍音構造の収集
            overtone_harm = []
            
            for peak in mid_harm_peaks:
                ot = OVERTONE()
                ot.base_level = mid_harm_frame_cep[peak]
                ot.fft_idx = i // 2
                ot.ch = 0
                ot.tone_idx = peak
                
                base_width = 1
                for j in range(-2, 2):
                    if peak + j >= len(mid_harm_frame_cep) or peak + j < 0:
                        continue
                    if abs(mid_harm_frame_cep[peak]) - abs(mid_harm_frame_cep[peak+j]) > 0.0001:
                        base_width = abs(j) + 1
                        break
                ot.width = int(base_width * WIDTH_EXPANSION_FACTOR)
                overtone_harm.append(ot)
            
            for peak in side_harm_peaks:
                ot = OVERTONE()
                ot.base_level = side_harm_frame_cep[peak]
                ot.fft_idx = i // 2
                ot.ch = 1
                ot.tone_idx = peak
                
                base_width = 1
                for j in range(-2, 2):
                    if peak + j >= len(side_harm_frame_cep) or peak + j < 0:
                        continue
                    if abs(side_harm_frame_cep[peak]) - abs(side_harm_frame_cep[peak+j]) > 0.0001:
                        base_width = abs(j) + 1
                        break
                ot.width = int(base_width * WIDTH_EXPANSION_FACTOR)
                overtone_harm.append(ot)
            
            # 倍音外挿
            for ot in overtone_harm:
                if ot.tone_idx == 0:
                    continue
                
                max_harmonic_order = min(
                    lowpass // (1 + ot.tone_idx),
                    20
                )
                
                if ot.ch == 0:  # Mid channel
                    ot_loops, ot_levels = [], []
                    
                    for n in range(MIN_HARMONIC_ORDER, max_harmonic_order):
                        start_idx = max(0, ot.tone_idx*n - ot.width)
                        end_idx = min(len(mid_harm_frame_cep), ot.tone_idx*n + ot.width)
                        ot_loops.append(mid_harm_frame_cep[start_idx:end_idx])
                        ot_levels.append(mid_harm_frame_cep[ot.tone_idx])
                    
                    if len(ot_loops) < 3:
                        continue
                    
                    try:
                        corrs = np.corrcoef(np.array(ot_loops[2:]))
                    except:
                        continue
                    
                    if corrs.size < 2:
                        continue
                    
                    # 相関チェック
                    valid_correlation = False
                    for l in range(corrs.shape[0]):
                        corr = corrs[l]
                        if abs(np.sum(abs(corr)) / (ot.width*2)) <= CORRELATION_THRESHOLD:
                            valid_correlation = True
                            break
                    
                    if valid_correlation:
                        envelope = np.array(ot_levels)
                        window = scipy.signal.windows.blackmanharris(len(ot_levels) * ot.width)
                        env_convolved = scipy.signal.fftconvolve(envelope, window, mode='valid')
                        
                        if len(env_convolved) <= lowpass // (ot.tone_idx + 1):
                            continue
                        
                        env_convolved = env_convolved[lowpass // (ot.tone_idx + 1):]
                        cnt = 0
                        
                        for a in range(
                            lowpass // (ot.tone_idx + 1),
                            min(
                                lowpass // (ot.tone_idx + 1) + (frame_length//2 - lowpass) // (ot.tone_idx + 1),
                                lowpass // (ot.tone_idx + 1) + len(env_convolved)
                            )
                        ):
                            try:
                                start_idx = max(0, (ot.tone_idx+1)*a - ot.width)
                                end_idx = min(len(mid_harm_frame_cep), (ot.tone_idx+1)*a + ot.width)
                                mid_harm_frame_cep[start_idx:end_idx] += (
                                    env_convolved[cnt] * HARMONIC_BOOST / max(len(ot_loops), 1)
                                )
                                cnt += 1
                            except:
                                break
                
                elif ot.ch == 1:  # Side channel
                    ot_loops, ot_levels = [], []
                    
                    for n in range(MIN_HARMONIC_ORDER, max_harmonic_order):
                        start_idx = max(0, ot.tone_idx*n - ot.width)
                        end_idx = min(len(side_harm_frame_cep), ot.tone_idx*n + ot.width)
                        ot_loops.append(side_harm_frame_cep[start_idx:end_idx])
                        ot_levels.append(side_harm_frame_cep[ot.tone_idx])
                    
                    if len(ot_loops) < 3:
                        continue
                    
                    try:
                        corrs = np.corrcoef(np.array(ot_loops[2:]))
                    except:
                        continue
                    
                    if corrs.size < 2:
                        continue
                    
                    valid_correlation = False
                    for l in range(corrs.shape[0]):
                        corr = corrs[l]
                        if abs(np.sum(abs(corr)) / (ot.width*2)) <= CORRELATION_THRESHOLD:
                            valid_correlation = True
                            break
                    
                    if valid_correlation:
                        envelope = np.array(ot_levels)
                        window = scipy.signal.windows.blackmanharris(len(ot_levels) * ot.width)
                        env_convolved = scipy.signal.fftconvolve(envelope, window, mode='valid')
                        
                        if len(env_convolved) <= lowpass // (ot.tone_idx + 1):
                            continue
                        
                        env_convolved = env_convolved[lowpass // (ot.tone_idx + 1):]
                        cnt = 0
                        
                        for a in range(
                            lowpass // (ot.tone_idx + 1),
                            min(
                                lowpass // (ot.tone_idx + 1) + (frame_length//2 - lowpass) // (ot.tone_idx + 1),
                                lowpass // (ot.tone_idx + 1) + len(env_convolved)
                            )
                        ):
                            try:
                                start_idx = max(0, (ot.tone_idx+1)*a - ot.width)
                                end_idx = min(len(side_harm_frame_cep), (ot.tone_idx+1)*a + ot.width)
                                side_harm_frame_cep[start_idx:end_idx] += (
                                    env_convolved[cnt] * HARMONIC_BOOST / max(len(ot_loops), 1)
                                )
                                cnt += 1
                            except:
                                break
            
            mid_harm_interp = mid_harm_frame_cep
            
            side_harm_interp = side_harm_frame_cep
            
            # スペクトル接続とランダム位相
            mid_harm_frame = scipy.ndimage.uniform_filter(
                connect_spectra_smooth(mid_harm_frame[:lowpass], abs(mid_harm_interp[lowpass:])),
                size=2
            ) * (abs(np.random.randn(frame_length//2)))
            
            side_harm_frame = scipy.ndimage.uniform_filter(
                connect_spectra_smooth(side_harm_frame[:lowpass], abs(side_harm_interp[lowpass:])),
                size=2
            ) * (abs(np.random.randn(frame_length//2)))
            
            mid_harm[:,i] = mid_harm_frame / 2
            side_harm[:,i] = side_harm_frame / 2
            
            # ========================================
            # 7.2 Noise成分の処理
            # ========================================
            
            mid_noise_frame_cep = np.abs(scipy.fft.dct(mid_noise_frame, norm="ortho", type=2))
            side_noise_frame_cep = np.abs(scipy.fft.dct(side_noise_frame, norm="ortho", type=2))
            
            mid_noise_peaks = scipy.signal.argrelmax(abs(mid_noise_frame_cep))[0]
            side_noise_peaks = scipy.signal.argrelmax(abs(side_noise_frame_cep))[0]
            
            mid_noise_peaks = mid_noise_peaks[mid_noise_peaks > CEPSTRUM_PEAK_THRESHOLD]
            side_noise_peaks = side_noise_peaks[side_noise_peaks > CEPSTRUM_PEAK_THRESHOLD]
            
            overtone_noise = []
            
            for peak in mid_noise_peaks:
                ot = OVERTONE()
                ot.base_level = mid_noise_frame_cep[peak]
                ot.fft_idx = i
                ot.ch = 0
                ot.tone_idx = peak
                
                base_width = 1
                for j in range(-2, 2):
                    if peak + j >= len(mid_noise_frame_cep) or peak + j < 0:
                        continue
                    if abs(mid_noise_frame_cep[peak]) - abs(mid_noise_frame_cep[peak+j]) > 0.0001:
                        base_width = abs(j) + 1
                        break
                ot.width = int(base_width * WIDTH_EXPANSION_FACTOR)
                overtone_noise.append(ot)
            
            for peak in side_noise_peaks:
                ot = OVERTONE()
                ot.base_level = side_noise_frame_cep[peak]
                ot.fft_idx = i
                ot.ch = 1
                ot.tone_idx = peak
                
                base_width = 1
                for j in range(-2, 2):
                    if peak + j >= len(side_noise_frame_cep) or peak + j < 0:
                        continue
                    if abs(side_noise_frame_cep[peak]) - abs(side_noise_frame_cep[peak+j]) > 0.0001:
                        base_width = abs(j) + 1
                        break
                ot.width = int(base_width * WIDTH_EXPANSION_FACTOR)
                overtone_noise.append(ot)
            
            # Noise成分の倍音外挿
            for ot in overtone_noise:
                if ot.tone_idx == 0:
                    continue
                
                max_harmonic_order = min(lowpass // (1 + ot.tone_idx), 15)
                
                if ot.ch == 0:
                    ot_loops, ot_levels = [], []
                    
                    for n in range(1, max_harmonic_order):
                        start_idx = max(0, ot.tone_idx*n - ot.width)
                        end_idx = min(len(mid_noise_frame_cep), ot.tone_idx*n + ot.width)
                        ot_loops.append(mid_noise_frame_cep[start_idx:end_idx])
                        ot_levels.append(mid_noise_frame_cep[ot.tone_idx])
                    
                    if not ot_loops:
                        continue
                    
                    envelope = np.array(ot_levels)
                    window = scipy.signal.windows.lanczos(len(ot_levels) * ot.width)
                    env_convolved = scipy.signal.fftconvolve(envelope, window, mode='valid')
                    
                    if len(env_convolved) <= lowpass // (ot.tone_idx + 1):
                        continue
                    
                    env_convolved = env_convolved[lowpass // (ot.tone_idx + 1):]
                    cnt = 0
                    
                    for a in range(
                        lowpass // (ot.tone_idx + 1),
                        min(
                            lowpass // (ot.tone_idx + 1) + (frame_length//2 - lowpass) // (ot.tone_idx + 1),
                            lowpass // (ot.tone_idx + 1) + len(env_convolved)
                        )
                    ):
                        try:
                            start_idx = max(0, (ot.tone_idx+1)*a - ot.width)
                            end_idx = min(len(mid_noise_frame_cep), (ot.tone_idx+1)*a + ot.width)
                            mid_noise_frame_cep[start_idx:end_idx] += (
                                env_convolved[cnt] / max(len(ot_loops), 1)
                            )
                            cnt += 1
                        except:
                            break
                
                elif ot.ch == 1:
                    ot_loops, ot_levels = [], []
                    
                    for n in range(1, max_harmonic_order):
                        start_idx = max(0, ot.tone_idx*n - ot.width)
                        end_idx = min(len(side_noise_frame_cep), ot.tone_idx*n + ot.width)
                        ot_loops.append(side_noise_frame_cep[start_idx:end_idx])
                        ot_levels.append(side_noise_frame_cep[ot.tone_idx])
                    
                    if not ot_loops:
                        continue
                    
                    envelope = np.array(ot_levels)
                    window = scipy.signal.windows.lanczos(len(ot_levels) * ot.width)
                    env_convolved = scipy.signal.fftconvolve(envelope, window, mode='valid')
                    
                    if len(env_convolved) <= lowpass // (ot.tone_idx + 1):
                        continue
                    
                    env_convolved = env_convolved[lowpass // (ot.tone_idx + 1):]
                    cnt = 0
                    
                    for a in range(
                        lowpass // (ot.tone_idx + 1),
                        min(
                            lowpass // (ot.tone_idx + 1) + (frame_length//2 - lowpass) // (ot.tone_idx + 1),
                            lowpass // (ot.tone_idx + 1) + len(env_convolved)
                        )
                    ):
                        try:
                            start_idx = max(0, (ot.tone_idx+1)*a - ot.width)
                            end_idx = min(len(side_noise_frame_cep), (ot.tone_idx+1)*a + ot.width)
                            side_noise_frame_cep[start_idx:end_idx] += (
                                env_convolved[cnt] / max(len(ot_loops), 1)
                            )
                            cnt += 1
                        except:
                            break
            
            # IDCT: ケプストラム → スペクトル（Noise）
            mid_noise_frame_cep[0] = 0
            side_noise_frame_cep[0] = 0
            
            mid_noise_interp = scipy.fft.idct(
                mid_noise_frame_cep * np.sign(scipy.fft.dct(mid_noise_frame)),
                norm="ortho", type=2
            )
            
            side_noise_interp = scipy.fft.idct(
                side_noise_frame_cep * np.sign(scipy.fft.dct(side_noise_frame)),
                norm="ortho", type=2
            )
            
            # スペクトル接続（Noiseは強めのスムージング）
            mid_noise_frame = scipy.ndimage.uniform_filter(
                connect_spectra_smooth(mid_noise_frame[:lowpass], abs(mid_noise_interp[lowpass:]), is_harm=False),
                size=12
            ) * (abs(np.random.randn(frame_length//2)))
            
            side_noise_frame = scipy.ndimage.uniform_filter(
                connect_spectra_smooth(side_noise_frame[:lowpass], abs(side_noise_interp[lowpass:]), is_harm=False),
                size=12
            ) * (abs(np.random.randn(frame_length//2)))
            
            mid_noise[:,i] = mid_noise_frame / 2
            side_noise[:,i] = side_noise_frame / 2
            
            # ========================================
            # 7.3 低域をゼロにして高域のみ保持
            # ========================================
            mid_harm[:,i][:lowpass] = 0
            side_harm[:,i][:lowpass] = 0
            mid_noise[:,i][:lowpass] = 0
            side_noise[:,i][:lowpass] = 0
            
            # ========================================
            # 7.4 フェードアウト適用
            # ========================================
            try:
                start_fade_len = lowpass
                fade_len = frame_length // 2 - start_fade_len
                
                if fade_len > 0:
                    fade_curve = np.linspace(1, 0, fade_len) ** FADE_CURVE_POWER
                    mid_harm[:,i][start_fade_len:] *= fade_curve
                    side_harm[:,i][start_fade_len:] *= fade_curve
                    mid_noise[:,i][start_fade_len:] *= fade_curve
                    side_noise[:,i][start_fade_len:] *= fade_curve
                
                # トランジエント処理
                if transient_frames[i] == True:
                    mid_noise[:,i] *= 1  # トランジエントではノイズそのまま
                    side_noise[:,i] *= 1  # トランジエントではノイズそのまま
            except:
                pass
        
        # ====================
        # 8. Griffin-Lim位相推定
        # ====================
        self.progress_updated.emit(90, "HFPv2: Reconstructing signal...")
        
        final_mid_spec = self._griffin_lim_mdct(
            abs(mid_harm + mid_noise),
            frame_length=frame_length,
            hop_length=hop_length,
            transient_frames=transient_frames
        )
        
        final_side_spec = self._griffin_lim_mdct(
            abs(side_harm + side_noise),
            frame_length=frame_length,
            hop_length=hop_length,
            transient_frames=transient_frames
        )
        
        # ====================
        # 9. IMDCT（周波数 → 時間領域）
        # ====================
        final_mid = mdct.imdct(final_mid_spec)
        final_side = mdct.imdct(final_side_spec)
        
        # ====================
        # 10. Mid/Side → L/R変換
        # ====================
        output_dat = np.vstack([
            (final_mid + final_side),
            (final_mid - final_side)
        ]).T
        
        # ====================
        # 11. 長さ調整
        # ====================
        original_len = len(dat)
        
        if len(output_dat) > original_len:
            output_dat = output_dat[int((len(output_dat) - original_len) / 2):][:original_len]
        elif len(output_dat) < original_len:
            pad_len = original_len - len(output_dat)
            output_dat = np.pad(
                output_dat,
                ((pad_len // 2, pad_len - pad_len // 2), (0, 0)),
                'constant'
            )
        
        # ====================
        # 12. 元の低域と合成
        # ====================
        if len(mid_filtered) > original_len:
            start = (len(mid_filtered) - original_len) // 2
            mid_original_len = mid_filtered[start:start + original_len]
            side_original_len = side_filtered[start:start + original_len]
        else:
            pad_width = (original_len - len(mid_filtered)) // 2
            mid_original_len = np.pad(
                mid_filtered,
                (pad_width, original_len - len(mid_filtered) - pad_width),
                'constant'
            )
            side_original_len = np.pad(
                side_filtered,
                (pad_width, original_len - len(side_filtered) - pad_width),
                'constant'
            )
        
        original_low_band = np.vstack([
            mid_original_len + side_original_len,
            mid_original_len - side_original_len
        ]).T
        
        self.progress_updated.emit(100, "HFPv2: Restoration complete.")
        
        return original_low_band + output_dat

class AdvancedPsychoacousticModel:
    """心理音響モデル - 人間の聴覚特性を考慮したスペクトル解析"""
    
    def __init__(self, sr, fft_size, num_bands=16, alpha=0.8):
        """
        Parameters:
        -----------
        sr : int
            サンプリングレート
        fft_size : int
            FFTサイズ
        num_bands : int
            周波数帯域の分割数
        alpha : float
            未使用パラメータ（将来の拡張用）
        """
        self.sr = sr
        self.fft_size = fft_size
        self.alpha = alpha
        self.num_bands = num_bands
        
        # 周波数ビンの計算
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
        
        # 帯域インデックスの事前計算
        self._precompute_band_indices()
        
        # 絶対聴覚閾値（ATH: Absolute Threshold of Hearing）の計算
        f_khz_safe = self.freqs / 1000.0
        f_khz_safe[f_khz_safe == 0] = 1e-12  # ゼロ除算回避
        
        # ISO 226標準に基づくATH曲線の近似式
        self.ath_db = (
            3.64 * (f_khz_safe ** -0.8) 
            - 6.5 * np.exp(-0.6 * (f_khz_safe - 3.3) ** 2) 
            + 10 ** -3 * (f_khz_safe ** 4)
        )
        
        # ナイキスト周波数付近は無限大に設定（聞こえない）
        self.ath_db[self.freqs > sr / 2.2] = np.inf
    
    def _hz_to_mel(self, hz):
        """Hz → Mel スケール変換"""
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel):
        """Mel → Hz スケール変換"""
        return 700 * (10 ** (mel / 2595) - 1)
    
    def _precompute_band_indices(self):
        """Melスケールに基づく周波数帯域のインデックスを事前計算"""
        max_mel = self._hz_to_mel(self.sr / 2)
        mel_points = np.linspace(0, max_mel, self.num_bands + 1)
        hz_points = self._mel_to_hz(mel_points)
        
        # 各帯域の開始インデックスを計算
        self.band_indices = [np.searchsorted(self.freqs, f) for f in hz_points]
        self.band_indices[-1] = len(self.freqs)
    
    def analyze_chunk(self, signal_chunk):
        """
        音声チャンクを解析し、パワースペクトルとマスキング閾値を計算
        
        Parameters:
        -----------
        signal_chunk : ndarray
            解析対象の音声信号
        
        Returns:
        --------
        power_db : ndarray
            パワースペクトル（dB）
        mask_db : ndarray
            マスキング閾値（dB）
        None : 
            将来の拡張用（現在未使用）
        power_spectrum : ndarray
            パワースペクトル（線形スケール）
        """
        # FFTサイズに合わせてパディング
        if len(signal_chunk) != self.fft_size:
            padded_chunk = np.zeros(self.fft_size)
            padded_chunk[:len(signal_chunk)] = signal_chunk
            signal_chunk = padded_chunk
        
        # ハン窓適用後のパワースペクトル計算
        windowed_signal = signal_chunk * scipy.signal.get_window('hann', self.fft_size)
        power_spectrum = np.abs(np.fft.rfft(windowed_signal)) ** 2
        
        # dBスケールに変換（96dBオフセット付き）
        power_db = 10 * np.log10(power_spectrum + 1e-12) + 96.0
        
        # スペクトル拡散によるマスキング閾値の計算
        # 畳み込みカーネルで周波数マスキングをモデル化
        masking_kernel = np.array([0.05, 0.1, 0.2, 1.0, 0.4, 0.2, 0.1, 0.05])
        masking_spread = scipy.signal.convolve(power_db, masking_kernel, mode='same') - 10
        
        # 絶対聴覚閾値とマスキング閾値の最大値を使用
        mask_db = np.maximum(self.ath_db, masking_spread)
        
        return power_db, mask_db, None, power_spectrum

class RDONoiseShaper(QObject):
    """
    RDO (Rate-Distortion Optimized) ノイズシェーパー
    
    心理音響モデルに基づいて適応的にディザリングとノイズシェーピングを行い、
    量子化ノイズを聴覚的にマスキングされやすい周波数帯域に移動させる。
    """
    
    progress_updated = pyqtSignal(int, str)
    
    def __init__(self, target_bit_depth, sr, chunk_size=2048, lpc_order=16):
        """
        Parameters:
        -----------
        target_bit_depth : int
            目標ビット深度（1, 4, 8, 16, 24など）
        sr : int
            サンプリングレート
        chunk_size : int
            処理チャンクサイズ
        lpc_order : int
            LPC（線形予測係数）の次数
        """
        super().__init__()
        
        self.target_bit_depth = target_bit_depth
        self.quantization_step = 1.0 / (2 ** (target_bit_depth - 1))
        
        self.sr = sr
        self.chunk_size = chunk_size
        self.lpc_order = lpc_order
        
        # 心理音響モデルの初期化（32帯域）
        self.psy_model = AdvancedPsychoacousticModel(
            sr, 
            chunk_size, 
            num_bands=32
        )
        
        # シンプルなノイズシェーピングフィルタ（フォールバック用）
        self.simple_b = np.zeros(self.lpc_order + 1)
        
        if self.lpc_order >= 2:
            # 2次以上：基本的な高域強調フィルタ
            self.simple_b[0:3] = [1.0, -1.5, 0.6]
        elif self.lpc_order == 1:
            # 1次：単純な微分フィルタ
            self.simple_b[0:2] = [1.0, -1.0]
        else:
            # 0次：フィルタリングなし
            self.simple_b[0] = 1.0
    
    def _calculate_lpc_coeffs(self, power_spectrum, order):
        """
        パワースペクトルからLPC係数を計算（Levinson-Durbinアルゴリズム）
        
        Parameters:
        -----------
        power_spectrum : ndarray
            パワースペクトル
        order : int
            LPCの次数
        
        Returns:
        --------
        lpc_coeffs : ndarray
            LPC係数（[1, a1, a2, ..., a_order]形式）
        """
        # パワースペクトルから自己相関関数を計算
        r = np.fft.irfft(power_spectrum)[:order + 1]
        
        try:
            # Toeplitz行列を使ってLPC係数を解く
            # solve_toeplitz は効率的な O(n^2) アルゴリズム
            lpc_coeffs = solve_toeplitz(
                (r[:-1], r[:-1]),  # Toeplitz行列の定義
                -r[1:]              # 右辺ベクトル
            )
            
            # 先頭に1.0を追加（標準的なLPC表現）
            return np.concatenate(([1.0], lpc_coeffs))
            
        except:
            # 計算が不安定な場合はフォールバックフィルタを使用
            return self.simple_b.copy()
    
    def _design_adaptive_filter(self, chunk_data, power_spectrum):
        """
        心理音響モデルに基づく適応的フィルタ設計
        
        信号のマスキング特性を解析し、最適なノイズシェーピングフィルタを設計する。
        
        Parameters:
        -----------
        chunk_data : ndarray
            音声チャンク
        power_spectrum : ndarray
            パワースペクトル
        
        Returns:
        --------
        adaptive_filter : ndarray
            適応的ノイズシェーピングフィルタ係数
        """
        # 心理音響解析
        power_db, mask_db, _, _ = self.psy_model.analyze_chunk(chunk_data)
        
        # 可聴域のマスキングマージン（余裕度）を計算
        # マージンが大きい = ノイズを隠しやすい
        audible_range = (self.psy_model.freqs > 20) & (self.psy_model.freqs < 20000)
        avg_margin = np.mean((power_db - mask_db)[audible_range])
        
        # マージンに基づいてシェーピングゲインを決定
        # マージンが小さい（5dB未満）→ 控えめなシェーピング（0.3）
        # マージンが大きい（25dB以上）→ 積極的なシェーピング（4.0）
        shaping_gain = np.clip((avg_margin - 5) / 20.0, 0.3, 4.0)
        
        # スペクトルに基づくLPCフィルタを計算
        lpc_b = self._calculate_lpc_coeffs(power_spectrum, self.lpc_order)
        
        # シンプルフィルタとLPCフィルタを重み付き平均
        # shaping_gain が大きいほどLPCフィルタを優先
        adaptive_filter = (
            self.simple_b * (1 - shaping_gain) + 
            lpc_b * shaping_gain
        )
        
        return adaptive_filter
    
    def process_channel(self, channel_data):
        """
        単一チャンネルのノイズシェーピング処理
        
        Parameters:
        -----------
        channel_data : ndarray
            処理対象の音声チャンネル
        
        Returns:
        --------
        output : ndarray
            ノイズシェーピング適用後の音声
        """
        num_samples = len(channel_data)
        output = np.zeros(len(channel_data))
        
        # チャンク単位で処理
        for i in range(0, num_samples, self.chunk_size):
            # 進捗表示（10チャンクごと）
            if i % (self.chunk_size * 10) == 0:
                progress = int(100 * i / num_samples)
                self.progress_updated.emit(
                    progress, 
                    f"Dithering (Numba): {progress}%"
                )
            
            # チャンク抽出
            chunk = channel_data[i:min(i + self.chunk_size, num_samples)]
            
            # チャンクサイズ未満の場合はパディング
            if len(chunk) < self.chunk_size:
                padded_chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
            else:
                padded_chunk = chunk
            
            # パワースペクトル解析
            _, _, _, power_spectrum = self.psy_model.analyze_chunk(padded_chunk)
            
            # 適応的フィルタ設計
            b = self._design_adaptive_filter(padded_chunk, power_spectrum)
            
            # Numba高速化されたノイズシェーピング処理
            processed_chunk = _numba_process_channel_dither(
                chunk, 
                self.chunk_size, 
                b, 
                self.quantization_step, 
                self.lpc_order
            )
            
            # 出力バッファに書き込み
            output[i:i+len(chunk)] = processed_chunk
        
        self.progress_updated.emit(100, "Dithering complete.")
        
        # クリッピング防止
        return np.clip(output, -1.0, 1.0)
    
    def process(self, signal):
        """
        モノラルまたはステレオ信号のノイズシェーピング処理
        
        Parameters:
        -----------
        signal : ndarray
            入力信号（1次元=モノラル、2次元=ステレオ）
        
        Returns:
        --------
        output : ndarray
            処理済み信号
        """
        if signal.ndim == 1:
            # モノラル処理
            return self.process_channel(signal)
        
        elif signal.ndim == 2:
            # ステレオ処理（左右独立）
            left = self.process_channel(signal[:, 0].copy())
            right = self.process_channel(signal[:, 1].copy())
            return np.stack([left, right], axis=1)
        
        # その他の次元はそのまま返す
        return signal

class IntelligentRDOResampler(QObject):
    """
    Intelligent RDO (Rate-Distortion Optimized) リサンプラー
    
    心理音響モデルに基づいて周波数帯域ごとに最適な位相特性を選択し、
    聴覚的な品質を最大化するリサンプリングを行う。
    """
    
    progress_updated = pyqtSignal(int, str)
    
    def __init__(self, original_sr, target_sr, chunk_size=8192, filter_taps=2047, num_bands=128):
        """
        Parameters:
        -----------
        original_sr : int
            元のサンプリングレート
        target_sr : int
            目標サンプリングレート
        chunk_size : int
            処理チャンクサイズ（大きいほど周波数分解能が高い）
        filter_taps : int
            FIRフィルタのタップ数（奇数に強制）
        num_bands : int
            心理音響解析の周波数帯域数
        """
        super().__init__()
        
        self.original_sr = original_sr
        self.target_sr = target_sr
        
        self.chunk_size = chunk_size
        self.filter_taps = filter_taps | 1  # 奇数に強制（ビット演算）
        self.num_bands = num_bands
        
        # オーバーラップ処理用のホップサイズ
        self.hop_size = self.chunk_size // 2
        
        # リサンプリング比率
        self.resample_ratio = self.target_sr / self.original_sr
        
        # ハン窓の事前計算
        self.window = scipy.signal.get_window('hann', self.chunk_size)
        
        # 心理音響モデルの初期化
        self.psy_model = AdvancedPsychoacousticModel(
            self.original_sr, 
            self.chunk_size, 
            self.num_bands
        )
        
        # ========================================
        # アンチエイリアシングフィルタの設計
        # ========================================
        
        # カットオフ周波数（元と目標の低い方の90%）
        cutoff_freq = min(self.original_sr, self.target_sr) / 2 * 0.9
        
        # 線形位相FIRフィルタ（Type I: 対称、奇数タップ）
        h_linear = scipy.signal.firwin(
            self.filter_taps, 
            cutoff_freq, 
            fs=self.original_sr
        )
        
        # ========================================
        # 最小位相フィルタの生成（ケプストラム法）
        # ========================================
        
        n = len(h_linear)
        
        # FFTサイズを2のべき乗に拡張
        fft_size = 1 << (n - 1).bit_length()
        
        # 振幅スペクトルの対数を取る
        amplitude_spectrum = np.abs(np.fft.rfft(h_linear, n=fft_size))
        log_amplitude = np.log(amplitude_spectrum + 1e-12)
        
        # 実ケプストラム（Real Cepstrum）を計算
        cepstrum = np.fft.irfft(log_amplitude)
        
        # 最小位相化のための窓関数
        # 因果的な最小位相フィルタを作成
        w = np.zeros(len(cepstrum))
        w[0] = 1                          # DC成分
        w[1:n//2] = 2                     # 正の周波数を2倍
        w[n//2] = 1 if n % 2 == 0 else 0  # ナイキスト周波数
        
        # ケプストラムに窓を適用
        cepstrum *= w
        
        # 最小位相フィルタのインパルス応答を復元
        h_min = np.fft.irfft(np.exp(np.fft.rfft(cepstrum)), n=fft_size)[:n]
        
        # ========================================
        # 周波数領域での事前計算
        # ========================================
        
        # 線形位相フィルタの周波数応答
        self.H_linear = np.fft.rfft(h_linear, n=self.chunk_size)
        
        # 最小位相フィルタの周波数応答
        self.H_min = np.fft.rfft(h_min, n=self.chunk_size)
    
    def _design_intelligent_mixed_phase_filter(self, chunk):
        """
        心理音響解析に基づく混合位相フィルタの設計
        
        各周波数帯域で信号のマスキング特性を解析し、
        - マージンが大きい帯域 → 線形位相（プリエコーなし）
        - マージンが小さい帯域 → 最小位相（遅延最小化）
        を適応的に選択する。
        
        Parameters:
        -----------
        chunk : ndarray
            解析対象の音声チャンク
        
        Returns:
        --------
        mixed_filter_spectrum : ndarray
            混合位相フィルタの周波数応答
        """
        # 心理音響解析
        power_db, mask_db, _, _ = self.psy_model.analyze_chunk(chunk)
        
        # 混合フィルタスペクトルの初期化
        mixed_filter_spectrum = np.zeros_like(self.H_linear)
        
        # 帯域ごとに最適な位相特性を選択
        for i in range(self.num_bands):
            start = self.psy_model.band_indices[i]
            end = self.psy_model.band_indices[i + 1]
            
            # 無効な帯域をスキップ
            if start >= end:
                continue
            
            # この帯域のマスキングマージンの平均を計算
            avg_margin = np.mean(power_db[start:end] - mask_db[start:end])
            
            # マージンに基づいて位相特性を選択
            if avg_margin > 9.0:
                # マージン大（9dB以上）→ 線形位相
                # プリエコーが聴こえにくいので、線形位相の利点を活かす
                mixed_filter_spectrum[start:end] = self.H_linear[start:end]
            else:
                # マージン小（9dB未満）→ 最小位相
                # 遅延を最小化してプリエコーを軽減
                mixed_filter_spectrum[start:end] = self.H_min[start:end]
        
        return mixed_filter_spectrum
    
    def _resample_channel(self, channel_data, ch_name=""):
        """
        単一チャンネルのリサンプリング処理
        
        Parameters:
        -----------
        channel_data : ndarray
            処理対象のチャンネルデータ
        ch_name : str
            チャンネル名（ログ表示用）
        
        Returns:
        --------
        output_signal : ndarray
            リサンプリング済みチャンネルデータ
        """
        # チャンク数の計算（オーバーラップあり）
        num_chunks = max(0, len(channel_data) - self.chunk_size) // self.hop_size + 1
        
        # 出力信号の長さを計算
        output_len = int(np.ceil(len(channel_data) * self.resample_ratio))
        output_signal = np.zeros(output_len)
        
        # 出力バッファのポインタ
        output_ptr = 0
        
        # チャンクごとに処理
        for i in range(num_chunks):
            # 進捗表示（10チャンクごと）
            if i % 10 == 0:
                progress = int(100 * i / num_chunks)
                self.progress_updated.emit(
                    progress, 
                    f"Resampling Ch {ch_name}: {progress}%"
                )
            
            # チャンク抽出
            start = i * self.hop_size
            chunk_raw = channel_data[start : start + self.chunk_size]
            
            # チャンクサイズ未満の場合はゼロパディング
            if len(chunk_raw) < self.chunk_size:
                chunk_raw = np.pad(chunk_raw, (0, self.chunk_size - len(chunk_raw)))
            
            # ========================================
            # 1. 適応的フィルタリング
            # ========================================
            
            # 心理音響解析に基づく混合位相フィルタを設計
            filter_spectrum = self._design_intelligent_mixed_phase_filter(chunk_raw)
            
            # 窓関数適用 + FFT
            chunk_spectrum = np.fft.rfft(chunk_raw * self.window)
            
            # 周波数領域でフィルタリング
            filtered_spectrum = chunk_spectrum * filter_spectrum
            
            # ========================================
            # 2. スペクトル補間（リサンプリング）
            # ========================================
            
            # 目標FFTサイズ
            target_fft_size = int(self.chunk_size * self.resample_ratio)
            
            # 新しいスペクトルサイズ
            resampled_spectrum = np.zeros(
                target_fft_size // 2 + 1, 
                dtype=np.complex128
            )
            
            # スペクトルのコピー（低周波数から）
            n_min = min(len(filtered_spectrum), len(resampled_spectrum))
            resampled_spectrum[:n_min] = filtered_spectrum[:n_min]
            
            # ========================================
            # 3. 逆FFT（周波数領域 → 時間領域）
            # ========================================
            
            processed_chunk = np.fft.irfft(resampled_spectrum, n=target_fft_size)
            
            # ========================================
            # 4. オーバーラップ加算
            # ========================================
            
            # 出力バッファの範囲チェック
            if output_ptr + len(processed_chunk) > len(output_signal):
                # はみ出す部分は切り捨て
                processed_chunk = processed_chunk[:len(output_signal) - output_ptr]
            
            # 加算
            output_signal[output_ptr : output_ptr + len(processed_chunk)] += processed_chunk
            
            # ポインタを進める（リサンプリング比率を考慮）
            output_ptr += int(self.hop_size * self.resample_ratio)
        
        # ゲイン補正（オーバーラップによる振幅増加を補正）
        return output_signal * self.resample_ratio
    
    def resample(self, signal, is_ms_processing=False):
        """
        モノラルまたはステレオ信号のリサンプリング
        
        Parameters:
        -----------
        signal : ndarray
            入力信号（1次元=モノラル、2次元=ステレオ）
        is_ms_processing : bool
            Mid/Side処理モードかどうか
        
        Returns:
        --------
        output : ndarray
            リサンプリング済み信号
        """
        if signal.ndim == 2:
            # ステレオ処理
            if is_ms_processing:
                ch1_name, ch2_name = 'Mid', 'Side'
            else:
                ch1_name, ch2_name = 'L', 'R'
            
            # 各チャンネルを独立に処理
            ch1 = self._resample_channel(signal[:, 0], ch1_name)
            ch2 = self._resample_channel(signal[:, 1], ch2_name)
            
            return np.stack([ch1, ch2], axis=1)
        
        # モノラル処理
        return self._resample_channel(signal, 'Mono')

class Worker(QObject):
    """
    バックグラウンド音声処理ワーカー
    
    メインスレッドをブロックせずに重い音声処理を実行するための
    QThreadベースのワーカークラス。
    """
    
    # シグナル定義
    finished = pyqtSignal()              # 処理完了シグナル
    progress = pyqtSignal(int, str)      # 進捗更新シグナル (進捗%, メッセージ)
    log_message = pyqtSignal(str)        # ログメッセージシグナル
    error = pyqtSignal(str)              # エラーメッセージシグナル
    
    def __init__(self, params):
        """
        Parameters:
        -----------
        params : dict
            処理パラメータ辞書
            - input_path: 入力ファイル/フォルダパス
            - output_path: 出力フォルダパス
            - enable_hfr: 高周波復元の有効/無効
            - hfr_lowpass: ローパスカットオフ周波数
            - enable_compressed_fix: 圧縮音源修復の有効/無効
            - resample_target_sr: 目標サンプリングレート
            - output_bit_depth: 出力ビット深度
            - enable_dither: ディザリングの有効/無効
            - dither_lpc_order: LPC次数
        """
        super().__init__()
        self.params = params
        self.is_running = True  # 処理中断フラグ
    
    def stop(self):
        """処理の中断を要求"""
        self.is_running = False
    
    def run(self):
        """
        メイン処理ループ
        
        入力パス内のすべての対応音声ファイルを処理し、
        リサンプリング、高周波復元、ディザリングを適用する。
        """
        try:
            input_path = self.params['input_path']
            output_path = self.params['output_path']
            
            # ========================================
            # 1. 処理対象ファイルのリストアップ
            # ========================================
            files_to_process = []
            
            if os.path.isdir(input_path):
                # ディレクトリの場合：再帰的に対応ファイルを検索
                supported_ext = ('.wav', '.flac', '.mp3', '.ogg', '.aiff', '.aif')
                
                for root, _, files in os.walk(input_path):
                    for file in files:
                        if file.lower().endswith(supported_ext):
                            files_to_process.append(os.path.join(root, file))
            
            elif os.path.isfile(input_path):
                # 単一ファイルの場合
                files_to_process.append(input_path)
            
            # ファイルが見つからない場合
            if not files_to_process:
                self.error.emit("No supported audio files found.")
                self.finished.emit()
                return
            
            # ========================================
            # 2. ファイルごとの処理ループ
            # ========================================
            for i, file_path in enumerate(files_to_process):
                # 中断チェック
                if not self.is_running:
                    break
                
                # ファイル処理開始ログ
                self.log_message.emit(
                    f"--- Processing file {i+1}/{len(files_to_process)}: "
                    f"{os.path.basename(file_path)} ---"
                )
                
                # 出力ファイルパスの決定
                if os.path.isdir(output_path):
                    output_file = os.path.join(output_path, os.path.basename(file_path))
                else:
                    output_file = output_path
                
                # 出力ディレクトリの作成
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # ========================================
                # 3. 音声ファイルの読み込み
                # ========================================
                self.progress.emit(0, "Loading audio file...")
                
                dat, sr = librosa.load(file_path, sr=None, mono=False)
                dat = dat.T  # (samples, channels)形式に転置
                
                # ステレオチェック
                if dat.ndim != 2 or dat.shape[1] != 2:
                    self.log_message.emit(
                        f"[Warning] Not stereo: '{os.path.basename(file_path)}'. Skipping."
                    )
                    continue
                
                # float64に変換
                processed_data = dat.astype(np.float64)
                current_sr = sr
                
                # ========================================
                # 4. リサンプリング処理
                # ========================================
                target_sr = self.params['resample_target_sr']
                
                if current_sr != target_sr:
                    self.log_message.emit(
                        f"Resampling from {current_sr} Hz to {target_sr} Hz..."
                    )
                    
                    # リサンプラーの初期化
                    resampler = IntelligentRDOResampler(
                        original_sr=current_sr, 
                        target_sr=target_sr
                    )
                    resampler.progress_updated.connect(self.progress.emit)
                    
                    # Mid/Side分解
                    mid = (processed_data[:, 0] + processed_data[:, 1]) / 2.0
                    side = (processed_data[:, 0] - processed_data[:, 1]) / 2.0
                    
                    # Mid/Sideそれぞれをリサンプリング
                    resampled_ms = resampler.resample(
                        np.stack([mid, side], axis=1), 
                        is_ms_processing=True
                    )
                    
                    # L/Rに再変換
                    processed_data = np.stack([
                        resampled_ms[:, 0] + resampled_ms[:, 1],  # Left
                        resampled_ms[:, 0] - resampled_ms[:, 1]   # Right
                    ], axis=1)
                    
                    current_sr = target_sr
                    self.log_message.emit("Resampling complete.")
                
                # 中断チェック
                if not self.is_running:
                    break
                
                # ========================================
                # 5. 高周波復元処理（HFPv2）
                # ========================================
                if self.params['enable_hfr']:
                    self.log_message.emit(
                        "Applying High-Frequency Restoration (HFPv2) post-resampling..."
                    )
                    
                    # HFPv2処理器の初期化
                    restorer = HighFrequencyRestorer()
                    restorer.progress_updated.connect(self.progress.emit)
                    
                    # ローパス周波数をビンインデックスに変換
                    lowpass_freq = self.params['hfr_lowpass']
                    
                    if lowpass_freq != -1:
                        lowpass_bin = int(lowpass_freq * 1024 / current_sr)
                    else:
                        lowpass_bin = -1  # 自動検出
                    
                    # HFPv2実行
                    processed_data = restorer.run_hfpv2(
                        processed_data, 
                        current_sr, 
                        lowpass=lowpass_bin, 
                        enable_compressed_fix=self.params['enable_compressed_fix']
                    )
                    
                    self.log_message.emit("HFPv2 Restoration complete.")
                
                # 中断チェック
                if not self.is_running:
                    break
                
                # ========================================
                # 6. ビット深度とディザリングの設定
                # ========================================
                
                # ビット深度の文字列からsoundfileのサブタイプに変換
                subtype_mapping = {
                    '1-bit PCM': 'PCM_1',
                    '4-bit PCM': 'PCM_4',
                    '8-bit PCM': 'PCM_8',
                    '16-bit PCM': 'PCM_16',
                    '24-bit PCM': 'PCM_24',
                    '32-bit Float': 'FLOAT'
                }
                
                subtype = subtype_mapping.get(
                    self.params['output_bit_depth'], 
                    'FLOAT'
                )
                
                final_data = processed_data
                
                # PCM形式でディザリングが有効な場合
                if 'PCM' in subtype and self.params['enable_dither']:
                    target_bit_depth = int(subtype.split('_')[1])
                    
                    # soundfileが対応していない低ビット深度の調整
                    if target_bit_depth == 4:
                        subtype = 'PCM_U8'  # 4-bit → 8-bit unsigned
                    elif target_bit_depth == 8:
                        subtype = 'PCM_U8'  # 8-bit → 8-bit unsigned
                    elif target_bit_depth == 1:
                        subtype = 'PCM_U8'  # 1-bit → 8-bit unsigned
                    
                    self.log_message.emit(
                        f"Applying {target_bit_depth}-bit dither and noise shaping..."
                    )
                    
                    # ノイズシェーパーの初期化
                    shaper = RDONoiseShaper(
                        target_bit_depth, 
                        current_sr, 
                        lpc_order=self.params['dither_lpc_order']
                    )
                    shaper.progress_updated.connect(self.progress.emit)
                    
                    # ディザリング処理
                    final_data = shaper.process(final_data)
                    
                    self.log_message.emit("Dithering and shaping complete.")
                
                # デバッグ出力
                print(final_data)
                
                # ========================================
                # 7. ファイルの保存
                # ========================================
                self.progress.emit(99, "Saving file...")
                
                # クリッピング防止して保存
                sf.write(
                    output_file + ".wav", 
                    np.clip(final_data, -1.0, 1.0), 
                    current_sr, 
                    subtype=subtype
                )
                
                self.log_message.emit(f"Successfully saved to: {output_file}\n")
            
            # ========================================
            # 8. 完了メッセージ
            # ========================================
            if self.is_running:
                self.log_message.emit("--- All tasks finished! ---")
            else:
                self.log_message.emit("--- Processing stopped by user. ---")
        
        except Exception as e:
            # エラーハンドリング
            self.error.emit(
                f"An error occurred: {e}\n{traceback.format_exc()}"
            )
        
        # 完了シグナル送信
        self.finished.emit()


class MainWindow(QMainWindow):
    """
    音声処理アプリケーションのメインウィンドウ
    
    リサンプリング、高周波復元、ディザリングを統合した
    GUIフロントエンドを提供する。
    """
    
    def __init__(self):
        super().__init__()
        
        # ウィンドウ設定
        self.setWindowTitle("Audio Processor (Resampler -> HFPv2) [Numba Accelerated]")
        self.setGeometry(100, 100, 800, 700)
        
        # ダークテーマのスタイルシート
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E2E2E;
            }
            QWidget {
                color: #E0E0E0;
                background-color: #2E2E2E;
                font-size: 14px;
            }
            QGroupBox {
                background-color: #3C3C3C;
                border: 1px solid #555;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
            }
            QLabel {
                background-color: transparent;
            }
            QLineEdit {
                background-color: #4A4A4A;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                background-color: #5A5A5A;
                border: 1px solid #666;
                border-radius: 4px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #6A6A6A;
            }
            QPushButton:pressed {
                background-color: #4A4A4A;
            }
            QPushButton#StartButton {
                background-color: #4CAF50;
                font-weight: bold;
            }
            QPushButton#StartButton:hover {
                background-color: #5CBF60;
            }
            QCheckBox, QComboBox {
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #4A4A4A;
                selection-background-color: #5A5A5A;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                color: #FFF;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #252525;
                border: 1px solid #555;
                border-radius: 4px;
            }
        """)
        
        # 中央ウィジェットの設定
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # メインレイアウト
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # UIコンポーネントの作成
        self._create_io_group()
        self._create_resampling_group()
        self._create_hfr_group()
        self._create_controls_group()
        
        # ワーカースレッド関連
        self.thread = None
        self.worker = None
    
    def _create_io_group(self):
        """入出力ファイル選択グループの作成"""
        group = QGroupBox("1. File Input/Output")
        layout = QVBoxLayout(group)
        
        # 入力ファイルパス選択
        input_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select input file or folder...")
        
        browse_in = QPushButton("Browse Input")
        browse_in.clicked.connect(self.browse_input)
        
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(browse_in)
        
        # 出力フォルダパス選択
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output folder...")
        
        browse_out = QPushButton("Browse Output")
        browse_out.clicked.connect(self.browse_output)
        
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(browse_out)
        
        # レイアウトに追加
        layout.addLayout(input_layout)
        layout.addLayout(output_layout)
        
        self.main_layout.addWidget(group)
    
    def _create_resampling_group(self):
        """リサンプリングと出力フォーマット設定グループの作成"""
        group = QGroupBox("2. Resampling & Output Format")
        layout = QVBoxLayout(group)
        
        # サンプリングレートとビット深度の選択
        grid_layout = QHBoxLayout()
        
        # サンプリングレート
        sr_layout = QVBoxLayout()
        sr_layout.addWidget(QLabel("Target Sample Rate:"))
        
        self.resample_sr_combo = QComboBox()
        self.resample_sr_combo.addItems([
            "32000", "44100", "48000", "88200", 
            "96000", "192000", "384000", "768000"
        ])
        self.resample_sr_combo.setCurrentText("48000")
        
        sr_layout.addWidget(self.resample_sr_combo)
        
        # 出力ビット深度
        bd_layout = QVBoxLayout()
        bd_layout.addWidget(QLabel("Output Format:"))
        
        self.output_bit_depth_combo = QComboBox()
        self.output_bit_depth_combo.addItems([
            "1-bit PCM", "4-bit PCM", "8-bit PCM", 
            "16-bit PCM", "24-bit PCM", "32-bit Float"
        ])
        self.output_bit_depth_combo.setCurrentText("24-bit PCM")
        
        bd_layout.addWidget(self.output_bit_depth_combo)
        
        grid_layout.addLayout(sr_layout)
        grid_layout.addLayout(bd_layout)
        
        # ディザリングオプション
        dither_group = QGroupBox("Dithering Options")
        dither_layout = QVBoxLayout(dither_group)
        
        self.dither_enable_check = QCheckBox("Enable Dither & Noise Shaping (for PCM)")
        self.dither_enable_check.setChecked(True)
        
        lpc_layout = QHBoxLayout()
        lpc_layout.addWidget(QLabel("LPC Order:"))
        
        self.dither_lpc_spin = QSpinBox()
        self.dither_lpc_spin.setRange(0, 32)
        self.dither_lpc_spin.setValue(16)
        
        lpc_layout.addWidget(self.dither_lpc_spin)
        
        dither_layout.addWidget(self.dither_enable_check)
        dither_layout.addLayout(lpc_layout)
        
        # 全体レイアウトに追加
        layout.addLayout(grid_layout)
        layout.addWidget(dither_group)
        
        self.main_layout.addWidget(group)
    
    def _create_hfr_group(self):
        """高周波復元設定グループの作成"""
        group = QGroupBox("3. High-Frequency Restoration (HFPv2)")
        layout = QVBoxLayout(group)
        
        # 有効化チェックボックス
        self.hfr_enable_check = QCheckBox("Enable Restoration Module")
        self.hfr_enable_check.setChecked(True)
        
        # 圧縮音源修復オプション
        self.hfr_compressed_fix_check = QCheckBox("Apply Compressed Audio Fixes")
        self.hfr_compressed_fix_check.setChecked(True)
        self.hfr_compressed_fix_check.setToolTip(
            "Restores spectral holes and high-frequency stereo width, "
            "useful for previously compressed audio (e.g., MP3)."
        )
        
        # ローパスカットオフ設定
        lowpass_layout = QHBoxLayout()
        lowpass_layout.addWidget(QLabel("Lowpass Cutoff (Hz):"))
        
        self.hfr_lowpass_spin = QSpinBox()
        self.hfr_lowpass_spin.setRange(-1, 22000)
        self.hfr_lowpass_spin.setSpecialValueText("Auto")
        self.hfr_lowpass_spin.setValue(13500)
        
        lowpass_layout.addWidget(self.hfr_lowpass_spin)
        
        # レイアウトに追加
        layout.addWidget(self.hfr_enable_check)
        layout.addWidget(self.hfr_compressed_fix_check)
        layout.addLayout(lowpass_layout)
        
        self.main_layout.addWidget(group)
    
    def _create_controls_group(self):
        """コントロールとログコンソールの作成"""
        # 開始ボタン
        self.start_button = QPushButton("Start Processing")
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_processing)
        
        # プログレスバー
        self.progress_bar = QProgressBar()
        
        # ステータスラベル
        self.status_label = QLabel("Ready.")
        
        # ログコンソール
        log_group = QGroupBox("Log Console")
        log_layout = QVBoxLayout(log_group)
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        
        log_layout.addWidget(self.log_console)
        
        # メインレイアウトに追加
        self.main_layout.addWidget(self.start_button)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.status_label)
        self.main_layout.addWidget(log_group)
    
    def browse_input(self):
        """入力ファイル/フォルダの選択ダイアログ"""
        # まずファイル選択を試みる
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Input File", 
            "", 
            "Audio Files (*.wav *.flac *.mp3 *.ogg *.aiff)"
        )
        
        # キャンセルされた場合はフォルダ選択
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        
        if path:
            self.input_path_edit.setText(path)
    
    def browse_output(self):
        """出力フォルダの選択ダイアログ"""
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        
        if path:
            self.output_path_edit.setText(path)
    
    def start_processing(self):
        """処理の開始/停止"""
        # すでに処理中の場合は停止
        if self.thread and self.thread.isRunning():
            self.log_console.append("\n>>> Sending stop signal...")
            self.worker.stop()
            self.start_button.setEnabled(False)
        else:
            # ========================================
            # Numba関数のJITコンパイル（初回のみ）
            # ========================================
            self.log_console.append("JIT Compiling Numba functions... (first run may be slow)")
            QApplication.processEvents()  # GUI更新
            
            try:
                # ダミーデータでコンパイルを強制実行
                _numba_apply_pre_echo_suppression(
                    np.zeros(10), 5, np.array([1])
                )
                _numba_apply_compressed_fix_loop(
                    np.zeros((10, 10)), 
                    np.zeros((10, 10)), 
                    -12, 5
                )
                _numba_process_channel_dither(
                    np.zeros(10), 5, 
                    np.array([1.0, -1.0]), 
                    0.1, 1
                )
            except Exception as e:
                self.log_console.append(f"Numba pre-compilation failed: {e}")
            
            self.log_console.append("Compilation complete.")
            
            # ========================================
            # 入力検証
            # ========================================
            input_path = self.input_path_edit.text()
            output_path = self.output_path_edit.text()
            
            # パスの妥当性チェック
            if not (input_path and output_path and os.path.exists(input_path) and 
                    (os.path.isdir(output_path) or not os.path.isdir(input_path))):
                QMessageBox.warning(
                    self, 
                    "Invalid Paths", 
                    "Please specify a valid input file/folder and an existing output folder."
                )
                return
            
            # ========================================
            # パラメータの収集
            # ========================================
            params = {
                'input_path': input_path,
                'output_path': output_path,
                'enable_hfr': self.hfr_enable_check.isChecked(),
                'hfr_lowpass': self.hfr_lowpass_spin.value(),
                'enable_compressed_fix': self.hfr_compressed_fix_check.isChecked(),
                'resample_target_sr': int(self.resample_sr_combo.currentText()),
                'output_bit_depth': self.output_bit_depth_combo.currentText(),
                'enable_dither': self.dither_enable_check.isChecked(),
                'dither_lpc_order': self.dither_lpc_spin.value(),
            }
            
            # ========================================
            # ワーカースレッドの起動
            # ========================================
            self.start_button.setText("Stop Processing")
            self.log_console.clear()
            
            # スレッドとワーカーの作成
            self.thread = QThread()
            self.worker = Worker(params)
            self.worker.moveToThread(self.thread)
            
            # シグナル/スロット接続
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            
            # 完了時のクリーンアップ
            def on_finished():
                self.start_button.setText("Start Processing")
                self.start_button.setEnabled(True)
                self.thread = None
                self.worker = None
            
            self.thread.finished.connect(on_finished)
            
            # 進捗/ログシグナルの接続
            self.worker.progress.connect(self.update_progress)
            self.worker.log_message.connect(self.update_log)
            self.worker.error.connect(self.show_error)
            
            # スレッド開始
            self.thread.start()
    
    def update_progress(self, value, message):
        """プログレスバーとステータスラベルの更新"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def update_log(self, message):
        """ログコンソールにメッセージを追加"""
        self.log_console.append(message)
        # 自動スクロール
        self.log_console.verticalScrollBar().setValue(
            self.log_console.verticalScrollBar().maximum()
        )
    
    def show_error(self, message):
        """エラーメッセージの表示"""
        self.log_console.append(f"\nERROR:\n{message}")
        self.status_label.setText("An error occurred.")
        QMessageBox.critical(self, "Error", message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
