"""
High-Frequency Restoration (HFPv2) module.

Implements the HFPv2 algorithm for restoring high-frequency content
in bandwidth-limited audio using harmonic extrapolation.
"""

import numpy as np
import scipy.signal
import scipy.ndimage
import scipy.fft
import librosa

# Import with fallback for both direct and package execution
try:
    from .mdct import mdct
    from .numba_utils import _numba_apply_compressed_fix_loop
    from .griffin_lim import griffin_lim_mdct
    from .spectra_utils import OVERTONE, connect_spectra_smooth
except ImportError:
    from mdct import mdct
    from numba_utils import _numba_apply_compressed_fix_loop
    from griffin_lim import griffin_lim_mdct
    from spectra_utils import OVERTONE, connect_spectra_smooth


class HighFrequencyRestorer:
    """
    高周波復元プロセッサ（HFPv2）
    
    倍音構造解析に基づいて帯域制限された音声の
    高周波成分を復元する。
    """
    
    def __init__(self):
        self._progress_callback = None
    
    @property
    def progress_updated(self):
        """Return self to allow .connect() pattern"""
        return self
    
    def connect(self, callback):
        """Set progress callback (PyQt compatibility)"""
        self._progress_callback = callback
    
    def emit(self, value, msg):
        """Emit progress update"""
        if self._progress_callback:
            self._progress_callback(value, msg)
    
    def _apply_compressed_fix(self, mid_spec, side_spec, threshold_db=-64, high_freq_bin_start_ratio=0.15):
        """
        圧縮音源の修復処理
        
        Parameters:
        -----------
        mid_spec : ndarray
            Mid channel spectrum
        side_spec : ndarray
            Side channel spectrum
        threshold_db : float
            Threshold in dB
        high_freq_bin_start_ratio : float
            Ratio for high frequency start bin
        
        Returns:
        --------
        mid_spec_proc, side_spec_proc : tuple
            Processed spectra
        """
        num_bins = mid_spec.shape[0]
        high_freq_bin_start = int(num_bins * high_freq_bin_start_ratio)
        mid_spec_proc, side_spec_proc = _numba_apply_compressed_fix_loop(
            mid_spec.copy(), side_spec.copy(), threshold_db, high_freq_bin_start)
        return mid_spec_proc, side_spec_proc

    def _griffin_lim_mdct(self, magnitudes, frame_length=2048, hop_length=1024, n_iter=35, transient_frames=None):
        """Wrapper for Griffin-Lim MDCT"""
        return griffin_lim_mdct(magnitudes, frame_length, hop_length, n_iter, transient_frames)

    def _connect_spectra_smooth(self, signal1, signal2, is_harm=True):
        """Wrapper for spectral connection"""
        if len(signal1) == 0 or len(signal2) == 0:
            return np.concatenate([signal1, signal2])
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
        CORRELATION_THRESHOLD = 0.75
        MIN_HARMONIC_ORDER = 2
        CEPSTRUM_PEAK_THRESHOLD = 64
        WIDTH_EXPANSION_FACTOR = 3
        FADE_CURVE_POWER = 1.22
        
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
        mid_noise = scipy.signal.medfilt2d(mid_ffted, kernel_size=(3,7))
        side_noise = scipy.signal.medfilt2d(side_ffted, kernel_size=(3,7))
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
            
            mid_harm[:,i] = mid_harm_frame 
            side_harm[:,i] = side_harm_frame
            
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
            
            mid_noise[:,i] = mid_noise_frame
            side_noise[:,i] = side_noise_frame
            
            # ========================================
            # 7.3 低域をゼロにして高域のみ保持
            # ========================================
            
            mid_harm[:,i][librosa.amplitude_to_db(mid_ffted[:,i])>-48] *= 1e-5
            side_harm[:,i][librosa.amplitude_to_db(side_ffted[:,i])>-48] *= 1e-5
            mid_noise[:,i][librosa.amplitude_to_db(mid_ffted[:,i])>-36] *= 1e-5
            side_noise[:,i][librosa.amplitude_to_db(side_ffted[:,i])>-36] *= 1e-5
            mid_harm[:,i][:128] *= 1e-6
            side_harm[:,i][:128] *= 1e-6
            mid_noise[:,i][:128] *= 1e-6
            side_noise[:,i][:128] *= 1e-6

            # ========================================
            # 7.4 フェードアウト適用
            # ========================================
            try:
                start_fade_len = lowpass
                fade_len = frame_length // 2 - start_fade_len
                
                if fade_len > 0:
                    fade_curve = np.linspace(FADE_CURVE_POWER, 0, fade_len) ** FADE_CURVE_POWER
                    mid_harm[:,i][start_fade_len:] *= fade_curve
                    side_harm[:,i][start_fade_len:] *= fade_curve
                    mid_noise[:,i][start_fade_len:] *= fade_curve
                    side_noise[:,i][start_fade_len:] *= fade_curve
                
                # トランジエント処理
                if transient_frames[i] == True:
                    mid_noise[:,i] *= 1.2  # トランジエントではノイズそのまま
                    side_noise[:,i] *= 1.2  # トランジエントではノイズそのまま
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
