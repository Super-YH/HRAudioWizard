"""
Advanced Psychoacoustic Model.

Implements human auditory system modeling including:
- Absolute Threshold of Hearing (ATH)
- Mel-scale frequency bands
- Spectral masking estimation
"""

import numpy as np
import scipy.signal


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
