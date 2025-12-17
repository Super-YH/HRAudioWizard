"""
MDCT (Modified Discrete Cosine Transform) module.

ScipyMDCT provides MDCT/IMDCT transforms using scipy DCT-IV.
Used for time-frequency analysis in audio processing.
"""

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


# Module-level default instance
mdct = ScipyMDCT()
