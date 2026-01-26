"""
RDO (Rate-Distortion Optimized) Noise Shaper module.

Implements psychoacoustically-informed noise shaping for
optimal dithering during quantization.
"""

import numpy as np
from scipy.linalg import solve_toeplitz
from PyQt6.QtCore import QObject, pyqtSignal

# Import with fallback for both direct and package execution
try:
    from .psychoacoustic import AdvancedPsychoacousticModel
    from .numba_utils import _numba_process_channel_dither
except ImportError:
    from psychoacoustic import AdvancedPsychoacousticModel
    from numba_utils import _numba_process_channel_dither


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
