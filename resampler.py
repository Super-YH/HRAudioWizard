"""
Intelligent RDO (Rate-Distortion Optimized) Resampler module.

Implements psychoacoustically-informed resampling with
adaptive mixed-phase filtering.
"""

import numpy as np
import scipy.signal
from PyQt6.QtCore import QObject, pyqtSignal

# Import with fallback for both direct and package execution
try:
    from .psychoacoustic import AdvancedPsychoacousticModel
except ImportError:
    from psychoacoustic import AdvancedPsychoacousticModel


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
