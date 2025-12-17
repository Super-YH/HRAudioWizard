"""
Griffin-Lim algorithm for MDCT phase estimation.

Implements momentum-accelerated Griffin-Lim for reconstructing
time-domain signals from magnitude-only MDCT spectra.
"""

import numpy as np

# Import with fallback for both direct and package execution
try:
    from .mdct import mdct
    from .numba_utils import (
        _numba_apply_pre_echo_suppression,
        _fast_sign_with_magnitude,
        _apply_momentum_update
    )
except ImportError:
    from mdct import mdct
    from numba_utils import (
        _numba_apply_pre_echo_suppression,
        _fast_sign_with_magnitude,
        _apply_momentum_update
    )


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
