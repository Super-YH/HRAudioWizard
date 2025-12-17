# HRAudioWizard

**高解像度オーディオ処理ツールキット**

先進的な信号処理アルゴリズムと心理音響モデルを用いて、オーディオ品質を向上・復元するプロフェッショナルグレードの音声処理アプリケーションです。

![バージョン](https://img.shields.io/badge/version-3.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![ライセンス](https://img.shields.io/badge/license-MIT-orange)

## 機能

### 🎵 高周波復元 (HFR)
非可逆圧縮や帯域制限によって失われた高周波成分を復元します。

- **HFP V2 (MDCT方式)**: 修正離散コサイン変換と倍音構造解析を使用
- **HFP V1 (STFT方式)**: 短時間フーリエ変換とHPSS分離を使用

### 🔊 インテリジェントリサンプリング
心理音響モデルに基づく適応型混合位相フィルタリングを備えたリサンプラー。

- 対応: 32kHz, 44.1kHz, 48kHz, 88.2kHz, 96kHz, 192kHz
- 知覚損失を最小化するレート歪み最適化 (RDO)

### 🎚️ RDOノイズシェーピング
心理音響ノイズシェーピングによる高度なディザリング。

- 16bit、24bit PCM および 32bit Float対応
- 適応型LPCベースのノイズシェーピングフィルタ
- 設定可能なLPC次数 (1-32)

### 💻 モダンUI
Fletフレームワークで構築されたクロスプラットフォームGUI。

- グラスモーフィズムを採用したダークテーマ
- ファイルリスト管理によるバッチ処理
- リアルタイム進捗表示
- 日本語・英語の2言語対応

## インストール

### 前提条件
- Python 3.8以上
- pipパッケージマネージャー

### 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 必須パッケージ
- `numpy` >= 1.21.0
- `scipy` >= 1.7.0
- `librosa` >= 0.9.0
- `soundfile` >= 0.10.0
- `numba` >= 0.55.0
- `flet` >= 0.20.0

## 使い方

### アプリケーションの起動

```bash
python -m HRAudioWizard
```

またはFlet UIを直接起動:

```bash
python flet_ui.py
```

### プログラムからの使用

```python
from HRAudioWizard import HighFrequencyRestorer, RDONoiseShaper, IntelligentRDOResampler
import librosa
import soundfile as sf

# オーディオ読み込み
audio, sr = librosa.load("input.mp3", sr=None, mono=False)
audio = audio.T  # (サンプル数, チャンネル数)

# 高周波復元
restorer = HighFrequencyRestorer()
restored = restorer.run_hfpv2(audio, sr, lowpass=-1, enable_compressed_fix=True)

# 96kHzにリサンプリング
resampler = IntelligentRDOResampler(sr, 96000)
resampled = resampler.resample(restored)

# 24bit出力用にノイズシェーピング適用
shaper = RDONoiseShaper(24, 96000, lpc_order=16)
output = shaper.process(resampled)

# 保存
sf.write("output.wav", output, 96000, subtype="PCM_24")
```

## プロジェクト構成

```
HRAudioWizard/
├── __init__.py           # パッケージ初期化
├── __main__.py           # エントリーポイント
├── flet_ui.py            # クロスプラットフォームGUI (Flet)
├── hfr.py                # HFPv2 高周波復元
├── hfp_v1.py             # HFP V1 (STFT方式) アルゴリズム
├── mdct.py               # MDCT/IMDCT実装
├── griffin_lim.py        # Griffin-Lim位相再構成
├── noise_shaper.py       # RDOノイズシェーピング/ディザリング
├── resampler.py          # インテリジェントRDOリサンプラー
├── psychoacoustic.py     # 心理音響モデル
├── spectra_utils.py      # スペクトルユーティリティ
├── numba_utils.py        # JITコンパイル関数
├── localization.py       # 多言語対応 (EN/JP)
└── requirements.txt      # 依存関係
```

## アルゴリズム

### HFPv2アルゴリズム
1. ステレオ処理のためのMid/Side分解
2. オンセット検出によるトランジェント検出
3. MDCT変換とHPSS分離
4. ケプストラムによる倍音構造解析
5. 相関分析に基づく倍音外挿
6. Griffin-Lim位相推定
7. スムーズなクロスフェードによるスペクトル接続

### ノイズシェーピング
1. マスキングカーブを用いた心理音響解析
2. LPC係数の計算 (Levinson-Durbinアルゴリズム)
3. マスキング閾値に基づく適応フィルタ設計
4. フィードバック誤差シェーピングを伴うTPDFディザリング

## 対応フォーマット

**入力**: WAV, FLAC, MP3, OGG, AIFF  
**出力**: WAV (16bit, 24bit PCM / 32bit Float)

## ライセンス

MITライセンス

## 作者

HRAudioWizard Team / SYH99999
