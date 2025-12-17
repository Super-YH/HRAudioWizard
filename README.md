# HRAudioWizard

**High-Resolution Audio Processing Toolkit**

A professional-grade audio processing application that enhances and restores audio quality using advanced signal processing algorithms and psychoacoustic models.

![Version](https://img.shields.io/badge/version-3.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## Features

### 🎵 High-Frequency Restoration (HFR)
Restores high-frequency content lost due to lossy compression or bandwidth limitation.

- **HFP V2 (MDCT-based)**: Uses Modified Discrete Cosine Transform with harmonic structure analysis
- **HFP V1 (STFT-based)**: Short-Time Fourier Transform based approach with HPSS separation

### 🔊 Intelligent Resampling
Psychoacoustically-informed resampler with adaptive mixed-phase filtering.

- Supports: 32kHz, 44.1kHz, 48kHz, 88.2kHz, 96kHz, 192kHz
- Rate-Distortion Optimization (RDO) for minimal perceptual loss

### 🎚️ RDO Noise Shaping
Advanced dithering with psychoacoustic noise shaping.

- Supports 16-bit, 24-bit PCM and 32-bit Float
- Adaptive LPC-based noise shaping filter
- Configurable LPC order (1-32)

### 💻 Modern UI
Cross-platform GUI built with Flet framework.

- Dark theme with glassmorphism design
- Batch processing with file list management
- Real-time progress tracking
- Bilingual support (English/Japanese)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
- `numpy` >= 1.21.0
- `scipy` >= 1.7.0
- `librosa` >= 0.9.0
- `soundfile` >= 0.10.0
- `numba` >= 0.55.0
- `flet` >= 0.20.0

## Usage

### Run the Application

```bash
python -m HRAudioWizard
```

Or run the Flet UI directly:

```bash
python flet_ui.py
```

### Programmatic Usage

```python
from HRAudioWizard import HighFrequencyRestorer, RDONoiseShaper, IntelligentRDOResampler
import librosa
import soundfile as sf

# Load audio
audio, sr = librosa.load("input.mp3", sr=None, mono=False)
audio = audio.T  # (samples, channels)

# High-frequency restoration
restorer = HighFrequencyRestorer()
restored = restorer.run_hfpv2(audio, sr, lowpass=-1, enable_compressed_fix=True)

# Resample to 96kHz
resampler = IntelligentRDOResampler(sr, 96000)
resampled = resampler.resample(restored)

# Apply noise shaping for 24-bit output
shaper = RDONoiseShaper(24, 96000, lpc_order=16)
output = shaper.process(resampled)

# Save
sf.write("output.wav", output, 96000, subtype="PCM_24")
```

## Project Structure

```
HRAudioWizard/
├── __init__.py           # Package initialization
├── __main__.py           # Entry point
├── flet_ui.py            # Cross-platform GUI (Flet)
├── hfr.py                # HFPv2 high-frequency restoration
├── hfp_v1.py             # HFP V1 (STFT-based) algorithm
├── mdct.py               # MDCT/IMDCT implementation
├── griffin_lim.py        # Griffin-Lim phase reconstruction
├── noise_shaper.py       # RDO noise shaping/dithering
├── resampler.py          # Intelligent RDO resampler
├── psychoacoustic.py     # Psychoacoustic model
├── spectra_utils.py      # Spectral utilities
├── numba_utils.py        # JIT-compiled functions
├── localization.py       # i18n support (EN/JP)
└── requirements.txt      # Dependencies
```

## Algorithms

### HFPv2 Algorithm
1. Mid/Side decomposition for stereo processing
2. Transient detection using onset detection
3. MDCT transformation and HPSS separation
4. Harmonic structure analysis via cepstrum
5. Overtone extrapolation based on correlation analysis
6. Griffin-Lim phase estimation
7. Spectral connection with smooth crossfade

### Noise Shaping
1. Psychoacoustic analysis using masking curves
2. LPC coefficient calculation (Levinson-Durbin)
3. Adaptive filter design based on masking threshold
4. TPDF dithering with feedback error shaping

## Supported Formats

**Input**: WAV, FLAC, MP3, OGG, AIFF  
**Output**: WAV (16-bit, 24-bit PCM / 32-bit Float)

## License

MIT License

## Author

HRAudioWizard Team / SYH99999
