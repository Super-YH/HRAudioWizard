"""
HRAudioWizard - High-Resolution Audio Processing Package

A professional audio processing toolkit featuring:
- ScipyMDCT: Modified Discrete Cosine Transform
- Griffin-Lim MDCT: Phase estimation algorithm
- HFPv2: High-Frequency Restoration
- RDO Noise Shaper: Psychoacoustic dithering
- Intelligent Resampler: Adaptive resampling

Usage:
    from HRAudioWizard import MainWindow
    # or run directly:
    # python -m HRAudioWizard
"""

from .mdct import ScipyMDCT, mdct
from .numba_utils import (
    _numba_tukey_window,
    _numba_apply_pre_echo_suppression,
    _numba_apply_compressed_fix_loop,
    _numba_process_channel_dither,
    _fast_sign_with_magnitude,
    _apply_momentum_update
)
from .griffin_lim import griffin_lim_mdct
from .spectra_utils import OVERTONE, connect_spectra_smooth, flatten_spectrum, griffin_lim_stft
from .psychoacoustic import AdvancedPsychoacousticModel
from .hfr import HighFrequencyRestorer
from .hfp_v1 import HighFrequencyRestorerV1
from .noise_shaper import RDONoiseShaper
from .resampler import IntelligentRDOResampler
from .localization import Localization, STRINGS
from .flet_ui import AudioWizardApp

__version__ = "3.0.0"
__author__ = "HRAudioWizard Team"

__all__ = [
    # Core
    "ScipyMDCT",
    "mdct",
    "griffin_lim_mdct",
    "OVERTONE",
    "connect_spectra_smooth",
    "flatten_spectrum",
    "griffin_lim_stft",
    "AdvancedPsychoacousticModel",
    
    # Processing
    "HighFrequencyRestorer",
    "HighFrequencyRestorerV1",
    "RDONoiseShaper", 
    "IntelligentRDOResampler",
    
    # UI
    "AudioWizardApp",
    "Localization",
    "STRINGS",
    
    # Numba utils
    "_numba_tukey_window",
    "_numba_apply_pre_echo_suppression",
    "_numba_apply_compressed_fix_loop",
    "_numba_process_channel_dither",
    "_fast_sign_with_magnitude",
    "_apply_momentum_update",
]
