"""
HRAudioWizard - Audio Processing Application

Entry point for the audio processing application featuring:
- Intelligent RDO Resampling
- High-Frequency Restoration (HFPv2 / HFP V1)
- Psychoacoustic Noise Shaping

Usage:
    python main.py              # Run Flet GUI
    python -m HRAudioWizard     # Run as package
"""

import sys
import os

# Add parent directory to path for direct execution
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Application entry point."""
    import flet as ft
    
    # Import with fallback
    try:
        from .flet_ui import main as flet_main
    except ImportError:
        from flet_ui import main as flet_main
    
    ft.app(target=flet_main)


if __name__ == '__main__':
    main()
