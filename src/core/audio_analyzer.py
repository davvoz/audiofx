"""
Audio analysis component - orchestrates frequency, beat, and section analysis.
"""

from typing import Optional
import numpy as np
import librosa

from ..models.data_models import AudioAnalysis
from .frequency_analyzer import FrequencyAnalyzer
from .beat_detector import BeatDetector
from .section_analyzer import SectionAnalyzer


class AudioAnalyzer:
    """Orchestrates audio analysis using specialized analyzers."""
    
    def __init__(self):
        """Initialize audio analyzer with sub-components."""
        self.frequency_analyzer = FrequencyAnalyzer()
        self.beat_detector = BeatDetector()
        self.section_analyzer = SectionAnalyzer()
    
    def load_and_analyze(
        self,
        audio_file: str,
        duration: Optional[float] = None,
        fps: int = 30
    ) -> AudioAnalysis:
        """
        Load audio file and perform spectral analysis.
        
        Args:
            audio_file: Path to audio file
            duration: Optional duration limit in seconds
            fps: Frames per second for analysis granularity
            
        Returns:
            AudioAnalysis object with all analysis results
        """
        # Load audio
        y, sr = librosa.load(audio_file, duration=duration)
        
        # STFT parameters
        hop_length = max(1, sr // fps)
        n_fft = 2048
        
        # Spectral analysis
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Extract frequency bands using FrequencyAnalyzer
        bass, mid, treble = self.frequency_analyzer.extract_bands(magnitude, frequencies)
        
        # Beat detection using BeatDetector
        beat_times = self.beat_detector.detect_beats(y, sr)
        
        return AudioAnalysis(
            audio_signal=y,
            sample_rate=sr,
            frequencies=frequencies,
            magnitude=magnitude,
            bass_energy=bass,
            mid_energy=mid,
            treble_energy=treble,
            beat_times=beat_times
        )
    
    def analyze_sections(self, audio_file: str) -> AudioAnalysis:
        """
        Analyze audio with section detection (for intelligent mode).
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            AudioAnalysis with sections information
        """
        # Load full audio
        y, sr = librosa.load(audio_file)
        
        # Perform standard analysis
        analysis = self.load_and_analyze(audio_file, fps=30)
        
        # Detect sections using SectionAnalyzer
        sections = self.section_analyzer.detect_sections(y, sr)
        analysis.sections = sections
        
        return analysis
