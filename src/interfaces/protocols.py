"""
Protocol definitions for dependency injection and loose coupling.
"""

from typing import Protocol, Optional
import numpy as np

from ..models.data_models import FrameContext, AudioAnalysis


class IEffect(Protocol):
    """Interface for visual effects."""
    
    def apply(self, context: FrameContext) -> np.ndarray:
        """Apply effect to frame."""
        ...


class IAudioAnalyzer(Protocol):
    """Interface for audio analysis."""
    
    def analyze(self, audio_file: str, duration: Optional[float], fps: int) -> AudioAnalysis:
        """Analyze audio file."""
        ...


class IVideoExporter(Protocol):
    """Interface for video export."""
    
    def export(
        self,
        frames: list,
        audio_file: str,
        output_file: str,
        fps: int,
        progress_callback=None
    ) -> None:
        """Export frames to video with audio."""
        ...
