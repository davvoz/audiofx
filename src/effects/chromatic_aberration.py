"""
Chromatic aberration effect.
"""

import numpy as np

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class ChromaticAberrationEffect(BaseEffect):
    """Chromatic aberration effect that shifts color channels."""
    
    def __init__(self, 
                 threshold: float = 0.3, 
                 max_shift: int = 15, 
                 **kwargs):
        """
        Initialize chromatic aberration effect.
        
        Args:
            threshold: Treble threshold to trigger effect
            max_shift: Maximum pixel shift
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.max_shift = max_shift
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply chromatic aberration effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with RGB channel separation
        """
        if context.treble < self.threshold:
            return frame
        
        result = frame.copy()
        shift = int((context.treble - self.threshold) * self.max_shift * self.intensity)
        
        # Shift red channel left, blue channel right
        result[:, :, 0] = np.roll(frame[:, :, 0], -shift, axis=1)
        result[:, :, 2] = np.roll(frame[:, :, 2], shift, axis=1)
        
        return result
