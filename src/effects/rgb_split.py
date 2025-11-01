"""
RGB split effect.
"""

import numpy as np

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class RGBSplitEffect(BaseEffect):
    """RGB split effect that vertically separates color channels."""
    
    def __init__(self, 
                 threshold: float = 0.6, 
                 max_split: int = 25, 
                 **kwargs):
        """
        Initialize RGB split effect.
        
        Args:
            threshold: Treble threshold to trigger split
            max_split: Maximum vertical split in pixels
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.max_split = max_split
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply RGB split effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with vertically split RGB channels
        """
        if context.treble < self.threshold:
            return frame
        
        result = frame.copy()
        split = int((context.treble - self.threshold) * self.max_split * self.intensity)
        
        # Shift red channel up, blue channel down
        result[:, :, 0] = np.roll(frame[:, :, 0], split, axis=0)
        result[:, :, 2] = np.roll(frame[:, :, 2], -split, axis=0)
        
        return result
