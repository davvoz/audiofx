"""
Glitch distortion effect.
"""

import numpy as np

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class GlitchEffect(BaseEffect):
    """Glitch distortion effect with random horizontal line shifts."""
    
    def __init__(self, threshold: float = 0.4, **kwargs):
        """
        Initialize glitch effect.
        
        Args:
            threshold: Intensity threshold to trigger glitch
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.threshold = threshold
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply glitch effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with glitch distortion
        """
        glitch_intensity = (context.bass + context.treble) / 2
        
        if glitch_intensity < self.threshold:
            return frame
        
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Randomly shift horizontal bands
        if np.random.random() < (glitch_intensity - self.threshold) * 2 * self.intensity:
            y1 = np.random.randint(0, h - 20)
            y2 = y1 + np.random.randint(5, 20)
            shift = np.random.randint(-30, 30)
            result[y1:y2] = np.roll(result[y1:y2], shift, axis=1)
        
        return result
