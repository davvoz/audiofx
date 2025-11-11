"""
Strobe negative effect - inverts colors on strong beats.
"""

from typing import List, Tuple
import numpy as np
import cv2

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class StrobeNegativeEffect(BaseEffect):
    """Strobe negative effect that inverts colors on high audio intensity."""
    
    def __init__(self, 
                 threshold: float = 0.8, 
                 invert_strength: float = 1.0,
                 **kwargs):
        """
        Initialize strobe negative effect.
        
        Args:
            threshold: Intensity threshold to trigger strobe
            invert_strength: Strength of color inversion (0.0-1.0)
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.invert_strength = invert_strength
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply strobe negative effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with color inversion on beats
        """
        # Use beat intensity for more responsive strobe
        total_intensity = max(context.bass, context.mid, context.treble)
        
        # Only trigger on strong beats
        if total_intensity < self.threshold or context.beat_intensity < 0.3:
            return frame
        
        # Calculate strobe strength based on how much we exceed threshold
        excess = total_intensity - self.threshold
        # Normalize excess (if threshold is 0.5 and we get 0.8, excess is 0.3)
        strobe_strength = min(excess * 2.0, 1.0) * self.intensity * self.invert_strength
        
        # Invert the frame colors
        inverted = 255 - frame
        
        # Blend between original and inverted based on strobe strength
        result = cv2.addWeighted(
            frame.astype(np.float32),
            1.0 - strobe_strength,
            inverted.astype(np.float32),
            strobe_strength,
            0
        )
        
        return np.clip(result, 0, 255).astype(np.uint8)
