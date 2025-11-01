"""
Strobe flash effect.
"""

from typing import List, Tuple
import numpy as np
import cv2

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class StrobeEffect(BaseEffect):
    """Strobe flash effect triggered by high audio intensity."""
    
    def __init__(self, 
                 colors: List[Tuple[float, float, float]], 
                 threshold: float = 0.8, 
                 **kwargs):
        """
        Initialize strobe effect.
        
        Args:
            colors: List of RGB color tuples (0.0-1.0 range)
            threshold: Intensity threshold to trigger strobe
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.colors = colors
        self.threshold = threshold
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply strobe effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with strobe flash
        """
        # Use beat intensity for more responsive strobe
        total_intensity = max(context.bass, context.mid, context.treble)
        
        # Only trigger on strong beats
        if total_intensity < self.threshold or context.beat_intensity < 0.3:
            return frame
        
        # Calculate strobe strength based on how much we exceed threshold
        excess = total_intensity - self.threshold
        # Normalize excess (if threshold is 0.5 and we get 0.8, excess is 0.3)
        strobe_strength = min(excess * 2.0, 1.0) * self.intensity
        
        # Use current color from palette
        color = self.colors[context.color_index % len(self.colors)]
        overlay = np.full(frame.shape, color, dtype=np.float32) * 255
        
        # Blend with strong flash
        flash_amount = strobe_strength * 0.7  # Up to 70% flash
        
        result = cv2.addWeighted(
            frame.astype(np.float32),
            1.0 - flash_amount,
            overlay,
            flash_amount,
            0
        )
        return np.clip(result, 0, 255).astype(np.uint8)
