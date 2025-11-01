"""
Color pulse effect based on audio frequency bands.
"""

import numpy as np
import cv2

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class ColorPulseEffect(BaseEffect):
    """Color pulse effect that modifies hue, saturation, and value based on audio."""
    
    def __init__(self, 
                 bass_threshold: float = 0.3,
                 mid_threshold: float = 0.2,
                 high_threshold: float = 0.15,
                 **kwargs):
        """
        Initialize color pulse effect.
        
        Args:
            bass_threshold: Threshold for bass activation
            mid_threshold: Threshold for mid activation
            high_threshold: Threshold for high frequency activation
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.bass_threshold = bass_threshold
        self.mid_threshold = mid_threshold
        self.high_threshold = high_threshold
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply color pulse effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with color pulse
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Bass affects brightness/value
        if context.bass > self.bass_threshold:
            bass_factor = (context.bass - self.bass_threshold) * 3 * self.intensity
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + bass_factor), 0, 255)
        
        # Mid affects saturation
        if context.mid > self.mid_threshold:
            mid_factor = (context.mid - self.mid_threshold) * 2 * self.intensity
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + mid_factor), 0, 255)
        
        # Treble affects hue
        if context.treble > self.high_threshold:
            treble_factor = (context.treble - self.high_threshold) * 50 * self.intensity
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + treble_factor, 0, 179)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
