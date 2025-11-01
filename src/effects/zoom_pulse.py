"""
Zoom pulse effect.
"""

import numpy as np
import cv2

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class ZoomPulseEffect(BaseEffect):
    """Zoom pulse effect that scales image based on bass."""
    
    def __init__(self, 
                 threshold: float = 0.3, 
                 max_zoom: float = 0.5, 
                 **kwargs):
        """
        Initialize zoom pulse effect.
        
        Args:
            threshold: Bass threshold to trigger zoom
            max_zoom: Maximum zoom factor multiplier
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.max_zoom = max_zoom
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply zoom pulse effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with zoom
        """
        if context.bass < self.threshold:
            return frame
        
        h, w = frame.shape[:2]
        zoom_factor = 1.0 + (context.bass - self.threshold) * self.max_zoom * self.intensity
        
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        zoomed = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop to original size from center
        start_x, start_y = (new_w - w) // 2, (new_h - h) // 2
        return zoomed[start_y:start_y+h, start_x:start_x+w]
