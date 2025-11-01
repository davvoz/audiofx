"""
Screen shake effect.
"""

import numpy as np
import cv2

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class ScreenShakeEffect(BaseEffect):
    """Screen shake effect that randomly offsets the entire frame."""
    
    def __init__(self, 
                 threshold: float = 0.5, 
                 max_shake: int = 30, 
                 **kwargs):
        """
        Initialize screen shake effect.
        
        Args:
            threshold: Mid frequency threshold to trigger shake
            max_shake: Maximum shake offset in pixels
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.max_shake = max_shake
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply screen shake effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with random offset
        """
        if context.mid < self.threshold:
            return frame
        
        h, w = frame.shape[:2]
        shake = int((context.mid - self.threshold) * self.max_shake * self.intensity)
        
        if shake <= 0:
            return frame
        
        # Random offset
        offset_x = np.random.randint(-shake, shake + 1)
        offset_y = np.random.randint(-shake, shake + 1)
        
        # Apply translation
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
