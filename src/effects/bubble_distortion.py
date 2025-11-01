"""
Bubble distortion effect.
"""

import numpy as np
import cv2

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class BubbleDistortionEffect(BaseEffect):
    """Bubble distortion effect that warps image radially."""
    
    def __init__(self, 
                 threshold: float = 0.4, 
                 max_strength: float = 30, 
                 **kwargs):
        """
        Initialize bubble distortion effect.
        
        Args:
            threshold: Bass threshold to trigger distortion
            max_strength: Maximum distortion strength
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.max_strength = max_strength
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply bubble distortion effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with radial distortion
        """
        if context.bass < self.threshold:
            return frame
        
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        factor = (context.bass - self.threshold) * self.max_strength * self.intensity
        
        # Create coordinate grids
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        dx, dy = x_coords - cx, y_coords - cy
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate radial deformation
        max_dist = np.sqrt(cx**2 + cy**2)
        norm_dist = distance / max_dist
        deformation = factor * np.sin(norm_dist * np.pi) * (1 - norm_dist)
        
        # Apply deformation
        angle = np.arctan2(dy, dx)
        map_x = x_coords + deformation * np.cos(angle)
        map_y = y_coords + deformation * np.sin(angle)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
