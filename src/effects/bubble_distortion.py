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
        # Cache for coordinate grids (PERFORMANCE OPTIMIZATION)
        self._cached_coords = None
        self._cached_shape = None
    
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
        
        # Use cached coordinate grids (PERFORMANCE OPTIMIZATION)
        if self._cached_shape != (h, w):
            y_coords, x_coords = np.indices((h, w), dtype=np.float32)
            dx, dy = x_coords - cx, y_coords - cy
            distance = np.sqrt(dx**2 + dy**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            norm_dist = distance / max_dist
            angle = np.arctan2(dy, dx)
            
            # Cache all expensive calculations
            self._cached_coords = {
                'x_coords': x_coords,
                'y_coords': y_coords,
                'norm_dist': norm_dist,
                'angle': angle
            }
            self._cached_shape = (h, w)
        
        x_coords = self._cached_coords['x_coords']
        y_coords = self._cached_coords['y_coords']
        norm_dist = self._cached_coords['norm_dist']
        angle = self._cached_coords['angle']
        
        # Calculate radial deformation (only this varies per frame)
        deformation = factor * np.sin(norm_dist * np.pi) * (1 - norm_dist)
        
        # Apply deformation (angle is pre-cached)
        map_x = x_coords + deformation * np.cos(angle)
        map_y = y_coords + deformation * np.sin(angle)
        
        # Ensure arrays are C-contiguous and of type CV_32FC1 (float32)
        map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        map_y = np.ascontiguousarray(map_y, dtype=np.float32)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
