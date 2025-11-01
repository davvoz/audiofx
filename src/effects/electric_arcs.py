"""
Electric arcs effect - colored lightning lines.
"""

from typing import List, Tuple
import numpy as np
import cv2

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class ElectricArcsEffect(BaseEffect):
    """Electric arcs effect - draws colorful lightning lines on beat."""
    
    def __init__(self, 
                 colors: List[Tuple[float, float, float]],
                 threshold: float = 0.7, 
                 **kwargs):
        """
        Initialize electric arcs effect.
        
        Args:
            colors: List of RGB color tuples (0.0-1.0 range) for the arcs
            threshold: Intensity threshold to trigger arcs
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.colors = colors
        self.threshold = threshold
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply electric arcs effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with electric arcs
        """
        # Calculate total intensity
        total_intensity = max(context.bass, context.mid, context.treble)
        
        # Only trigger on strong beats or high intensity
        if total_intensity < self.threshold or context.beat_intensity < 0.4:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Number of arcs based on intensity
        num_arcs = int((total_intensity - self.threshold) * 25 * self.intensity)
        num_arcs = max(1, min(num_arcs, 20))  # Clamp between 1-20
        
        for _ in range(num_arcs):
            # Generate random endpoints
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            
            # Use current color from palette
            color_float = self.colors[context.color_index % len(self.colors)]
            arc_color = tuple(int(c * 255) for c in color_float)
            
            # Line thickness based on intensity
            thickness = np.random.randint(1, 4)
            
            # Draw main arc line
            cv2.line(result, (x1, y1), (x2, y2), arc_color, thickness)
            
            # Add glow effect for high intensity
            if total_intensity > 0.85:
                cv2.line(result, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # Blend with original frame for transparency effect
        alpha = 0.7
        result = cv2.addWeighted(frame, 1 - alpha, result, alpha, 0)
        
        return result
