"""
Vortex distortion effect - creates an aesthetic spiral distortion from center.
"""

import numpy as np
import cv2
from typing import Optional

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class VortexDistortionEffect(BaseEffect):
    """Creates a vortex/spiral distortion effect from the center that pulses with audio."""
    
    def __init__(self, 
                 threshold: float = 0.2,
                 max_angle: float = 35.0,
                 radius_falloff: float = 1.8,
                 rotation_speed: float = 3.0,
                 smoothing: float = 0.3,
                 **kwargs):
        """
        Initialize vortex distortion effect.
        
        Args:
            threshold: Audio intensity threshold to trigger effect
            max_angle: Maximum rotation angle in degrees at center
            radius_falloff: Controls how quickly the effect falls off from center (higher = faster falloff)
            rotation_speed: Speed of vortex rotation over time
            smoothing: Smoothing factor for rotation transitions (0-1, higher = smoother)
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.max_angle = max_angle
        self.radius_falloff = radius_falloff
        self.rotation_speed = rotation_speed
        self.smoothing = smoothing
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._center_x: Optional[int] = None
        self._center_y: Optional[int] = None
        self._max_radius: Optional[float] = None
        self._accumulated_rotation: float = 0.0
        self._smooth_intensity: float = 0.0
    
    def _initialize_maps(self, height: int, width: int) -> None:
        """Initialize coordinate maps for the vortex effect."""
        if self._map_x is not None and self._map_x.shape == (height, width):
            return
        
        # Cache center and max radius
        self._center_x = width // 2
        self._center_y = height // 2
        self._max_radius = np.sqrt(self._center_x**2 + self._center_y**2)
        
        # Pre-compute coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
        
        # Store as instance variables for reuse
        self._base_x = x_coords
        self._base_y = y_coords
        
        # Calculate distances from center
        dx = x_coords - self._center_x
        dy = y_coords - self._center_y
        self._distances = np.sqrt(dx**2 + dy**2)
        
        # Calculate angles
        self._angles = np.arctan2(dy, dx)
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply vortex distortion effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with vortex distortion
        """
        height, width = frame.shape[:2]
        
        # Initialize maps if needed
        self._initialize_maps(height, width)
        
        # Calculate vortex strength based on audio
        # Use combination of bass and mid frequencies for more dynamic effect
        audio_intensity = (context.bass * 0.6 + context.mid * 0.4)
        
        # Calculate target effect strength (0.0 to 1.0)
        if audio_intensity < self.threshold:
            target_strength = 0.0
        else:
            target_strength = min((audio_intensity - self.threshold) / (1.0 - self.threshold), 1.0)
            target_strength *= self.intensity
            
            # Add beat boost for more dramatic effect on beats
            if context.beat_intensity > 0.3:
                target_strength *= (1.0 + context.beat_intensity * 0.3)
        
        # Smooth the intensity transitions using exponential moving average
        self._smooth_intensity += (target_strength - self._smooth_intensity) * self.smoothing
        effect_strength = self._smooth_intensity
        
        # If effect is too weak, return original frame (but keep accumulating rotation)
        if effect_strength < 0.005:
            # Still accumulate minimal rotation to keep vortex moving
            if target_strength > 0:
                self._accumulated_rotation += self.rotation_speed * 0.1
            return frame
        
        # Accumulate rotation over time based on audio intensity
        # This makes the vortex spin continuously instead of oscillating
        rotation_increment = self.rotation_speed * (0.5 + effect_strength * 0.5)  # Always rotate at least 50%
        self._accumulated_rotation += rotation_increment
        
        # Calculate normalized distance from center (0 at center, 1 at edges)
        normalized_dist = self._distances / self._max_radius
        
        # Apply radius falloff - effect is stronger at center
        # Using exponential falloff for smooth aesthetic transition
        falloff = np.exp(-normalized_dist * self.radius_falloff)
        
        # Calculate rotation amount based on distance - more rotation at center
        # Create spiral effect: rotation increases towards center (inverted distance)
        spiral_factor = (1.0 - normalized_dist) ** 0.7  # More rotation at center
        rotation_per_pixel = self._accumulated_rotation * spiral_factor * effect_strength
        
        # Convert to radians and scale by max_angle
        rotation_radians = np.deg2rad(rotation_per_pixel * self.max_angle / 50.0)
        
        # Apply rotation to angles
        rotated_angles = self._angles + rotation_radians
        
        # Calculate new coordinates with vortex distortion
        new_x = self._center_x + self._distances * np.cos(rotated_angles)
        new_y = self._center_y + self._distances * np.sin(rotated_angles)
        
        # Clip to valid range first
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
        
        # Ensure arrays are C-contiguous and of type CV_32FC1 (float32)
        new_x = np.ascontiguousarray(new_x, dtype=np.float32)
        new_y = np.ascontiguousarray(new_y, dtype=np.float32)
        
        # Apply remap to create vortex distortion
        # Use LINEAR interpolation instead of CUBIC for 3-4x speedup
        distorted = cv2.remap(frame, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Blend with original based on effect strength for smoother transition
        if effect_strength < 1.0:
            distorted = cv2.addWeighted(
                frame.astype(np.float32),
                1.0 - effect_strength,
                distorted.astype(np.float32),
                effect_strength,
                0
            ).astype(np.uint8)
        
        # Optional: add subtle color enhancement at the vortex center for aesthetic appeal
        if effect_strength > 0.3:
            # Create a radial gradient mask (stronger at center)
            center_mask = (1.0 - normalized_dist) * effect_strength * 0.2
            center_mask = np.clip(center_mask, 0, 1)
            
            # Enhance saturation slightly at center for aesthetic effect
            hsv = cv2.cvtColor(distorted, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + center_mask[:, :] * 0.3), 0, 255)
            # Slightly brighten the center
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + center_mask[:, :] * 0.15), 0, 255)
            distorted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return distorted
