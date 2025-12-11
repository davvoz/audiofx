"""
3D Dimensional Warp effect with rhythmic distortions.
"""

import numpy as np
import cv2
from typing import Tuple

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class DimensionalWarpEffect(BaseEffect):
    """
    Creates stunning 3D dimensional warps with perspective distortions.
    
    This effect creates rhythmic 3D distortions that warp the image in multiple
    dimensions, creating a sense of depth and spatial transformation synchronized
    with the music's beat and frequency bands.
    """
    
    def __init__(self,
                 bass_threshold: float = 0.3,
                 mid_threshold: float = 0.2,
                 warp_strength: float = 25.0,
                 rotation_speed: float = 0.3,
                 perspective_depth: float = 120.0,
                 wave_frequency: float = 1.5,
                 layer_count: int = 1,  # REDUCED from 3 to 1 for performance
                 smoothing: float = 0.85,
                 **kwargs):
        """
        Initialize 3D dimensional warp effect.
        
        Args:
            bass_threshold: Threshold for bass-triggered warps
            mid_threshold: Threshold for mid-frequency effects
            warp_strength: Base strength of spatial warping (0-100)
            rotation_speed: Speed of rotational warping
            perspective_depth: Depth of perspective transformation
            wave_frequency: Frequency of wave distortions
            layer_count: Number of layered distortion passes
            smoothing: Smoothing factor for temporal coherence (0-1)
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.bass_threshold = bass_threshold
        self.mid_threshold = mid_threshold
        self.warp_strength = warp_strength
        self.rotation_speed = rotation_speed
        self.perspective_depth = perspective_depth
        self.wave_frequency = wave_frequency
        self.layer_count = max(1, min(5, layer_count))
        self.smoothing = smoothing
        self.phase = 0.0
        # Smoothed values for fluid transitions
        self._smooth_bass = 0.0
        self._smooth_mid = 0.0
        self._smooth_treble = 0.0
        self._smooth_beat = 0.0
    
    def _create_perspective_warp(self, 
                                 frame: np.ndarray, 
                                 angle_x: float, 
                                 angle_y: float, 
                                 depth: float) -> np.ndarray:
        """
        Create smooth perspective warp transformation with elegant boundaries.
        
        Args:
            frame: Input frame
            angle_x: Rotation angle around X axis
            angle_y: Rotation angle around Y axis
            depth: Perspective depth
            
        Returns:
            Perspective warped frame
        """
        h, w = frame.shape[:2]
        
        # Softer angles for smoother effect
        rad_x = np.radians(angle_x * 0.7)
        rad_y = np.radians(angle_y * 0.7)
        
        # Calculate smooth perspective points with reduced distortion
        src_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
        
        # Apply gentle 3D perspective transformation
        offset_x = depth * np.sin(rad_y) * 0.5
        offset_y = depth * np.sin(rad_x) * 0.5
        scale_x = 1 + abs(np.sin(rad_y)) * 0.15
        scale_y = 1 + abs(np.sin(rad_x)) * 0.15
        
        # More balanced corner transformations
        dst_points = np.float32([
            [offset_x * 0.8, offset_y * 0.8],
            [w - offset_x * scale_x * 0.8, offset_y * scale_x * 0.9],
            [w - offset_x * scale_x * 0.9, h - offset_y * scale_y * 0.8],
            [offset_x * scale_y * 0.9, h - offset_y * scale_y * 0.9]
        ])
        
        # Get perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        return cv2.warpPerspective(frame, matrix, (w, h), 
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
    
    def _create_spherical_warp(self, 
                               frame: np.ndarray, 
                               strength: float,
                               center_offset: Tuple[float, float] = (0, 0)) -> np.ndarray:
        """
        Create elegant spherical/bulge warp with smooth falloff.
        
        Args:
            frame: Input frame
            strength: Warp strength
            center_offset: Offset from center for warp origin
            
        Returns:
            Spherically warped frame
        """
        h, w = frame.shape[:2]
        cx = w // 2 + int(center_offset[0] * 0.5)
        cy = h // 2 + int(center_offset[1] * 0.5)
        
        # Create coordinate grids
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        dx = x_coords - cx
        dy = y_coords - cy
        distance = np.sqrt(dx**2 + dy**2)
        
        # Spherical distortion with smooth falloff
        max_dist = np.sqrt(cx**2 + cy**2)
        norm_dist = np.clip(distance / max_dist, 0, 1)
        
        # Smooth bulge/pinch with cosine interpolation for elegance
        smooth_strength = strength * 0.4
        warp_curve = 1.0 + smooth_strength * (np.cos(norm_dist * np.pi) - 1)
        warp_factor = np.power(norm_dist, warp_curve)
        
        # Avoid distortion at very center
        warp_factor = np.where(distance < 5, 1.0, warp_factor)
        
        # Calculate new positions
        map_x = cx + (dx * warp_factor)
        map_y = cy + (dy * warp_factor)
        
        # Ensure arrays are C-contiguous and of type CV_32FC1 (float32)
        map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        map_y = np.ascontiguousarray(map_y, dtype=np.float32)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_REFLECT_101)
    
    def _create_wave_distortion(self, 
                                frame: np.ndarray, 
                                amplitude: float,
                                frequency: float,
                                phase: float,
                                direction: str = 'horizontal') -> np.ndarray:
        """
        Create smooth wave distortion with elegant flow.
        
        Args:
            frame: Input frame
            amplitude: Wave amplitude
            frequency: Wave frequency
            phase: Wave phase offset
            direction: 'horizontal', 'vertical', or 'radial'
            
        Returns:
            Wave-distorted frame
        """
        h, w = frame.shape[:2]
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        
        # Reduce amplitude for subtler, more elegant waves
        smooth_amp = amplitude * 0.5
        smooth_freq = frequency * 0.8
        
        if direction == 'horizontal':
            wave = smooth_amp * np.sin(2 * np.pi * smooth_freq * y_coords / h + phase)
            map_x = x_coords + wave
            map_y = y_coords
        elif direction == 'vertical':
            wave = smooth_amp * np.sin(2 * np.pi * smooth_freq * x_coords / w + phase)
            map_x = x_coords
            map_y = y_coords + wave
        else:  # radial - more elegant spiral
            cx, cy = w // 2, h // 2
            dx, dy = x_coords - cx, y_coords - cy
            distance = np.sqrt(dx**2 + dy**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            norm_dist = distance / max_dist
            
            # Smooth radial wave with falloff
            wave = smooth_amp * np.sin(2 * np.pi * smooth_freq * norm_dist + phase) * (1 - norm_dist * 0.3)
            angle = np.arctan2(dy, dx)
            map_x = x_coords + wave * np.cos(angle)
            map_y = y_coords + wave * np.sin(angle)
        
        # Ensure arrays are C-contiguous and of type CV_32FC1 (float32)
        map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        map_y = np.ascontiguousarray(map_y, dtype=np.float32)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    
    def _create_tunnel_warp(self, 
                           frame: np.ndarray, 
                           depth: float,
                           twist: float) -> np.ndarray:
        """
        Create elegant tunnel/vortex warp with smooth spiral.
        
        Args:
            frame: Input frame
            depth: Tunnel depth factor
            twist: Rotational twist amount
            
        Returns:
            Tunnel-warped frame
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Create polar coordinates
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        dx = x_coords - cx
        dy = y_coords - cy
        
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # Tunnel transformation with smooth curves
        max_dist = np.sqrt(cx**2 + cy**2)
        norm_dist = np.clip(distance / max_dist, 0, 1)
        
        # Smooth depth with cosine easing
        smooth_depth = depth * 0.6
        depth_curve = 1.0 - np.cos(norm_dist * np.pi / 2)
        radius_warp = norm_dist * (1.0 + smooth_depth * depth_curve * (1.0 - norm_dist))
        
        # Smooth twist with elegant spiral
        smooth_twist = twist * 0.5
        angle_warp = angle + smooth_twist * np.sin(norm_dist * np.pi) * (1.0 - norm_dist)
        
        # Convert back to cartesian
        map_x = cx + radius_warp * max_dist * np.cos(angle_warp)
        map_y = cy + radius_warp * max_dist * np.sin(angle_warp)
        
        # Ensure arrays are C-contiguous and of type CV_32FC1 (float32)
        map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        map_y = np.ascontiguousarray(map_y, dtype=np.float32)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    
    def _create_multi_point_warp(self, 
                                 frame: np.ndarray,
                                 points: list,
                                 strength: float) -> np.ndarray:
        """
        Create dynamic warp with multiple moving focal points.
        
        Args:
            frame: Input frame
            points: List of (x, y, radius, strength) tuples
            strength: Overall warp strength
            
        Returns:
            Warped frame with multiple distortion centers
        """
        h, w = frame.shape[:2]
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        
        # Start with identity mapping
        map_x = x_coords.copy()
        map_y = y_coords.copy()
        
        # Apply each warp point
        for px, py, radius, point_strength in points:
            dx = x_coords - px
            dy = y_coords - py
            distance = np.sqrt(dx**2 + dy**2)
            
            # Smooth falloff based on radius
            influence = np.exp(-distance / (radius * 2))
            warp_factor = point_strength * strength * influence
            
            # Apply radial displacement
            angle = np.arctan2(dy, dx)
            map_x += warp_factor * np.cos(angle) * distance * 0.1
            map_y += warp_factor * np.sin(angle) * distance * 0.1
        
        # Ensure arrays are C-contiguous and of type CV_32FC1 (float32)
        map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        map_y = np.ascontiguousarray(map_y, dtype=np.float32)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    
    def _create_depth_layers(self, 
                            frame: np.ndarray,
                            depth_map: np.ndarray,
                            intensity: float) -> np.ndarray:
        """
        Create depth-based warping with varying zoom at different regions.
        
        Args:
            frame: Input frame
            depth_map: Normalized depth values (0-1)
            intensity: Warp intensity
            
        Returns:
            Depth-warped frame
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        dx = x_coords - cx
        dy = y_coords - cy
        
        # Apply depth-based zoom
        zoom_factor = 1.0 + (depth_map - 0.5) * intensity * 0.3
        
        map_x = cx + dx * zoom_factor
        map_y = cy + dy * zoom_factor
        
        # Ensure arrays are C-contiguous and of type CV_32FC1 (float32)
        map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        map_y = np.ascontiguousarray(map_y, dtype=np.float32)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)

    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply dynamic 3D dimensional warp with moving distortion points.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with dynamic spatial warps
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Smooth audio values for fluid transitions
        self._smooth_bass = self.smoothing * self._smooth_bass + (1 - self.smoothing) * context.bass
        self._smooth_mid = self.smoothing * self._smooth_mid + (1 - self.smoothing) * context.mid
        self._smooth_treble = self.smoothing * self._smooth_treble + (1 - self.smoothing) * context.treble
        self._smooth_beat = self.smoothing * self._smooth_beat + (1 - self.smoothing) * context.beat_intensity
        
        # Update phase for smooth animation
        phase_speed = self.rotation_speed * (0.5 + self._smooth_beat * 0.5)
        self.phase += phase_speed
        
        # Calculate smooth audio-reactive parameters
        bass_active = self._smooth_bass > self.bass_threshold
        mid_active = self._smooth_mid > self.mid_threshold
        
        # Compute smooth intensities
        bass_intensity = max(0, self._smooth_bass - self.bass_threshold) * self.intensity
        mid_intensity = max(0, self._smooth_mid - self.mid_threshold) * self.intensity
        treble_factor = self._smooth_treble * 0.6
        
        # Create dynamic depth map based on audio
        x_norm = np.linspace(0, 1, w)
        y_norm = np.linspace(0, 1, h)
        x_grid, y_grid = np.meshgrid(x_norm, y_norm)
        
        depth_map = (
            0.5 + 
            0.2 * np.sin(x_grid * np.pi * 2 + self.phase * 0.5) * bass_intensity +
            0.2 * np.cos(y_grid * np.pi * 2 + self.phase * 0.3) * mid_intensity +
            0.1 * np.sin((x_grid + y_grid) * np.pi + self.phase * 0.7) * treble_factor
        )
        depth_map = np.clip(depth_map, 0, 1).astype(np.float32)
        
        # Layer 1: Dynamic depth-based warping
        if self.layer_count >= 1:
            depth_intensity = bass_intensity * 1.5 + mid_intensity * 0.5
            if depth_intensity > 0.1:
                result = self._create_depth_layers(result, depth_map, depth_intensity)
        
        # Layer 2: Multiple moving warp points (bass-driven)
        if bass_active and self.layer_count >= 2:
            # Create 3-4 moving distortion points
            num_points = 3 if bass_intensity > 0.5 else 2
            warp_points = []
            
            for i in range(num_points):
                angle = self.phase * 0.6 + i * (2 * np.pi / num_points)
                radius_factor = 0.3 + 0.2 * np.sin(self.phase * 0.4 + i)
                
                px = w // 2 + np.cos(angle) * w * radius_factor * bass_intensity
                py = h // 2 + np.sin(angle) * h * radius_factor * bass_intensity
                point_radius = 80 + 40 * np.sin(self.phase * 0.5 + i * 0.5)
                point_strength = 0.8 + 0.4 * np.sin(self.phase * 0.8 + i * 0.3)
                
                warp_points.append((px, py, point_radius, point_strength))
            
            result = self._create_multi_point_warp(result, warp_points, bass_intensity * 2.0)
        
        # Layer 3: Traveling waves (mid-driven)
        if mid_active and self.layer_count >= 3:
            wave_amp = self.warp_strength * mid_intensity * 0.6
            wave_freq = self.wave_frequency * (0.8 + treble_factor * 0.4)
            
            # Alternate between directions smoothly
            direction_blend = (np.sin(self.phase * 0.3) + 1) / 2
            
            if direction_blend < 0.33:
                result = self._create_wave_distortion(result, wave_amp, wave_freq, 
                                                      self.phase * 1.5, 'horizontal')
            elif direction_blend < 0.66:
                result = self._create_wave_distortion(result, wave_amp, wave_freq, 
                                                      self.phase * 1.5, 'vertical')
            else:
                result = self._create_wave_distortion(result, wave_amp * 0.8, wave_freq, 
                                                      self.phase * 1.5, 'radial')
        
        # Layer 4: Orbiting spherical distortion (treble-driven)
        if self.layer_count >= 4 and treble_factor > 0.2:
            # Create moving spherical distortion point
            orbit_angle = self.phase * 0.8
            orbit_radius = 0.25 + 0.15 * np.sin(self.phase * 0.4)
            
            sphere_x = w // 2 + np.cos(orbit_angle) * w * orbit_radius
            sphere_y = h // 2 + np.sin(orbit_angle) * h * orbit_radius
            
            sphere_strength = treble_factor * 0.8 * (0.8 + self._smooth_beat * 0.4)
            
            # Offset from calculated position
            offset_x = sphere_x - w // 2
            offset_y = sphere_y - h // 2
            
            result = self._create_spherical_warp(result, sphere_strength, (offset_x, offset_y))
        
        # Layer 5: Beat-synchronized perspective shifts
        if self._smooth_beat > 0.6 and self.layer_count >= 5:
            # Rotate perspective based on beat
            angle_x = np.sin(self.phase * 1.2) * 10 * self._smooth_beat
            angle_y = np.cos(self.phase * 0.9) * 10 * self._smooth_beat
            depth = self.perspective_depth * 0.5 * self._smooth_beat * bass_intensity
            result = self._create_perspective_warp(result, angle_x, angle_y, depth)
        
        # Dynamic chromatic aberration following movement
        if bass_active or mid_active:
            # Direction of chromatic shift follows phase
            shift_angle = self.phase * 0.5
            shift_amount = int(4 * (bass_intensity * 0.7 + mid_intensity * 0.3))
            
            if shift_amount > 0 and shift_amount < min(w, h) // 4:
                shift_x = int(np.cos(shift_angle) * shift_amount)
                shift_y = int(np.sin(shift_angle) * shift_amount)
                
                result_shifted = result.copy()
                
                # Apply directional chromatic shift
                if abs(shift_x) > 0:
                    if shift_x > 0:
                        result[:, shift_x:, 2] = result_shifted[:, :-shift_x, 2]
                    else:
                        result[:, :shift_x, 2] = result_shifted[:, -shift_x:, 2]
                
                if abs(shift_y) > 0:
                    if shift_y > 0:
                        result[shift_y:, :, 0] = result_shifted[:-shift_y, :, 0]
                    else:
                        result[:shift_y, :, 0] = result_shifted[-shift_y:, :, 0]
        
        return result
