"""
Texture Stretch effect - creates rhythmic texture stretching distortions.

This effect distorts the image texture with a rhythm that is 4 times slower
than the music beat, creating hypnotic, flowing stretching patterns with
random variations in shape and position.
"""

import numpy as np
import cv2
from typing import Optional
import random

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class TextureStretchEffect(BaseEffect):
    """
    Creates fantastic rhythmic texture stretching with its own slower rhythm.
    
    This effect stretches and distorts the image texture in multiple directions,
    creating flowing, organic deformations that pulse at 1/4 the music's tempo.
    The result is a hypnotic, mesmerizing visual that appears to breathe and flow
    independently while still remaining synchronized to the audio.
    """
    
    def __init__(self,
                 bass_threshold: float = 0.25,
                 mid_threshold: float = 0.2,
                 max_stretch: float = 45.0,
                 wave_complexity: int = 3,
                 flow_speed: float = 0.15,
                 stretch_smoothness: float = 0.92,
                 direction_change_speed: float = 0.08,
                 texture_grain: float = 2.0,
                 **kwargs):
        """
        Initialize texture stretch effect.
        
        Args:
            bass_threshold: Bass threshold to trigger stretching
            mid_threshold: Mid-frequency threshold for variation
            max_stretch: Maximum stretch amplitude in pixels
            wave_complexity: Number of wave harmonics (1-5)
            flow_speed: Speed of texture flow animation
            stretch_smoothness: Temporal smoothing (0-1, higher = smoother)
            direction_change_speed: Speed of directional shifts
            texture_grain: Texture detail granularity
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.bass_threshold = bass_threshold
        self.mid_threshold = mid_threshold
        self.max_stretch = max_stretch
        self.wave_complexity = max(1, min(5, wave_complexity))
        self.flow_speed = flow_speed
        self.stretch_smoothness = stretch_smoothness
        self.direction_change_speed = direction_change_speed
        self.texture_grain = texture_grain
        
        # Internal state for rhythm tracking
        self._slow_phase: float = 0.0  # Main phase at 1/4 speed
        self._fast_phase: float = 0.0  # Full speed phase for audio reactivity
        self._stretch_history: list = []  # History for smoothing
        self._smooth_bass: float = 0.0
        self._smooth_mid: float = 0.0
        self._smooth_treble: float = 0.0
        self._beat_accumulator: float = 0.0
        
        # Direction vectors for multi-directional stretching
        self._direction_phase: float = 0.0
        self._last_beat_time: float = 0.0
        
        # Random variation seeds and offsets
        self._random_seed: int = random.randint(0, 10000)
        self._random_offset_x: float = random.uniform(-0.3, 0.3)
        self._random_offset_y: float = random.uniform(-0.3, 0.3)
        self._random_shape_factor: float = random.uniform(0.7, 1.3)
        self._random_wave_offset: float = random.uniform(0, 2 * np.pi)
        self._randomness_change_counter: int = 0
        self._random_direction_bias: float = random.uniform(0, 2 * np.pi)
        
        # Cache for coordinate maps
        self._map_cache: Optional[tuple] = None
        self._last_shape: Optional[tuple] = None
    
    def _calculate_slow_rhythm(self, context: FrameContext) -> float:
        """
        Calculate the slow rhythm (1/4 tempo) based on beat detection.
        
        Args:
            context: Frame context with beat information
            
        Returns:
            Slow rhythm intensity (0-1)
        """
        # Accumulate beats to create slower rhythm
        if context.beat_intensity > 0.5:
            self._beat_accumulator += context.beat_intensity
            
            # On strong beats, randomly change some parameters
            if context.beat_intensity > 0.7:
                self._update_random_variations()
        
        # Create a slow pulsing rhythm at 1/4 the beat rate
        # Use sine wave with accumulated phase
        slow_pulse = (np.sin(self._slow_phase * np.pi * 2) + 1) / 2
        
        # Modulate with accumulated beats
        beat_mod = np.clip(self._beat_accumulator * 0.25, 0, 1)
        
        # Decay accumulator slowly
        self._beat_accumulator *= 0.98
        
        return slow_pulse * (0.7 + beat_mod * 0.3)
    
    def _update_random_variations(self):
        """Update random variation parameters periodically for unpredictability."""
        self._randomness_change_counter += 1
        
        # Change random parameters every few beats
        if self._randomness_change_counter % 4 == 0:
            # Smoothly transition random offsets
            self._random_offset_x += random.uniform(-0.1, 0.1)
            self._random_offset_y += random.uniform(-0.1, 0.1)
            self._random_offset_x = np.clip(self._random_offset_x, -0.5, 0.5)
            self._random_offset_y = np.clip(self._random_offset_y, -0.5, 0.5)
            
            # Vary shape factor
            self._random_shape_factor += random.uniform(-0.15, 0.15)
            self._random_shape_factor = np.clip(self._random_shape_factor, 0.5, 1.5)
            
            # Change direction bias
            self._random_direction_bias += random.uniform(-0.3, 0.3)
            
            # Occasionally add a random wave offset
            if random.random() > 0.7:
                self._random_wave_offset += random.uniform(-np.pi/4, np.pi/4)
    
    def _should_regenerate_cache(self, shape: tuple) -> bool:
        """Check if coordinate cache needs regeneration."""
        return self._map_cache is None or self._last_shape != shape
    
    def _create_multi_directional_stretch(self,
                                          frame: np.ndarray,
                                          stretch_intensity: float,
                                          direction_mix: float) -> np.ndarray:
        """
        Create multi-directional texture stretching with random variations.
        
        Args:
            frame: Input frame
            stretch_intensity: Overall stretch strength
            direction_mix: Blend factor between different directions
            
        Returns:
            Stretched frame
        """
        h, w = frame.shape[:2]
        
        # Create or retrieve coordinate grids
        if self._last_shape != (h, w):
            y_coords, x_coords = np.indices((h, w), dtype=np.float32)
            self._map_cache = (x_coords, y_coords)
            self._last_shape = (h, w)
        else:
            x_coords, y_coords = self._map_cache
        
        # Normalize coordinates to -1 to 1 with random offset
        center_x = 0.5 + self._random_offset_x
        center_y = 0.5 + self._random_offset_y
        x_norm = (x_coords / w - center_x) * 2
        y_norm = (y_coords / h - center_y) * 2
        
        # Calculate distance from random center for radial effects
        distance = np.sqrt(x_norm**2 + y_norm**2)
        angle = np.arctan2(y_norm, x_norm) + self._random_direction_bias
        
        # Initialize displacement maps
        disp_x = np.zeros_like(x_coords)
        disp_y = np.zeros_like(y_coords)
        
        # Generate multiple wave harmonics for complex stretching with randomization
        for harmonic in range(1, self.wave_complexity + 1):
            # Add random variation to frequency
            freq_variation = 1.0 + random.uniform(-0.2, 0.2) * self._random_shape_factor
            freq = harmonic * self.texture_grain * freq_variation
            amplitude = stretch_intensity * self.max_stretch / harmonic * self._random_shape_factor
            
            # Random phase offsets for each harmonic
            random_phase_h = self._random_wave_offset * harmonic * 0.1
            random_phase_v = self._random_wave_offset * harmonic * 0.15
            random_phase_d = self._random_wave_offset * harmonic * 0.2
            
            # Horizontal waves with phase offset and random variation
            phase_h = self._slow_phase * (1.0 / harmonic) + self._direction_phase * 0.3 + random_phase_h
            wave_h = np.sin(y_norm * np.pi * freq + phase_h * np.pi * 2)
            # Add random modulation
            wave_h *= (1.0 + 0.2 * np.sin(x_norm * freq * 0.5 + self._random_wave_offset))
            disp_x += wave_h * amplitude * (1.0 - abs(x_norm) * 0.3)
            
            # Vertical waves with different phase and random variation
            phase_v = self._slow_phase * (1.0 / harmonic) - self._direction_phase * 0.2 + random_phase_v
            wave_v = np.sin(x_norm * np.pi * freq + phase_v * np.pi * 2)
            # Add random modulation
            wave_v *= (1.0 + 0.2 * np.cos(y_norm * freq * 0.5 - self._random_wave_offset))
            disp_y += wave_v * amplitude * (1.0 - abs(y_norm) * 0.3)
            
            # Diagonal waves for more complexity with randomness
            phase_d = self._slow_phase * (1.0 / harmonic) + self._direction_phase * 0.5 + random_phase_d
            wave_d = np.sin((x_norm + y_norm) * np.pi * freq * 0.7 + phase_d * np.pi * 2)
            # Random directional bias
            random_cos = np.cos(direction_mix * np.pi + self._random_direction_bias)
            random_sin = np.sin(direction_mix * np.pi + self._random_direction_bias)
            disp_x += wave_d * amplitude * 0.4 * random_cos
            disp_y += wave_d * amplitude * 0.4 * random_sin
        
        # Add radial stretching component with random variations
        radial_phase = self._slow_phase * 0.5 + self._fast_phase * 0.1 + self._random_wave_offset * 0.5
        radial_stretch = np.sin(distance * np.pi * 1.5 * self._random_shape_factor + radial_phase * np.pi * 2)
        radial_stretch *= stretch_intensity * self.max_stretch * 0.5
        radial_stretch *= (1.0 - distance * 0.5)  # Falloff at edges
        
        # Apply radial component with random angle offset
        disp_x += radial_stretch * np.cos(angle) * 0.6
        disp_y += radial_stretch * np.sin(angle) * 0.6
        
        # Add circular flow for organic feel with randomness
        flow_phase = self._slow_phase * 0.8 + self._random_wave_offset * 0.3
        flow_strength = stretch_intensity * self.max_stretch * 0.3 * self._random_shape_factor
        # Random flow direction changes
        flow_direction = 1.0 if random.random() > 0.3 else -1.0
        circular_flow_x = -np.sin(angle + flow_phase * np.pi) * flow_strength * distance * 0.5 * flow_direction
        circular_flow_y = np.cos(angle + flow_phase * np.pi) * flow_strength * distance * 0.5 * flow_direction
        
        disp_x += circular_flow_x
        disp_y += circular_flow_y
        
        # Add random localized distortions (hot spots)
        if random.random() > 0.6:  # 40% chance to add hot spot
            hotspot_x = random.uniform(0.2, 0.8) * w
            hotspot_y = random.uniform(0.2, 0.8) * h
            hotspot_radius = random.uniform(80, 200)
            hotspot_strength = stretch_intensity * random.uniform(10, 25)
            
            dx_hotspot = x_coords - hotspot_x
            dy_hotspot = y_coords - hotspot_y
            dist_hotspot = np.sqrt(dx_hotspot**2 + dy_hotspot**2)
            hotspot_influence = np.exp(-dist_hotspot / hotspot_radius)
            
            angle_hotspot = np.arctan2(dy_hotspot, dx_hotspot) + self._random_direction_bias
            disp_x += hotspot_influence * hotspot_strength * np.cos(angle_hotspot + self._slow_phase)
            disp_y += hotspot_influence * hotspot_strength * np.sin(angle_hotspot + self._slow_phase)
        
        # Calculate final mapping
        map_x = x_coords + disp_x
        map_y = y_coords + disp_y
        
        # Ensure arrays are C-contiguous and of type CV_32FC1
        map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        map_y = np.ascontiguousarray(map_y, dtype=np.float32)
        
        # Apply remapping - use INTER_LINEAR for 3-4x speedup vs CUBIC
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
        
        return result
    
    def _create_texture_grain_distortion(self,
                                        frame: np.ndarray,
                                        intensity: float) -> np.ndarray:
        """
        Add fine-grain texture distortion for enhanced detail with randomness.
        
        Args:
            frame: Input frame
            intensity: Distortion intensity
            
        Returns:
            Frame with grain distortion
        """
        if intensity < 0.1:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create fine-grain noise pattern with random variations
        grain_freq = self.texture_grain * 4 * self._random_shape_factor
        x_norm = np.linspace(0, grain_freq, w)
        y_norm = np.linspace(0, grain_freq, h)
        x_grid, y_grid = np.meshgrid(x_norm, y_norm)
        
        # Multi-octave noise for organic texture with randomness
        noise = (
            np.sin(x_grid + self._fast_phase * 3 + self._random_wave_offset) * 0.5 +
            np.sin(y_grid - self._fast_phase * 2.5 - self._random_wave_offset * 0.7) * 0.5 +
            np.sin((x_grid + y_grid) * 0.7 + self._slow_phase * 4 + self._random_wave_offset * 1.3) * 0.3
        )
        
        # Add random turbulence
        if random.random() > 0.5:
            turbulence = np.sin(x_grid * y_grid * 0.01 + self._fast_phase * 2) * 0.2
            noise += turbulence
        
        # Scale noise to displacement with random variation
        noise_amp = intensity * 3.0 * (0.8 + random.random() * 0.4)
        disp_x = noise * noise_amp
        disp_y = np.roll(noise, shift=int(w * 0.1 * self._random_shape_factor), axis=1) * noise_amp
        
        # Create displacement maps
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        map_x = x_coords + disp_x.astype(np.float32)
        map_y = y_coords + disp_y.astype(np.float32)
        
        # Ensure arrays are C-contiguous
        map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        map_y = np.ascontiguousarray(map_y, dtype=np.float32)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)
    
    def _apply_chromatic_stretching(self,
                                   frame: np.ndarray,
                                   intensity: float) -> np.ndarray:
        """
        Apply chromatic aberration aligned with stretch direction and random variations.
        
        Args:
            frame: Input frame
            intensity: Aberration intensity
            
        Returns:
            Frame with chromatic stretching
        """
        if intensity < 0.15:
            return frame
        
        # Calculate shift based on slow rhythm direction with randomness
        shift_angle = self._direction_phase * np.pi * 2 + self._random_direction_bias
        shift_amount = int(intensity * 5 * (0.8 + random.random() * 0.4))
        
        if shift_amount > 0:
            shift_x = int(np.cos(shift_angle) * shift_amount)
            shift_y = int(np.sin(shift_angle) * shift_amount)
            
            result = frame.copy()
            h, w = frame.shape[:2]
            
            # Randomly choose which channels to shift
            if random.random() > 0.5:
                # Shift red channel
                if abs(shift_x) > 0:
                    if shift_x > 0:
                        result[:, shift_x:, 0] = frame[:, :-shift_x, 0]
                    else:
                        result[:, :shift_x, 0] = frame[:, -shift_x:, 0]
                
                # Shift blue channel in opposite direction
                if abs(shift_y) > 0:
                    if shift_y > 0:
                        result[shift_y:, :, 2] = frame[:-shift_y, :, 2]
                    else:
                        result[:shift_y, :, 2] = frame[-shift_y:, :, 2]
            else:
                # Alternative: shift green and blue
                if abs(shift_x) > 0:
                    if shift_x > 0:
                        result[:, shift_x:, 1] = frame[:, :-shift_x, 1]
                    else:
                        result[:, :shift_x, 1] = frame[:, -shift_x:, 1]
                
                if abs(shift_y) > 0:
                    if shift_y > 0:
                        result[shift_y:, :, 2] = frame[:-shift_y, :, 2]
                    else:
                        result[:shift_y, :, 2] = frame[-shift_y:, :, 2]
            
            return result
        
        return frame
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply fantastic rhythmic texture stretching.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Frame with texture stretching effect
        """
        # Smooth audio values for fluid motion
        alpha = 1.0 - self.stretch_smoothness
        self._smooth_bass = self.stretch_smoothness * self._smooth_bass + alpha * context.bass
        self._smooth_mid = self.stretch_smoothness * self._smooth_mid + alpha * context.mid
        self._smooth_treble = self.stretch_smoothness * self._smooth_treble + alpha * context.treble
        
        # Update fast phase (full speed for reactive elements)
        self._fast_phase += self.flow_speed * (0.8 + self._smooth_bass * 0.4)
        
        # Update slow phase (1/4 speed - the main rhythm)
        slow_speed_factor = 0.25  # This makes it 4x slower
        self._slow_phase += self.flow_speed * slow_speed_factor * (0.5 + self._smooth_mid * 0.5)
        
        # Update direction phase for organic movement
        self._direction_phase += self.direction_change_speed * (0.7 + self._smooth_treble * 0.3)
        
        # Calculate slow rhythm intensity
        slow_rhythm = self._calculate_slow_rhythm(context)
        
        # Calculate audio-reactive stretch intensity
        bass_active = self._smooth_bass > self.bass_threshold
        mid_active = self._smooth_mid > self.mid_threshold
        
        # Compute base intensity from audio
        bass_contribution = max(0, self._smooth_bass - self.bass_threshold) * 1.2
        mid_contribution = max(0, self._smooth_mid - self.mid_threshold) * 0.8
        treble_contribution = self._smooth_treble * 0.4
        
        # Combine audio reactivity with slow rhythm
        audio_intensity = (bass_contribution + mid_contribution + treble_contribution) * self.intensity
        stretch_intensity = audio_intensity * slow_rhythm
        
        # Apply beat boost but modulated by slow rhythm
        if context.beat_intensity > 0.4:
            beat_boost = context.beat_intensity * slow_rhythm * 0.4
            stretch_intensity *= (1.0 + beat_boost)
        
        # If intensity too low, return original
        if stretch_intensity < 0.05:
            return frame
        
        # Calculate direction mix from phases
        direction_mix = (np.sin(self._direction_phase * np.pi * 2) + 1) / 2
        
        # Start with original frame
        result = frame.copy()
        
        # Apply main texture stretching
        result = self._create_multi_directional_stretch(result, stretch_intensity, direction_mix)
        
        # Add fine texture grain distortion if mid frequencies are active
        if mid_active and stretch_intensity > 0.2:
            grain_intensity = mid_contribution * self.intensity * slow_rhythm * 0.6
            result = self._create_texture_grain_distortion(result, grain_intensity)
        
        # Add chromatic stretching for enhanced aesthetic
        if stretch_intensity > 0.3:
            chroma_intensity = stretch_intensity * 0.7
            result = self._apply_chromatic_stretching(result, chroma_intensity)
        
        # Blend with original based on intensity for smooth transitions
        if stretch_intensity < 0.95:
            blend_factor = np.clip(stretch_intensity, 0, 1)
            result = cv2.addWeighted(
                frame.astype(np.float32),
                1.0 - blend_factor,
                result.astype(np.float32),
                blend_factor,
                0
            ).astype(np.uint8)
        
        return result
