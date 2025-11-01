"""
Advanced Glitch effect - dynamic, aesthetic digital distortion.
Features: RGB channel separation, block displacement, scanlines, noise, pixelation.
"""

from typing import Tuple
import numpy as np
import cv2

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class AdvancedGlitchEffect(BaseEffect):
    """
    Advanced glitch effect with multiple distortion techniques:
    - RGB channel separation and displacement
    - Block displacement with varied shapes
    - Digital artifacts and scanlines
    - Pixelation zones
    - Color corruption
    - Noise injection
    """
    
    def __init__(self, 
                 threshold: float = 0.5,
                 channel_shift_amount: int = 8,
                 block_size_range: Tuple[int, int] = (10, 80),
                 **kwargs):
        """
        Initialize advanced glitch effect.
        
        Args:
            threshold: Intensity threshold to trigger glitch
            channel_shift_amount: Maximum pixel shift for RGB channels
            block_size_range: Range of block sizes for displacement (min, max)
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.channel_shift_amount = channel_shift_amount
        self.block_size_range = block_size_range
    
    def _apply_rgb_shift(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Shift RGB channels independently for chromatic aberration effect.
        
        Args:
            frame: Input frame
            intensity: Effect intensity
            
        Returns:
            Frame with shifted RGB channels
        """
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Extract channels
        r, g, b = cv2.split(result)
        
        # Random shift amounts based on intensity
        shift_max = int(self.channel_shift_amount * intensity * self.intensity)
        
        if shift_max > 0:
            # Shift red channel
            r_shift_x = np.random.randint(-shift_max, shift_max)
            r_shift_y = np.random.randint(-shift_max // 2, shift_max // 2)
            M_r = np.float32([[1, 0, r_shift_x], [0, 1, r_shift_y]])
            r = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_WRAP)
            
            # Shift blue channel
            b_shift_x = np.random.randint(-shift_max, shift_max)
            b_shift_y = np.random.randint(-shift_max // 2, shift_max // 2)
            M_b = np.float32([[1, 0, b_shift_x], [0, 1, b_shift_y]])
            b = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_WRAP)
        
        # Merge back
        result = cv2.merge([r, g, b])
        return result
    
    def _apply_block_displacement(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Displace random blocks of the image.
        
        Args:
            frame: Input frame
            intensity: Effect intensity
            
        Returns:
            Frame with displaced blocks
        """
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Number of blocks based on intensity
        num_blocks = int((intensity - self.threshold) * 15 * self.intensity)
        num_blocks = max(1, min(num_blocks, 20))
        
        for _ in range(num_blocks):
            # Random block size
            block_w = np.random.randint(self.block_size_range[0], 
                                       min(self.block_size_range[1], w // 2))
            block_h = np.random.randint(self.block_size_range[0] // 2, 
                                       min(self.block_size_range[1] // 2, h // 2))
            
            # Random source position
            src_x = np.random.randint(0, max(1, w - block_w))
            src_y = np.random.randint(0, max(1, h - block_h))
            
            # Random displacement
            displacement_x = np.random.randint(-w // 4, w // 4)
            displacement_y = np.random.randint(-h // 8, h // 8)
            
            dest_x = src_x + displacement_x
            dest_y = src_y + displacement_y
            
            # Extract block
            block = frame[src_y:src_y + block_h, src_x:src_x + block_w].copy()
            
            # Apply random transformations
            transform_type = np.random.choice(['flip', 'mirror', 'duplicate', 'corrupt'])
            
            if transform_type == 'flip':
                block = cv2.flip(block, 1)  # Horizontal flip
            elif transform_type == 'mirror':
                block = cv2.flip(block, 0)  # Vertical flip
            elif transform_type == 'corrupt':
                # Color corruption
                corruption = np.random.randint(-50, 50, block.shape, dtype=np.int16)
                block = np.clip(block.astype(np.int16) + corruption, 0, 255).astype(np.uint8)
            
            # Place block at destination (with bounds checking)
            dest_x = max(0, min(dest_x, w - block_w))
            dest_y = max(0, min(dest_y, h - block_h))
            
            # Blend with random alpha for ghosting effect
            alpha = np.random.uniform(0.6, 1.0)
            result[dest_y:dest_y + block_h, dest_x:dest_x + block_w] = \
                cv2.addWeighted(
                    result[dest_y:dest_y + block_h, dest_x:dest_x + block_w],
                    1 - alpha,
                    block,
                    alpha,
                    0
                )
        
        return result
    
    def _apply_scanlines(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Apply digital scanline artifacts.
        
        Args:
            frame: Input frame
            intensity: Effect intensity
            
        Returns:
            Frame with scanlines
        """
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Random scanline thickness and spacing
        num_scanlines = int(intensity * 10 * self.intensity)
        
        for _ in range(num_scanlines):
            y = np.random.randint(0, h)
            thickness = np.random.randint(1, 4)
            
            # Random scanline effect
            effect_type = np.random.choice(['dark', 'bright', 'color_shift'])
            
            y_end = min(y + thickness, h)
            
            if effect_type == 'dark':
                result[y:y_end] = (result[y:y_end] * 0.3).astype(np.uint8)
            elif effect_type == 'bright':
                result[y:y_end] = np.clip(result[y:y_end] * 1.5, 0, 255).astype(np.uint8)
            else:  # color_shift
                # Shift color channels
                temp = result[y:y_end].copy()
                result[y:y_end, :, 0] = temp[:, :, 2]  # R = B
                result[y:y_end, :, 2] = temp[:, :, 0]  # B = R
        
        return result
    
    def _apply_pixelation(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Apply pixelation to random zones.
        
        Args:
            frame: Input frame
            intensity: Effect intensity
            
        Returns:
            Frame with pixelated zones
        """
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Number of pixelated zones
        num_zones = int((intensity - self.threshold) * 8 * self.intensity)
        num_zones = max(1, min(num_zones, 10))
        
        for _ in range(num_zones):
            # Random zone size
            zone_w = np.random.randint(w // 8, w // 3)
            zone_h = np.random.randint(h // 8, h // 3)
            
            # Random position
            x = np.random.randint(0, max(1, w - zone_w))
            y = np.random.randint(0, max(1, h - zone_h))
            
            # Extract zone
            zone = result[y:y + zone_h, x:x + zone_w].copy()
            
            # Pixelation factor
            pixel_size = np.random.randint(8, 25)
            
            # Downscale
            small_h = max(1, zone_h // pixel_size)
            small_w = max(1, zone_w // pixel_size)
            small = cv2.resize(zone, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            
            # Upscale back with nearest neighbor for blocky effect
            pixelated = cv2.resize(small, (zone_w, zone_h), interpolation=cv2.INTER_NEAREST)
            
            # Place back
            result[y:y + zone_h, x:x + zone_w] = pixelated
        
        return result
    
    def _apply_noise(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Add digital noise artifacts.
        
        Args:
            frame: Input frame
            intensity: Effect intensity
            
        Returns:
            Frame with noise
        """
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Noise strength based on intensity
        noise_strength = int(intensity * 40 * self.intensity)
        
        if noise_strength > 0:
            # Generate noise
            noise = np.random.randint(-noise_strength, noise_strength, 
                                     (h, w, 3), dtype=np.int16)
            
            # Apply noise to random areas (not full frame)
            mask = np.random.random((h, w)) < (intensity * 0.3)
            mask = np.stack([mask] * 3, axis=2)
            
            result = np.where(mask, 
                            np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8),
                            result)
        
        return result
    
    def _apply_horizontal_tears(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Apply horizontal tearing/displacement effects.
        
        Args:
            frame: Input frame
            intensity: Effect intensity
            
        Returns:
            Frame with horizontal tears
        """
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Number of tears
        num_tears = int((intensity - self.threshold) * 12 * self.intensity)
        num_tears = max(1, min(num_tears, 15))
        
        for _ in range(num_tears):
            y = np.random.randint(0, h - 2)
            height = np.random.randint(1, 8)
            shift = np.random.randint(-w // 3, w // 3)
            
            y_end = min(y + height, h)
            
            # Shift the section
            result[y:y_end] = np.roll(result[y:y_end], shift, axis=1)
            
            # Add subtle color distortion on edges
            if abs(shift) > 10:
                edge_mask = np.zeros((h, w, 3), dtype=np.uint8)
                edge_mask[y:y_end, :5] = [0, 255, 255]  # Cyan on left edge
                edge_mask[y:y_end, -5:] = [255, 0, 255]  # Magenta on right edge
                result = cv2.addWeighted(result, 0.9, edge_mask, 0.1, 0)
        
        return result
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply advanced glitch effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with advanced glitch distortion
        """
        # Calculate glitch intensity from multiple audio bands
        glitch_intensity = max(context.bass, context.mid, context.treble)
        
        # Only trigger if intensity is above threshold
        if glitch_intensity < self.threshold:
            return frame
        
        result = frame.copy()
        
        # Calculate which effects to apply based on intensity
        effects_intensity = (glitch_intensity - self.threshold) / (1.0 - self.threshold)
        
        # Apply effects probabilistically based on intensity and audio features
        if np.random.random() < effects_intensity * 0.8:
            result = self._apply_rgb_shift(result, glitch_intensity)
        
        if np.random.random() < effects_intensity * 0.9 or context.beat_intensity > 0.6:
            result = self._apply_block_displacement(result, glitch_intensity)
        
        if np.random.random() < effects_intensity * 0.6:
            result = self._apply_horizontal_tears(result, glitch_intensity)
        
        if np.random.random() < effects_intensity * 0.5:
            result = self._apply_scanlines(result, glitch_intensity)
        
        if np.random.random() < effects_intensity * 0.4:
            result = self._apply_pixelation(result, glitch_intensity)
        
        if np.random.random() < effects_intensity * 0.7:
            result = self._apply_noise(result, glitch_intensity)
        
        # Blend with original for less jarring effect
        alpha = 0.8 * self.intensity
        result = cv2.addWeighted(frame, 1 - alpha, result, alpha, 0)
        
        return result
