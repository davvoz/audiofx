"""
Fashion Lightning effect - intricate lightning bolts radiating from specific colors.
Music-driven electric art with branching, fractal-like lightning patterns.
"""

from typing import List, Tuple
import numpy as np
import cv2

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class FashionLightningEffect(BaseEffect):
    """
    Fashion Lightning effect - creates intricate, branching lightning bolts
    that radiate from specific color points based on audio frequencies.
    """
    
    def __init__(self, 
                 colors: List[Tuple[float, float, float]],
                 threshold: float = 0.65,
                 branching_probability: float = 0.6,
                 max_branches: int = 5,
                 segment_length_min: int = 5,
                 segment_length_max: int = 20,
                 **kwargs):
        """
        Initialize fashion lightning effect.
        
        Args:
            colors: List of RGB color tuples (0.0-1.0 range) for lightning colors
            threshold: Audio intensity threshold to trigger lightning
            branching_probability: Probability of lightning branching (0.0-1.0)
            max_branches: Maximum number of branches per lightning bolt
            segment_length_min: Minimum length of each lightning segment
            segment_length_max: Maximum length of each lightning segment
            **kwargs: Additional arguments for BaseEffect
        """
        super().__init__(**kwargs)
        self.colors = colors
        self.threshold = threshold
        self.branching_probability = max(0.0, min(1.0, branching_probability))
        self.max_branches = max(1, max_branches)
        
        # Ensure min is not greater than max
        if segment_length_min > segment_length_max:
            segment_length_min, segment_length_max = segment_length_max, segment_length_min
        
        self.segment_length_min = max(1, segment_length_min)
        self.segment_length_max = max(self.segment_length_min, segment_length_max)
    
    def _generate_lightning_branch(self, 
                                   start_point: Tuple[int, int],
                                   direction: Tuple[float, float],
                                   remaining_length: int,
                                   color: Tuple[int, int, int],
                                   frame: np.ndarray,
                                   intensity: float,
                                   branches_created: int = 0) -> None:
        """
        Recursively generate a branching lightning bolt with intricate paths.
        
        Args:
            start_point: Starting (x, y) coordinates
            direction: Direction vector (dx, dy) normalized
            remaining_length: Remaining length to draw
            color: RGB color for this branch
            frame: Frame to draw on
            intensity: Audio intensity affecting lightning appearance
            branches_created: Number of branches already created
        """
        if remaining_length <= 0 or branches_created >= self.max_branches:
            return
        
        h, w = frame.shape[:2]
        x, y = start_point
        
        # Calculate segment length
        max_len = min(self.segment_length_max, remaining_length)
        if self.segment_length_min >= max_len:
            segment_length = max_len
        else:
            segment_length = np.random.randint(self.segment_length_min, max_len)
        
        # Add randomness to direction for natural lightning look
        angle_variation = np.random.uniform(-0.5, 0.5) * intensity
        dx, dy = direction
        new_dx = dx * np.cos(angle_variation) - dy * np.sin(angle_variation)
        new_dy = dx * np.sin(angle_variation) + dy * np.cos(angle_variation)
        
        # Normalize direction
        norm = np.sqrt(new_dx**2 + new_dy**2)
        if norm > 0:
            new_dx /= norm
            new_dy /= norm
        
        # Calculate end point
        end_x = int(x + new_dx * segment_length)
        end_y = int(y + new_dy * segment_length)
        
        # Clamp to frame boundaries
        end_x = max(0, min(w - 1, end_x))
        end_y = max(0, min(h - 1, end_y))
        
        # Sample color from destination point and blend with original
        dest_color = frame[end_y, end_x].astype(int)
        # Blend 70% original color + 30% destination color for smooth transition
        blended_color = tuple(int(color[i] * 0.7 + dest_color[i] * 0.3) for i in range(3))
        
        # Draw main lightning segment with varying thickness
        thickness = max(1, int(2 + intensity * 3))
        cv2.line(frame, (x, y), (end_x, end_y), blended_color, thickness, cv2.LINE_AA)
        
        # Add glow effect for high intensity
        if intensity > 0.8:
            # Outer glow with blended color
            glow_color = tuple(min(255, int(c * 1.3)) for c in blended_color)
            cv2.line(frame, (x, y), (end_x, end_y), glow_color, thickness + 2, cv2.LINE_AA)
            
            # Inner bright core
            cv2.line(frame, (x, y), (end_x, end_y), (255, 255, 255), 1, cv2.LINE_AA)
        
        # Create branches (OPTIMIZED - reduced branching)
        if np.random.random() < self.branching_probability * 0.5 and branches_created < self.max_branches:
            num_branches = 1  # Single branch (was 1-2)
            
            for _ in range(num_branches):
                # Branch direction varies from main direction
                branch_angle = np.random.uniform(-np.pi/3, np.pi/3)
                branch_dx = new_dx * np.cos(branch_angle) - new_dy * np.sin(branch_angle)
                branch_dy = new_dx * np.sin(branch_angle) + new_dy * np.cos(branch_angle)
                
                # Normalize
                branch_norm = np.sqrt(branch_dx**2 + branch_dy**2)
                if branch_norm > 0:
                    branch_dx /= branch_norm
                    branch_dy /= branch_norm
                
                # Branch from a random point along the segment
                branch_t = np.random.uniform(0.3, 0.7)
                branch_x = int(x + (end_x - x) * branch_t)
                branch_y = int(y + (end_y - y) * branch_t)
                
                # Branch length is shorter than main bolt
                branch_length = int(remaining_length * np.random.uniform(0.3, 0.6))
                
                # Use blended color for branch continuity
                # Recursively draw branch
                self._generate_lightning_branch(
                    (branch_x, branch_y),
                    (branch_dx, branch_dy),
                    branch_length,
                    blended_color,  # Pass blended color to maintain context
                    frame,
                    intensity * 0.8,  # Slightly lower intensity for branches
                    branches_created + 1
                )
        
        # Continue main bolt with blended color
        new_remaining = remaining_length - segment_length
        if new_remaining > 0:
            self._generate_lightning_branch(
                (end_x, end_y),
                (new_dx, new_dy),
                new_remaining,
                blended_color,  # Pass blended color for smooth color transition
                frame,
                intensity,
                branches_created
            )
    
    def _get_origin_points(self, frame: np.ndarray, context: FrameContext) -> List[Tuple[int, int, Tuple[int, int, int]]]:
        """
        Determine origin points for lightning based on audio frequencies and colors.
        Returns list of (x, y, color) tuples.
        Lightning colors are extracted from the frame at the origin point.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            List of origin points with colors
        """
        h, w = frame.shape[:2]
        origins = []
        
        # Bass-driven lightning from bottom (warm colors from bottom area)
        if context.bass > self.threshold:
            num_bolts = int((context.bass - self.threshold) * 8 * self.intensity)
            num_bolts = max(1, min(num_bolts, 4))
            
            for _ in range(num_bolts):
                x = np.random.randint(0, max(1, w))
                y_min = int(h * 0.7)
                y_max = h
                y = np.random.randint(y_min, max(y_min + 1, y_max)) if y_min < y_max else y_min
                
                # Extract color from frame at this position
                pixel_color = frame[y, x].astype(int)
                # Boost intensity for electric effect
                color = tuple(min(255, int(c * 1.3)) for c in pixel_color)
                origins.append((x, y, color))
        
        # Mid-driven lightning from sides (colors from left/right edges)
        if context.mid > self.threshold:
            num_bolts = int((context.mid - self.threshold) * 6 * self.intensity)
            num_bolts = max(1, min(num_bolts, 3))
            
            for _ in range(num_bolts):
                side = np.random.choice(['left', 'right'])
                if side == 'left':
                    x_max = max(1, int(w * 0.2))
                    x = np.random.randint(0, x_max) if x_max > 0 else 0
                else:
                    x_min = int(w * 0.8)
                    x = np.random.randint(x_min, max(x_min + 1, w)) if x_min < w else x_min
                y = np.random.randint(0, max(1, h))
                
                # Extract color from frame at this position
                pixel_color = frame[y, x].astype(int)
                # Boost intensity for electric effect
                color = tuple(min(255, int(c * 1.3)) for c in pixel_color)
                origins.append((x, y, color))
        
        # Treble-driven lightning from top (cool colors from top area)
        if context.treble > self.threshold:
            num_bolts = int((context.treble - self.threshold) * 10 * self.intensity)
            num_bolts = max(1, min(num_bolts, 5))
            
            for _ in range(num_bolts):
                x = np.random.randint(0, max(1, w))
                y_max = max(1, int(h * 0.3))
                y = np.random.randint(0, y_max) if y_max > 0 else 0
                
                # Extract color from frame at this position
                pixel_color = frame[y, x].astype(int)
                # Boost intensity for electric effect
                color = tuple(min(255, int(c * 1.3)) for c in pixel_color)
                origins.append((x, y, color))
        
        # Beat-driven center lightning (colors from center area)
        if context.beat_intensity > 0.7:
            num_bolts = int(context.beat_intensity * 4 * self.intensity)
            num_bolts = max(1, min(num_bolts, 3))
            
            for i in range(num_bolts):
                x_min = int(w * 0.3)
                x_max = int(w * 0.7)
                x = np.random.randint(x_min, max(x_min + 1, x_max)) if x_min < x_max else x_min
                
                y_min = int(h * 0.3)
                y_max = int(h * 0.7)
                y = np.random.randint(y_min, max(y_min + 1, y_max)) if y_min < y_max else y_min
                
                # Extract color from frame at this position
                pixel_color = frame[y, x].astype(int)
                # Strong boost for beat-driven bolts
                color = tuple(min(255, int(c * 1.5)) for c in pixel_color)
                origins.append((x, y, color))
        
        return origins
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Apply fashion lightning effect.
        
        Args:
            frame: Input frame
            context: Frame context with audio data
            
        Returns:
            Processed frame with intricate lightning bolts
        """
        # Calculate total intensity
        total_intensity = max(context.bass, context.mid, context.treble)
        
        # Only trigger if intensity is above threshold
        if total_intensity < self.threshold:
            return frame
        
        # Create overlay for lightning
        h, w = frame.shape[:2]
        lightning_layer = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Get origin points for lightning
        origins = self._get_origin_points(frame, context)
        
        if not origins:
            return frame
        
        # Generate lightning from each origin (OPTIMIZED - reduced bolt count)
        for origin_x, origin_y, color in origins:
            # Single bolt per origin for performance (was 1-3)
            num_bolts = 1
            
            for _ in range(num_bolts):
                # Random initial direction
                angle = np.random.uniform(0, 2 * np.pi)
                direction = (np.cos(angle), np.sin(angle))
                
                # Lightning length based on intensity
                length = int(50 + total_intensity * 150 * self.intensity)
                
                # Generate the branching lightning bolt
                self._generate_lightning_branch(
                    (origin_x, origin_y),
                    direction,
                    length,
                    color,
                    lightning_layer,
                    total_intensity
                )
        
        # Blend lightning layer with original frame
        # Use additive blending for electric effect
        alpha = 0.8 * self.intensity
        result = cv2.addWeighted(frame, 1.0, lightning_layer, alpha, 0)
        
        # Add slight glow/bloom effect on high intensity
        if total_intensity > 0.85:
            blurred = cv2.GaussianBlur(lightning_layer, (15, 15), 0)
            result = cv2.addWeighted(result, 1.0, blurred, 0.3, 0)
        
        return result
