"""
Image loading and preprocessing utilities.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class ImageLoader:
    """Handles image loading and preprocessing."""
    
    @staticmethod
    def load_and_prepare(
        image_file: str,
        target_resolution: Optional[Tuple[int, int]] = (720, 720)
    ) -> np.ndarray:
        """
        Load and prepare base image.
        
        Args:
            image_file: Path to image file
            target_resolution: Target resolution (width, height). 
                             If None, uses native image resolution.
            
        Returns:
            Processed image array (RGB)
            
        Raises:
            FileNotFoundError: If image file not found
        """
        img = cv2.imread(image_file)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_file}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize only if target_resolution is specified
        if target_resolution is not None:
            img = cv2.resize(img, target_resolution)
        
        return img
