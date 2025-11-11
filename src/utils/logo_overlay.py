"""
Logo overlay utility for adding watermarks to video frames.
"""

import cv2
import numpy as np
from typing import Tuple, Literal


def apply_logo_to_frame(
    frame: np.ndarray,
    logo: np.ndarray,
    position: Literal["top-left", "top-right", "bottom-left", "bottom-right", "center"] = "top-right",
    scale: float = 0.15,
    opacity: float = 1.0,
    margin: int = 12
) -> np.ndarray:
    """
    Apply logo overlay to a frame.
    
    Args:
        frame: Input frame (RGB)
        logo: Logo image (RGBA or RGB)
        position: Logo position on frame
        scale: Logo scale relative to frame width (0.0-1.0)
        opacity: Logo opacity (0.0-1.0)
        margin: Margin from edges in pixels
        
    Returns:
        Frame with logo overlay
    """
    frame_h, frame_w = frame.shape[:2]
    
    # Calculate logo dimensions
    logo_width = int(frame_w * scale)
    logo_aspect = logo.shape[0] / logo.shape[1]
    logo_height = int(logo_width * logo_aspect)
    
    # Resize logo
    logo_resized = cv2.resize(logo, (logo_width, logo_height), interpolation=cv2.INTER_AREA)
    
    # Calculate position
    if position == "top-left":
        x, y = margin, margin
    elif position == "top-right":
        x, y = frame_w - logo_width - margin, margin
    elif position == "bottom-left":
        x, y = margin, frame_h - logo_height - margin
    elif position == "bottom-right":
        x, y = frame_w - logo_width - margin, frame_h - logo_height - margin
    elif position == "center":
        x, y = (frame_w - logo_width) // 2, (frame_h - logo_height) // 2
    else:
        x, y = frame_w - logo_width - margin, margin  # Default to top-right
    
    # Make sure logo fits in frame
    if x < 0 or y < 0 or x + logo_width > frame_w or y + logo_height > frame_h:
        return frame
    
    # Create a copy of the frame
    result = frame.copy()
    
    # Extract the region where logo will be placed
    roi = result[y:y+logo_height, x:x+logo_width]
    
    # Check if logo has alpha channel
    if logo_resized.shape[2] == 4:
        # Logo has alpha channel (RGBA)
        logo_rgb = logo_resized[:, :, :3]
        alpha_channel = logo_resized[:, :, 3] / 255.0 * opacity
        alpha_channel = alpha_channel[:, :, np.newaxis]
        
        # Blend using alpha channel
        blended = (logo_rgb * alpha_channel + roi * (1 - alpha_channel)).astype(np.uint8)
    else:
        # Logo has no alpha channel (RGB)
        # Use opacity for blending
        blended = cv2.addWeighted(logo_resized, opacity, roi, 1 - opacity, 0)
    
    # Place blended logo back to frame
    result[y:y+logo_height, x:x+logo_width] = blended
    
    return result


def load_logo(logo_path: str) -> np.ndarray:
    """
    Load logo image with alpha channel support.
    
    Args:
        logo_path: Path to logo file
        
    Returns:
        Logo image in RGB or RGBA format
    """
    # Try to load with alpha channel
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    
    if logo is None:
        raise FileNotFoundError(f"Could not load logo: {logo_path}")
    
    # Convert BGR to RGB (or BGRA to RGBA)
    if logo.shape[2] == 4:
        # Has alpha channel
        logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2RGBA)
    else:
        # No alpha channel
        logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
    
    return logo
