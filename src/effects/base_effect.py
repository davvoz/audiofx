"""
Base effect class providing template method pattern.
"""

from abc import ABC, abstractmethod
import numpy as np

from ..models.data_models import FrameContext


class BaseEffect(ABC):
    """Base class for all visual effects using template method pattern."""
    
    def __init__(self, intensity: float = 1.0, enabled: bool = True):
        """
        Initialize effect.
        
        Args:
            intensity: Effect strength multiplier (0.0-2.0)
            enabled: Whether effect is active
        """
        self.intensity = intensity
        self.enabled = enabled
    
    @abstractmethod
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """
        Process frame with effect (implement in subclass).
        
        Args:
            frame: Input frame
            context: Frame context information
            
        Returns:
            Processed frame
        """
        pass
    
    def apply(self, context: FrameContext) -> np.ndarray:
        """
        Apply effect if enabled (template method).
        
        Args:
            context: Frame context with frame and metadata
            
        Returns:
            Processed or original frame
        """
        if not self.enabled:
            return context.frame
        return self.process(context.frame, context)
    
    def set_intensity(self, intensity: float) -> None:
        """Set effect intensity."""
        self.intensity = max(0.0, min(2.0, intensity))
    
    def enable(self) -> None:
        """Enable effect."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable effect."""
        self.enabled = False
    
    def toggle(self) -> None:
        """Toggle effect enabled state."""
        self.enabled = not self.enabled
