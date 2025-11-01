"""
Effect pipeline for composing multiple effects.
"""

from typing import List
import numpy as np

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class EffectPipeline:
    """Manages a pipeline of effects using chain of responsibility pattern."""
    
    def __init__(self):
        """Initialize empty pipeline."""
        self.effects: List[BaseEffect] = []
    
    def add_effect(self, effect: BaseEffect) -> 'EffectPipeline':
        """
        Add effect to pipeline (builder pattern).
        
        Args:
            effect: Effect to add
            
        Returns:
            Self for method chaining
        """
        self.effects.append(effect)
        return self
    
    def remove_effect(self, effect_type: type) -> 'EffectPipeline':
        """
        Remove all effects of given type.
        
        Args:
            effect_type: Type of effect to remove
            
        Returns:
            Self for method chaining
        """
        self.effects = [e for e in self.effects if not isinstance(e, effect_type)]
        return self
    
    def apply(self, context: FrameContext) -> np.ndarray:
        """
        Apply all effects in sequence.
        
        Args:
            context: Frame context
            
        Returns:
            Frame after all effects applied
        """
        result = context.frame.copy()
        for effect in self.effects:
            if effect.enabled:
                # Update context with current frame
                context.frame = result
                result = effect.apply(context)
        return result
    
    def clear(self) -> None:
        """Remove all effects from pipeline."""
        self.effects.clear()
    
    def get_effects(self) -> List[BaseEffect]:
        """Get list of all effects."""
        return self.effects.copy()
    
    def count(self) -> int:
        """Get number of effects in pipeline."""
        return len(self.effects)
    
    def enable_all(self) -> None:
        """Enable all effects."""
        for effect in self.effects:
            effect.enable()
    
    def disable_all(self) -> None:
        """Disable all effects."""
        for effect in self.effects:
            effect.disable()
