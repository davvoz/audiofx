"""
Standard pipeline builder.
"""

from ..models.data_models import EffectConfig
from ..effects import (
    EffectPipeline,
    ColorPulseEffect,
    StrobeEffect,
    ZoomPulseEffect,
    GlitchEffect,
)


class StandardPipelineBuilder:
    """Builds standard effect pipeline."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Create standard effect pipeline.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with standard effects
        """
        pipeline = EffectPipeline()
        pipeline.add_effect(ColorPulseEffect(
            config.bass_threshold, 
            config.mid_threshold, 
            config.high_threshold
        ))
        pipeline.add_effect(StrobeEffect(config.colors, threshold=0.5, intensity=1.0))  # Lowered threshold
        pipeline.add_effect(ZoomPulseEffect(threshold=0.3))
        pipeline.add_effect(GlitchEffect(threshold=0.4))
        return pipeline
