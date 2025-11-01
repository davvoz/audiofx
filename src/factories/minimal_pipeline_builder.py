"""
Minimal pipeline builder.
"""

from ..models.data_models import EffectConfig
from ..effects import (
    EffectPipeline,
    ColorPulseEffect,
    ZoomPulseEffect,
)


class MinimalPipelineBuilder:
    """Builds minimal effect pipeline for performance mode."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Create minimal effect pipeline (performance mode).
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with minimal effects
        """
        pipeline = EffectPipeline()
        pipeline.add_effect(ColorPulseEffect(
            config.bass_threshold, 
            config.mid_threshold, 
            config.high_threshold,
            intensity=0.8
        ))
        pipeline.add_effect(ZoomPulseEffect(threshold=0.4, intensity=0.7))
        return pipeline
