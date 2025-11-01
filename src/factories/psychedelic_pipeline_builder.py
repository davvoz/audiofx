"""
Psychedelic pipeline builder.
"""

from ..models.data_models import EffectConfig
from ..effects import (
    EffectPipeline,
    ColorPulseEffect,
    StrobeEffect,
    ZoomPulseEffect,
    BubbleDistortionEffect,
    ChromaticAberrationEffect,
)


class PsychedelicPipelineBuilder:
    """Builds psychedelic effect pipeline with warping effects."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Create psychedelic effect pipeline with warping effects.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with psychedelic effects
        """
        pipeline = EffectPipeline()
        pipeline.add_effect(ColorPulseEffect(
            config.bass_threshold * 0.8, 
            config.mid_threshold * 0.8, 
            config.high_threshold * 0.8,
            intensity=1.2
        ))
        pipeline.add_effect(BubbleDistortionEffect(threshold=0.3, intensity=1.1))
        pipeline.add_effect(ChromaticAberrationEffect(threshold=0.2, intensity=0.9))
        pipeline.add_effect(ZoomPulseEffect(threshold=0.4, intensity=0.8))
        pipeline.add_effect(StrobeEffect(config.colors, threshold=0.5, intensity=1.2))  # Lowered threshold, increased intensity
        return pipeline
