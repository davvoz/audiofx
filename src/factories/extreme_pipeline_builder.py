"""
Extreme pipeline builder.
"""

from ..models.data_models import EffectConfig
from ..effects import (
    EffectPipeline,
    ColorPulseEffect,
    StrobeEffect,
    ZoomPulseEffect,
    BubbleDistortionEffect,
    ScreenShakeEffect,
    ChromaticAberrationEffect,
    RGBSplitEffect,
    GlitchEffect,
)


class ExtremePipelineBuilder:
    """Builds extreme effect pipeline with maximum intensity."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Create extreme effect pipeline with higher intensity.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with extreme effects
        """
        pipeline = EffectPipeline()
        pipeline.add_effect(ColorPulseEffect(
            config.bass_threshold, 
            config.mid_threshold, 
            config.high_threshold,
            intensity=1.2
        ))
        #pipeline.add_effect(ZoomPulseEffect(threshold=0.3, intensity=1.2))
        pipeline.add_effect(BubbleDistortionEffect(threshold=0.4))
        pipeline.add_effect(ScreenShakeEffect(threshold=0.5, intensity=1.3))
        pipeline.add_effect(ChromaticAberrationEffect(threshold=0.3, intensity=1.1))
        pipeline.add_effect(RGBSplitEffect(threshold=0.6, intensity=1.2))
        pipeline.add_effect(StrobeEffect(config.colors, threshold=0.45, intensity=1.3))  # Lowered threshold, increased intensity
        pipeline.add_effect(GlitchEffect(threshold=0.4, intensity=0.8))
        return pipeline
