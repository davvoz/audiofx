"""
Retro Wave effect pipeline builder.
"""

from ..effects.effect_pipeline import EffectPipeline
from ..effects.color_pulse import ColorPulseEffect
from ..effects.strobe import StrobeEffect
from ..effects.chromatic_aberration import ChromaticAberrationEffect
from ..effects.zoom_pulse import ZoomPulseEffect
from ..models.data_models import EffectConfig


class RetroWavePipelineBuilder:
    """Builder for retro wave effect pipeline."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Build retro wave pipeline with 80s aesthetic colors.
        
        Args:
            config: Effect configuration
            
        Returns:
            Configured effect pipeline
        """
        pipeline = EffectPipeline()
        
        # Retro wave colors: magenta, cyan, pink, purple
        if not config.colors:
            config.colors = [
                (1.0, 0.0, 1.0),    # Magenta
                (0.0, 1.0, 1.0),    # Cyan
                (1.0, 0.4, 0.8),    # Pink
                (0.6, 0.0, 1.0),    # Purple
            ]
        
        # Retro wave effects - smooth, nostalgic, molto zoom
        pipeline.add_effect(ZoomPulseEffect(threshold=0.25, intensity=1.0))  # Zoom prominente
        pipeline.add_effect(ColorPulseEffect(
            bass_threshold=0.28,
            mid_threshold=0.18,
            high_threshold=0.12,
            intensity=1.0
        ))
        pipeline.add_effect(ChromaticAberrationEffect(threshold=0.4, intensity=0.5))  # Aberrazione soft
        pipeline.add_effect(StrobeEffect(colors=config.colors, threshold=0.5, intensity=1.1))  # Lowered threshold, increased intensity
        # NO glitch, NO shake - mantiene l'estetica smooth anni '80
        
        return pipeline
