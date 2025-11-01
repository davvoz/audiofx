"""
Acid House effect pipeline builder.
"""

from ..effects.effect_pipeline import EffectPipeline
from ..effects.color_pulse import ColorPulseEffect
from ..effects.strobe import StrobeEffect
from ..effects.zoom_pulse import ZoomPulseEffect
from ..effects.bubble_distortion import BubbleDistortionEffect
from ..effects.chromatic_aberration import ChromaticAberrationEffect
from ..models.data_models import EffectConfig


class AcidHousePipelineBuilder:
    """Builder for acid house effect pipeline."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Build acid house pipeline with vibrant acidic colors and warping.
        
        Args:
            config: Effect configuration
            
        Returns:
            Configured effect pipeline
        """
        pipeline = EffectPipeline()
        
        # Acid house colors: acidic yellow, lime green, magenta, cyan
        if not config.colors:
            config.colors = [
                (1.0, 1.0, 0.0),    # Acid yellow
                (0.0, 1.0, 0.5),    # Lime green
                (1.0, 0.0, 1.0),    # Magenta
                (0.5, 1.0, 0.0),    # Acid green
                (1.0, 0.5, 1.0),    # Pink
                (0.0, 1.0, 1.0),    # Cyan
            ]
        
        # Acid house effects - trippy, fluid, molto distorsione
        pipeline.add_effect(BubbleDistortionEffect(threshold=0.15, intensity=1.3))  # Bolle sui bassi sempre
        pipeline.add_effect(ColorPulseEffect(
            bass_threshold=0.2,
            mid_threshold=0.15,
            high_threshold=0.1,
            intensity=1.3
        ))
        #pipeline.add_effect(ZoomPulseEffect(threshold=0.25, intensity=1.1))  # Zoom pulsante
        pipeline.add_effect(StrobeEffect(colors=config.colors, threshold=0.5, intensity=1.4))  # Lowered threshold, increased intensity
        pipeline.add_effect(ChromaticAberrationEffect(threshold=0.3, intensity=0.7))  # Aberrazione trippy
        
        return pipeline
