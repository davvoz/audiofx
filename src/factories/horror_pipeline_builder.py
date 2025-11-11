"""
Horror effect pipeline builder.
"""

from ..effects.effect_pipeline import EffectPipeline
from ..effects.color_pulse import ColorPulseEffect
from ..effects.strobe import StrobeEffect
from ..effects.strobe_negative import StrobeNegativeEffect
from ..effects.glitch import GlitchEffect
from ..effects.screen_shake import ScreenShakeEffect
from ..effects.rgb_split import RGBSplitEffect
from ..models.data_models import EffectConfig


class HorrorPipelineBuilder:
    """Builder for horror effect pipeline."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Build horror pipeline with dark, bloody colors and disturbing effects.
        
        Args:
            config: Effect configuration
            
        Returns:
            Configured effect pipeline
        """
        pipeline = EffectPipeline()
        
        # Horror colors: blood red, black, grey, dark red
        if not config.colors:
            config.colors = [
                (0.8, 0.0, 0.0),    # Blood red
                (0.0, 0.0, 0.0),    # Black
                (0.5, 0.5, 0.5),    # Grey
                (0.3, 0.0, 0.0),    # Dark red
            ]
        
        # Horror effects - disturbing, random glitch, split RGB estremo
        pipeline.add_effect(RGBSplitEffect(threshold=0.35, intensity=1.4))  # RGB split molto forte e spesso
        pipeline.add_effect(GlitchEffect(threshold=0.3, intensity=1.3))  # Glitch frequenti e forti
        pipeline.add_effect(ScreenShakeEffect(threshold=0.45, intensity=1.2))  # Shake inquietante
        pipeline.add_effect(ColorPulseEffect(
            bass_threshold=0.35,
            mid_threshold=0.25,
            high_threshold=0.18,
            intensity=1.1
        ))
        pipeline.add_effect(StrobeEffect(colors=config.colors, threshold=0.5, intensity=1.3))  # Lowered threshold, increased intensity
        pipeline.add_effect(StrobeNegativeEffect(threshold=0.6, intensity=1.2))  # Negative strobe for disturbing effect
        # Focus su effetti disturbanti: split, glitch, shake, negative strobe
        
        return pipeline
