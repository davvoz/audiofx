"""
Industrial effect pipeline builder.
"""

from ..effects.effect_pipeline import EffectPipeline
from ..effects.color_pulse import ColorPulseEffect
from ..effects.strobe import StrobeEffect
from ..effects.glitch import GlitchEffect
from ..effects.screen_shake import ScreenShakeEffect
from ..effects.rgb_split import RGBSplitEffect
from ..models.data_models import EffectConfig


class IndustrialPipelineBuilder:
    """Builder for industrial effect pipeline."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Build industrial pipeline with metallic colors and harsh effects.
        
        Args:
            config: Effect configuration
            
        Returns:
            Configured effect pipeline
        """
        pipeline = EffectPipeline()
        
        # Industrial colors: metallic greys, rust orange, dark red
        if not config.colors:
            config.colors = [
                (0.7, 0.7, 0.7),    # Metallic grey
                (0.8, 0.3, 0.0),    # Rust orange
                (0.5, 0.0, 0.0),    # Dark red
                (0.0, 0.5, 0.0),    # Military green
                (0.3, 0.3, 0.3),    # Dark grey
                (0.6, 0.4, 0.2),    # Metallic brown
            ]
        
        # Industrial effects - harsh, mechanical, molto shake
        pipeline.add_effect(ScreenShakeEffect(threshold=0.4, intensity=1.5))  # Shake fortissimo sui bassi
        pipeline.add_effect(ColorPulseEffect(
            bass_threshold=0.35,
            mid_threshold=0.25,
            high_threshold=0.2,
            intensity=0.9
        ))
        pipeline.add_effect(GlitchEffect(threshold=0.3, intensity=1.2))  # Glitch molto aggressivi
        pipeline.add_effect(StrobeEffect(colors=config.colors, threshold=0.45, intensity=1.0))  # Lowered threshold
        pipeline.add_effect(RGBSplitEffect(threshold=0.6, intensity=0.6))  # Split meccanico
        
        return pipeline
