"""
Cyberpunk effect pipeline builder.
"""

from ..effects.effect_pipeline import EffectPipeline
from ..effects.color_pulse import ColorPulseEffect
from ..effects.strobe import StrobeEffect
from ..effects.chromatic_aberration import ChromaticAberrationEffect
from ..effects.glitch import GlitchEffect
from ..effects.screen_shake import ScreenShakeEffect
from ..effects.rgb_split import RGBSplitEffect
from ..models.data_models import EffectConfig


class CyberpunkPipelineBuilder:
    """Builder for cyberpunk effect pipeline."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Build cyberpunk pipeline with neon colors and glitch effects.
        
        Args:
            config: Effect configuration
            
        Returns:
            Configured effect pipeline
        """
        pipeline = EffectPipeline()
        
        # Cyberpunk colors: neon cyan, magenta, yellow
        if not config.colors:
            config.colors = [
                (0.0, 1.0, 1.0),    # Cyan neon
                (1.0, 0.0, 1.0),    # Magenta neon
                (1.0, 1.0, 0.0),    # Yellow neon
                (0.0, 0.8, 1.0),    # Blue electric
                (1.0, 0.0, 0.5),    # Pink neon
            ]
        
        # Cyberpunk effects - focus on glitch and chromatic aberration
        pipeline.add_effect(ChromaticAberrationEffect(threshold=0.25, intensity=1.3))  # Sempre attivo
        pipeline.add_effect(ColorPulseEffect(
            bass_threshold=0.25,
            mid_threshold=0.2,
            high_threshold=0.15,
            intensity=1.1
        ))
        pipeline.add_effect(GlitchEffect(threshold=0.35, intensity=0.9))  # Glitch frequenti
        pipeline.add_effect(StrobeEffect(colors=config.colors, threshold=0.5, intensity=1.2))  # Lowered threshold
        pipeline.add_effect(RGBSplitEffect(threshold=0.5, intensity=0.8))  # RGB split sui beat
        
        return pipeline
