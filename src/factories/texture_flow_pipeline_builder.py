"""
Texture Flow pipeline builder - showcases the TextureStretchEffect.
"""

from ..models.data_models import EffectConfig
from ..effects import (
    EffectPipeline,
    TextureStretchEffect,
    ChromaticAberrationEffect,
    ColorPulseEffect,
    VortexDistortionEffect,
)


class TextureFlowPipelineBuilder:
    """Builds texture flow effect pipeline centered on texture stretching."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Create texture flow effect pipeline with slow-rhythm stretching.
        
        This pipeline showcases the TextureStretchEffect with complementary
        effects that enhance the flowing, organic texture distortions.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with texture flow effects
        """
        pipeline = EffectPipeline()
        
        # Subtle color pulse to enhance the organic feel
        pipeline.add_effect(ColorPulseEffect(
            config.bass_threshold * 0.9, 
            config.mid_threshold * 0.9, 
            config.high_threshold * 0.9,
            intensity=0.7
        ))
        
        # Main texture stretch effect with slow rhythm (4x slower than music)
        pipeline.add_effect(TextureStretchEffect(
            bass_threshold=0.25,
            mid_threshold=0.2,
            max_stretch=45.0,
            wave_complexity=3,
            flow_speed=0.15,
            stretch_smoothness=0.92,
            direction_change_speed=0.08,
            texture_grain=2.0,
            intensity=1.0
        ))
        
        # Light chromatic aberration to enhance depth
        pipeline.add_effect(ChromaticAberrationEffect(
            threshold=0.3, 
            intensity=0.5
        ))
        
        # Optional subtle vortex for additional motion
        pipeline.add_effect(VortexDistortionEffect(
            threshold=0.4,
            max_angle=15.0,
            radius_falloff=2.2,
            rotation_speed=1.5,
            smoothing=0.4,
            intensity=0.6
        ))
        
        return pipeline


class IntenseTextureFlowPipelineBuilder:
    """Builds intense texture flow pipeline with stronger effects."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Create intense texture flow pipeline with dramatic stretching.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with intense texture flow effects
        """
        pipeline = EffectPipeline()
        
        # Stronger color pulse
        pipeline.add_effect(ColorPulseEffect(
            config.bass_threshold * 0.7, 
            config.mid_threshold * 0.7, 
            config.high_threshold * 0.7,
            intensity=1.2
        ))
        
        # Intense texture stretch with more complexity
        pipeline.add_effect(TextureStretchEffect(
            bass_threshold=0.2,
            mid_threshold=0.15,
            max_stretch=65.0,
            wave_complexity=4,
            flow_speed=0.2,
            stretch_smoothness=0.88,
            direction_change_speed=0.12,
            texture_grain=2.5,
            intensity=1.3
        ))
        
        # Stronger chromatic effect
        pipeline.add_effect(ChromaticAberrationEffect(
            threshold=0.25, 
            intensity=0.8
        ))
        
        # More pronounced vortex
        pipeline.add_effect(VortexDistortionEffect(
            threshold=0.3,
            max_angle=25.0,
            radius_falloff=1.8,
            rotation_speed=2.5,
            smoothing=0.35,
            intensity=0.9
        ))
        
        return pipeline


class MinimalTextureFlowPipelineBuilder:
    """Builds minimal texture flow pipeline focusing only on stretching."""
    
    @staticmethod
    def build(config: EffectConfig) -> EffectPipeline:
        """
        Create minimal texture flow pipeline with pure stretching.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with minimal texture flow effects
        """
        pipeline = EffectPipeline()
        
        # Only the texture stretch effect for pure, undistracted flow
        pipeline.add_effect(TextureStretchEffect(
            bass_threshold=0.22,
            mid_threshold=0.18,
            max_stretch=40.0,
            wave_complexity=2,
            flow_speed=0.12,
            stretch_smoothness=0.95,
            direction_change_speed=0.06,
            texture_grain=1.8,
            intensity=1.0
        ))
        
        # Very subtle chromatic aberration
        pipeline.add_effect(ChromaticAberrationEffect(
            threshold=0.4, 
            intensity=0.3
        ))
        
        return pipeline
