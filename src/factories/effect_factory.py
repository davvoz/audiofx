"""
Effect factory for creating preset pipelines - delegates to specialized builders.
"""

from typing import List

from ..models.data_models import EffectConfig
from ..effects import BaseEffect, EffectPipeline
from .standard_pipeline_builder import StandardPipelineBuilder
from .extreme_pipeline_builder import ExtremePipelineBuilder
from .psychedelic_pipeline_builder import PsychedelicPipelineBuilder
from .minimal_pipeline_builder import MinimalPipelineBuilder
from .cyberpunk_pipeline_builder import CyberpunkPipelineBuilder
from .industrial_pipeline_builder import IndustrialPipelineBuilder
from .acid_house_pipeline_builder import AcidHousePipelineBuilder
from .retro_wave_pipeline_builder import RetroWavePipelineBuilder
from .horror_pipeline_builder import HorrorPipelineBuilder
from .floating_text_pipeline_builder import FloatingTextPipelineBuilder
from .texture_flow_pipeline_builder import (
    TextureFlowPipelineBuilder,
    IntenseTextureFlowPipelineBuilder,
    MinimalTextureFlowPipelineBuilder
)


class EffectFactory:
    """Factory for creating effect pipelines - delegates to specialized builders."""
    
    @staticmethod
    def create_standard_pipeline(config: EffectConfig) -> EffectPipeline:
        """
        Create standard effect pipeline using StandardPipelineBuilder.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with standard effects
        """
        return StandardPipelineBuilder.build(config)
    
    @staticmethod
    def create_extreme_pipeline(config: EffectConfig) -> EffectPipeline:
        """
        Create extreme effect pipeline using ExtremePipelineBuilder.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with extreme effects
        """
        return ExtremePipelineBuilder.build(config)
    
    @staticmethod
    def create_psychedelic_pipeline(config: EffectConfig) -> EffectPipeline:
        """
        Create psychedelic effect pipeline using PsychedelicPipelineBuilder.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with psychedelic effects
        """
        return PsychedelicPipelineBuilder.build(config)
    
    @staticmethod
    def create_minimal_pipeline(config: EffectConfig) -> EffectPipeline:
        """
        Create minimal effect pipeline using MinimalPipelineBuilder.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with minimal effects
        """
        return MinimalPipelineBuilder.build(config)
    
    @staticmethod
    def create_cyberpunk_pipeline(config: EffectConfig) -> EffectPipeline:
        """Create cyberpunk effect pipeline."""
        return CyberpunkPipelineBuilder.build(config)
    
    @staticmethod
    def create_industrial_pipeline(config: EffectConfig) -> EffectPipeline:
        """Create industrial effect pipeline."""
        return IndustrialPipelineBuilder.build(config)
    
    @staticmethod
    def create_acid_house_pipeline(config: EffectConfig) -> EffectPipeline:
        """Create acid house effect pipeline."""
        return AcidHousePipelineBuilder.build(config)
    
    @staticmethod
    def create_retro_wave_pipeline(config: EffectConfig) -> EffectPipeline:
        """Create retro wave effect pipeline."""
        return RetroWavePipelineBuilder.build(config)
    
    @staticmethod
    def create_horror_pipeline(config: EffectConfig) -> EffectPipeline:
        """Create horror effect pipeline."""
        return HorrorPipelineBuilder.build(config)
    
    @staticmethod
    def create_floating_text_pipeline(config: EffectConfig, text: str = "MUSIC",
                                     font_size: int = 120,
                                     color_scheme: str = "rainbow",
                                     animation_style: str = "wave") -> EffectPipeline:
        """
        Create floating text effect pipeline.
        
        Args:
            config: Effect configuration
            text: Text to display
            font_size: Base font size
            color_scheme: Color scheme ('rainbow', 'fire', 'ice', 'neon', 'gold')
            animation_style: Animation style ('wave', 'bounce', 'spin', 'pulse', 'glitch')
            
        Returns:
            EffectPipeline with floating text
        """
        return FloatingTextPipelineBuilder.build(
            config=config,
            text=text,
            font_size=font_size,
            color_scheme=color_scheme,
            animation_style=animation_style
        )
    
    @staticmethod
    def create_texture_flow_pipeline(config: EffectConfig) -> EffectPipeline:
        """
        Create texture flow effect pipeline with slow-rhythm stretching (4x slower).
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with texture flow effects
        """
        return TextureFlowPipelineBuilder.build(config)
    
    @staticmethod
    def create_intense_texture_flow_pipeline(config: EffectConfig) -> EffectPipeline:
        """
        Create intense texture flow effect pipeline with dramatic stretching.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with intense texture flow effects
        """
        return IntenseTextureFlowPipelineBuilder.build(config)
    
    @staticmethod
    def create_minimal_texture_flow_pipeline(config: EffectConfig) -> EffectPipeline:
        """
        Create minimal texture flow effect pipeline focusing on pure stretching.
        
        Args:
            config: Effect configuration
            
        Returns:
            EffectPipeline with minimal texture flow effects
        """
        return MinimalTextureFlowPipelineBuilder.build(config)
    
    @staticmethod
    def create_custom_pipeline(effects: List[BaseEffect]) -> EffectPipeline:
        """
        Create custom effect pipeline from list of effects.
        
        Args:
            effects: List of effect instances
            
        Returns:
            EffectPipeline with custom effects
        """
        pipeline = EffectPipeline()
        for effect in effects:
            pipeline.add_effect(effect)
        return pipeline
