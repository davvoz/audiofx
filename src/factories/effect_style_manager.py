"""
Effect style manager for managing different effect presets.
"""

from typing import Dict

from ..models.data_models import EffectConfig, EffectStyle
from ..effects import EffectPipeline
from .effect_factory import EffectFactory


class EffectStyleManager:
    """Manages effect styles and their configurations."""
    
    def __init__(self, config: EffectConfig):
        """
        Initialize effect style manager.
        
        Args:
            config: Effect configuration
        """
        self.config = config
        self.pipelines: Dict[EffectStyle, EffectPipeline] = {}
        self._initialize_pipelines()
    
    def _initialize_pipelines(self):
        """Initialize effect pipelines for each style."""
        self.pipelines[EffectStyle.STANDARD] = EffectFactory.create_standard_pipeline(self.config)
        self.pipelines[EffectStyle.EXTREME] = EffectFactory.create_extreme_pipeline(self.config)
        self.pipelines[EffectStyle.PSYCHEDELIC] = EffectFactory.create_psychedelic_pipeline(self.config)
        self.pipelines[EffectStyle.MINIMAL] = EffectFactory.create_minimal_pipeline(self.config)
        self.pipelines[EffectStyle.CYBERPUNK] = EffectFactory.create_cyberpunk_pipeline(self.config)
        self.pipelines[EffectStyle.INDUSTRIAL] = EffectFactory.create_industrial_pipeline(self.config)
        self.pipelines[EffectStyle.ACID_HOUSE] = EffectFactory.create_acid_house_pipeline(self.config)
        self.pipelines[EffectStyle.RETRO_WAVE] = EffectFactory.create_retro_wave_pipeline(self.config)
        self.pipelines[EffectStyle.HORROR] = EffectFactory.create_horror_pipeline(self.config)
        # SOLIDS_3D and INTELLIGENT styles can be added here when implemented
    
    def get_pipeline(self, style: EffectStyle) -> EffectPipeline:
        """
        Get pipeline for given style.
        
        Args:
            style: Effect style enum
            
        Returns:
            EffectPipeline for the style
        """
        return self.pipelines.get(style, self.pipelines[EffectStyle.STANDARD])
    
    def add_custom_pipeline(self, style: EffectStyle, pipeline: EffectPipeline):
        """
        Add custom pipeline for a style.
        
        Args:
            style: Effect style enum
            pipeline: Custom effect pipeline
        """
        self.pipelines[style] = pipeline
    
    def set_config(self, config: EffectConfig):
        """
        Update configuration and reinitialize pipelines.
        
        Args:
            config: New effect configuration
        """
        self.config = config
        self._initialize_pipelines()
    
    def list_styles(self) -> list:
        """
        Get list of available styles.
        
        Returns:
            List of EffectStyle enums
        """
        return list(self.pipelines.keys())
