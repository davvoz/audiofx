"""
Service layer for business logic.
Separates concerns following Single Responsibility and Dependency Inversion principles.
"""

import os
from typing import List, Optional, Tuple
from .models import VideoGenerationConfig, AudioThresholds, EffectsConfiguration


class ValidationService:
    """Handles validation logic."""
    
    @staticmethod
    def validate_video_config(config: VideoGenerationConfig) -> Tuple[bool, Optional[str]]:
        """
        Validate video generation configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate audio input - skip if using video audio
        if not (config.mode == "video" and config.use_video_audio):
            # Only validate audio if NOT using video audio
            if not config.audio_path or not os.path.exists(config.audio_path):
                return False, "Seleziona un file audio valido"
        
        # Validate media input based on mode
        if config.mode == "image":
            if not config.image_path or not os.path.exists(config.image_path):
                return False, "Seleziona un file immagine valido"
        else:  # video mode
            if not config.video_path or not os.path.exists(config.video_path):
                return False, "Seleziona un file video valido"
        
        # Validate output
        if not config.output_path:
            return False, "Specifica un file di output"
        
        # Validate duration if specified
        if config.duration:
            try:
                duration = float(config.duration)
                if duration <= 0:
                    return False, "La durata deve essere maggiore di 0"
            except ValueError:
                return False, "Durata non valida (deve essere un numero)"
        
        return True, None
    
    @staticmethod
    def validate_effects(effects_config: EffectsConfiguration) -> Tuple[bool, Optional[str]]:
        """
        Validate effects configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not effects_config.is_any_effect_enabled():
            return False, "Seleziona almeno un effetto"
        
        return True, None


class EffectFactoryService:
    """Creates effect instances from configuration."""
    
    @staticmethod
    def create_effects_from_config(
        effects_config: EffectsConfiguration,
        thresholds: AudioThresholds
    ) -> List:
        """
        Create effect instances from configuration.
        
        Returns:
            List of effect instances in configured order
        """
        from src.effects import (
            ColorPulseEffect, ZoomPulseEffect, StrobeEffect, StrobeNegativeEffect,
            GlitchEffect, ChromaticAberrationEffect, BubbleDistortionEffect,
            ScreenShakeEffect, RGBSplitEffect, ElectricArcsEffect,
            FashionLightningEffect, AdvancedGlitchEffect, DimensionalWarpEffect,
            VortexDistortionEffect, FloatingText, GhostParticlesEffect,
            TextureStretchEffect
        )
        
        # Map effect names to factory functions
        effect_factories = {
            "ColorPulse": lambda: ColorPulseEffect(
                bass_threshold=thresholds.bass,
                mid_threshold=thresholds.mid,
                high_threshold=thresholds.high,
                intensity=effects_config.effects["ColorPulse"].intensity
            ),
            "ZoomPulse": lambda: ZoomPulseEffect(
                threshold=thresholds.bass,
                intensity=effects_config.effects["ZoomPulse"].intensity
            ),
            "Strobe": lambda: StrobeEffect(
                colors=[(1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0)],
                threshold=0.8,
                intensity=effects_config.effects["Strobe"].intensity
            ),
            "StrobeNegative": lambda: StrobeNegativeEffect(
                threshold=0.8,
                intensity=effects_config.effects["StrobeNegative"].intensity
            ),
            "Glitch": lambda: GlitchEffect(
                threshold=thresholds.mid,
                intensity=effects_config.effects["Glitch"].intensity
            ),
            "ChromaticAberration": lambda: ChromaticAberrationEffect(
                threshold=thresholds.high,
                intensity=effects_config.effects["ChromaticAberration"].intensity
            ),
            "BubbleDistortion": lambda: BubbleDistortionEffect(
                threshold=thresholds.bass,
                intensity=effects_config.effects["BubbleDistortion"].intensity
            ),
            "ScreenShake": lambda: ScreenShakeEffect(
                threshold=thresholds.mid,
                intensity=effects_config.effects["ScreenShake"].intensity
            ),
            "RGBSplit": lambda: RGBSplitEffect(
                threshold=thresholds.high,
                intensity=effects_config.effects["RGBSplit"].intensity
            ),
            "ElectricArcs": lambda: ElectricArcsEffect(
                colors=[(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)],
                threshold=0.7,
                intensity=effects_config.effects["ElectricArcs"].intensity
            ),
            "FashionLightning": lambda: FashionLightningEffect(
                colors=[(1.0, 0.0, 0.8), (0.0, 0.9, 1.0), (0.8, 1.0, 0.0)],
                threshold=0.65,
                branching_probability=0.6,
                max_branches=5,
                segment_length_min=5,
                segment_length_max=20,
                intensity=effects_config.effects["FashionLightning"].intensity
            ),
            "AdvancedGlitch": lambda: AdvancedGlitchEffect(
                threshold=0.5,
                channel_shift_amount=8,
                block_size_range=(10, 80),
                intensity=effects_config.effects["AdvancedGlitch"].intensity
            ),
            "DimensionalWarp": lambda: DimensionalWarpEffect(
                bass_threshold=thresholds.bass,
                mid_threshold=thresholds.mid,
                warp_strength=45.0,
                rotation_speed=0.5,
                perspective_depth=200.0,
                wave_frequency=2.0,
                layer_count=3,
                intensity=effects_config.effects["DimensionalWarp"].intensity
            ),
            "VortexDistortion": lambda: VortexDistortionEffect(
                threshold=0.2,
                max_angle=35.0,
                radius_falloff=1.8,
                rotation_speed=3.0,
                smoothing=0.3,
                intensity=effects_config.effects["VortexDistortion"].intensity
            ),
            "FloatingText": lambda: FloatingText(
                text=effects_config.floating_text_config.content,
                font_size=effects_config.floating_text_config.font_size,
                color_scheme=effects_config.floating_text_config.color_scheme,
                animation_style=effects_config.floating_text_config.animation,
                start_time=float(effects_config.floating_text_config.start_time) if effects_config.floating_text_config.start_time else None,
                end_time=float(effects_config.floating_text_config.end_time) if effects_config.floating_text_config.end_time else None
            ),
            "GhostParticles": lambda: GhostParticlesEffect(
                sample_density=18,
                explosion_threshold=0.5,
                particle_lifetime=70.0,
                max_particles=600,
                intensity=effects_config.effects["GhostParticles"].intensity
            ),
            "TextureStretch": lambda: TextureStretchEffect(
                bass_threshold=0.25,
                mid_threshold=0.2,
                max_stretch=45.0,
                wave_complexity=3,
                flow_speed=0.15,
                stretch_smoothness=0.92,
                direction_change_speed=0.08,
                texture_grain=2.0,
                intensity=effects_config.effects["TextureStretch"].intensity
            ),
        }
        
        # Create effects in configured order
        custom_effects = []
        for effect_name in effects_config.effect_order:
            if effect_name in effect_factories and effect_name in effects_config.effects:
                if effects_config.effects[effect_name].enabled:
                    custom_effects.append(effect_factories[effect_name]())
        
        return custom_effects


class FileTypeService:
    """Manages file type definitions."""
    
    @staticmethod
    def get_audio_types() -> List[Tuple[str, str]]:
        """Get supported audio file types."""
        return [
            ("Audio", "*.mp3 *.wav *.flac *.m4a"),
            ("Tutti i file", "*.*")
        ]
    
    @staticmethod
    def get_image_types() -> List[Tuple[str, str]]:
        """Get supported image file types."""
        return [
            ("Immagini", "*.jpg *.jpeg *.png"),
            ("Tutti i file", "*.*")
        ]
    
    @staticmethod
    def get_video_types() -> List[Tuple[str, str]]:
        """Get supported video file types."""
        return [
            ("Video", "*.mp4 *.avi *.mov *.mkv *.flv"),
            ("Tutti i file", "*.*")
        ]
    
    @staticmethod
    def get_output_types() -> List[Tuple[str, str]]:
        """Get supported output file types."""
        return [
            ("MP4", "*.mp4"),
            ("Tutti i file", "*.*")
        ]
