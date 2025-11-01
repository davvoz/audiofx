"""
Scalable OOP Audio-Reactive Visual Generator

A modular, extensible system for creating audio-reactive visualizations.

Basic Usage:
    from src import AudioVisualGenerator, EffectStyle
    
    generator = AudioVisualGenerator(
        audio_file="track.mp3",
        image_file="background.jpg",
        output_file="output.mp4",
        effect_style=EffectStyle.EXTREME
    )
    generator.generate()

Advanced Usage:
    from src import AudioVisualGenerator, EffectConfig
    from src.effects import ColorPulseEffect, ZoomPulseEffect
    
    # Custom configuration
    config = EffectConfig(
        colors=[(1.0, 0.0, 1.0), (0.0, 1.0, 1.0)],
        bass_threshold=0.25
    )
    
    # Create generator
    generator = AudioVisualGenerator(
        audio_file="track.mp3",
        image_file="background.jpg",
        effect_config=config
    )
    
    # Add custom effects
    generator.add_custom_effect(ColorPulseEffect(intensity=1.5))
    generator.add_custom_effect(ZoomPulseEffect(threshold=0.2))
    
    generator.generate()
"""

# Main generator
from .audio_visual_generator import AudioVisualGenerator

# Video audio sync
from .video_audio_sync import VideoAudioSync

# Data models and enums
from .models import (
    AudioAnalysis,
    EffectConfig,
    FrameContext,
    EffectStyle,
    SectionType,
    ProgressCallback,
)

# Interfaces
from .interfaces import (
    IEffect,
    IAudioAnalyzer,
    IVideoExporter,
)

# Effects
from .effects import (
    BaseEffect,
    EffectPipeline,
    ColorPulseEffect,
    StrobeEffect,
    ZoomPulseEffect,
    ChromaticAberrationEffect,
    GlitchEffect,
    BubbleDistortionEffect,
    ScreenShakeEffect,
    RGBSplitEffect,
)

# Core components
from .core import (
    AudioAnalyzer,
    FrameGenerator,
    VideoExporter,
)

# Factories
from .factories import (
    EffectFactory,
    EffectStyleManager,
    StandardPipelineBuilder,
    ExtremePipelineBuilder,
    PsychedelicPipelineBuilder,
    MinimalPipelineBuilder,
    CyberpunkPipelineBuilder,
    IndustrialPipelineBuilder,
    AcidHousePipelineBuilder,
    RetroWavePipelineBuilder,
    HorrorPipelineBuilder,
)

# Utils
from .utils import (
    ImageLoader,
)

# Core utilities
from .core import (
    FrequencyAnalyzer,
    BeatDetector,
    SectionAnalyzer,
)

__version__ = "2.0.0"
__author__ = "AudioFX Team"

__all__ = [
    # Main classes
    'AudioVisualGenerator',
    'VideoAudioSync',
    
    # Data models
    'AudioAnalysis',
    'EffectConfig',
    'FrameContext',
    'EffectStyle',
    'SectionType',
    'ProgressCallback',
    
    # Interfaces
    'IEffect',
    'IAudioAnalyzer',
    'IVideoExporter',
    
    # Base classes
    'BaseEffect',
    'EffectPipeline',
    
    # Concrete effects
    'ColorPulseEffect',
    'StrobeEffect',
    'ZoomPulseEffect',
    'ChromaticAberrationEffect',
    'GlitchEffect',
    'BubbleDistortionEffect',
    'ScreenShakeEffect',
    'RGBSplitEffect',
    
    # Core components
    'AudioAnalyzer',
    'FrameGenerator',
    'VideoExporter',
    
    # Factories
    'EffectFactory',
    'EffectStyleManager',
    'StandardPipelineBuilder',
    'ExtremePipelineBuilder',
    'PsychedelicPipelineBuilder',
    'MinimalPipelineBuilder',
    'CyberpunkPipelineBuilder',
    'IndustrialPipelineBuilder',
    'AcidHousePipelineBuilder',
    'RetroWavePipelineBuilder',
    'HorrorPipelineBuilder',
    
    # Utils
    'ImageLoader',
    
    # Core utilities
    'FrequencyAnalyzer',
    'BeatDetector',
    'SectionAnalyzer',
]
