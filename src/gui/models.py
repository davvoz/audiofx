"""
Models for GUI state management.
Follows Single Responsibility Principle - each model handles one concern.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import tkinter as tk


@dataclass
class VideoGenerationConfig:
    """Configuration for video generation."""
    mode: str = "image"  # "image" or "video"
    audio_path: str = ""
    image_path: str = ""
    video_path: str = ""
    output_path: str = "custom_output.mp4"
    fps: int = 30
    duration: str = ""  # Empty = full duration
    video_mode: str = "loop"  # "loop" or "stretch"
    use_video_audio: bool = False
    use_native_resolution: bool = True
    
    # Logo settings
    logo_path: str = ""
    logo_position: str = "top-right"
    logo_scale: float = 0.15
    logo_opacity: float = 1.0
    logo_margin: int = 12


@dataclass
class AudioThresholds:
    """Audio analysis thresholds."""
    bass: float = 0.3
    mid: float = 0.2
    high: float = 0.15


@dataclass
class EffectSettings:
    """Settings for a single effect."""
    enabled: bool
    intensity: float


@dataclass
class FloatingTextConfig:
    """Configuration for floating text effect."""
    content: str = "MUSIC"
    color_scheme: str = "rainbow"
    animation: str = "wave"
    font_size: int = 120
    start_time: str = ""  # Tempo di inizio in secondi (vuoto = dall'inizio)
    end_time: str = ""    # Tempo di fine in secondi (vuoto = fino alla fine)


@dataclass
class EffectsConfiguration:
    """Complete effects configuration."""
    effects: Dict[str, EffectSettings] = field(default_factory=dict)
    effect_order: List[str] = field(default_factory=list)
    floating_text_config: FloatingTextConfig = field(default_factory=FloatingTextConfig)
    
    def __post_init__(self):
        """Initialize default effects if empty."""
        if not self.effects:
            self._initialize_default_effects()
        if not self.effect_order:
            self.effect_order = list(self.effects.keys())
    
    def _initialize_default_effects(self):
        """Set up default effects configuration."""
        default_effects = {
            "ColorPulse": EffectSettings(enabled=True, intensity=1.0),
            "ZoomPulse": EffectSettings(enabled=True, intensity=1.0),
            "Strobe": EffectSettings(enabled=False, intensity=1.0),
            "StrobeNegative": EffectSettings(enabled=False, intensity=1.0),
            "Glitch": EffectSettings(enabled=False, intensity=1.0),
            "ChromaticAberration": EffectSettings(enabled=False, intensity=1.0),
            "BubbleDistortion": EffectSettings(enabled=False, intensity=1.0),
            "ScreenShake": EffectSettings(enabled=False, intensity=1.0),
            "RGBSplit": EffectSettings(enabled=False, intensity=1.0),
            "ElectricArcs": EffectSettings(enabled=False, intensity=1.0),
            "FashionLightning": EffectSettings(enabled=False, intensity=1.0),
            "AdvancedGlitch": EffectSettings(enabled=False, intensity=1.0),
            "DimensionalWarp": EffectSettings(enabled=False, intensity=1.0),
            "VortexDistortion": EffectSettings(enabled=False, intensity=1.0),
            "FloatingText": EffectSettings(enabled=False, intensity=1.0),
        }
        self.effects = default_effects
    
    def get_enabled_effects(self) -> List[str]:
        """Get list of enabled effects in order."""
        return [name for name in self.effect_order 
                if name in self.effects and self.effects[name].enabled]
    
    def is_any_effect_enabled(self) -> bool:
        """Check if at least one effect is enabled."""
        return any(effect.enabled for effect in self.effects.values())


@dataclass
class ApplicationState:
    """Complete application state."""
    video_config: VideoGenerationConfig = field(default_factory=VideoGenerationConfig)
    audio_thresholds: AudioThresholds = field(default_factory=AudioThresholds)
    effects_config: EffectsConfiguration = field(default_factory=EffectsConfiguration)
    is_processing: bool = False
    cancel_requested: bool = False
    
    def reset_processing_state(self):
        """Reset processing flags."""
        self.is_processing = False
        self.cancel_requested = False
