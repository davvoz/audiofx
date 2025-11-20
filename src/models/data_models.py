"""
Data classes and enumerations for audio-visual generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple, Dict, Any
import numpy as np


# Type definitions
ProgressCallback = Optional[Callable[[str, dict], None]]


class EffectStyle(Enum):
    """Effect style presets."""
    STANDARD = "standard"              # Dark Techno
    EXTREME = "extreme"                # Extreme Vibrant
    INTELLIGENT = "intelligent"        # Intelligent Adaptive
    PSYCHEDELIC = "psychedelic"        # Psychedelic Refraction
    MINIMAL = "minimal"                # Minimal/Performance
    SOLIDS_3D = "3d_solids"           # 3D Rotating Solids
    CYBERPUNK = "cyberpunk"           # Cyberpunk
    INDUSTRIAL = "industrial"          # Industrial
    ACID_HOUSE = "acid_house"         # Acid House
    RETRO_WAVE = "retro_wave"         # Retro Wave
    HORROR = "horror"                  # Horror
    TEXTURE_FLOW = "texture_flow"      # Texture Flow (Slow Rhythm)
    INTENSE_TEXTURE_FLOW = "intense_texture_flow"  # Intense Texture Flow
    MINIMAL_TEXTURE_FLOW = "minimal_texture_flow"  # Minimal Texture Flow


class SectionType(Enum):
    """Music section types for intelligent analysis."""
    INTRO = "intro"
    BUILDUP = "buildup"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    OUTRO = "outro"
    STEADY = "steady"


@dataclass
class AudioAnalysis:
    """Container for audio analysis results."""
    audio_signal: np.ndarray
    sample_rate: int
    frequencies: np.ndarray
    magnitude: np.ndarray
    bass_energy: np.ndarray
    mid_energy: np.ndarray
    treble_energy: np.ndarray
    beat_times: np.ndarray
    sections: Optional[List[Dict[str, Any]]] = None


@dataclass
class EffectConfig:
    """Configuration for visual effects."""
    colors: List[Tuple[float, float, float]]
    bass_threshold: float = 0.3
    mid_threshold: float = 0.2
    high_threshold: float = 0.15
    transition_duration: float = 3.0
    effect_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameContext:
    """Context information for a single frame."""
    frame: np.ndarray
    time: float
    frame_index: int
    bass: float
    mid: float
    treble: float
    beat_intensity: float
    section_type: Optional[str] = None
    color_index: int = 0
