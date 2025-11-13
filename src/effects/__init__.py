"""
Visual effects system with base classes and concrete implementations.
"""

from .base_effect import BaseEffect
from .effect_pipeline import EffectPipeline

# Concrete effects
from .color_pulse import ColorPulseEffect
from .strobe import StrobeEffect
from .strobe_negative import StrobeNegativeEffect
from .zoom_pulse import ZoomPulseEffect
from .chromatic_aberration import ChromaticAberrationEffect
from .glitch import GlitchEffect
from .bubble_distortion import BubbleDistortionEffect
from .screen_shake import ScreenShakeEffect
from .rgb_split import RGBSplitEffect
from .electric_arcs import ElectricArcsEffect
from .fashion_lightning import FashionLightningEffect
from .advanced_glitch import AdvancedGlitchEffect
from .dimensional_warp import DimensionalWarpEffect
from .vortex_distortion import VortexDistortionEffect
from .floating_text import FloatingText

__all__ = [
    # Base classes
    'BaseEffect',
    'EffectPipeline',
    # Concrete effects
    'ColorPulseEffect',
    'StrobeEffect',
    'StrobeNegativeEffect',
    'ZoomPulseEffect',
    'ChromaticAberrationEffect',
    'GlitchEffect',
    'BubbleDistortionEffect',
    'ScreenShakeEffect',
    'RGBSplitEffect',
    'ElectricArcsEffect',
    'FashionLightningEffect',
    'AdvancedGlitchEffect',
    'DimensionalWarpEffect',
    'VortexDistortionEffect',
    'FloatingText',
]
