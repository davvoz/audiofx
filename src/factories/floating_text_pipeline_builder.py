"""
Floating Text Pipeline Builder - Crea pipeline con effetto testo fluttuante
"""

from ..effects import EffectPipeline, FloatingText, ColorPulseEffect
from ..models.data_models import EffectConfig


class FloatingTextPipelineBuilder:
    """Builder per pipeline con floating text personalizzabile"""
    
    @staticmethod
    def build(config: EffectConfig, 
              text: str = "MUSIC",
              font_size: int = 120,
              color_scheme: str = "rainbow",
              animation_style: str = "wave",
              add_background_effects: bool = True,
              start_time: float = None,
              end_time: float = None) -> EffectPipeline:
        """
        Crea una pipeline con floating text.
        
        Args:
            config: Configurazione effetti
            text: Testo da visualizzare
            font_size: Dimensione font base
            color_scheme: Schema colori ('rainbow', 'fire', 'ice', 'neon', 'gold')
            animation_style: Stile animazione ('wave', 'bounce', 'spin', 'pulse', 'glitch')
            add_background_effects: Se True, aggiunge effetti di background sottili
            start_time: Tempo in secondi quando il testo inizia ad apparire (None = dall'inizio)
            end_time: Tempo in secondi quando il testo scompare (None = fino alla fine)
            
        Returns:
            EffectPipeline configurata
        """
        pipeline = EffectPipeline()
        
        # Effetto di background sottile (opzionale)
        if add_background_effects:
            bg_pulse = ColorPulseEffect(
                intensity=0.2,  # Molto sottile
                color=(100, 100, 150)
            )
            pipeline.add_effect(bg_pulse)
        
        # Effetto principale: Floating Text
        floating_text = FloatingText(
            text=text,
            font_size=font_size,
            color_scheme=color_scheme,
            animation_style=animation_style,
            start_time=start_time,
            end_time=end_time
        )
        pipeline.add_effect(floating_text)
        
        return pipeline
    
    @staticmethod
    def build_minimal(config: EffectConfig, text: str = "BEATS") -> EffectPipeline:
        """Versione minimal con solo testo bianco che pulsa"""
        return FloatingTextPipelineBuilder.build(
            config=config,
            text=text,
            font_size=100,
            color_scheme="default",
            animation_style="pulse",
            add_background_effects=False
        )
    
    @staticmethod
    def build_party(config: EffectConfig, text: str = "PARTY") -> EffectPipeline:
        """Versione party con rainbow e animazione wave"""
        return FloatingTextPipelineBuilder.build(
            config=config,
            text=text,
            font_size=140,
            color_scheme="rainbow",
            animation_style="wave",
            add_background_effects=True
        )
    
    @staticmethod
    def build_fire(config: EffectConfig, text: str = "FIRE") -> EffectPipeline:
        """Versione fuoco con colori caldi e bounce"""
        return FloatingTextPipelineBuilder.build(
            config=config,
            text=text,
            font_size=130,
            color_scheme="fire",
            animation_style="bounce",
            add_background_effects=True
        )
    
    @staticmethod
    def build_neon(config: EffectConfig, text: str = "NEON") -> EffectPipeline:
        """Versione neon con glitch effect"""
        return FloatingTextPipelineBuilder.build(
            config=config,
            text=text,
            font_size=150,
            color_scheme="neon",
            animation_style="glitch",
            add_background_effects=True
        )
    
    @staticmethod
    def build_ice(config: EffectConfig, text: str = "CHILL") -> EffectPipeline:
        """Versione ice con colori freddi e spin"""
        return FloatingTextPipelineBuilder.build(
            config=config,
            text=text,
            font_size=120,
            color_scheme="ice",
            animation_style="spin",
            add_background_effects=True
        )
