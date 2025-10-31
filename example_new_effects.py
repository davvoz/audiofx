"""
Esempio di utilizzo dei nuovi effetti visivi.

Questo script dimostra come integrare i nuovi effetti nel video generator:
- Passaggi sfumati (gradient blend)
- Bianco e nero (black and white)
- Negativo (negative)
- Distorsione triangolare (triangular distortion)
- Distorsione geometrica (geometric distortion)

Per utilizzare questi effetti, puoi modificare il metodo _apply_section_effects
o creare un nuovo preset in config.py che li includa.
"""

from video_generator import AudioVisualFX
import numpy as np


def example_custom_effects_preset():
    """
    Esempio di come creare un preset personalizzato che usa i nuovi effetti.
    """
    
    # Configurazione custom che include i nuovi effetti
    custom_config = {
        'name': 'New Effects Showcase',
        'colors': [
            (1.0, 0.0, 0.5),    # Rosa
            (0.0, 1.0, 0.8),    # Turchese
            (0.8, 0.0, 1.0),    # Viola
            (1.0, 0.5, 0.0),    # Arancione
        ],
        'thresholds': {
            'bass': 0.25,
            'mid': 0.15,
            'high': 0.12
        },
        'effects': {
            'strobe_intensity': 0.7,
            'distortion_threshold': 0.4,
            'noise_level': 0.1,
            'flash_duration': 0.03,
            
            # Nuovi effetti
            'use_gradient_blend': True,
            'gradient_direction': 'radial',      # horizontal, vertical, radial, diagonal
            'use_black_and_white': True,
            'bw_intensity': 0.5,                 # 0 = colore, 1 = completamente B&W
            'use_negative': True,
            'negative_intensity': 0.3,           # 0 = normale, 1 = completamente negativo
            'use_triangular_distortion': True,
            'triangular_intensity': 0.6,
            'use_geometric_distortion': True,
            'geometric_mode': 'swirl',           # pinch, barrel, pincushion, swirl
            'geometric_intensity': 0.5,
        }
    }
    
    return custom_config


def demo_gradient_effects():
    """Dimostra i vari tipi di gradiente."""
    print("\n🎨 Demo Passaggi Sfumati")
    print("=" * 60)
    print("I passaggi sfumati possono essere usati per:")
    print("  • Transizioni smooth tra colori")
    print("  • Effetti di fade graduale")
    print("  • Creazione di atmosfere")
    print("\nDirezioni disponibili:")
    print("  - horizontal: sfumatura da sinistra a destra")
    print("  - vertical: sfumatura dall'alto in basso")
    print("  - radial: sfumatura dal centro verso l'esterno")
    print("  - diagonal: sfumatura diagonale")
    
    # Esempio di codice per applicare l'effetto
    print("\n💻 Codice esempio:")
    print("""
    frame = AudioVisualFX._gradient_blend(
        frame, 
        intensity=bass[i] * 0.8,  # Reattivo ai bassi
        direction='radial'
    )
    """)


def demo_black_and_white():
    """Dimostra l'effetto bianco e nero."""
    print("\n⚫ Demo Bianco e Nero")
    print("=" * 60)
    print("L'effetto bianco e nero può essere usato per:")
    print("  • Creare contrasto drammatico")
    print("  • Evidenziare forme e strutture")
    print("  • Effetti vintage o cinematografici")
    print("\nControllo intensità:")
    print("  - 0.0: immagine a colori originale")
    print("  - 0.5: blend 50% tra colore e B&W")
    print("  - 1.0: completamente in bianco e nero")
    
    # Esempio di codice
    print("\n💻 Codice esempio:")
    print("""
    # B&W sui breakdown (sezioni melodiche)
    if section_type == 'breakdown':
        frame = AudioVisualFX._black_and_white(
            frame, 
            intensity=0.7  # Molto desaturato
        )
    
    # B&W progressivo su fade out
    if section_type == 'outro':
        bw_intensity = min(current_time / audio_duration, 1.0)
        frame = AudioVisualFX._black_and_white(frame, bw_intensity)
    """)


def demo_negative():
    """Dimostra l'effetto negativo."""
    print("\n🔄 Demo Negativo")
    print("=" * 60)
    print("L'effetto negativo inverte i colori e può essere usato per:")
    print("  • Effetti psichedelici")
    print("  • Flash drammatici sui beat")
    print("  • Transizioni impattanti")
    print("\nControllo intensità:")
    print("  - 0.0: colori originali")
    print("  - 0.5: blend tra normale e negativo")
    print("  - 1.0: completamente invertito")
    
    # Esempio di codice
    print("\n💻 Codice esempio:")
    print("""
    # Negativo flash sui beat
    if beat_intensity > 0.8:
        frame = AudioVisualFX._negative(
            frame,
            intensity=beat_intensity  # Intensità legata al beat
        )
    
    # Negativo sugli alti (hi-hat, cymbals)
    if treble[i] > 0.7:
        frame = AudioVisualFX._negative(
            frame,
            intensity=treble[i] * 0.6
        )
    """)


def demo_triangular_distortion():
    """Dimostra la distorsione triangolare."""
    print("\n🔺 Demo Distorsione Triangolare")
    print("=" * 60)
    print("La distorsione triangolare crea pattern geometrici e può essere usata per:")
    print("  • Effetti glitch geometrici")
    print("  • Pattern ritmici visivi")
    print("  • Sincronizzazione con synth e lead")
    print("\nCaratteristiche:")
    print("  - Animata nel tempo (usa frame_time)")
    print("  - Pattern di onde triangolari (non sinusoidali)")
    print("  - Ideale per musica elettronica")
    
    # Esempio di codice
    print("\n💻 Codice esempio:")
    print("""
    # Distorsione triangolare sui mid (synth, lead)
    if mid[i] > 0.4:
        frame = AudioVisualFX._triangular_distortion(
            frame,
            intensity=mid[i] * 1.1,
            frame_time=current_time
        )
    
    # Distorsione crescente nel buildup
    if section_type == 'buildup':
        buildup_progress = (current_time - section_start) / section_duration
        frame = AudioVisualFX._triangular_distortion(
            frame,
            intensity=0.3 + buildup_progress * 0.6,
            frame_time=current_time
        )
    """)


def demo_geometric_distortion():
    """Dimostra le distorsioni geometriche."""
    print("\n📐 Demo Distorsione Geometrica")
    print("=" * 60)
    print("Le distorsioni geometriche offrono 4 modalità diverse:")
    print("\n1. PINCH - Compressione verso il centro")
    print("   • Effetto 'succhiato' verso il centro")
    print("   • Ottimo per enfasi sui bassi pesanti")
    
    print("\n2. BARREL - Rigonfiamento (fisheye)")
    print("   • Effetto lente fish-eye")
    print("   • Crea senso di espansione")
    
    print("\n3. PINCUSHION - Compressione dei bordi")
    print("   • Opposto del barrel")
    print("   • Effetto 'risucchio' dai bordi")
    
    print("\n4. SWIRL - Rotazione a spirale")
    print("   • Rotazione crescente dal centro")
    print("   • Effetto vortice/turbine")
    
    # Esempio di codice
    print("\n💻 Codice esempio:")
    print("""
    # Pinch sui bassi (kick drum)
    if bass[i] > 0.6:
        frame = AudioVisualFX._geometric_distortion(
            frame,
            intensity=bass[i] * 0.8,
            mode='pinch'
        )
    
    # Swirl sui drop
    if section_type == 'drop':
        frame = AudioVisualFX._geometric_distortion(
            frame,
            intensity=total_intensity * 0.7,
            mode='swirl'
        )
    
    # Barrel sui breakdown per effetto espansivo
    if section_type == 'breakdown':
        frame = AudioVisualFX._geometric_distortion(
            frame,
            intensity=0.5,
            mode='barrel'
        )
    """)


def demo_combined_usage():
    """Dimostra come combinare i nuovi effetti."""
    print("\n🌈 Demo Combinazione Effetti")
    print("=" * 60)
    print("Gli effetti possono essere combinati per risultati unici:")
    
    print("\n💡 Esempio 1: Intro Cinematografica")
    print("""
    if section_type == 'intro':
        # B&W per atmosfera cinematografica
        frame = AudioVisualFX._black_and_white(frame, 0.8)
        # Gradiente radiale per focus centrale
        frame = AudioVisualFX._gradient_blend(frame, 0.4, 'radial')
        # Distorsione geometrica leggera
        frame = AudioVisualFX._geometric_distortion(frame, 0.3, 'pincushion')
    """)
    
    print("\n💡 Esempio 2: Drop Psichedelico")
    print("""
    if section_type == 'drop' and beat_intensity > 0.7:
        # Negativo flash sul beat
        frame = AudioVisualFX._negative(frame, beat_intensity)
        # Distorsione triangolare
        frame = AudioVisualFX._triangular_distortion(frame, 0.8, current_time)
        # Swirl per caos
        frame = AudioVisualFX._geometric_distortion(frame, 0.6, 'swirl')
    """)
    
    print("\n💡 Esempio 3: Breakdown Melodico")
    print("""
    if section_type == 'breakdown':
        # Gradiente diagonale soft
        frame = AudioVisualFX._gradient_blend(frame, 0.5, 'diagonal')
        # B&W parziale per tono melodico
        frame = AudioVisualFX._black_and_white(frame, 0.4)
        # Barrel distortion delicato
        frame = AudioVisualFX._geometric_distortion(frame, 0.3, 'barrel')
    """)
    
    print("\n💡 Esempio 4: Outro Fade")
    print("""
    if section_type == 'outro':
        # Calcola progress (0 → 1)
        progress = (current_time - section_start) / section_duration
        
        # B&W crescente
        frame = AudioVisualFX._black_and_white(frame, progress * 0.9)
        # Gradiente radiale per fade
        frame = AudioVisualFX._gradient_blend(frame, progress * 0.7, 'radial')
    """)


def integration_guide():
    """Guida per integrare i nuovi effetti."""
    print("\n📚 Guida Integrazione")
    print("=" * 60)
    print("\n1️⃣  MODO RAPIDO: Modifica _apply_section_effects")
    print("   Apri video_generator.py e aggiungi gli effetti nel metodo")
    print("   _apply_section_effects per applicarli automaticamente")
    print("   in base alle sezioni musicali.")
    
    print("\n2️⃣  MODO CUSTOM: Crea un nuovo effect_style")
    print("   Aggiungi un nuovo ramo nel metodo _generate_frames:")
    print("""
    elif self.effect_style == "my_custom_style":
        # I tuoi effetti personalizzati qui
        frame = self._gradient_blend(frame, bass[i] * 0.8, 'radial')
        frame = self._black_and_white(frame, mid[i] * 0.6)
        # ... altri effetti
    """)
    
    print("\n3️⃣  MODO PRESET: Configura in config.py")
    print("   Aggiungi un nuovo preset in config.py simile a")
    print("   EXTREME_VIBRANT_CONFIG o PSYCHEDELIC_REFRACTION_CONFIG")
    print("   includendo i parametri per i nuovi effetti.")
    
    print("\n4️⃣  Esempi di mappatura audio-reattiva:")
    print("   • Bass → Geometric distortion (pinch/swirl)")
    print("   • Mid → Triangular distortion")
    print("   • Treble → Negative flash")
    print("   • Low energy → Black & White")
    print("   • Transitions → Gradient blend")


def main():
    """Esegue tutte le demo."""
    print("\n" + "=" * 60)
    print("🎬 GUIDA AI NUOVI EFFETTI VISIVI")
    print("=" * 60)
    
    demo_gradient_effects()
    demo_black_and_white()
    demo_negative()
    demo_triangular_distortion()
    demo_geometric_distortion()
    demo_combined_usage()
    integration_guide()
    
    print("\n" + "=" * 60)
    print("✅ FINE DELLA GUIDA")
    print("=" * 60)
    print("\n📝 Prossimi passi:")
    print("  1. Scegli gli effetti che vuoi utilizzare")
    print("  2. Decidi dove integrarli (section-based o custom style)")
    print("  3. Modifica video_generator.py di conseguenza")
    print("  4. Testa con python gui.py o audio_visual_fx.py")
    print("\n💡 Suggerimento: inizia con combinazioni semplici")
    print("   e aumenta gradualmente la complessità!")
    print("\n🎉 Buon divertimento con i nuovi effetti!")


if __name__ == "__main__":
    main()
