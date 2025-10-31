"""
Preset che dimostra l'uso dei nuovi effetti visivi.
Questo preset utilizza tutti i 5 nuovi effetti in modo intelligente.
"""

from config import *


# Nuovo preset che usa TUTTI i nuovi effetti
GEOMETRIC_DREAMS_CONFIG = {
    'name': 'Geometric Dreams',
    'colors': [
        (0.9, 0.1, 0.9),    # Magenta vibrante
        (0.1, 0.9, 0.9),    # Cyan elettrico
        (0.9, 0.5, 0.1),    # Arancione caldo
        (0.5, 0.1, 0.9),    # Viola profondo
        (0.9, 0.9, 0.1),    # Giallo brillante
        (0.1, 0.9, 0.5),    # Verde vivace
    ],
    'thresholds': {
        'bass': 0.22,        # Sensibile per geometric distortion
        'mid': 0.14,         # Sensibile per triangular distortion
        'high': 0.11         # Sensibile per negative flash
    },
    'effects': {
        'strobe_intensity': 0.75,
        'distortion_threshold': 0.35,
        'noise_level': 0.08,
        'flash_duration': 0.02,
        
        # ===== NUOVI EFFETTI =====
        
        # Gradient Blend
        'use_gradient_blend': True,
        'gradient_direction': 'radial',      # horizontal, vertical, radial, diagonal
        'gradient_on_transition': True,      # Attiva su cambio sezione
        'gradient_intensity_base': 0.4,
        
        # Black & White
        'use_black_and_white': True,
        'bw_on_low_energy': True,            # B&W quando energia è bassa
        'bw_on_breakdown': 0.7,              # Intensità su breakdown
        'bw_on_intro': 0.6,                  # Intensità su intro
        'bw_on_outro': 'progressive',        # Progressivo su outro
        
        # Negative
        'use_negative': True,
        'negative_on_beat': True,            # Flash negativo sui beat
        'negative_on_treble': True,          # Negativo sugli alti
        'negative_intensity_max': 0.7,       # Intensità massima
        
        # Triangular Distortion
        'use_triangular_distortion': True,
        'triangular_on_mid': True,           # Attiva sulle mid frequencies
        'triangular_intensity': 0.75,
        'triangular_on_buildup': True,       # Crescente nel buildup
        
        # Geometric Distortion
        'use_geometric_distortion': True,
        'geometric_mode_bass': 'pinch',      # pinch sui bassi
        'geometric_mode_drop': 'swirl',      # swirl sui drop
        'geometric_mode_breakdown': 'barrel', # barrel sui breakdown
        'geometric_mode_intro': 'pincushion', # pincushion sulle intro
        'geometric_intensity_max': 0.85,
    }
}


CINEMATIC_BW_CONFIG = {
    'name': 'Cinematic Black & White',
    'colors': [
        (0.9, 0.9, 0.9),    # Bianco sporco
        (0.7, 0.7, 0.7),    # Grigio chiaro
        (0.4, 0.4, 0.4),    # Grigio medio
        (0.2, 0.2, 0.2),    # Grigio scuro
    ],
    'thresholds': {
        'bass': 0.30,
        'mid': 0.20,
        'high': 0.15
    },
    'effects': {
        'strobe_intensity': 0.5,
        'distortion_threshold': 0.5,
        'noise_level': 0.15,           # Noise per texture film
        'flash_duration': 0.04,
        
        # Focus su B&W e gradienti
        'use_gradient_blend': True,
        'gradient_direction': 'radial',
        'gradient_intensity_base': 0.6,
        
        'use_black_and_white': True,
        'bw_base_intensity': 0.9,      # Principalmente B&W
        'bw_color_flash_on_beat': True, # Flash colore sui beat
        
        'use_geometric_distortion': True,
        'geometric_mode': 'pincushion',
        'geometric_subtle': True,       # Distorsioni sottili
    }
}


NEGATIVE_FLASH_CONFIG = {
    'name': 'Negative Flash Madness',
    'colors': [
        (1.0, 0.0, 0.5),    # Rosa shock
        (0.0, 1.0, 0.8),    # Turchese brillante
        (0.8, 0.0, 1.0),    # Viola elettrico
        (1.0, 0.5, 0.0),    # Arancione vibrante
    ],
    'thresholds': {
        'bass': 0.20,
        'mid': 0.12,
        'high': 0.10
    },
    'effects': {
        'strobe_intensity': 0.85,
        'distortion_threshold': 0.30,
        'noise_level': 0.12,
        'flash_duration': 0.015,
        
        # Focus su negative e distorsioni
        'use_negative': True,
        'negative_aggressive': True,
        'negative_on_every_beat': True,
        'negative_intensity_max': 0.95,
        
        'use_triangular_distortion': True,
        'triangular_intensity': 0.9,
        
        'use_geometric_distortion': True,
        'geometric_mode': 'swirl',
        'geometric_chaotic': True,
    }
}


TRIANGULAR_TECHNO_CONFIG = {
    'name': 'Triangular Techno',
    'colors': [
        (0.0, 1.0, 1.0),    # Cyan puro
        (1.0, 0.0, 1.0),    # Magenta puro
        (1.0, 1.0, 0.0),    # Giallo puro
        (0.0, 1.0, 0.0),    # Verde puro
    ],
    'thresholds': {
        'bass': 0.25,
        'mid': 0.15,
        'high': 0.12
    },
    'effects': {
        'strobe_intensity': 0.80,
        'distortion_threshold': 0.35,
        'noise_level': 0.10,
        'flash_duration': 0.02,
        
        # Focus su distorsione triangolare
        'use_triangular_distortion': True,
        'triangular_primary_effect': True,
        'triangular_intensity': 0.95,
        'triangular_on_all_frequencies': True,
        
        'use_geometric_distortion': True,
        'geometric_mode': 'pinch',
        'geometric_on_kick': True,
        
        'use_gradient_blend': True,
        'gradient_direction': 'diagonal',
        'gradient_animated': True,
    }
}


GEOMETRIC_SWIRL_CONFIG = {
    'name': 'Geometric Swirl Vortex',
    'colors': [
        (0.8, 0.0, 0.8),    # Magenta
        (0.0, 0.8, 0.8),    # Cyan
        (0.8, 0.4, 0.0),    # Arancione
        (0.4, 0.0, 0.8),    # Viola
    ],
    'thresholds': {
        'bass': 0.28,
        'mid': 0.18,
        'high': 0.13
    },
    'effects': {
        'strobe_intensity': 0.75,
        'distortion_threshold': 0.40,
        'noise_level': 0.08,
        'flash_duration': 0.03,
        
        # Focus su distorsioni geometriche
        'use_geometric_distortion': True,
        'geometric_primary_effect': True,
        'geometric_modes_rotation': ['swirl', 'pinch', 'barrel'],  # Rotazione tra modi
        'geometric_intensity_max': 0.90,
        'geometric_on_all_sections': True,
        
        'use_triangular_distortion': True,
        'triangular_intensity': 0.6,
        'triangular_complementary': True,  # Complementare a geometric
        
        'use_gradient_blend': True,
        'gradient_direction': 'radial',
        'gradient_subtle': True,
    }
}


# Aggiorna la funzione get_preset_config per includere i nuovi preset
def get_preset_config_extended(preset_name: str) -> dict:
    """
    Ottieni configurazione preset per nome (versione estesa con nuovi preset).
    
    Args:
        preset_name: Nome del preset
        
    Returns:
        Dizionario con la configurazione
    """
    presets = {
        # Preset originali
        'dark_techno': DARK_TECHNO_CONFIG,
        'cyberpunk': CYBERPUNK_CONFIG,
        'industrial': INDUSTRIAL_CONFIG,
        'acid_house': ACID_HOUSE_CONFIG,
        'extreme_vibrant': EXTREME_VIBRANT_CONFIG,
        'psychedelic_refraction': PSYCHEDELIC_REFRACTION_CONFIG,
        'intelligent_adaptive': INTELLIGENT_ADAPTIVE_CONFIG,
        
        # Nuovi preset con effetti geometrici
        'geometric_dreams': GEOMETRIC_DREAMS_CONFIG,
        'cinematic_bw': CINEMATIC_BW_CONFIG,
        'negative_flash': NEGATIVE_FLASH_CONFIG,
        'triangular_techno': TRIANGULAR_TECHNO_CONFIG,
        'geometric_swirl': GEOMETRIC_SWIRL_CONFIG,
    }
    
    return presets.get(preset_name.lower(), DARK_TECHNO_CONFIG)


# Mappa preset → descrizione
PRESET_DESCRIPTIONS = {
    'geometric_dreams': 'Tutti i nuovi effetti combinati in modo bilanciato',
    'cinematic_bw': 'Focus su bianco e nero cinematografico con gradienti',
    'negative_flash': 'Negativi aggressivi con flash sui beat',
    'triangular_techno': 'Distorsioni triangolari per techno/elettronica',
    'geometric_swirl': 'Vortici e distorsioni geometriche intense',
}


def print_new_presets_info():
    """Stampa informazioni sui nuovi preset."""
    print("\n" + "=" * 70)
    print("🆕 NUOVI PRESET CON EFFETTI AVANZATI")
    print("=" * 70)
    
    for preset_name, description in PRESET_DESCRIPTIONS.items():
        config = get_preset_config_extended(preset_name)
        print(f"\n📦 {config['name']}")
        print(f"   ID: '{preset_name}'")
        print(f"   📝 {description}")
        print(f"   🎨 Colori: {len(config['colors'])}")
        print(f"   🎛️  Effetti speciali:")
        
        effects = config['effects']
        
        # Mostra quali nuovi effetti sono attivi
        if effects.get('use_gradient_blend'):
            direction = effects.get('gradient_direction', 'radial')
            print(f"      ✓ Gradient Blend ({direction})")
        
        if effects.get('use_black_and_white'):
            print(f"      ✓ Black & White")
        
        if effects.get('use_negative'):
            print(f"      ✓ Negative Flash")
        
        if effects.get('use_triangular_distortion'):
            print(f"      ✓ Triangular Distortion")
        
        if effects.get('use_geometric_distortion'):
            mode = effects.get('geometric_mode', 'swirl')
            print(f"      ✓ Geometric Distortion ({mode})")
    
    print("\n" + "=" * 70)
    print("💡 Per usare un preset, specifica il nome nella GUI o nel codice:")
    print("   fx = AudioVisualFX(..., effect_style='geometric_dreams')")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_new_presets_info()
    
    print("\n🧪 Test caricamento preset...")
    for preset_name in PRESET_DESCRIPTIONS.keys():
        config = get_preset_config_extended(preset_name)
        print(f"  ✓ {preset_name:20} → {config['name']}")
    
    print("\n✅ Tutti i preset caricati correttamente!")
