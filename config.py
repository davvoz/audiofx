"""
Configurazione avanzata per Audio Visual FX Generator
Personalizza gli effetti e i parametri per diversi stili musicali
"""

# =============================================================================
# CONFIGURAZIONI PRESET PER GENERI MUSICALI
# =============================================================================

DARK_TECHNO_CONFIG = {
    'name': 'Dark Techno',
    'colors': [
        (0.8, 0.0, 0.8),    # Magenta scuro
        (0.0, 0.8, 0.8),    # Cyan scuro
        (0.8, 0.3, 0.0),    # Arancione scuro
        (0.5, 0.0, 0.8),    # Viola
        (0.8, 0.0, 0.3),    # Rosso scuro
        (0.0, 0.8, 0.3),    # Verde acido
    ],
    'thresholds': {
        'bass': 0.3,
        'mid': 0.2,
        'high': 0.15
    },
    'effects': {
        'strobe_intensity': 0.8,
        'distortion_threshold': 0.6,
        'noise_level': 0.1,
        'flash_duration': 0.05
    }
}

CYBERPUNK_CONFIG = {
    'name': 'Cyberpunk',
    'colors': [
        (1.0, 0.0, 1.0),    # Magenta neon
        (0.0, 1.0, 1.0),    # Cyan neon
        (1.0, 0.0, 0.5),    # Rosa neon
        (0.5, 0.0, 1.0),    # Viola neon
        (0.0, 1.0, 0.0),    # Verde neon
        (1.0, 0.5, 0.0),    # Arancione neon
    ],
    'thresholds': {
        'bass': 0.25,
        'mid': 0.15,
        'high': 0.1
    },
    'effects': {
        'strobe_intensity': 0.9,
        'distortion_threshold': 0.5,
        'noise_level': 0.15,
        'flash_duration': 0.03
    }
}

INDUSTRIAL_CONFIG = {
    'name': 'Industrial',
    'colors': [
        (0.7, 0.7, 0.7),    # Grigio metallico
        (0.8, 0.3, 0.0),    # Arancione ruggine
        (0.5, 0.0, 0.0),    # Rosso scuro
        (0.0, 0.5, 0.0),    # Verde militare
        (0.3, 0.3, 0.3),    # Grigio scuro
        (0.6, 0.4, 0.2),    # Marrone metallico
    ],
    'thresholds': {
        'bass': 0.4,
        'mid': 0.3,
        'high': 0.2
    },
    'effects': {
        'strobe_intensity': 0.7,
        'distortion_threshold': 0.4,
        'noise_level': 0.2,
        'flash_duration': 0.08
    }
}

ACID_HOUSE_CONFIG = {
    'name': 'Acid House',
    'colors': [
        (1.0, 1.0, 0.0),    # Giallo acido
        (0.0, 1.0, 0.5),    # Verde lime
        (1.0, 0.0, 1.0),    # Magenta
        (0.5, 1.0, 0.0),    # Verde acido
        (1.0, 0.5, 1.0),    # Rosa
        (0.0, 1.0, 1.0),    # Cyan
    ],
    'thresholds': {
        'bass': 0.2,
        'mid': 0.15,
        'high': 0.1
    },
    'effects': {
        'strobe_intensity': 0.95,
        'distortion_threshold': 0.3,
        'noise_level': 0.05,
        'flash_duration': 0.02
    }
}

EXTREME_VIBRANT_CONFIG = {
    'name': 'Extreme Vibrant',
    'colors': [
        (1.0, 0.0, 1.0),    # Magenta shock
        (0.0, 1.0, 1.0),    # Cyan elettrico
        (1.0, 0.2, 0.8),    # Rosa neon aggressivo
        (0.2, 1.0, 0.0),    # Verde lime vibrante
        (1.0, 0.0, 0.5),    # Rosso neon
        (0.5, 0.0, 1.0),    # Viola intenso
        (1.0, 1.0, 0.0),    # Giallo flash
        (0.0, 0.8, 1.0),    # Azzurro elettrico
    ],
    'thresholds': {
        'bass': 0.15,        # Molto sensibile sui bassi per effetto bolleggiante
        'mid': 0.10,         # Super sensibile sullo snare
        'high': 0.12         # Iper-reattivo sugli alti
    },
    'effects': {
        'strobe_intensity': 1.0,        # Massimo strobe
        'distortion_threshold': 0.2,    # Distorsioni aggressive
        'noise_level': 0.25,            # Noise alto per texture
        'flash_duration': 0.01,         # Flash rapidissimi
        'zoom_intensity': 0.95,         # Zoom estremo
        'vibration_amount': 0.9,        # Vibrazioni intense sullo snare
        'bubble_effect': 0.85,          # Effetto bolleggiante sui bassi
        'color_shift_speed': 0.98,      # Cambio colori rapidissimo
        'saturation_boost': 1.5,        # Saturazione oltre il limite
        'contrast_extreme': 1.8,        # Contrasto esagerato
        'pulse_multiplier': 3.0,        # Pulsazioni triple
        'shake_amount': 0.7,            # Screen shake forte
        'chromatic_aberration': 0.6,    # Aberrazione cromatica cyberpunk
        'glitch_probability': 0.15,     # 15% probabilità glitch per frame
    }
}

PSYCHEDELIC_REFRACTION_CONFIG = {
    'name': 'Psychedelic Refraction',
    'colors': [
        (1.0, 0.0, 0.5),    # Rosa psichedelico
        (0.0, 1.0, 0.8),    # Turchese brillante
        (0.8, 0.0, 1.0),    # Viola elettrico
        (1.0, 0.5, 0.0),    # Arancione acido
        (0.2, 1.0, 0.3),    # Verde lime
        (1.0, 0.8, 0.0),    # Giallo dorato
        (0.0, 0.6, 1.0),    # Blu cristallo
        (1.0, 0.0, 0.8),    # Magenta prismatico
    ],
    'thresholds': {
        'bass': 0.18,        # Sensibile per attivare rifrazioni
        'mid': 0.12,         # Rifrazioni medie su mid
        'high': 0.10         # Rifrazioni sottili sugli alti
    },
    'effects': {
        'strobe_intensity': 0.75,           # Strobe moderato
        'distortion_threshold': 0.25,       # Distorsioni fluide
        'noise_level': 0.08,                # Noise leggero
        'flash_duration': 0.03,             # Flash morbidi
        'refraction_intensity': 0.95,       # Intensità rifrazione massima
        'refraction_waves': 0.85,           # Onde di rifrazione
        'kaleidoscope_segments': 6,         # Segmenti caleidoscopio
        'prismatic_split': 0.90,            # Split prismatico RGB
        'liquid_flow': 0.88,                # Flusso liquido
        'crystal_distortion': 0.92,         # Distorsione cristallina
        'color_dispersion': 0.95,           # Dispersione cromatica
        'wave_frequency': 0.85,             # Frequenza onde
        'pixel_shift_chaos': 0.80,          # Caos shift pixel
    }
}

INTELLIGENT_ADAPTIVE_CONFIG = {
    'name': 'Intelligent Adaptive',
    'colors': [
        # Palette versatile che si adatta a ogni sezione
        (0.9, 0.1, 0.9),    # Magenta energetico
        (0.1, 0.9, 0.9),    # Cyan fresco
        (0.9, 0.5, 0.1),    # Arancione caldo
        (0.5, 0.1, 0.9),    # Viola profondo
        (0.9, 0.9, 0.1),    # Giallo brillante
        (0.1, 0.9, 0.5),    # Verde vivace
    ],
    'thresholds': {
        'bass': 0.20,        # Bilanciato per analisi
        'mid': 0.15,         # Sensibile ai cambi
        'high': 0.12         # Reattivo agli alti
    },
    'effects': {
        'strobe_intensity': 0.85,           # Variabile per sezione
        'distortion_threshold': 0.30,       # Adattivo
        'noise_level': 0.12,                # Medio
        'flash_duration': 0.02,             # Rapido
        # Parametri di analisi intelligente (VERSIONE CLASSICA)
        'section_analysis': True,           # Abilita analisi sezioni
        'energy_window': 2.0,               # Finestra analisi energia (secondi)
        'transition_smoothing': 0.5,        # Smooth tra sezioni
        'transition_duration': 3.0,         # Durata transizione smooth (secondi) - ULTRA SMOOTH
        'adaptive_intensity': 0.90,         # Intensità adattamento
        # Soglie riconoscimento sezioni
        'intro_threshold': 0.35,            # Energia bassa = intro
        'buildup_slope': 0.15,              # Pendenza energia = buildup
        'drop_impact': 0.75,                # Impatto improvviso = drop
        'break_sparsity': 0.40,             # Bassa densità = break
        'outro_decay': 0.30,                # Decadimento = outro
    }
}

INTELLIGENT_ADAPTIVE_PRO_CONFIG = {
    'name': 'Intelligent Adaptive Pro',
    'colors': [
        # Palette versatile che si adatta a ogni sezione
        (0.9, 0.1, 0.9),    # Magenta energetico
        (0.1, 0.9, 0.9),    # Cyan fresco
        (0.9, 0.5, 0.1),    # Arancione caldo
        (0.5, 0.1, 0.9),    # Viola profondo
        (0.9, 0.9, 0.1),    # Giallo brillante
        (0.1, 0.9, 0.5),    # Verde vivace
        (0.9, 0.1, 0.5),    # Rosa intenso
        (0.1, 0.5, 0.9),    # Blu elettrico
    ],
    'thresholds': {
        'bass': 0.20,        # Bilanciato per analisi
        'mid': 0.15,         # Sensibile ai cambi
        'high': 0.12         # Reattivo agli alti
    },
    'effects': {
        'strobe_intensity': 0.85,           # Variabile per sezione
        'distortion_threshold': 0.30,       # Adattivo
        'noise_level': 0.12,                # Medio
        'flash_duration': 0.02,             # Rapido
        # Parametri di analisi intelligente
        'section_analysis': True,           # Abilita analisi sezioni
        'energy_window': 2.0,               # Finestra analisi energia (secondi)
        'transition_smoothing': 0.5,        # Smooth tra sezioni
        'transition_duration': 2.5,         # Durata transizione smooth Pro - ULTRA SMOOTH
        'adaptive_intensity': 0.90,         # Intensità adattamento
        # Soglie riconoscimento sezioni
        'intro_threshold': 0.35,            # Energia bassa = intro
        'buildup_slope': 0.15,              # Pendenza energia = buildup
        'drop_impact': 0.75,                # Impatto improvviso = drop
        'break_sparsity': 0.40,             # Bassa densità = break
        'outro_decay': 0.30,                # Decadimento = outro
        
        # ===== NUOVI EFFETTI INTELLIGENTI (PRO) =====
        # Gradient Blend - Transizioni smooth tra sezioni
        'use_gradient_blend': True,
        'gradient_on_transition': True,     # Attiva su cambio sezione
        'gradient_direction_intro': 'radial',      # Radial per intro
        'gradient_direction_buildup': 'diagonal',  # Diagonal per buildup
        'gradient_direction_drop': 'horizontal',   # Horizontal per drop
        'gradient_direction_breakdown': 'vertical',# Vertical per breakdown
        'gradient_intensity': 0.5,
        
        # Black & White - Atmosfera su sezioni specifiche
        'use_black_and_white': True,
        'bw_on_intro': 0.6,                 # B&W parziale su intro
        'bw_on_breakdown': 0.7,             # B&W forte su breakdown
        'bw_on_outro': 'progressive',       # Fade progressivo su outro
        'bw_on_low_energy': True,           # B&W quando energia < threshold
        'bw_threshold': 0.35,
        
        # Negative - Flash drammatici
        'use_negative': True,
        'negative_on_beat': True,           # Flash su beat forti
        'negative_beat_threshold': 0.75,    # Solo beat > 0.75
        'negative_on_drop': True,           # Attivo sui drop
        'negative_intensity_max': 0.6,      # Intensità massima controllata
        
        # Triangular Distortion - Dinamica su mid frequencies
        'use_triangular_distortion': True,
        'triangular_on_mid': True,          # Attiva su mid (synth/lead)
        'triangular_mid_threshold': 0.4,    # Soglia mid
        'triangular_buildup_boost': 1.3,    # Boost nel buildup
        'triangular_intensity_base': 0.6,
        
        # Geometric Distortion - Adattivo per sezione
        'use_geometric_distortion': True,
        'geometric_adaptive_mode': True,    # Cambia modalità per sezione
        'geometric_intro_mode': 'pincushion',     # Intro: compressione sottile
        'geometric_buildup_mode': 'pinch',        # Buildup: tensione crescente
        'geometric_drop_mode': 'swirl',           # Drop: caos controllato
        'geometric_breakdown_mode': 'barrel',     # Breakdown: espansione
        'geometric_outro_mode': 'pincushion',     # Outro: chiusura
        'geometric_bass_boost': True,       # Boost intensità sui bassi
        'geometric_intensity_base': 0.6,
        'geometric_intensity_max': 0.85,
    }
}

# =============================================================================
# CONFIGURAZIONI EFFETTI AVANZATI
# =============================================================================

ADVANCED_EFFECTS_CONFIG = {
    'frequency_bands': {
        'sub_bass': (20, 60),      # Sub bass
        'bass': (60, 200),         # Bass
        'low_mid': (200, 500),     # Low mids
        'mid': (500, 2000),        # Mids
        'high_mid': (2000, 5000),  # High mids
        'presence': (5000, 8000),  # Presence
        'brilliance': (8000, 20000) # Brilliance
    },
    
    'effect_mappings': {
        'sub_bass': ['room_shake', 'heavy_pulse'],
        'bass': ['brightness_pulse', 'color_saturation'],
        'low_mid': ['hue_shift', 'contrast_boost'],
        'mid': ['saturation_boost', 'edge_enhancement'],
        'high_mid': ['sparkle_effect', 'sharp_flash'],
        'presence': ['strobe_effect', 'rapid_pulse'],
        'brilliance': ['white_flash', 'extreme_strobe']
    },
    
    'beat_sync_effects': {
        'kick': ['full_screen_flash', 'color_invert'],
        'snare': ['edge_flash', 'brightness_spike'],
        'hihat': ['sparkle', 'high_freq_pulse'],
        'crash': ['white_out', 'screen_shake']
    }
}

# =============================================================================
# CONFIGURAZIONI VIDEO OUTPUT
# =============================================================================

VIDEO_QUALITY_PRESETS = {
    'draft': {
        'resolution': (640, 360),
        'fps': 15,
        'bitrate': '500k',
        'quality': 'draft'
    },
    'preview': {
        'resolution': (720, 720),
        'fps': 30,
        'bitrate': '2M',
        'quality': 'medium'
    },
    'final': {
        'resolution': (720, 720),
        'fps': 60,
        'bitrate': '8M',
        'quality': 'high'
    },
    '4k': {
        'resolution': (3840, 2160),
        'fps': 60,
        'bitrate': '20M',
        'quality': 'ultra'
    }
}

# =============================================================================
# CONFIGURAZIONI PERFORMANCE
# =============================================================================

PERFORMANCE_CONFIG = {
    'low_memory': {
        'chunk_size': 100,
        'temp_cleanup': True,
        'compress_frames': True,
        'max_resolution': (720, 720)
    },
    'balanced': {
        'chunk_size': 500,
        'temp_cleanup': True,
        'compress_frames': False,
        'max_resolution': (720, 720)
    },
    'high_quality': {
        'chunk_size': 1000,
        'temp_cleanup': False,
        'compress_frames': False,
        'max_resolution': (3840, 2160)
    }
}

# =============================================================================
# FUNZIONI DI UTILITÀ
# =============================================================================

def get_preset_config(preset_name: str) -> dict:
    """
    Ottieni configurazione preset per nome
    
    Args:
        preset_name: Nome del preset ('dark_techno', 'cyberpunk', 'industrial', 'acid_house', 
                     'extreme_vibrant', 'psychedelic_refraction', 'intelligent_adaptive', 
                     'intelligent_adaptive_pro', '3d_solids')
        
    Returns:
        Dizionario con la configurazione
    """
    presets = {
        'dark_techno': DARK_TECHNO_CONFIG,
        'cyberpunk': CYBERPUNK_CONFIG,
        'industrial': INDUSTRIAL_CONFIG,
        'acid_house': ACID_HOUSE_CONFIG,
        'extreme_vibrant': EXTREME_VIBRANT_CONFIG,
        'psychedelic_refraction': PSYCHEDELIC_REFRACTION_CONFIG,
        'intelligent_adaptive': INTELLIGENT_ADAPTIVE_CONFIG,
        'intelligent_adaptive_pro': INTELLIGENT_ADAPTIVE_PRO_CONFIG,
        '3d_solids': ROTATING_3D_SOLIDS_CONFIG,
        'rotating_3d_solids': ROTATING_3D_SOLIDS_CONFIG,
    }
    
    return presets.get(preset_name.lower(), DARK_TECHNO_CONFIG)

def get_quality_preset(quality: str) -> dict:
    """
    Ottieni preset qualità video
    
    Args:
        quality: Livello qualità ('draft', 'preview', 'final', '4k')
        
    Returns:
        Dizionario con le impostazioni video
    """
    return VIDEO_QUALITY_PRESETS.get(quality.lower(), VIDEO_QUALITY_PRESETS['preview'])

def get_performance_config(mode: str) -> dict:
    """
    Ottieni configurazione performance
    
    Args:
        mode: Modalità performance ('low_memory', 'balanced', 'high_quality')
        
    Returns:
        Dizionario con le impostazioni performance
    """
    return PERFORMANCE_CONFIG.get(mode.lower(), PERFORMANCE_CONFIG['balanced'])

# =============================================================================
# CONFIGURAZIONI CUSTOM
# =============================================================================

def create_custom_config(name: str, 
                        colors: list, 
                        bass_threshold: float = 0.3,
                        mid_threshold: float = 0.2, 
                        high_threshold: float = 0.15) -> dict:
    """
    Crea una configurazione personalizzata
    
    Args:
        name: Nome della configurazione
        colors: Lista di colori RGB (0-1)
        bass_threshold: Soglia per frequenze basse
        mid_threshold: Soglia per frequenze medie
        high_threshold: Soglia per frequenze acute
        
    Returns:
        Configurazione personalizzata
    """
    return {
        'name': name,
        'colors': colors,
        'thresholds': {
            'bass': bass_threshold,
            'mid': mid_threshold,
            'high': high_threshold
        },
        'effects': {
            'strobe_intensity': 0.8,
            'distortion_threshold': 0.6,
            'noise_level': 0.1,
            'flash_duration': 0.05
        }
    }

# Esempi di configurazioni personalizzate
RETRO_WAVE_CONFIG = create_custom_config(
    name='Retro Wave',
    colors=[
        (1.0, 0.0, 1.0),    # Magenta
        (0.0, 1.0, 1.0),    # Cyan
        (1.0, 0.4, 0.8),    # Rosa
        (0.6, 0.0, 1.0),    # Viola
    ],
    bass_threshold=0.25,
    mid_threshold=0.15,
    high_threshold=0.1
)

HORROR_CONFIG = create_custom_config(
    name='Horror',
    colors=[
        (0.8, 0.0, 0.0),    # Rosso sangue
        (0.0, 0.0, 0.0),    # Nero
        (0.5, 0.5, 0.5),    # Grigio
        (0.3, 0.0, 0.0),    # Rosso scuro
    ],
    bass_threshold=0.4,
    mid_threshold=0.3,
    high_threshold=0.2
)

ROTATING_3D_SOLIDS_CONFIG = {
    'name': '3D Rotating Solids',
    'colors': [
        (1.0, 0.0, 1.0),    # Magenta elettrico
        (0.0, 1.0, 1.0),    # Cyan brillante
        (1.0, 0.5, 0.0),    # Arancione neon
        (0.5, 0.0, 1.0),    # Viola profondo
        (0.0, 1.0, 0.5),    # Verde lime
        (1.0, 0.0, 0.5),    # Rosa shock
        (0.5, 1.0, 0.0),    # Giallo-verde
        (0.0, 0.5, 1.0),    # Blu elettrico
    ],
    'thresholds': {
        'bass': 0.20,        # Sensibile per pulsazioni
        'mid': 0.15,         # Controllo rotazione
        'high': 0.12         # Frammentazione dinamica
    },
    'effects': {
        'strobe_intensity': 0.75,
        'distortion_threshold': 0.30,
        'noise_level': 0.10,
        'flash_duration': 0.02,
        
        # ===== PARAMETRI SPECIFICI 3D SOLIDS =====
        '3d_solids_enabled': True,
        '3d_solids_intensity': 0.85,            # Intensità globale effetto
        
        # Rotazione
        'rotation_speed_base': 2.0,             # Velocità base rotazione
        'rotation_speed_mid_multiplier': 4.0,   # Boost da mid frequencies
        'rotation_dual_axis': True,             # Rotazione su assi multipli
        
        # Pulsazione (sui bassi)
        'pulse_enabled': True,
        'pulse_amount_base': 0.1,
        'pulse_amount_bass_multiplier': 0.4,
        'pulse_frequency': 8.0,                 # Hz della pulsazione
        
        # Frammentazione (numero di solidi)
        'num_solids_base': 3,                   # Minimo numero solidi
        'num_solids_treble_multiplier': 5,     # Solidi aggiunti dal treble
        'num_solids_max': 12,                   # Massimo numero solidi
        
        # Collisioni (sui beat)
        'collision_enabled': True,
        'collision_threshold': 0.6,             # Soglia beat per collisioni
        'collision_intensity': 0.5,             # Intensità esplosione
        'collision_interference': True,         # Interferenze tra solidi
        
        # Proiezione 3D
        'perspective_enabled': True,
        'depth_variation': 0.7,                 # Variazione profondità (0-1)
        'perspective_scale_min': 0.7,           # Scala minima prospettiva
        'perspective_scale_max': 1.0,           # Scala massima prospettiva
        
        # Distorsione dimensionale
        'dimensional_warp_enabled': True,
        'warp_intensity_multiplier': 2.0,
        'warp_frequency_x': 0.05,
        'warp_frequency_y': 0.04,
        
        # Bordi luminosi
        'edge_glow_enabled': True,
        'edge_glow_threshold': 0.6,             # Intensità minima per glow
        'edge_glow_brightness': 2.5,
        'edge_glow_treble_reactive': True,      # Colore bordi cambia con treble
        
        # Combinazioni con altri effetti
        'combine_with_chromatic': True,         # Aberrazione cromatica
        'chromatic_intensity': 0.9,
        'combine_with_color_pulse': True,       # Color pulse
        'color_pulse_intensity': 1.1,
        'combine_with_zoom': True,              # Zoom pulse leggero
        'zoom_intensity': 0.4,
        'combine_with_glitch': True,            # Glitch occasionale
        'glitch_probability': 0.15,
        
        # Effetti elettrici sui beat
        'electric_arcs_on_beat': True,
        'electric_arcs_threshold': 0.7,
        'electric_arcs_intensity': 0.6,
        
        # Mappatura sezioni musicali (per modalità intelligente)
        'section_mapping': {
            'intro': {
                'intensity': 0.4,
                'rotation_speed': 0.6,
                'num_solids': 3,
                'pulse_amount': 0.5,
            },
            'buildup': {
                'intensity': 0.7,
                'rotation_speed': 1.0,
                'num_solids': 5,
                'pulse_amount': 0.8,
            },
            'drop': {
                'intensity': 0.95,
                'rotation_speed': 1.3,
                'num_solids': 8,
                'pulse_amount': 1.0,
            },
            'breakdown': {
                'intensity': 0.5,
                'rotation_speed': 0.7,
                'num_solids': 6,
                'pulse_amount': 0.6,
            },
            'outro': {
                'intensity': 0.3,
                'rotation_speed': 0.5,
                'num_solids': 4,
                'pulse_amount': 0.4,
            },
        }
    }
}