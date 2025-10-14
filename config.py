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
        preset_name: Nome del preset ('dark_techno', 'cyberpunk', 'industrial', 'acid_house')
        
    Returns:
        Dizionario con la configurazione
    """
    presets = {
        'dark_techno': DARK_TECHNO_CONFIG,
        'cyberpunk': CYBERPUNK_CONFIG,
        'industrial': INDUSTRIAL_CONFIG,
        'acid_house': ACID_HOUSE_CONFIG
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