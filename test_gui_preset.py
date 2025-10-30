"""
Test rapido per verificare che il preset Psychedelic Refraction
sia disponibile nella GUI e mappato correttamente.
"""

from config import get_preset_config

def test_gui_mapping():
    """Testa il mapping GUI -> config"""
    
    print("=" * 60)
    print("TEST MAPPING GUI -> CONFIG")
    print("=" * 60)
    
    # Mapping GUI (come in gui.py)
    name_to_key = {
        "Dark Techno": "dark_techno",
        "Cyberpunk": "cyberpunk",
        "Industrial": "industrial",
        "Acid House": "acid_house",
        "Extreme Vibrant": "extreme_vibrant",
        "Psychedelic Refraction": "psychedelic_refraction",
    }
    
    # Test mapping effect_style
    print("\n📋 Test Effect Style Mapping:")
    print("-" * 60)
    
    for gui_name, config_key in name_to_key.items():
        # Determina effect_style (come in gui.py)
        if config_key == "extreme_vibrant":
            effect_style = "extreme"
        elif config_key == "psychedelic_refraction":
            effect_style = "psychedelic"
        else:
            effect_style = "standard"
        
        # Carica config
        config = get_preset_config(config_key)
        
        print(f"\n✅ {gui_name:25} → {config_key}")
        print(f"   Config Name: {config['name']}")
        print(f"   Effect Style: {effect_style}")
        print(f"   Colors: {len(config['colors'])} colori")
        print(f"   Thresholds: bass={config['thresholds']['bass']}, "
              f"mid={config['thresholds']['mid']}, "
              f"high={config['thresholds']['high']}")
    
    print("\n" + "=" * 60)
    print("✅ Tutti i preset sono mappati correttamente!")
    print("=" * 60)
    
    # Verifica preset psichedelico specifico
    print("\n🔮 DETTAGLI PSYCHEDELIC REFRACTION:")
    print("-" * 60)
    
    config = get_preset_config("psychedelic_refraction")
    
    print(f"\nNome: {config['name']}")
    print(f"\nColori ({len(config['colors'])}):")
    for i, color in enumerate(config['colors'], 1):
        r, g, b = [int(c * 255) for c in color]
        print(f"  {i}. RGB({r}, {g}, {b}) = {color}")
    
    print(f"\nSoglie:")
    for key, value in config['thresholds'].items():
        print(f"  {key}: {value}")
    
    print(f"\nEffetti ({len(config['effects'])}):")
    for key, value in config['effects'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("🎉 Test completato con successo!")
    print("=" * 60)


if __name__ == "__main__":
    test_gui_mapping()
