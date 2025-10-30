"""
Test del nuovo preset Psychedelic Refraction
Effetti di rifrazione intelligente psichedelica tra pixels
"""

from config import PSYCHEDELIC_REFRACTION_CONFIG, get_preset_config
from video_generator import AudioVisualFX


def test_psychedelic_preset():
    """Test del preset psichedelico"""
    
    print("=" * 60)
    print("TEST PRESET: PSYCHEDELIC REFRACTION")
    print("=" * 60)
    
    # Carica configurazione
    config = get_preset_config('psychedelic_refraction')
    
    print(f"\n📋 Nome Preset: {config['name']}")
    print(f"\n🎨 Palette Colori ({len(config['colors'])} colori):")
    for i, color in enumerate(config['colors'], 1):
        print(f"  {i}. RGB: {color}")
    
    print(f"\n🎚️ Soglie Audio:")
    print(f"  Bass:  {config['thresholds']['bass']}")
    print(f"  Mid:   {config['thresholds']['mid']}")
    print(f"  High:  {config['thresholds']['high']}")
    
    print(f"\n✨ Effetti Speciali:")
    for effect, value in config['effects'].items():
        print(f"  {effect:.<30} {value}")
    
    print("\n" + "=" * 60)
    print("CARATTERISTICHE EFFETTO RIFRAZIONE:")
    print("=" * 60)
    print("""
    🔮 EFFETTO RIFRAZIONE PSICHEDELICA:
    
    1. 🌊 Rifrazione Ondulata Multi-Direzionale
       - Onde sinusoidali complesse simulano rifrazione attraverso prismi
       - Pattern fluidi che si muovono in direzioni multiple
       - Intensità reattiva ai bassi e agli alti
    
    2. 🌈 Dispersione Cromatica Prismatica
       - Separazione RGB con offset diversi per ogni canale
       - Simula rifrazione della luce attraverso un prisma
       - Canale rosso deviato verso l'esterno
       - Canale blu deviato verso l'interno
       - Effetto arcobaleno sui bordi degli oggetti
    
    3. 💎 Effetto Cristallo (Pixel Shift Intelligente)
       - Zone locali di rifrazione simulano facce di cristallo
       - Shift radiale dal centro di ogni "cristallo"
       - Blending fluido tra zone rifratte
       - 6-8 zone cristalline per frame ad alta intensità
    
    4. 🔶 Kaleidoscope (Intensità Alta)
       - Pattern speculare a 6 segmenti
       - Coordinate polari trasformate in pattern geometrico
       - Attivato su beat intensi e alte frequenze
    
    5. 🌀 Flusso Liquido
       - Distorsione fluida con movimento continuo
       - Simula rifrazione attraverso liquido in movimento
       - Turbolenza multi-onda per realismo
    
    6. 🎭 Split Prismatico Avanzato
       - Pattern sinusoidale per split RGB non uniforme
       - Effetto ondulato lungo l'immagine
       - Split verticale e orizzontale combinati
    
    📊 REATTIVITÀ AUDIO:
    - Bassi (0.18): Attiva flusso liquido e bolle
    - Mid (0.12): Intensifica rifrazioni
    - Alti (0.10): Split prismatico e dispersione
    - Beat: Kaleidoscope flash
    
    🎨 PALETTE CROMATICA:
    - 8 colori psichedelici ad alta saturazione
    - Rosa, turchese, viola, arancione acido
    - Giallo dorato, blu cristallo, magenta
    - Cambio colore fluido ogni 0.5 secondi
    """)
    
    print("\n" + "=" * 60)
    print("ESEMPIO D'USO:")
    print("=" * 60)
    print("""
    # Usa il preset con AudioVisualFX
    
    config = get_preset_config('psychedelic_refraction')
    
    fx = AudioVisualFX(
        audio_file="track.mp3",
        image_file="artwork.jpg",
        output_file="psychedelic_output.mp4",
        fps=30,
        colors=config['colors'],
        thresholds=(
            config['thresholds']['bass'],
            config['thresholds']['mid'],
            config['thresholds']['high']
        ),
        effect_style="psychedelic"  # IMPORTANTE: usa stile psychedelic!
    )
    
    fx.create_video()
    """)
    
    print("\n✅ Test completato!")
    print("=" * 60)


def compare_with_other_presets():
    """Confronta il preset psichedelico con gli altri"""
    
    print("\n" + "=" * 60)
    print("CONFRONTO PRESET")
    print("=" * 60)
    
    presets = [
        'dark_techno',
        'cyberpunk', 
        'industrial',
        'acid_house',
        'extreme_vibrant',
        'psychedelic_refraction'
    ]
    
    print("\n{:<25} {:>10} {:>10} {:>10}".format("Preset", "Bass", "Mid", "High"))
    print("-" * 60)
    
    for preset_name in presets:
        config = get_preset_config(preset_name)
        thresholds = config['thresholds']
        print("{:<25} {:>10.2f} {:>10.2f} {:>10.2f}".format(
            config['name'],
            thresholds['bass'],
            thresholds['mid'],
            thresholds['high']
        ))
    
    print("\n" + "=" * 60)
    print("CARATTERISTICHE UNICHE DEI PRESET:")
    print("=" * 60)
    print("""
    🌑 Dark Techno: Sobrio, colori scuri, strobe pesante
    🌃 Cyberpunk: Neon vibranti, aberrazioni cromatiche
    🏭 Industrial: Metallico, noise alto, distorsioni pesanti
    💊 Acid House: Colori acidi, strobe estremo, veloce
    ⚡ Extreme Vibrant: Massima energia, tutti gli effetti al massimo
    🔮 Psychedelic Refraction: Rifrazione intelligente, effetti prismatici fluidi
    """)


if __name__ == "__main__":
    test_psychedelic_preset()
    compare_with_other_presets()
