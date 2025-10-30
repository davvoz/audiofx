"""
Test del preset Intelligent Adaptive
Sistema di riconoscimento automatico delle sezioni musicali
"""

from config import INTELLIGENT_ADAPTIVE_CONFIG, get_preset_config


def test_intelligent_preset():
    """Test del preset intelligente"""
    
    print("=" * 70)
    print("🧠 TEST PRESET: INTELLIGENT ADAPTIVE")
    print("=" * 70)
    
    # Carica configurazione
    config = get_preset_config('intelligent_adaptive')
    
    print(f"\n📋 Nome Preset: {config['name']}")
    print(f"\n🎨 Palette Colori ({len(config['colors'])} colori versatili):")
    for i, color in enumerate(config['colors'], 1):
        r, g, b = [int(c * 255) for c in color]
        print(f"  {i}. RGB({r:3}, {g:3}, {b:3}) = {color}")
    
    print(f"\n🎚️ Soglie Audio:")
    print(f"  Bass:  {config['thresholds']['bass']}")
    print(f"  Mid:   {config['thresholds']['mid']}")
    print(f"  High:  {config['thresholds']['high']}")
    
    print(f"\n✨ Parametri Effetti:")
    for effect, value in config['effects'].items():
        print(f"  {effect:.<35} {value}")
    
    print("\n" + "=" * 70)
    print("🧠 SISTEMA INTELLIGENTE - Come Funziona")
    print("=" * 70)
    print("""
📊 ANALISI AUTOMATICA SEZIONI:
Il sistema analizza l'intera traccia audio per riconoscere automaticamente
le diverse parti musicali e applica effetti specifici per ogni sezione.

🎵 TIPI DI SEZIONE RICONOSCIUTI:

1. 🎵 INTRO (Introduzione)
   Caratteristiche:
   - Bassa energia all'inizio del brano (primi 15 secondi)
   - Energia < 35%
   
   Effetti Applicati:
   - Color pulse soft (60% intensità)
   - Zoom molto leggero (40% intensità)
   - Liquid flow dolce
   - Strobe minimo (50% intensità)
   
   Mood: Minimalista, graduale, atmosferico

2. 📈 BUILDUP (Crescita Tensione)
   Caratteristiche:
   - Energia crescente rapidamente (slope > 15%)
   - Energia < 70%
   
   Effetti Applicati:
   - Color pulse intenso (120% bass, 110% mid)
   - Zoom crescente (90% intensità)
   - Distorsione crescente (80% intensità)
   - Chromatic aberration
   - Screen shake crescente
   - Strobe crescente (85% intensità)
   
   Mood: Tensione crescente, anticipazione

3. 💥 DROP (Climax Energetico)
   Caratteristiche:
   - Impatto improvviso (slope > 25% e energia > 60%)
   - Alta energia (> 75%)
   
   Effetti Applicati:
   - Color pulse ESTREMO (140% bass, 130% mid, 120% treble)
   - Zoom esplosivo (130% intensità)
   - Bubble distortion sui bassi forti
   - Screen shake intenso
   - Distorsione MASSIMA
   - Chromatic aberration forte (120% intensità)
   - RGB split
   - Strobe massimo
   - Glitch sui beat
   - Scariche elettriche
   
   Mood: ENERGIA MASSIMA, esplosivo, caotico

4. 🎹 BREAKDOWN (Calo Energia)
   Caratteristiche:
   - Calo energia dopo drop (slope < -10%)
   - Energia < 50%
   
   Effetti Applicati:
   - Color pulse melodico (70% bass, 90% mid, 110% treble)
   - Liquid flow prominente
   - Prism split (90% intensità)
   - Distorsione leggera (50% intensità)
   - Chromatic aberration soft
   - Scan lines
   
   Mood: Melodico, fluido, rilassato

5. 🌅 OUTRO (Conclusione)
   Caratteristiche:
   - Decadimento finale (ultimi 20 secondi)
   - Energia < 30%
   
   Effetti Applicati:
   - Color pulse decrescente (50% bass, 40% mid, 60% treble)
   - Liquid flow lento
   - Zoom minimo (30% intensità)
   - Distorsione minima (30% intensità)
   - VHS distortion occasionale (effetto "fine cassetta")
   
   Mood: Atmosferico, decadente, conclusivo

6. 🔄 STEADY (Sezione Stabile)
   Caratteristiche:
   - Energia costante
   - Nessun cambio significativo
   
   Effetti Applicati:
   - Color pulse normale
   - Zoom moderato (70% intensità)
   - Distorsione media (60% intensità)
   - Chromatic aberration
   - Strobe medio (70% intensità)
   - Glitch occasionale
   
   Mood: Bilanciato, stabile, consistente

🔬 ALGORITMO DI RICONOSCIMENTO:

1. Estrazione Features Audio:
   - Energia totale per frame (bass + mid + treble) / 3
   - Smoothing con filtro Gaussiano (finestra ~2 secondi)
   - Derivata dell'energia (rate of change)

2. Classificazione Sezioni:
   - Analizza tempo, energia e slope per ogni frame
   - Applica regole euristiche per classificare
   - Merge sezioni troppo corte (< 2 secondi)

3. Applicazione Effetti:
   - Per ogni frame, determina sezione corrente
   - Applica effetti specifici basati sul tipo sezione
   - Transizioni fluide tra sezioni

📊 VANTAGGI:

✅ Adattamento automatico al brano
✅ Nessuna configurazione manuale necessaria
✅ Effetti sempre appropriati al momento musicale
✅ Risultati professionali su qualsiasi genere
✅ Massimizza l'impatto visivo
✅ Sincronizzazione perfetta con la struttura del brano

💡 IDEALE PER:

- Tracce con struttura chiara (intro-buildup-drop-breakdown)
- Musica elettronica (house, techno, trance, dubstep)
- Mix e DJ set
- Produzioni con dinamica variabile
- Quando vuoi risultati ottimali senza tweaking manuale

⚠️ MENO ADATTO PER:

- Tracce molto sperimentali senza struttura
- Ambient/drone (poca variazione energia)
- Spoken word/podcast (non musicale)

🎯 CASO D'USO PERFETTO:

"Ho una traccia techno di 5 minuti con intro soft, buildup lungo,
drop potente, breakdown melodico e outro. Voglio che gli effetti
si adattino automaticamente a ogni parte per massimizzare l'impatto
visivo senza dover fare setup manuale."

→ Intelligent Adaptive è la scelta perfetta! 🎉
""")
    
    print("\n" + "=" * 70)
    print("📝 ESEMPIO D'USO:")
    print("=" * 70)
    print("""
from config import get_preset_config
from video_generator import AudioVisualFX

config = get_preset_config('intelligent_adaptive')

fx = AudioVisualFX(
    audio_file="my_track.mp3",
    image_file="artwork.jpg",
    output_file="intelligent_output.mp4",
    fps=30,
    colors=config['colors'],
    thresholds=(
        config['thresholds']['bass'],
        config['thresholds']['mid'],
        config['thresholds']['high']
    ),
    effect_style="intelligent"  # 🔑 Attiva analisi intelligente!
)

fx.create_video()

# Output:
# 🧠 Analisi intelligente sezioni traccia...
# 📊 Sezioni rilevate: 6
#   1. 🎵 INTRO       |   0.0s -  12.5s | Durata:  12.5s | Energia: 0.28
#   2. 📈 BUILDUP     |  12.5s -  28.3s | Durata:  15.8s | Energia: 0.54
#   3. 💥 DROP        |  28.3s -  58.7s | Durata:  30.4s | Energia: 0.89
#   4. 🎹 BREAKDOWN   |  58.7s -  78.2s | Durata:  19.5s | Energia: 0.42
#   5. 💥 DROP        |  78.2s - 108.9s | Durata:  30.7s | Energia: 0.91
#   6. 🌅 OUTRO       | 108.9s - 120.0s | Durata:  11.1s | Energia: 0.25
""")
    
    print("\n✅ Test completato!")
    print("=" * 70)


def compare_intelligent_with_others():
    """Confronta Intelligent Adaptive con altri preset"""
    
    print("\n" + "=" * 70)
    print("📊 CONFRONTO: INTELLIGENT ADAPTIVE vs ALTRI PRESET")
    print("=" * 70)
    
    presets = [
        'dark_techno',
        'extreme_vibrant',
        'psychedelic_refraction',
        'intelligent_adaptive'
    ]
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:<20}".format(
        "Preset", "Bass", "Mid", "High", "Caratteristica"
    ))
    print("-" * 80)
    
    for preset_name in presets:
        config = get_preset_config(preset_name)
        th = config['thresholds']
        
        caratteristica = {
            'dark_techno': 'Fisso, scuro',
            'extreme_vibrant': 'Fisso, estremo',
            'psychedelic_refraction': 'Fisso, psichedelico',
            'intelligent_adaptive': '🧠 ADATTIVO'
        }.get(preset_name, '')
        
        print("{:<25} {:>10.2f} {:>10.2f} {:>10.2f} {:<20}".format(
            config['name'],
            th['bass'],
            th['mid'],
            th['high'],
            caratteristica
        ))
    
    print("\n" + "=" * 70)
    print("🎯 DIFFERENZE CHIAVE:")
    print("=" * 70)
    print("""
PRESET FISSI (Dark Techno, Extreme Vibrant, Psychedelic Refraction):
✓ Effetti sempre uguali per tutta la traccia
✓ Stile visivo consistente
✓ Prevedibile
✓ Buono per aesthetic specifico
✗ Non si adatta alla struttura del brano
✗ Può essere ripetitivo

INTELLIGENT ADAPTIVE:
✓ Effetti cambiano automaticamente con la musica
✓ Si adatta alla struttura del brano (intro/buildup/drop/etc)
✓ Massimizza impatto visivo per ogni sezione
✓ Risultati professionali senza configurazione
✓ Perfetto per tracce con dinamica variabile
✗ Meno consistente visivamente
✗ Richiede più risorse computazionali

QUANDO USARE INTELLIGENT ADAPTIVE:
→ Vuoi risultati ottimali automaticamente
→ Il brano ha struttura chiara
→ Vuoi effetti appropriati per ogni momento
→ Non vuoi fare tweaking manuale

QUANDO USARE PRESET FISSI:
→ Vuoi uno stile visivo specifico
→ Il brano è ambient/drone (poca dinamica)
→ Vuoi consistenza visiva totale
→ Hai già un'idea precisa del risultato
""")


if __name__ == "__main__":
    test_intelligent_preset()
    compare_intelligent_with_others()
