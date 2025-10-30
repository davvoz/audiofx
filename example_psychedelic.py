"""
Esempio pratico di utilizzo del preset Psychedelic Refraction
Genera un video con effetti di rifrazione psichedelica
"""

import os
from config import get_preset_config
from video_generator import AudioVisualFX


def create_psychedelic_video(
    audio_file: str,
    image_file: str,
    output_file: str = "psychedelic_output.mp4"
):
    """
    Crea un video con effetti psichedelici di rifrazione
    
    Args:
        audio_file: Path al file audio
        image_file: Path all'immagine di base
        output_file: Path al file video di output
    """
    
    # Verifica file input
    if not os.path.exists(audio_file):
        print(f"❌ File audio non trovato: {audio_file}")
        return
    
    if not os.path.exists(image_file):
        print(f"❌ File immagine non trovato: {image_file}")
        return
    
    print("=" * 70)
    print("🔮 GENERATORE VIDEO PSYCHEDELIC REFRACTION")
    print("=" * 70)
    
    # Carica configurazione preset
    config = get_preset_config('psychedelic_refraction')
    
    print(f"\n📋 Configurazione: {config['name']}")
    print(f"🎵 Audio: {audio_file}")
    print(f"🖼️  Immagine: {image_file}")
    print(f"📹 Output: {output_file}")
    print(f"🎨 Colori: {len(config['colors'])} nella palette psichedelica")
    
    # Callback per progress
    def progress_callback(event: str, payload: dict):
        if event == "status":
            print(f"⏳ {payload.get('message', '')}")
        elif event == "start":
            print(f"🎬 Frame totali da generare: {payload.get('total_frames')}")
        elif event == "frame":
            i = payload.get('index')
            t = payload.get('total')
            if i and t and i % max(1, t // 20) == 0:
                progress = (i / t) * 100
                print(f"   📊 Progresso: {i}/{t} ({progress:.1f}%)")
        elif event == "done":
            print(f"✅ Video completato: {payload.get('output')}")
    
    print("\n🚀 Avvio generazione video...\n")
    
    try:
        # Crea istanza AudioVisualFX con preset psichedelico
        fx = AudioVisualFX(
            audio_file=audio_file,
            image_file=image_file,
            output_file=output_file,
            fps=30,
            duration=None,  # Usa durata completa dell'audio
            progress_cb=progress_callback,
            colors=config['colors'],
            thresholds=(
                config['thresholds']['bass'],
                config['thresholds']['mid'],
                config['thresholds']['high']
            ),
            target_resolution=(720, 720),
            effect_style="psychedelic",  # IMPORTANTE: attiva effetti psichedelici!
        )
        
        # Genera il video
        fx.create_video()
        
        print("\n" + "=" * 70)
        print("🎉 GENERAZIONE COMPLETATA CON SUCCESSO! 🎉")
        print("=" * 70)
        print(f"\n📹 Il tuo video psichedelico è pronto: {output_file}")
        print("\n✨ Effetti applicati:")
        print("   🌊 Rifrazione ondulata multi-direzionale")
        print("   🌈 Dispersione cromatica prismatica")
        print("   💎 Shift intelligente pixel (cristalli)")
        print("   🔶 Kaleidoscope su beat intensi")
        print("   🌀 Flusso liquido dinamico")
        print("   🎭 Split prismatico RGB avanzato")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante la generazione: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Entry point esempio"""
    
    print("\n🔮 Esempio: Psychedelic Refraction Video Generator\n")
    
    # Esempio con file di default (modificare con i tuoi file)
    audio_file = "your_audio.mp3"
    image_file = "your_image.jpg"
    output_file = "psychedelic_refraction.mp4"
    
    print("📝 Istruzioni:")
    print("   1. Modifica le variabili audio_file e image_file")
    print("   2. Esegui questo script")
    print("   3. Attendi la generazione")
    print("   4. Goditi il risultato psichedelico! ✨")
    print()
    
    # Controllo se i file esistono
    if not os.path.exists(audio_file):
        print(f"⚠️  File audio '{audio_file}' non trovato.")
        print("   Modifica la variabile 'audio_file' con il path corretto.")
        print("\n💡 Esempio:")
        print('   audio_file = "C:/Music/techno_track.mp3"')
        return
    
    if not os.path.exists(image_file):
        print(f"⚠️  File immagine '{image_file}' non trovato.")
        print("   Modifica la variabile 'image_file' con il path corretto.")
        print("\n💡 Esempio:")
        print('   image_file = "C:/Images/artwork.jpg"')
        return
    
    # Genera il video
    create_psychedelic_video(audio_file, image_file, output_file)


if __name__ == "__main__":
    main()
