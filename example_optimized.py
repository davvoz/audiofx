#!/usr/bin/env python3
"""
Esempio semplice di utilizzo con le ottimizzazioni.
Usa sempre generate_streaming() per massima efficienza.
"""

from src.audio_visual_generator import AudioVisualGenerator
from src.models.data_models import EffectStyle, EffectConfig


def create_optimized_video(
    audio_file: str,
    image_file: str,
    output_file: str = "output.mp4",
    style: EffectStyle = EffectStyle.CYBERPUNK,
    duration: float = None,
    resolution: tuple = (1280, 720),
    use_multicore: bool = True
):
    """
    Crea video ottimizzato usando streaming mode.
    
    Args:
        audio_file: Path al file audio (mp3, wav, etc.)
        image_file: Path all'immagine di background (jpg, png, etc.)
        output_file: Nome file output (default: output.mp4)
        style: Stile effetti (CYBERPUNK, PSYCHEDELIC, HORROR, etc.)
        duration: Durata in secondi (None = usa tutta la canzone)
        resolution: Risoluzione video (default: 720p)
        use_multicore: Usa elaborazione parallela (default: True)
    """
    
    print(f"🎬 Creating video with optimized streaming mode...")
    print(f"   Audio: {audio_file}")
    print(f"   Image: {image_file}")
    print(f"   Style: {style.value}")
    print(f"   Resolution: {resolution[0]}x{resolution[1]}")
    print(f"   Multicore: {'Yes' if use_multicore else 'No'}")
    print()
    
    # Progress callback
    def on_progress(event_type, data):
        if event_type == "status":
            print(f"ℹ️  {data.get('message', '')}")
        elif event_type == "progress":
            msg = data.get('message', '')
            if msg:
                print(f"   {msg}")
        elif event_type == "done":
            print(f"✅ Video saved: {data.get('output', output_file)}")
        elif event_type == "error":
            print(f"❌ Error: {data.get('message', '')}")
    
    # Custom colors (optional)
    config = EffectConfig(
        colors=[
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.5, 0.0),  # Orange
            (0.0, 1.0, 0.5),  # Teal
        ]
    )
    
    # Create generator
    generator = AudioVisualGenerator(
        audio_file=audio_file,
        image_file=image_file,
        output_file=output_file,
        fps=30,
        duration=duration,
        effect_config=config,
        effect_style=style,
        target_resolution=resolution,
        progress_cb=on_progress,
        use_multiprocessing=use_multicore,
        num_workers=None  # Auto-detect (CPU count - 1)
    )
    
    # IMPORTANT: Use streaming mode for memory efficiency!
    # This writes frames directly to disk instead of accumulating in RAM
    generator.generate_streaming()
    
    print("\n🎉 Done! Video created successfully.")


# ==============================================================================
# ESEMPI DI USO
# ==============================================================================

def example_quick_video():
    """Esempio: Video veloce 30 secondi."""
    create_optimized_video(
        audio_file="my_song.mp3",
        image_file="my_image.jpg",
        output_file="quick_video.mp4",
        style=EffectStyle.STANDARD,
        duration=30.0,  # Solo 30 secondi
        resolution=(720, 720)
    )


def example_long_video():
    """Esempio: Video lungo (5+ minuti) senza problemi RAM."""
    create_optimized_video(
        audio_file="long_song.mp3",
        image_file="background.jpg",
        output_file="long_video.mp4",
        style=EffectStyle.PSYCHEDELIC,
        duration=None,  # Tutta la canzone
        resolution=(1280, 720),  # 720p
        use_multicore=True
    )


def example_high_quality():
    """Esempio: Alta qualità 1080p."""
    create_optimized_video(
        audio_file="track.mp3",
        image_file="artwork.png",
        output_file="hq_video.mp4",
        style=EffectStyle.CYBERPUNK,
        duration=None,
        resolution=(1920, 1080),  # Full HD
        use_multicore=True
    )


def example_all_styles():
    """Crea video con tutti gli stili disponibili."""
    styles = [
        EffectStyle.STANDARD,
        EffectStyle.MINIMAL,
        EffectStyle.PSYCHEDELIC,
        EffectStyle.CYBERPUNK,
        EffectStyle.HORROR,
        EffectStyle.RETRO_WAVE,
        EffectStyle.INDUSTRIAL,
        EffectStyle.ACID_HOUSE,
        EffectStyle.EXTREME,
    ]
    
    audio_file = "song.mp3"
    image_file = "image.jpg"
    
    for style in styles:
        output = f"output_{style.value}.mp4"
        print(f"\n{'='*60}")
        print(f"Creating {style.value} version...")
        print(f"{'='*60}")
        
        create_optimized_video(
            audio_file=audio_file,
            image_file=image_file,
            output_file=output,
            style=style,
            duration=60.0,  # 1 minuto per test veloce
            resolution=(720, 720)
        )


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        # Command line usage
        audio = sys.argv[1]
        image = sys.argv[2]
        output = sys.argv[3] if len(sys.argv) >= 4 else "output.mp4"
        
        create_optimized_video(
            audio_file=audio,
            image_file=image,
            output_file=output,
            style=EffectStyle.CYBERPUNK,
            resolution=(1280, 720)
        )
    else:
        # Show usage
        print("Usage:")
        print("  python example_optimized.py <audio_file> <image_file> [output_file]")
        print()
        print("Example:")
        print("  python example_optimized.py song.mp3 background.jpg my_video.mp4")
        print()
        print("Or edit this file and uncomment one of the example functions below:")
        print("  - example_quick_video()")
        print("  - example_long_video()")
        print("  - example_high_quality()")
        print("  - example_all_styles()")
        
        # Uncomment to run examples:
        # example_quick_video()
        # example_long_video()
        # example_high_quality()
        # example_all_styles()
