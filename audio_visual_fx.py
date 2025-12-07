"""
Audio Visual FX Generator (CLI)
Refactored to delegate to `video_generator.AudioVisualFX` for scalability.
"""

import argparse
import os

from video_generator import AudioVisualFX


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera video dark techno da audio e immagine")
    parser.add_argument("--audio", "-a", required=True, help="File audio input")
    parser.add_argument("--image", "-i", required=True, help="File immagine input")
    parser.add_argument("--output", "-o", default="dark_techno_fx.mp4", help="File video output")
    parser.add_argument("--fps", type=int, default=30, help="Frame per secondo")
    parser.add_argument("--duration", "-d", type=float, help="Durata del video in secondi")
    parser.add_argument("--native-resolution", action="store_true", 
                       help="Usa le dimensioni native dell'immagine (default: 720x720)")
    # Logo options
    parser.add_argument("--logo", help="File immagine logo da sovrapporre (PNG consigliato)")
    parser.add_argument(
        "--logo-position",
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        default="top-right",
        help="Posizione del logo",
    )
    parser.add_argument("--logo-scale", type=float, default=0.15, help="Larghezza del logo rispetto al frame (0-1)")
    parser.add_argument("--logo-opacity", type=float, default=1.0, help="Opacità del logo (0-1)")
    parser.add_argument("--logo-margin", type=int, default=12, help="Margine del logo dai bordi in pixel")
    # Performance options
    parser.add_argument("--no-multiprocessing", action="store_true", 
                       help="Disabilita multiprocessing (più lento ma usa meno RAM)")
    parser.add_argument("--workers", type=int, help="Numero di worker paralleli (default: CPU count - 1)")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Errore: File audio '{args.audio}' non trovato")
        return
    if not os.path.exists(args.image):
        print(f"Errore: File immagine '{args.image}' non trovato")
        return

    def _progress(event: str, payload: dict) -> None:
        if event == "status":
            print(payload.get("message", ""))
        elif event == "start":
            print(f"Frame totali stimati: {payload.get('total_frames')}")
        elif event == "frame":
            i = payload.get("index"); t = payload.get("total")
            if i and t and i % max(1, t // 10) == 0:
                print(f"Progresso: {i}/{t}")
        elif event == "done":
            print(f"Video creato: {payload.get('output')}")

    # Set target resolution based on flag
    target_res = None if args.native_resolution else (720, 720)

    fx = AudioVisualFX(
        audio_file=args.audio,
        image_file=args.image,
        output_file=args.output,
        fps=args.fps,
        duration=args.duration,
        target_resolution=target_res,
        progress_cb=_progress,
        logo_file=args.logo,
        logo_position=args.logo_position,
        logo_scale=args.logo_scale,
        logo_opacity=args.logo_opacity,
        logo_margin=args.logo_margin,
        use_multiprocessing=not args.no_multiprocessing,
        num_workers=args.workers
    )

    try:
        fx.create_video()
        print("\n🎉 Video generato con successo! 🎉")
    except Exception as e:  # keep simple error surfacing on CLI
        print(f"Errore durante la generazione del video: {e}")


if __name__ == "__main__":
    main()