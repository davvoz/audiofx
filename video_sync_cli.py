"""
Video Audio Sync CLI
Sincronizza un video con un audio. La durata dell'audio comanda:
- Video troppo lungo: viene tagliato
- Video troppo corto: viene messo in loop
"""

import argparse
import os

from video_generator import VideoAudioFX


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sincronizza video con audio - l'audio comanda la durata"
    )
    parser.add_argument("--audio", "-a", required=True, help="File audio input")
    parser.add_argument("--video", "-v", required=True, help="File video input")
    parser.add_argument(
        "--output", "-o", default="synced_video.mp4", help="File video output"
    )
    parser.add_argument(
        "--short-video-mode",
        choices=["loop", "stretch"],
        default="loop",
        help="Come gestire video più corti dell'audio: 'loop' ripete il video, 'stretch' lo rallenta (default: loop)",
    )
    # Logo options
    parser.add_argument(
        "--logo", help="File immagine logo da sovrapporre (PNG consigliato)"
    )
    parser.add_argument(
        "--logo-position",
        choices=["top-left", "top-right", "bottom-left", "bottom-right"],
        default="top-right",
        help="Posizione del logo",
    )
    parser.add_argument(
        "--logo-scale",
        type=float,
        default=0.15,
        help="Larghezza del logo rispetto al frame (0-1)",
    )
    parser.add_argument(
        "--logo-opacity", type=float, default=1.0, help="Opacità del logo (0-1)"
    )
    parser.add_argument(
        "--logo-margin", type=int, default=12, help="Margine del logo dai bordi in pixel"
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Errore: File audio '{args.audio}' non trovato")
        return
    if not os.path.exists(args.video):
        print(f"Errore: File video '{args.video}' non trovato")
        return

    def _progress(event: str, payload: dict) -> None:
        if event == "status":
            print(payload.get("message", ""))
        elif event == "start":
            print(f"Frame totali da processare: {payload.get('total_frames')}")
        elif event == "frame":
            i = payload.get("index")
            t = payload.get("total")
            if i and t and i % max(1, t // 10) == 0:
                print(f"Progresso: {i}/{t}")
        elif event == "done":
            print(f"Video sincronizzato creato: {payload.get('output')}")

    fx = VideoAudioFX(
        audio_file=args.audio,
        video_file=args.video,
        output_file=args.output,
        progress_cb=_progress,
        short_video_mode=args.short_video_mode,
        logo_file=args.logo,
        logo_position=args.logo_position,
        logo_scale=args.logo_scale,
        logo_opacity=args.logo_opacity,
        logo_margin=args.logo_margin,
    )

    try:
        fx.create_video()
        print("\n🎉 Video sincronizzato con successo! 🎉")
        print(f"\nOutput: {args.output}")
    except Exception as e:
        print(f"Errore durante la sincronizzazione: {e}")


if __name__ == "__main__":
    main()
