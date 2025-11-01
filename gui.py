"""
Simple GUI for Audio Visual FX using Tkinter.
Now using modular architecture from src/ package.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

# Import from new modular architecture
from src import (
    AudioVisualGenerator,
    VideoAudioSync,
    EffectStyle,
    EffectConfig,
)
from config import get_preset_config


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Audio Visual FX Generator")
        self.geometry("900x700")
        self.resizable(True, True)  # Finestra ridimensionabile

        # Tab 1: Audio + Image
        self.audio_path = tk.StringVar()
        self.image_path = tk.StringVar()
        self.output_path = tk.StringVar(value="output.mp4")
        self.fps_var = tk.IntVar(value=30)
        self.preset_var = tk.StringVar(value="Dark Techno")
        
        # Tab 2: Audio + Video Sync
        self.sync_audio_path = tk.StringVar()
        self.sync_video_path = tk.StringVar()
        self.sync_output_path = tk.StringVar(value="synced_output.mp4")
        self.sync_preset_var = tk.StringVar(value="Dark Techno")
        self.sync_short_mode = tk.StringVar(value="Loop")
        
        # Logo controls (shared)
        self.logo_path = tk.StringVar()
        self.logo_position = tk.StringVar(value="Top-Right")
        self.logo_scale = tk.DoubleVar(value=0.15)
        self.logo_opacity = tk.DoubleVar(value=1.0)
        self.logo_margin = tk.IntVar(value=12)
        
        # Tab 3: Custom Preset Controls
        self.custom_mode = tk.StringVar(value="image")
        self.custom_audio_path = tk.StringVar()
        self.custom_image_path = tk.StringVar()
        self.custom_video_path = tk.StringVar()
        self.custom_output_path = tk.StringVar(value="custom_output.mp4")
        self.custom_fps_var = tk.IntVar(value=30)
        self.custom_video_mode = tk.StringVar(value="loop")
        
        # Custom effect checkboxes
        self.effect_color_pulse = tk.BooleanVar(value=True)
        self.effect_zoom_pulse = tk.BooleanVar(value=True)
        self.effect_strobe = tk.BooleanVar(value=False)
        self.effect_glitch = tk.BooleanVar(value=False)
        self.effect_chromatic = tk.BooleanVar(value=False)
        self.effect_bubble = tk.BooleanVar(value=False)
        self.effect_screen_shake = tk.BooleanVar(value=False)
        self.effect_rgb_split = tk.BooleanVar(value=False)
        self.effect_electric_arcs = tk.BooleanVar(value=False)
        
        # Custom effect intensities
        self.intensity_color_pulse = tk.DoubleVar(value=1.0)
        self.intensity_zoom_pulse = tk.DoubleVar(value=1.0)
        self.intensity_strobe = tk.DoubleVar(value=1.0)
        self.intensity_glitch = tk.DoubleVar(value=1.0)
        self.intensity_chromatic = tk.DoubleVar(value=1.0)
        self.intensity_bubble = tk.DoubleVar(value=1.0)
        self.intensity_screen_shake = tk.DoubleVar(value=1.0)
        self.intensity_rgb_split = tk.DoubleVar(value=1.0)
        self.intensity_electric_arcs = tk.DoubleVar(value=1.0)
        
        # Custom thresholds
        self.threshold_bass = tk.DoubleVar(value=0.3)
        self.threshold_mid = tk.DoubleVar(value=0.2)
        self.threshold_high = tk.DoubleVar(value=0.15)

        self._build_ui()
        self._worker = None  # type: ignore[assignment]
        self._cancel = False

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Audio + Image
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Audio + Immagine")
        self._build_audio_image_tab()
        
        # Tab 2: Audio + Video Sync
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Audio + Video Sync")
        self._build_video_sync_tab()
        
        # Tab 3: Custom Preset Builder
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Custom Preset")
        self._build_custom_preset_tab()
    
    def _build_audio_image_tab(self) -> None:
        """Build UI for audio + image tab."""
        pad = {"padx": 10, "pady": 6}
        frame = self.tab1

        # Preset - Now using EffectStyle from modular architecture
        tk.Label(frame, text="Preset:").grid(row=0, column=0, sticky="e", **pad)
        self.preset_combo = ttk.Combobox(
            frame,
            textvariable=self.preset_var,
            values=["Standard", "Extreme", "Psychedelic", "Minimal", "Cyberpunk", 
                    "Industrial", "Acid House", "Retro Wave", "Horror"],
            state="readonly",
            width=30,
        )
        self.preset_combo.current(0)
        self.preset_combo.grid(row=0, column=1, sticky="w", **pad)

        # Audio
        tk.Label(frame, text="Audio:").grid(row=1, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.audio_path, width=45).grid(row=1, column=1, **pad)
        tk.Button(frame, text="Sfoglia", command=self.browse_audio).grid(row=1, column=2, **pad)

        # Image
        tk.Label(frame, text="Immagine:").grid(row=2, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.image_path, width=45).grid(row=2, column=1, **pad)
        tk.Button(frame, text="Sfoglia", command=self.browse_image).grid(row=2, column=2, **pad)

        # Output
        tk.Label(frame, text="Output:").grid(row=3, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.output_path, width=45).grid(row=3, column=1, **pad)
        tk.Button(frame, text="Scegli", command=self.choose_output).grid(row=3, column=2, **pad)
        tk.Button(frame, text="Cartella", command=self.choose_output_dir).grid(row=3, column=3, **pad)

        # FPS
        tk.Label(frame, text="FPS:").grid(row=4, column=0, sticky="e", **pad)
        tk.Spinbox(frame, from_=1, to=120, textvariable=self.fps_var, width=10).grid(
            row=4, column=1, sticky="w", **pad
        )

        # Logo group
        sep = ttk.Separator(frame, orient="horizontal")
        sep.grid(row=5, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 2))

        tk.Label(frame, text="Logo (opzionale):").grid(row=6, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.logo_path, width=45).grid(row=6, column=1, **pad)
        tk.Button(frame, text="Sfoglia", command=self.browse_logo).grid(row=6, column=2, **pad)

        tk.Label(frame, text="Posizione:").grid(row=7, column=0, sticky="e", **pad)
        ttk.Combobox(
            frame,
            textvariable=self.logo_position,
            values=["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"],
            state="readonly",
            width=20,
        ).grid(row=7, column=1, sticky="w", **pad)

        tk.Label(frame, text="Scala (larghezza %):").grid(row=8, column=0, sticky="e", **pad)
        tk.Scale(frame, variable=self.logo_scale, from_=0.05, to=0.5, resolution=0.01, orient="horizontal", length=220).grid(
            row=8, column=1, sticky="w", **pad
        )

        tk.Label(frame, text="Opacità:").grid(row=9, column=0, sticky="e", **pad)
        tk.Scale(frame, variable=self.logo_opacity, from_=0.0, to=1.0, resolution=0.01, orient="horizontal", length=220).grid(
            row=9, column=1, sticky="w", **pad
        )

        tk.Label(frame, text="Margine (px):").grid(row=10, column=0, sticky="e", **pad)
        tk.Spinbox(frame, from_=0, to=200, textvariable=self.logo_margin, width=10).grid(
            row=10, column=1, sticky="w", **pad
        )

        # Progress
        self.progress = ttk.Progressbar(frame, mode="determinate", length=620)
        self.progress.grid(row=11, column=0, columnspan=4, **pad)
        self.status_lbl = tk.Label(frame, text="Pronto")
        self.status_lbl.grid(row=12, column=0, columnspan=4, sticky="w", **pad)

        # Actions
        self.run_btn = tk.Button(frame, text="Genera Video", command=self.on_run)
        self.run_btn.grid(row=13, column=2, sticky="e", **pad)
        self.cancel_btn = tk.Button(frame, text="Annulla", command=self.on_cancel, state=tk.DISABLED)
        self.cancel_btn.grid(row=13, column=3, sticky="w", **pad)
    
    def _build_video_sync_tab(self) -> None:
        """Build UI for video sync tab."""
        pad = {"padx": 10, "pady": 6}
        frame = self.tab2
        
        # Description
        desc = tk.Label(
            frame,
            text="Sincronizza un video con un audio e applica effetti audio-reattivi.\n"
                 "La durata dell'audio comanda: video più lungo → tagliato | video più corto → loop",
            justify=tk.LEFT,
            font=("Arial", 9, "italic")
        )
        desc.grid(row=0, column=0, columnspan=4, sticky="w", **pad)
        
        # Preset - Now using EffectStyle from modular architecture
        tk.Label(frame, text="Preset Effetti:").grid(row=1, column=0, sticky="e", **pad)
        self.sync_preset_combo = ttk.Combobox(
            frame,
            textvariable=self.sync_preset_var,
            values=["Standard", "Extreme", "Psychedelic", "Minimal", "Cyberpunk", 
                    "Industrial", "Acid House", "Retro Wave", "Horror"],
            state="readonly",
            width=30,
        )
        self.sync_preset_combo.current(0)
        self.sync_preset_combo.grid(row=1, column=1, sticky="w", **pad)
        
        # Short video mode
        tk.Label(frame, text="Video corto:").grid(row=1, column=2, sticky="e", padx=(20, 5), pady=6)
        self.sync_mode_combo = ttk.Combobox(
            frame,
            textvariable=self.sync_short_mode,
            values=["Loop", "Stretch"],
            state="readonly",
            width=10,
        )
        self.sync_mode_combo.current(0)
        self.sync_mode_combo.grid(row=1, column=3, sticky="w", pady=6)
        
        # Tooltip/help text
        mode_help = tk.Label(
            frame,
            text="Loop: ripete video | Stretch: rallenta video",
            font=("Arial", 8),
            fg="gray"
        )
        mode_help.grid(row=2, column=2, columnspan=2, sticky="w", padx=(20, 5), pady=(0, 6))
        
        # Audio
        tk.Label(frame, text="Audio:").grid(row=3, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.sync_audio_path, width=45).grid(row=3, column=1, **pad)
        tk.Button(frame, text="Sfoglia", command=self.browse_sync_audio).grid(row=3, column=2, **pad)
        
        # Video
        tk.Label(frame, text="Video:").grid(row=4, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.sync_video_path, width=45).grid(row=4, column=1, **pad)
        tk.Button(frame, text="Sfoglia", command=self.browse_sync_video).grid(row=4, column=2, **pad)
        
        # Output
        tk.Label(frame, text="Output:").grid(row=5, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.sync_output_path, width=45).grid(row=5, column=1, **pad)
        tk.Button(frame, text="Scegli", command=self.choose_sync_output).grid(row=5, column=2, **pad)
        tk.Button(frame, text="Cartella", command=self.choose_sync_output_dir).grid(row=5, column=3, **pad)
        
        # Logo group
        sep = ttk.Separator(frame, orient="horizontal")
        sep.grid(row=6, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 2))
        
        tk.Label(frame, text="Logo (opzionale):").grid(row=7, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.logo_path, width=45).grid(row=7, column=1, **pad)
        tk.Button(frame, text="Sfoglia", command=self.browse_logo).grid(row=7, column=2, **pad)
        
        tk.Label(frame, text="Posizione:").grid(row=8, column=0, sticky="e", **pad)
        ttk.Combobox(
            frame,
            textvariable=self.logo_position,
            values=["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"],
            state="readonly",
            width=20,
        ).grid(row=8, column=1, sticky="w", **pad)
        
        tk.Label(frame, text="Scala (larghezza %):").grid(row=9, column=0, sticky="e", **pad)
        tk.Scale(frame, variable=self.logo_scale, from_=0.05, to=0.5, resolution=0.01, orient="horizontal", length=220).grid(
            row=9, column=1, sticky="w", **pad
        )
        
        tk.Label(frame, text="Opacità:").grid(row=10, column=0, sticky="e", **pad)
        tk.Scale(frame, variable=self.logo_opacity, from_=0.0, to=1.0, resolution=0.01, orient="horizontal", length=220).grid(
            row=10, column=1, sticky="w", **pad
        )
        
        tk.Label(frame, text="Margine (px):").grid(row=11, column=0, sticky="e", **pad)
        tk.Spinbox(frame, from_=0, to=200, textvariable=self.logo_margin, width=10).grid(
            row=11, column=1, sticky="w", **pad
        )
        
        # Progress
        self.sync_progress = ttk.Progressbar(frame, mode="determinate", length=620)
        self.sync_progress.grid(row=12, column=0, columnspan=4, **pad)
        self.sync_status_lbl = tk.Label(frame, text="Pronto")
        self.sync_status_lbl.grid(row=13, column=0, columnspan=4, sticky="w", **pad)
        
        # Actions
        self.sync_run_btn = tk.Button(frame, text="Sincronizza Video", command=self.on_sync_run)
        self.sync_run_btn.grid(row=14, column=2, sticky="e", **pad)
        self.sync_cancel_btn = tk.Button(frame, text="Annulla", command=self.on_cancel, state=tk.DISABLED)
        self.sync_cancel_btn.grid(row=14, column=3, sticky="w", **pad)

    def browse_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona file audio",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a"), ("Tutti i file", "*.*")],
        )
        if path:
            self.audio_path.set(path)

    def browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona immagine",
            filetypes=[("Immagini", "*.jpg *.jpeg *.png"), ("Tutti i file", "*.*")],
        )
        if path:
            self.image_path.set(path)

    def browse_logo(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona logo (PNG consigliato)",
            filetypes=[("Immagini", "*.png *.jpg *.jpeg"), ("Tutti i file", "*.*")],
        )
        if path:
            self.logo_path.set(path)
    
    def browse_sync_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona file audio",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a"), ("Tutti i file", "*.*")],
        )
        if path:
            self.sync_audio_path.set(path)
    
    def browse_sync_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.flv"), ("Tutti i file", "*.*")],
        )
        if path:
            self.sync_video_path.set(path)

    def _suggest_output_filename(self) -> str:
        current = self.output_path.get().strip()
        if current:
            return os.path.basename(current)
        audio = self.audio_path.get().strip()
        if audio:
            return f"{Path(audio).stem}_fx.mp4"
        return "output.mp4"

    def choose_output(self) -> None:
        init_dir = None
        audio = self.audio_path.get().strip()
        if audio and os.path.exists(audio):
            init_dir = os.path.dirname(audio)
        elif self.output_path.get().strip():
            init_dir = os.path.dirname(self.output_path.get().strip())
        path = filedialog.asksaveasfilename(
            title="Scegli file di output",
            defaultextension=".mp4",
            initialdir=init_dir,
            initialfile=self._suggest_output_filename(),
            filetypes=[("MP4", "*.mp4"), ("Tutti i file", "*.*")],
        )
        if path:
            self.output_path.set(path)

    def choose_output_dir(self) -> None:
        init_dir = None
        audio = self.audio_path.get().strip()
        if audio and os.path.exists(audio):
            init_dir = os.path.dirname(audio)
        elif self.output_path.get().strip():
            init_dir = os.path.dirname(self.output_path.get().strip())
        directory = filedialog.askdirectory(title="Scegli cartella di output", initialdir=init_dir)
        if directory:
            filename = self._suggest_output_filename()
            self.output_path.set(os.path.join(directory, filename))
    
    def _suggest_sync_output_filename(self) -> str:
        current = self.sync_output_path.get().strip()
        if current:
            return os.path.basename(current)
        video = self.sync_video_path.get().strip()
        if video:
            return f"{Path(video).stem}_synced.mp4"
        return "synced_output.mp4"
    
    def choose_sync_output(self) -> None:
        init_dir = None
        video = self.sync_video_path.get().strip()
        if video and os.path.exists(video):
            init_dir = os.path.dirname(video)
        elif self.sync_output_path.get().strip():
            init_dir = os.path.dirname(self.sync_output_path.get().strip())
        path = filedialog.asksaveasfilename(
            title="Scegli file di output",
            defaultextension=".mp4",
            initialdir=init_dir,
            initialfile=self._suggest_sync_output_filename(),
            filetypes=[("MP4", "*.mp4"), ("Tutti i file", "*.*")],
        )
        if path:
            self.sync_output_path.set(path)
    
    def choose_sync_output_dir(self) -> None:
        init_dir = None
        video = self.sync_video_path.get().strip()
        if video and os.path.exists(video):
            init_dir = os.path.dirname(video)
        elif self.sync_output_path.get().strip():
            init_dir = os.path.dirname(self.sync_output_path.get().strip())
        directory = filedialog.askdirectory(title="Scegli cartella di output", initialdir=init_dir)
        if directory:
            filename = self._suggest_sync_output_filename()
            self.sync_output_path.set(os.path.join(directory, filename))

    def on_run(self) -> None:
        audio = self.audio_path.get().strip()
        image = self.image_path.get().strip()
        output = self.output_path.get().strip()
        fps = int(self.fps_var.get())

        if not audio or not os.path.exists(audio):
            messagebox.showerror("Errore", "Seleziona un file audio valido")
            return
        if not image or not os.path.exists(image):
            messagebox.showerror("Errore", "Seleziona un file immagine valido")
            return
        if not output:
            messagebox.showerror("Errore", "Specifica un file di output")
            return

        # Map GUI preset names to EffectStyle enum
        name_to_style = {
            "Standard": EffectStyle.STANDARD,
            "Extreme": EffectStyle.EXTREME,
            "Psychedelic": EffectStyle.PSYCHEDELIC,
            "Minimal": EffectStyle.MINIMAL,
            "Cyberpunk": EffectStyle.CYBERPUNK,
            "Industrial": EffectStyle.INDUSTRIAL,
            "Acid House": EffectStyle.ACID_HOUSE,
            "Retro Wave": EffectStyle.RETRO_WAVE,
            "Horror": EffectStyle.HORROR,
        }
        effect_style = name_to_style.get(self.preset_var.get(), EffectStyle.STANDARD)
        
        # Get colors from config if available
        preset_key_map = {
            "Standard": "dark_techno",
            "Extreme": "extreme_vibrant",
            "Psychedelic": "psychedelic_refraction",
            "Minimal": "dark_techno",
            "Cyberpunk": "cyberpunk",
            "Industrial": "industrial",
            "Acid House": "acid_house",
            "Retro Wave": "retro_wave",
            "Horror": "horror",
        }
        preset_key = preset_key_map.get(self.preset_var.get(), "dark_techno")
        preset = get_preset_config(preset_key)
        colors = preset.get("colors")

        self._cancel = False
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.status_lbl.config(text="In esecuzione...")
        self.progress.config(value=0, maximum=100)

        def progress_cb(event: str, payload: dict) -> None:
            if event == "start":
                total = max(1, int(payload.get("total_frames", 100)))
                self._total_frames = total
                self.after(0, lambda: self.progress.config(value=0, maximum=total))
            elif event == "frame":
                idx = int(payload.get("index", 0))
                self.after(0, lambda i=idx: self.progress.config(value=i))
            elif event == "status":
                msg = payload.get("message", "")
                self.after(0, lambda m=msg: self.status_lbl.config(text=m))
            elif event == "done":
                out = payload.get("output", "output.mp4")
                self.after(0, lambda o=out: self.status_lbl.config(text=f"Fatto: {o}"))

        def worker() -> None:
            try:
                # Create effect config with custom colors
                effect_config = None
                if colors:
                    effect_config = EffectConfig(colors=colors)
                
                # Create generator using modular architecture
                generator = AudioVisualGenerator(
                    audio_file=audio,
                    image_file=image,
                    output_file=output,
                    fps=fps,
                    duration=None,
                    effect_style=effect_style,
                    effect_config=effect_config,
                    progress_cb=progress_cb,
                )
                
                # Generate video
                generator.generate()
                self.after(0, lambda: messagebox.showinfo("Completato", f"Video creato: {output}"))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: messagebox.showerror("Errore", msg))
            finally:
                self.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.cancel_btn.config(state=tk.DISABLED))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def on_sync_run(self) -> None:
        """Handler per sincronizzazione video con audio."""
        audio = self.sync_audio_path.get().strip()
        video = self.sync_video_path.get().strip()
        output = self.sync_output_path.get().strip()
        
        if not audio or not os.path.exists(audio):
            messagebox.showerror("Errore", "Seleziona un file audio valido")
            return
        if not video or not os.path.exists(video):
            messagebox.showerror("Errore", "Seleziona un file video valido")
            return
        if not output:
            messagebox.showerror("Errore", "Specifica un file di output")
            return
        
        # Map GUI preset names to EffectStyle enum
        name_to_style = {
            "Standard": EffectStyle.STANDARD,
            "Extreme": EffectStyle.EXTREME,
            "Psychedelic": EffectStyle.PSYCHEDELIC,
            "Minimal": EffectStyle.MINIMAL,
            "Cyberpunk": EffectStyle.CYBERPUNK,
            "Industrial": EffectStyle.INDUSTRIAL,
            "Acid House": EffectStyle.ACID_HOUSE,
            "Retro Wave": EffectStyle.RETRO_WAVE,
            "Horror": EffectStyle.HORROR,
        }
        effect_style = name_to_style.get(self.sync_preset_var.get(), EffectStyle.STANDARD)
        
        # Get colors from config if available
        preset_key_map = {
            "Standard": "dark_techno",
            "Extreme": "extreme_vibrant",
            "Psychedelic": "psychedelic_refraction",
            "Minimal": "dark_techno",
            "Cyberpunk": "cyberpunk",
            "Industrial": "industrial",
            "Acid House": "acid_house",
            "Retro Wave": "retro_wave",
            "Horror": "horror",
        }
        preset_key = preset_key_map.get(self.sync_preset_var.get(), "dark_techno")
        preset = get_preset_config(preset_key)
        colors = preset.get("colors")
        
        self._cancel = False
        self.sync_run_btn.config(state=tk.DISABLED)
        self.sync_cancel_btn.config(state=tk.NORMAL)
        self.sync_status_lbl.config(text="In esecuzione...")
        self.sync_progress.config(value=0, maximum=100)
        
        def progress_cb(event: str, payload: dict) -> None:
            if event == "start":
                total = max(1, int(payload.get("total_frames", 100)))
                self._total_frames = total
                self.after(0, lambda: self.sync_progress.config(value=0, maximum=total))
            elif event == "frame":
                idx = int(payload.get("index", 0))
                self.after(0, lambda i=idx: self.sync_progress.config(value=i))
            elif event == "status":
                msg = payload.get("message", "")
                self.after(0, lambda m=msg: self.sync_status_lbl.config(text=m))
            elif event == "done":
                out = payload.get("output", "output.mp4")
                self.after(0, lambda o=out: self.sync_status_lbl.config(text=f"Fatto: {o}"))
        
        def worker() -> None:
            try:
                # Get logo settings
                logo = self.logo_path.get().strip() if self.logo_path.get().strip() else None
                logo_pos = self.logo_position.get().lower()
                logo_scale_val = float(self.logo_scale.get())
                logo_opacity_val = float(self.logo_opacity.get())
                logo_margin_val = int(self.logo_margin.get())
                
                # Get short video mode
                short_mode = self.sync_short_mode.get().lower()
                
                # Create effect config with custom colors
                effect_config = None
                if colors:
                    effect_config = EffectConfig(colors=colors)
                
                # Use VideoAudioSync class
                syncer = VideoAudioSync(
                    audio_file=audio,
                    video_file=video,
                    output_file=output,
                    effect_style=effect_style,
                    effect_config=effect_config,
                    short_video_mode=short_mode,
                    logo_file=logo,
                    logo_position=logo_pos,
                    logo_scale=logo_scale_val,
                    logo_opacity=logo_opacity_val,
                    logo_margin=logo_margin_val,
                    progress_cb=progress_cb,
                )
                syncer.sync()
                
                self.after(0, lambda: messagebox.showinfo("Completato", f"Video sincronizzato con effetti: {output}"))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: messagebox.showerror("Errore", msg))
            finally:
                self.after(0, lambda: self.sync_run_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.sync_cancel_btn.config(state=tk.DISABLED))
        
        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
    
    def _build_custom_preset_tab(self) -> None:
        """Build UI for custom preset configuration."""
        pad = {"padx": 10, "pady": 6}
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(self.tab3)
        scrollbar = ttk.Scrollbar(self.tab3, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        frame = scrollable_frame
        
        # Description
        desc = tk.Label(
            frame,
            text="Configura il tuo preset personalizzato con Audio+Immagine o Audio+Video",
            justify=tk.LEFT,
            font=("Arial", 9, "italic")
        )
        desc.grid(row=0, column=0, columnspan=4, sticky="w", **pad)
        
        # Mode selection: Image or Video
        tk.Label(frame, text="Modalità:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky="e", **pad)
        self.custom_mode = tk.StringVar(value="image")
        tk.Radiobutton(frame, text="Audio + Immagine", variable=self.custom_mode, 
                      value="image", command=self._toggle_custom_mode).grid(row=1, column=1, sticky="w", **pad)
        tk.Radiobutton(frame, text="Audio + Video", variable=self.custom_mode, 
                      value="video", command=self._toggle_custom_mode).grid(row=1, column=2, sticky="w", **pad)
        
        # Audio
        tk.Label(frame, text="Audio:").grid(row=2, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.custom_audio_path, width=45).grid(row=2, column=1, **pad)
        tk.Button(frame, text="Sfoglia", command=lambda: self._browse_custom_audio()).grid(row=2, column=2, **pad)
        
        # Image (shown only in image mode)
        self.custom_image_label = tk.Label(frame, text="Immagine:")
        self.custom_image_label.grid(row=3, column=0, sticky="e", **pad)
        self.custom_image_entry = tk.Entry(frame, textvariable=self.custom_image_path, width=45)
        self.custom_image_entry.grid(row=3, column=1, **pad)
        self.custom_image_btn = tk.Button(frame, text="Sfoglia", command=lambda: self._browse_custom_image())
        self.custom_image_btn.grid(row=3, column=2, **pad)
        
        # Video (shown only in video mode)
        self.custom_video_label = tk.Label(frame, text="Video:")
        self.custom_video_entry = tk.Entry(frame, textvariable=self.custom_video_path, width=45)
        self.custom_video_btn = tk.Button(frame, text="Sfoglia", command=lambda: self._browse_custom_video())
        
        # Video mode selection (shown only in video mode)
        self.custom_video_mode_label = tk.Label(frame, text="Video corto:")
        self.custom_video_mode_combo = ttk.Combobox(
            frame,
            textvariable=self.custom_video_mode,
            values=["loop", "stretch"],
            state="readonly",
            width=10
        )
        self.custom_video_mode_combo.current(0)
        
        # Output
        tk.Label(frame, text="Output:").grid(row=4, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.custom_output_path, width=45).grid(row=4, column=1, **pad)
        tk.Button(frame, text="Scegli", command=lambda: self._choose_custom_output()).grid(row=4, column=2, **pad)
        
        # FPS (not shown in video mode as we use video FPS)
        self.custom_fps_label = tk.Label(frame, text="FPS:")
        self.custom_fps_label.grid(row=5, column=0, sticky="e", **pad)
        self.custom_fps_spinbox = tk.Spinbox(frame, from_=1, to=120, textvariable=self.custom_fps_var, width=10)
        self.custom_fps_spinbox.grid(row=5, column=1, sticky="w", **pad)
        
        # Initialize visibility based on mode
        self._toggle_custom_mode()
        
        # Separator
        sep = ttk.Separator(frame, orient="horizontal")
        sep.grid(row=6, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 2))
        
        # Thresholds section
        tk.Label(frame, text="Soglie Audio:", font=("Arial", 9, "bold")).grid(row=7, column=0, sticky="w", **pad)
        
        tk.Label(frame, text="Bass:").grid(row=8, column=0, sticky="e", **pad)
        tk.Scale(frame, variable=self.threshold_bass, from_=0.0, to=1.0, resolution=0.01, 
                orient="horizontal", length=150).grid(row=8, column=1, sticky="w", **pad)
        
        tk.Label(frame, text="Mid:").grid(row=8, column=2, sticky="e", padx=(20,5))
        tk.Scale(frame, variable=self.threshold_mid, from_=0.0, to=1.0, resolution=0.01, 
                orient="horizontal", length=150).grid(row=8, column=3, sticky="w", **pad)
        
        tk.Label(frame, text="High:").grid(row=9, column=0, sticky="e", **pad)
        tk.Scale(frame, variable=self.threshold_high, from_=0.0, to=1.0, resolution=0.01, 
                orient="horizontal", length=150).grid(row=9, column=1, sticky="w", **pad)
        
        # Separator
        sep2 = ttk.Separator(frame, orient="horizontal")
        sep2.grid(row=10, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 2))
        
        # Effects section
        tk.Label(frame, text="Effetti (seleziona e imposta intensità):", font=("Arial", 9, "bold")).grid(
            row=11, column=0, columnspan=2, sticky="w", **pad
        )
        
        # Create scrollable frame for effects
        effects_frame = tk.Frame(frame)
        effects_frame.grid(row=12, column=0, columnspan=4, sticky="ew", **pad)
        
        # Effect checkboxes with intensity sliders
        effects = [
            ("ColorPulse", self.effect_color_pulse, self.intensity_color_pulse),
            ("ZoomPulse", self.effect_zoom_pulse, self.intensity_zoom_pulse),
            ("Strobe", self.effect_strobe, self.intensity_strobe),
            ("Glitch", self.effect_glitch, self.intensity_glitch),
            ("ChromaticAberration", self.effect_chromatic, self.intensity_chromatic),
            ("BubbleDistortion", self.effect_bubble, self.intensity_bubble),
            ("ScreenShake", self.effect_screen_shake, self.intensity_screen_shake),
            ("RGBSplit", self.effect_rgb_split, self.intensity_rgb_split),
            ("ElectricArcs", self.effect_electric_arcs, self.intensity_electric_arcs),
        ]
        
        for idx, (name, var, intensity_var) in enumerate(effects):
            row_offset = idx // 2
            col_offset = (idx % 2) * 2
            
            tk.Checkbutton(effects_frame, text=name, variable=var).grid(
                row=row_offset, column=col_offset, sticky="w", padx=5, pady=2
            )
            tk.Scale(effects_frame, variable=intensity_var, from_=0.0, to=2.0, resolution=0.1,
                    orient="horizontal", length=100, label="").grid(
                row=row_offset, column=col_offset + 1, sticky="w", padx=5, pady=2
            )
        
        # Progress
        self.custom_progress = ttk.Progressbar(frame, mode="determinate", length=620)
        self.custom_progress.grid(row=13, column=0, columnspan=4, **pad)
        self.custom_status_lbl = tk.Label(frame, text="Pronto")
        self.custom_status_lbl.grid(row=14, column=0, columnspan=4, sticky="w", **pad)
        
        # Actions
        self.custom_run_btn = tk.Button(frame, text="Genera Video Custom", command=self.on_custom_run)
        self.custom_run_btn.grid(row=15, column=2, sticky="e", **pad)
        self.custom_cancel_btn = tk.Button(frame, text="Annulla", command=self.on_cancel, state=tk.DISABLED)
        self.custom_cancel_btn.grid(row=15, column=3, sticky="w", **pad)
    
    def _browse_custom_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona file audio",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a"), ("Tutti i file", "*.*")],
        )
        if path:
            self.custom_audio_path.set(path)
    
    def _browse_custom_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona immagine",
            filetypes=[("Immagini", "*.jpg *.jpeg *.png"), ("Tutti i file", "*.*")],
        )
        if path:
            self.custom_image_path.set(path)
    
    def _browse_custom_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.flv"), ("Tutti i file", "*.*")],
        )
        if path:
            self.custom_video_path.set(path)
    
    def _toggle_custom_mode(self) -> None:
        """Toggle visibility of image/video controls based on selected mode."""
        mode = self.custom_mode.get()
        pad = {"padx": 10, "pady": 6}
        
        if mode == "image":
            # Show image controls
            self.custom_image_label.grid(row=3, column=0, sticky="e", **pad)
            self.custom_image_entry.grid(row=3, column=1, **pad)
            self.custom_image_btn.grid(row=3, column=2, **pad)
            # Show FPS control
            self.custom_fps_label.grid(row=5, column=0, sticky="e", **pad)
            self.custom_fps_spinbox.grid(row=5, column=1, sticky="w", **pad)
            
            # Hide video controls
            self.custom_video_label.grid_forget()
            self.custom_video_entry.grid_forget()
            self.custom_video_btn.grid_forget()
            self.custom_video_mode_label.grid_forget()
            self.custom_video_mode_combo.grid_forget()
        else:
            # Hide image controls
            self.custom_image_label.grid_forget()
            self.custom_image_entry.grid_forget()
            self.custom_image_btn.grid_forget()
            # Hide FPS control (we'll use video FPS)
            self.custom_fps_label.grid_forget()
            self.custom_fps_spinbox.grid_forget()
            
            # Show video controls
            self.custom_video_label.grid(row=3, column=0, sticky="e", **pad)
            self.custom_video_entry.grid(row=3, column=1, **pad)
            self.custom_video_btn.grid(row=3, column=2, **pad)
            self.custom_video_mode_label.grid(row=5, column=0, sticky="e", **pad)
            self.custom_video_mode_combo.grid(row=5, column=1, sticky="w", **pad)
    
    def _choose_custom_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Scegli file di output",
            defaultextension=".mp4",
            initialfile="custom_output.mp4",
            filetypes=[("MP4", "*.mp4"), ("Tutti i file", "*.*")],
        )
        if path:
            self.custom_output_path.set(path)
    
    def on_custom_run(self) -> None:
        """Handler for custom preset video generation."""
        audio = self.custom_audio_path.get().strip()
        output = self.custom_output_path.get().strip()
        mode = self.custom_mode.get()
        
        if not audio or not os.path.exists(audio):
            messagebox.showerror("Errore", "Seleziona un file audio valido")
            return
        
        if mode == "image":
            image = self.custom_image_path.get().strip()
            if not image or not os.path.exists(image):
                messagebox.showerror("Errore", "Seleziona un file immagine valido")
                return
            fps = int(self.custom_fps_var.get())
        else:  # video mode
            video = self.custom_video_path.get().strip()
            if not video or not os.path.exists(video):
                messagebox.showerror("Errore", "Seleziona un file video valido")
                return
            short_mode = self.custom_video_mode.get()
        
        if not output:
            messagebox.showerror("Errore", "Specifica un file di output")
            return
        
        # Check at least one effect is selected
        if not any([
            self.effect_color_pulse.get(),
            self.effect_zoom_pulse.get(),
            self.effect_strobe.get(),
            self.effect_glitch.get(),
            self.effect_chromatic.get(),
            self.effect_bubble.get(),
            self.effect_screen_shake.get(),
            self.effect_rgb_split.get(),
            self.effect_electric_arcs.get(),
        ]):
            messagebox.showerror("Errore", "Seleziona almeno un effetto")
            return
        
        self._cancel = False
        self.custom_run_btn.config(state=tk.DISABLED)
        self.custom_cancel_btn.config(state=tk.NORMAL)
        self.custom_status_lbl.config(text="In esecuzione...")
        self.custom_progress.config(value=0, maximum=100)
        
        def progress_cb(event: str, payload: dict) -> None:
            if event == "start":
                total = max(1, int(payload.get("total_frames", 100)))
                self._total_frames = total
                self.after(0, lambda: self.custom_progress.config(value=0, maximum=total))
            elif event == "frame":
                idx = int(payload.get("index", 0))
                self.after(0, lambda i=idx: self.custom_progress.config(value=i))
            elif event == "status":
                msg = payload.get("message", "")
                self.after(0, lambda m=msg: self.custom_status_lbl.config(text=m))
            elif event == "done":
                out = payload.get("output", "output.mp4")
                self.after(0, lambda o=out: self.custom_status_lbl.config(text=f"Fatto: {o}"))
        
        def worker() -> None:
            try:
                # Import effects
                from src.effects import (
                    ColorPulseEffect, ZoomPulseEffect, StrobeEffect, GlitchEffect,
                    ChromaticAberrationEffect, BubbleDistortionEffect, ScreenShakeEffect, RGBSplitEffect,
                    ElectricArcsEffect
                )
                from src.factories import EffectFactory
                
                # Build custom effect list
                custom_effects = []
                
                if self.effect_color_pulse.get():
                    custom_effects.append(ColorPulseEffect(
                        bass_threshold=self.threshold_bass.get(),
                        mid_threshold=self.threshold_mid.get(),
                        high_threshold=self.threshold_high.get(),
                        intensity=self.intensity_color_pulse.get()
                    ))
                
                if self.effect_zoom_pulse.get():
                    custom_effects.append(ZoomPulseEffect(
                        threshold=self.threshold_bass.get(),
                        intensity=self.intensity_zoom_pulse.get()
                    ))
                
                if self.effect_strobe.get():
                    # Default colors for strobe
                    default_colors = [(1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0)]
                    custom_effects.append(StrobeEffect(
                        colors=default_colors,
                        threshold=0.8,
                        intensity=self.intensity_strobe.get()
                    ))
                
                if self.effect_glitch.get():
                    custom_effects.append(GlitchEffect(
                        threshold=self.threshold_mid.get(),
                        intensity=self.intensity_glitch.get()
                    ))
                
                if self.effect_chromatic.get():
                    custom_effects.append(ChromaticAberrationEffect(
                        threshold=self.threshold_high.get(),
                        intensity=self.intensity_chromatic.get()
                    ))
                
                if self.effect_bubble.get():
                    custom_effects.append(BubbleDistortionEffect(
                        threshold=self.threshold_bass.get(),
                        intensity=self.intensity_bubble.get()
                    ))
                
                if self.effect_screen_shake.get():
                    custom_effects.append(ScreenShakeEffect(
                        threshold=self.threshold_mid.get(),
                        intensity=self.intensity_screen_shake.get()
                    ))
                
                if self.effect_rgb_split.get():
                    custom_effects.append(RGBSplitEffect(
                        threshold=self.threshold_high.get(),
                        intensity=self.intensity_rgb_split.get()
                    ))
                
                if self.effect_electric_arcs.get():
                    # Default colors for electric arcs
                    default_colors = [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
                    custom_effects.append(ElectricArcsEffect(
                        colors=default_colors,
                        threshold=0.7,
                        intensity=self.intensity_electric_arcs.get()
                    ))
                
                # Create custom pipeline
                custom_pipeline = EffectFactory.create_custom_pipeline(custom_effects)
                
                if mode == "image":
                    # Image mode: manual implementation with custom pipeline
                    from src.core.audio_analyzer import AudioAnalyzer
                    from src.core.frame_generator import FrameGenerator
                    from src.core.video_exporter import VideoExporter
                    from src.utils.image_loader import ImageLoader
                    
                    # Analyze audio
                    progress_cb("status", {"message": "Analisi audio..."})
                    audio_analyzer = AudioAnalyzer()
                    audio_data = audio_analyzer.load_and_analyze(audio, duration=None, fps=fps)
                    
                    # Load image
                    progress_cb("status", {"message": "Caricamento immagine..."})
                    base_image = ImageLoader.load_and_prepare(image, (720, 720))
                    
                    # Create frame generator with custom pipeline
                    progress_cb("status", {"message": "Setup effetti custom..."})
                    frame_generator = FrameGenerator(
                        base_image=base_image,
                        effect_pipeline=custom_pipeline,
                        fps=fps
                    )
                    
                    # Generate frames
                    num_frames = len(audio_data.bass_energy)
                    progress_cb("start", {"total_frames": num_frames})
                    
                    output_frames = []
                    for idx in range(num_frames):
                        progress_cb("frame", {"index": idx + 1, "total": num_frames})
                        
                        frame = frame_generator.generate_frame(
                            frame_index=idx,
                            audio_analysis=audio_data,
                            color_index=idx % 6
                        )
                        output_frames.append(frame)
                    
                    # Export video
                    progress_cb("status", {"message": "Encoding video..."})
                    exporter = VideoExporter(output, fps)
                    exporter.export(output_frames, audio, progress_cb)
                    
                    progress_cb("done", {"output": output})
                else:
                    # Video mode: manual implementation with custom pipeline
                    from src.video_audio_sync import VideoAudioSync
                    from src.models.data_models import EffectConfig
                    import cv2
                    from src.core.audio_analyzer import AudioAnalyzer
                    from src.core.frame_generator import FrameGenerator
                    from src.core.video_exporter import VideoExporter
                    
                    # Load video to get FPS and frames
                    progress_cb("status", {"message": "Caricamento video..."})
                    cap = cv2.VideoCapture(video)
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    video_frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cap.release()
                    
                    if not video_frames:
                        raise ValueError("Impossibile caricare frames dal video")
                    
                    resolution = (video_frames[0].shape[1], video_frames[0].shape[0])
                    
                    # Load and analyze audio
                    progress_cb("status", {"message": "Analisi audio..."})
                    audio_analyzer = AudioAnalyzer()
                    audio_data = audio_analyzer.load_and_analyze(audio, fps=int(video_fps))
                    audio_duration = len(audio_data.audio_signal) / audio_data.sample_rate
                    
                    # Sync video frames to audio duration
                    required_frames = int(audio_duration * video_fps)
                    available_frames = len(video_frames)
                    
                    if required_frames <= available_frames:
                        frame_indices = list(range(required_frames))
                    elif short_mode == "loop":
                        frame_indices = []
                        while len(frame_indices) < required_frames:
                            remaining = required_frames - len(frame_indices)
                            frame_indices.extend(range(min(available_frames, remaining)))
                    else:  # stretch
                        stretch_factor = required_frames / available_frames
                        frame_indices = []
                        for i in range(available_frames):
                            repeat_count = int(stretch_factor)
                            if (i * stretch_factor) % 1 >= (1 - stretch_factor % 1):
                                repeat_count += 1
                            frame_indices.extend([i] * repeat_count)
                        frame_indices = frame_indices[:required_frames]
                    
                    # Generate frames with custom effects
                    total_frames = len(frame_indices)
                    progress_cb("start", {"total_frames": total_frames})
                    
                    output_frames = []
                    for idx, frame_idx in enumerate(frame_indices):
                        progress_cb("frame", {"index": idx + 1, "total": total_frames})
                        
                        base_frame = video_frames[frame_idx].copy()
                        
                        frame_generator = FrameGenerator(
                            base_image=base_frame,
                            effect_pipeline=custom_pipeline,
                            fps=int(video_fps)
                        )
                        
                        frame_with_effects = frame_generator.generate_frame(
                            frame_index=idx,
                            audio_analysis=audio_data,
                            color_index=0
                        )
                        
                        output_frames.append(frame_with_effects)
                    
                    # Export video
                    progress_cb("status", {"message": "Encoding video..."})
                    exporter = VideoExporter(output, int(video_fps))
                    exporter.export(output_frames, audio, progress_cb)
                    
                    progress_cb("done", {"output": output})
                
                self.after(0, lambda: messagebox.showinfo("Completato", f"Video custom creato: {output}"))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: messagebox.showerror("Errore", msg))
            finally:
                self.after(0, lambda: self.custom_run_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.custom_cancel_btn.config(state=tk.DISABLED))
        
        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
    
    def on_cancel(self) -> None:
        # Soft cancel: just inform the user; full cooperative cancel would require plumbing
        self._cancel = True
        messagebox.showinfo("Annulla", "Annullamento non immediato. Chiudi la finestra per interrompere.")


if __name__ == "__main__":
    App().mainloop()
