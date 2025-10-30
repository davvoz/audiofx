"""
Simple GUI for Audio Visual FX using Tkinter.
Allows selecting audio, image, FPS, output, and a visual preset, then runs generation.
Includes flexible output location selection (save-as dialog or pick a folder).
Now includes video sync functionality.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from video_generator import generate_video, sync_video_with_audio
from config import get_preset_config


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Audio Visual FX Generator")
        self.geometry("800x600")
        self.resizable(False, False)

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
    
    def _build_audio_image_tab(self) -> None:
        """Build UI for audio + image tab."""
        pad = {"padx": 10, "pady": 6}
        frame = self.tab1

        # Preset
        tk.Label(frame, text="Preset:").grid(row=0, column=0, sticky="e", **pad)
        self.preset_combo = ttk.Combobox(
            frame,
            textvariable=self.preset_var,
            values=["Dark Techno", "Cyberpunk", "Industrial", "Acid House"],
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
        
        # Preset
        tk.Label(frame, text="Preset Effetti:").grid(row=1, column=0, sticky="e", **pad)
        self.sync_preset_combo = ttk.Combobox(
            frame,
            textvariable=self.sync_preset_var,
            values=["Dark Techno", "Cyberpunk", "Industrial", "Acid House"],
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

        name_to_key = {
            "Dark Techno": "dark_techno",
            "Cyberpunk": "cyberpunk",
            "Industrial": "industrial",
            "Acid House": "acid_house",
        }
        preset_key = name_to_key.get(self.preset_var.get(), "dark_techno")
        preset = get_preset_config(preset_key)
        colors = preset.get("colors")
        th = preset.get("thresholds", {})
        thresholds = (
            float(th.get("bass", 0.3)),
            float(th.get("mid", 0.2)),
            float(th.get("high", 0.15)),
        )

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
                pos_map = {
                    "Top-Left": "top-left",
                    "Top-Right": "top-right",
                    "Bottom-Left": "bottom-left",
                    "Bottom-Right": "bottom-right",
                }
                pos = pos_map.get(self.logo_position.get(), "top-right")
                generate_video(
                    audio,
                    image,
                    output,
                    fps=fps,
                    duration=None,
                    progress_cb=progress_cb,
                    colors=colors,
                    thresholds=thresholds,
                    logo=self.logo_path.get().strip() or None,
                    logo_position=pos,
                    logo_scale=float(self.logo_scale.get()),
                    logo_opacity=float(self.logo_opacity.get()),
                    logo_margin=int(self.logo_margin.get()),
                )
                self.after(0, lambda: messagebox.showinfo("Completato", f"Video creato: {output}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Errore", str(e)))
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
        
        # Get preset configuration
        name_to_key = {
            "Dark Techno": "dark_techno",
            "Cyberpunk": "cyberpunk",
            "Industrial": "industrial",
            "Acid House": "acid_house",
        }
        preset_key = name_to_key.get(self.sync_preset_var.get(), "dark_techno")
        preset = get_preset_config(preset_key)
        colors = preset.get("colors")
        th = preset.get("thresholds", {})
        thresholds = (
            float(th.get("bass", 0.3)),
            float(th.get("mid", 0.2)),
            float(th.get("high", 0.15)),
        )
        
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
                pos_map = {
                    "Top-Left": "top-left",
                    "Top-Right": "top-right",
                    "Bottom-Left": "bottom-left",
                    "Bottom-Right": "bottom-right",
                }
                pos = pos_map.get(self.logo_position.get(), "top-right")
                
                mode_map = {
                    "Loop": "loop",
                    "Stretch": "stretch",
                }
                short_mode = mode_map.get(self.sync_short_mode.get(), "loop")
                
                sync_video_with_audio(
                    audio,
                    video,
                    output,
                    progress_cb=progress_cb,
                    colors=colors,
                    thresholds=thresholds,
                    short_video_mode=short_mode,
                    logo=self.logo_path.get().strip() or None,
                    logo_position=pos,
                    logo_scale=float(self.logo_scale.get()),
                    logo_opacity=float(self.logo_opacity.get()),
                    logo_margin=int(self.logo_margin.get()),
                )
                self.after(0, lambda: messagebox.showinfo("Completato", f"Video sincronizzato con effetti: {output}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Errore", str(e)))
            finally:
                self.after(0, lambda: self.sync_run_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.sync_cancel_btn.config(state=tk.DISABLED))
        
        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
    
    def on_cancel(self) -> None:
        # Soft cancel: just inform the user; full cooperative cancel would require plumbing
        self._cancel = True
        messagebox.showinfo("Annulla", "Annullamento non immediato. Chiudi la finestra per interrompere.")


if __name__ == "__main__":
    App().mainloop()
