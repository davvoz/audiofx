"""
Simple GUI for Audio Visual FX using Tkinter.
Now using modular architecture from src/ package.
"""

import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Audio Visual FX Generator")
        self.geometry("900x800")
        self.resizable(True, True)  # Finestra ridimensionabile

    # Custom Preset Controls
        self.custom_mode = tk.StringVar(value="image")
        self.custom_audio_path = tk.StringVar()
        self.custom_image_path = tk.StringVar()
        self.custom_video_path = tk.StringVar()
        self.custom_output_path = tk.StringVar(value="custom_output.mp4")
        self.custom_fps_var = tk.IntVar(value=30)
        self.custom_video_mode = tk.StringVar(value="loop")
        self.use_video_audio = tk.BooleanVar(value=False)
        self.use_native_resolution = tk.BooleanVar(value=True)  # NEW: use native resolution by default
        
        # Custom effect checkboxes
        self.effect_color_pulse = tk.BooleanVar(value=True)
        self.effect_zoom_pulse = tk.BooleanVar(value=True)
        self.effect_strobe = tk.BooleanVar(value=False)
        self.effect_strobe_negative = tk.BooleanVar(value=False)
        self.effect_glitch = tk.BooleanVar(value=False)
        self.effect_chromatic = tk.BooleanVar(value=False)
        self.effect_bubble = tk.BooleanVar(value=False)
        self.effect_screen_shake = tk.BooleanVar(value=False)
        self.effect_rgb_split = tk.BooleanVar(value=False)
        self.effect_electric_arcs = tk.BooleanVar(value=False)
        self.effect_fashion_lightning = tk.BooleanVar(value=False)
        self.effect_advanced_glitch = tk.BooleanVar(value=False)
        self.effect_dimensional_warp = tk.BooleanVar(value=False)
        self.effect_vortex_distortion = tk.BooleanVar(value=False)
        
        # Custom effect intensities
        self.intensity_color_pulse = tk.DoubleVar(value=1.0)
        self.intensity_zoom_pulse = tk.DoubleVar(value=1.0)
        self.intensity_strobe = tk.DoubleVar(value=1.0)
        self.intensity_strobe_negative = tk.DoubleVar(value=1.0)
        self.intensity_glitch = tk.DoubleVar(value=1.0)
        self.intensity_chromatic = tk.DoubleVar(value=1.0)
        self.intensity_bubble = tk.DoubleVar(value=1.0)
        self.intensity_screen_shake = tk.DoubleVar(value=1.0)
        self.intensity_rgb_split = tk.DoubleVar(value=1.0)
        self.intensity_electric_arcs = tk.DoubleVar(value=1.0)
        self.intensity_fashion_lightning = tk.DoubleVar(value=1.0)
        self.intensity_advanced_glitch = tk.DoubleVar(value=1.0)
        self.intensity_dimensional_warp = tk.DoubleVar(value=1.0)
        self.intensity_vortex_distortion = tk.DoubleVar(value=1.0)
        
        # Custom thresholds
        self.threshold_bass = tk.DoubleVar(value=0.3)
        self.threshold_mid = tk.DoubleVar(value=0.2)
        self.threshold_high = tk.DoubleVar(value=0.15)
        
        # Logo overlay settings
        self.logo_path = tk.StringVar()
        self.logo_position = tk.StringVar(value="top-right")
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

        # Single Tab: Custom Preset Builder (only)
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Custom Preset")
        self._build_custom_preset_tab()
    
    def _build_audio_image_tab(self) -> None:
        # Legacy tab removed
        pass
    
    def _build_video_sync_tab(self) -> None:
        # Legacy tab removed
        pass

    # Legacy browse/output helpers removed with old tabs

    def on_run(self) -> None:
        # Legacy handler removed
        pass

    def on_sync_run(self) -> None:
        # Legacy handler removed
        pass
    
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
        self.custom_audio_entry = tk.Entry(frame, textvariable=self.custom_audio_path, width=45)
        self.custom_audio_entry.grid(row=2, column=1, **pad)
        self.custom_audio_btn = tk.Button(frame, text="Sfoglia", command=lambda: self._browse_custom_audio())
        self.custom_audio_btn.grid(row=2, column=2, **pad)
        self.use_video_audio_check = tk.Checkbutton(frame, text="Usa audio del video", 
                                                     variable=self.use_video_audio,
                                                     command=self._toggle_video_audio)
        self.use_video_audio_check.grid(row=2, column=3, sticky="w", **pad)
        
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
        
        # Native resolution checkbox
        tk.Checkbutton(frame, text="Usa dimensioni native dell'immagine/video", 
                      variable=self.use_native_resolution).grid(row=4, column=3, sticky="w", **pad)
        
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
        
        # Logo overlay section
        tk.Label(frame, text="Logo Overlay (opzionale):", font=("Arial", 9, "bold")).grid(
            row=11, column=0, sticky="w", **pad
        )
        
        tk.Label(frame, text="Logo:").grid(row=12, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.logo_path, width=35).grid(row=12, column=1, **pad)
        tk.Button(frame, text="Sfoglia", command=self._browse_logo).grid(row=12, column=2, **pad)
        
        tk.Label(frame, text="Posizione:").grid(row=13, column=0, sticky="e", **pad)
        logo_pos_combo = ttk.Combobox(
            frame,
            textvariable=self.logo_position,
            values=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
            state="readonly",
            width=12
        )
        logo_pos_combo.grid(row=13, column=1, sticky="w", **pad)
        
        tk.Label(frame, text="Scala:").grid(row=13, column=2, sticky="e", padx=(20,5))
        tk.Scale(frame, variable=self.logo_scale, from_=0.05, to=0.5, resolution=0.05,
                orient="horizontal", length=100).grid(row=13, column=3, sticky="w", **pad)
        
        tk.Label(frame, text="Opacità:").grid(row=14, column=0, sticky="e", **pad)
        tk.Scale(frame, variable=self.logo_opacity, from_=0.0, to=1.0, resolution=0.1,
                orient="horizontal", length=150).grid(row=14, column=1, sticky="w", **pad)
        
        tk.Label(frame, text="Margine:").grid(row=14, column=2, sticky="e", padx=(20,5))
        tk.Spinbox(frame, from_=0, to=100, textvariable=self.logo_margin, width=10).grid(
            row=14, column=3, sticky="w", **pad
        )
        
        # Separator
        sep3 = ttk.Separator(frame, orient="horizontal")
        sep3.grid(row=15, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 2))
        
        # Effects section
        tk.Label(frame, text="Effetti (seleziona e imposta intensità):", font=("Arial", 9, "bold")).grid(
            row=16, column=0, columnspan=2, sticky="w", **pad
        )
        
        # Create scrollable frame for effects
        effects_frame = tk.Frame(frame)
        effects_frame.grid(row=17, column=0, columnspan=4, sticky="ew", **pad)
        
        # Effect checkboxes with intensity sliders
        effects = [
            ("ColorPulse", self.effect_color_pulse, self.intensity_color_pulse),
            ("ZoomPulse", self.effect_zoom_pulse, self.intensity_zoom_pulse),
            ("Strobe", self.effect_strobe, self.intensity_strobe),
            ("StrobeNegative", self.effect_strobe_negative, self.intensity_strobe_negative),
            ("Glitch", self.effect_glitch, self.intensity_glitch),
            ("ChromaticAberration", self.effect_chromatic, self.intensity_chromatic),
            ("BubbleDistortion", self.effect_bubble, self.intensity_bubble),
            ("ScreenShake", self.effect_screen_shake, self.intensity_screen_shake),
            ("RGBSplit", self.effect_rgb_split, self.intensity_rgb_split),
            ("ElectricArcs", self.effect_electric_arcs, self.intensity_electric_arcs),
            ("FashionLightning", self.effect_fashion_lightning, self.intensity_fashion_lightning),
            ("AdvancedGlitch", self.effect_advanced_glitch, self.intensity_advanced_glitch),
            ("DimensionalWarp", self.effect_dimensional_warp, self.intensity_dimensional_warp),
            ("VortexDistortion", self.effect_vortex_distortion, self.intensity_vortex_distortion),
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
        self.custom_progress.grid(row=18, column=0, columnspan=4, **pad)
        self.custom_status_lbl = tk.Label(frame, text="Pronto")
        self.custom_status_lbl.grid(row=19, column=0, columnspan=4, sticky="w", **pad)
        
        # Actions
        self.custom_run_btn = tk.Button(frame, text="Genera Video Custom", command=self.on_custom_run)
        self.custom_run_btn.grid(row=20, column=2, sticky="e", **pad)
        self.custom_cancel_btn = tk.Button(frame, text="Annulla", command=self.on_cancel, state=tk.DISABLED)
        self.custom_cancel_btn.grid(row=20, column=3, sticky="w", **pad)
    
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
    
    def _browse_logo(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleziona logo",
            filetypes=[("Immagini", "*.png *.jpg *.jpeg"), ("Tutti i file", "*.*")],
        )
        if path:
            self.logo_path.set(path)
    
    def _toggle_video_audio(self) -> None:
        """Toggle audio input controls based on use_video_audio checkbox."""
        if self.use_video_audio.get():
            # Disable audio file selection when using video audio
            self.custom_audio_entry.config(state=tk.DISABLED)
            self.custom_audio_btn.config(state=tk.DISABLED)
            self.custom_audio_path.set("(audio dal video)")
        else:
            # Enable audio file selection
            self.custom_audio_entry.config(state=tk.NORMAL)
            self.custom_audio_btn.config(state=tk.NORMAL)
            if self.custom_audio_path.get() == "(audio dal video)":
                self.custom_audio_path.set("")
    
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
            
            # Hide "use video audio" checkbox in image mode
            self.use_video_audio_check.grid_forget()
            self.use_video_audio.set(False)
            self._toggle_video_audio()
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
            
            # Show "use video audio" checkbox in video mode
            self.use_video_audio_check.grid(row=2, column=3, sticky="w", **pad)
    
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
        use_video_audio = self.use_video_audio.get()
        
        # Validate audio input
        if mode == "video" and use_video_audio:
            # Will extract audio from video later
            audio = None
        else:
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
            self.effect_strobe_negative.get(),
            self.effect_glitch.get(),
            self.effect_chromatic.get(),
            self.effect_bubble.get(),
            self.effect_screen_shake.get(),
            self.effect_rgb_split.get(),
            self.effect_electric_arcs.get(),
            self.effect_fashion_lightning.get(),
            self.effect_advanced_glitch.get(),
            self.effect_dimensional_warp.get(),
            self.effect_vortex_distortion.get(),
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
                    ColorPulseEffect, ZoomPulseEffect, StrobeEffect, StrobeNegativeEffect, GlitchEffect,
                    ChromaticAberrationEffect, BubbleDistortionEffect, ScreenShakeEffect, RGBSplitEffect,
                    ElectricArcsEffect, FashionLightningEffect, AdvancedGlitchEffect, DimensionalWarpEffect,
                    VortexDistortionEffect
                )
                from src.factories import EffectFactory
                
                # Build custom effect list
                custom_effects = []

                # Preserve original audio path into a local variable used below.
                # We must not reassign the outer name `audio` inside this worker,
                # otherwise Python treats it as a local and raises UnboundLocalError.
                audio_source = audio
                
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
                
                if self.effect_strobe_negative.get():
                    custom_effects.append(StrobeNegativeEffect(
                        threshold=0.8,
                        intensity=self.intensity_strobe_negative.get()
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
                
                if self.effect_fashion_lightning.get():
                    # Fashion colors for lightning - more vibrant and trendy
                    fashion_colors = [(1.0, 0.0, 0.8), (0.0, 0.9, 1.0), (0.8, 1.0, 0.0)]
                    custom_effects.append(FashionLightningEffect(
                        colors=fashion_colors,
                        threshold=0.65,
                        branching_probability=0.6,
                        max_branches=5,
                        segment_length_min=5,
                        segment_length_max=20,
                        intensity=self.intensity_fashion_lightning.get()
                    ))
                
                if self.effect_advanced_glitch.get():
                    custom_effects.append(AdvancedGlitchEffect(
                        threshold=0.5,
                        channel_shift_amount=8,
                        block_size_range=(10, 80),
                        intensity=self.intensity_advanced_glitch.get()
                    ))
                
                if self.effect_dimensional_warp.get():
                    custom_effects.append(DimensionalWarpEffect(
                        bass_threshold=self.threshold_bass.get(),
                        mid_threshold=self.threshold_mid.get(),
                        warp_strength=45.0,
                        rotation_speed=0.5,
                        perspective_depth=200.0,
                        wave_frequency=2.0,
                        layer_count=3,
                        intensity=self.intensity_dimensional_warp.get()
                    ))
                
                if self.effect_vortex_distortion.get():
                    custom_effects.append(VortexDistortionEffect(
                        threshold=0.2,
                        max_angle=35.0,
                        radius_falloff=1.8,
                        rotation_speed=3.0,
                        smoothing=0.3,
                        intensity=self.intensity_vortex_distortion.get()
                    ))
                
                # Create custom pipeline
                custom_pipeline = EffectFactory.create_custom_pipeline(custom_effects)
                
                if mode == "image":
                    # Image mode: optimized streaming implementation
                    from src.core.audio_analyzer import AudioAnalyzer
                    from src.core.frame_generator import FrameGenerator
                    from src.utils.image_loader import ImageLoader
                    from src.utils.logo_overlay import load_logo, apply_logo_to_frame
                    import cv2
                    import subprocess
                    import os
                    import tempfile
                    import shutil
                    
                    # Analyze audio
                    progress_cb("status", {"message": "Analisi audio..."})
                    audio_analyzer = AudioAnalyzer()
                    audio_data = audio_analyzer.load_and_analyze(audio_source, duration=None, fps=fps)
                    
                    # Load image - use native resolution if checkbox is selected
                    progress_cb("status", {"message": "Caricamento immagine..."})
                    target_res = None if self.use_native_resolution.get() else (720, 720)
                    base_image = ImageLoader.load_and_prepare(image, target_resolution=target_res)
                    height, width = base_image.shape[:2]
                    progress_cb("status", {"message": f"Risoluzione immagine: {width}x{height}"})
                    
                    # Create frame generator with custom pipeline
                    progress_cb("status", {"message": "Setup effetti custom..."})
                    frame_generator = FrameGenerator(
                        base_image=base_image,
                        effect_pipeline=custom_pipeline,
                        fps=fps
                    )
                    
                    # Generate frames with streaming
                    num_frames = len(audio_data.bass_energy)
                    progress_cb("start", {"total_frames": num_frames})
                    progress_cb("status", {"message": "Processing con streaming..."})
                    
                    # Load logo if provided
                    logo_img = None
                    logo_pos = self.logo_position.get()
                    logo_sc = self.logo_scale.get()
                    logo_op = self.logo_opacity.get()
                    logo_mg = self.logo_margin.get()
                    
                    logo_path_str = self.logo_path.get().strip()
                    if logo_path_str and os.path.exists(logo_path_str):
                        try:
                            logo_img = load_logo(logo_path_str)
                            progress_cb("status", {"message": f"Logo caricato: {os.path.basename(logo_path_str)}"})
                        except Exception as e:
                            progress_cb("status", {"message": f"Errore caricamento logo: {e}"})
                    
                    # Write directly to temporary video file (use AVI for temp, then convert)
                    import tempfile
                    temp_video_avi = tempfile.mktemp(suffix='_temp.avi', dir=os.path.dirname(output))
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    writer = cv2.VideoWriter(temp_video_avi, fourcc, fps, (width, height))
                    
                    for idx in range(num_frames):
                        progress_cb("frame", {"index": idx + 1, "total": num_frames})
                        
                        frame = frame_generator.generate_frame(
                            frame_index=idx,
                            audio_analysis=audio_data,
                            color_index=idx % 6
                        )
                        
                        # Apply logo if available
                        if logo_img is not None:
                            frame = apply_logo_to_frame(frame, logo_img, logo_pos, logo_sc, logo_op, logo_mg)
                        
                        # Write directly (convert RGB to BGR for cv2)
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        
                        # Free memory immediately
                        del frame
                    
                    writer.release()
                    
                    # Add audio with ffmpeg or moviepy
                    progress_cb("status", {"message": "Aggiunta audio al video..."})
                    
                    audio_added = False
                    
                    # Try ffmpeg first (fastest)
                    try:
                        result = subprocess.run([
                            'ffmpeg', '-y', '-i', temp_video_avi, '-i', audio_source,
                            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', output
                        ], check=True, capture_output=True)
                        os.remove(temp_video_avi)
                        audio_added = True
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        # Fallback to moviepy
                        try:
                            # Try MoviePy 2.x import first
                            try:
                                from moviepy import VideoFileClip, AudioFileClip
                            except ImportError:
                                from moviepy.editor import VideoFileClip, AudioFileClip
                            
                            progress_cb("status", {"message": "Aggiunta audio con moviepy..."})
                            video_clip = VideoFileClip(temp_video_avi)
                            audio_clip = AudioFileClip(audio_source)
                            final_clip = video_clip.with_audio(audio_clip)
                            final_clip.write_videofile(output, codec='libx264', audio_codec='aac', logger=None)
                            video_clip.close()
                            audio_clip.close()
                            final_clip.close()
                            os.remove(temp_video_avi)
                            audio_added = True
                        except Exception as e2:
                            # Last resort: convert video without audio
                            try:
                                subprocess.run(['ffmpeg', '-y', '-i', temp_video_avi, '-c:v', 'libx264', output], check=False)
                                os.remove(temp_video_avi)
                            except:
                                pass
                    
                    progress_cb("done", {"output": output})
                else:
                    # Video mode: manual implementation with custom pipeline
                    from src.video_audio_sync import VideoAudioSync
                    from src.models.data_models import EffectConfig
                    import cv2
                    import numpy as np
                    from src.core.audio_analyzer import AudioAnalyzer
                    from src.core.frame_generator import FrameGenerator
                    from src.core.video_exporter import VideoExporter
                    from src.utils.logo_overlay import load_logo, apply_logo_to_frame
                    import subprocess
                    import os
                    import tempfile
                    import shutil
                    
                    # Get video metadata without loading all frames
                    progress_cb("status", {"message": "Analisi video..."})
                    cap = cv2.VideoCapture(video)
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    available_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if available_frames == 0:
                        cap.release()
                        raise ValueError("Impossibile leggere il video")
                    
                    # Extract or use provided audio
                    if use_video_audio:
                        # Extract audio from video
                        progress_cb("status", {"message": "Estrazione audio dal video..."})
                        temp_audio = tempfile.mktemp(suffix='.wav', dir=os.path.dirname(output))
                        try:
                            # Try ffmpeg first
                            subprocess.run([
                                'ffmpeg', '-y', '-i', video, '-vn', '-acodec', 'pcm_s16le', 
                                '-ar', '44100', '-ac', '2', temp_audio
                            ], check=True, capture_output=True)
                            audio_source = temp_audio
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            # Fallback to moviepy
                            try:
                                try:
                                    from moviepy import VideoFileClip
                                except ImportError:
                                    from moviepy.editor import VideoFileClip
                                
                                video_clip = VideoFileClip(video)
                                video_clip.audio.write_audiofile(temp_audio, logger=None)
                                video_clip.close()
                                audio_source = temp_audio
                            except Exception as e:
                                if os.path.exists(temp_audio):
                                    os.remove(temp_audio)
                                raise ValueError(f"Impossibile estrarre audio dal video: {e}")
                    
                    # Load and analyze audio
                    progress_cb("status", {"message": "Analisi audio..."})
                    audio_analyzer = AudioAnalyzer()
                    audio_data = audio_analyzer.load_and_analyze(audio_source, fps=int(video_fps))
                    audio_duration = len(audio_data.audio_signal) / audio_data.sample_rate
                    
                    # Calculate frame mapping (without loading frames)
                    required_frames = int(audio_duration * video_fps)
                    
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
                    
                    # Process video with streaming (frame by frame)
                    total_frames = len(frame_indices)
                    progress_cb("start", {"total_frames": total_frames})
                    progress_cb("status", {"message": "Processing video con streaming..."})
                    
                    # Load logo if provided
                    logo_img = None
                    logo_pos = self.logo_position.get()
                    logo_sc = self.logo_scale.get()
                    logo_op = self.logo_opacity.get()
                    logo_mg = self.logo_margin.get()
                    
                    logo_path_str = self.logo_path.get().strip()
                    if logo_path_str and os.path.exists(logo_path_str):
                        try:
                            logo_img = load_logo(logo_path_str)
                            progress_cb("status", {"message": f"Logo caricato: {os.path.basename(logo_path_str)}"})
                        except Exception as e:
                            progress_cb("status", {"message": f"Errore caricamento logo: {e}"})
                    
                    # Initialize video writer for direct streaming (use AVI temp)
                    temp_video_avi = tempfile.mktemp(suffix='_temp.avi', dir=os.path.dirname(output))
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    writer = cv2.VideoWriter(temp_video_avi, fourcc, int(video_fps), (width, height))
                    
                    # Cache for loop mode - store only necessary frames
                    frame_cache = {}
                    last_frame_idx = -1
                    cached_frame = None
                    
                    for idx, frame_idx in enumerate(frame_indices):
                        progress_cb("frame", {"index": idx + 1, "total": total_frames})
                        
                        # Load frame only if needed (optimize for sequential and loop access)
                        if frame_idx in frame_cache:
                            # Get from cache (for loop mode)
                            base_frame = frame_cache[frame_idx].copy()
                        elif frame_idx == last_frame_idx and cached_frame is not None:
                            # Reuse last frame (for stretch mode with repeated frames)
                            base_frame = cached_frame.copy()
                        else:
                            # Seek and read frame from video
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                            if not ret:
                                # Fallback to last cached frame or black frame
                                if cached_frame is not None:
                                    base_frame = cached_frame.copy()
                                else:
                                    base_frame = np.zeros((height, width, 3), dtype=np.uint8)
                            else:
                                base_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Cache frame if in loop mode and not too many cached
                                if short_mode == "loop" and len(frame_cache) < 300:
                                    frame_cache[frame_idx] = base_frame.copy()
                        
                        # Update cache for stretch mode
                        cached_frame = base_frame.copy()
                        last_frame_idx = frame_idx
                        
                        # Apply effects directly on the current frame
                        from src.models.data_models import FrameContext
                        
                        # Get audio features for this frame
                        time = idx / video_fps
                        bass = audio_data.bass_energy[idx] if idx < len(audio_data.bass_energy) else 0.0
                        mid = audio_data.mid_energy[idx] if idx < len(audio_data.mid_energy) else 0.0
                        treble = audio_data.treble_energy[idx] if idx < len(audio_data.treble_energy) else 0.0
                        
                        # Calculate beat intensity
                        beat_intensity = 0.0
                        for beat_time in audio_data.beat_times:
                            if abs(beat_time - time) < 0.1:
                                beat_intensity = 1.0
                                break
                        
                        # Create frame context for this frame
                        frame_context = FrameContext(
                            frame=base_frame,
                            time=time,
                            frame_index=idx,
                            bass=bass,
                            mid=mid,
                            treble=treble,
                            beat_intensity=beat_intensity,
                            color_index=0
                        )
                        
                        # Apply custom pipeline effects
                        frame_with_effects = custom_pipeline.apply(frame_context)
                        
                        # Apply logo if available
                        if logo_img is not None:
                            frame_with_effects = apply_logo_to_frame(frame_with_effects, logo_img, logo_pos, logo_sc, logo_op, logo_mg)
                        
                        # Write directly to video (convert back to BGR)
                        writer.write(cv2.cvtColor(frame_with_effects, cv2.COLOR_RGB2BGR))
                        
                        # Clear frame to free memory
                        del base_frame
                        del frame_with_effects
                    
                    # Release resources
                    cap.release()
                    writer.release()
                    frame_cache.clear()
                    
                    # Add audio to video
                    progress_cb("status", {"message": "Aggiunta audio al video..."})
                    
                    audio_added = False
                    
                    # Try ffmpeg first (fastest)
                    try:
                        result = subprocess.run([
                            'ffmpeg', '-y', '-i', temp_video_avi, '-i', audio_source,
                            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', output
                        ], check=True, capture_output=True)
                        os.remove(temp_video_avi)
                        audio_added = True
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        # Fallback to moviepy
                        try:
                            # Try MoviePy 2.x import first
                            try:
                                from moviepy import VideoFileClip, AudioFileClip
                            except ImportError:
                                from moviepy.editor import VideoFileClip, AudioFileClip
                            
                            progress_cb("status", {"message": "Aggiunta audio con moviepy..."})
                            video_clip = VideoFileClip(temp_video_avi)
                            audio_clip = AudioFileClip(audio_source)
                            final_clip = video_clip.with_audio(audio_clip)
                            final_clip.write_videofile(output, codec='libx264', audio_codec='aac', logger=None)
                            video_clip.close()
                            audio_clip.close()
                            final_clip.close()
                            os.remove(temp_video_avi)
                            audio_added = True
                        except Exception as e2:
                            # Last resort: convert video without audio
                            try:
                                subprocess.run(['ffmpeg', '-y', '-i', temp_video_avi, '-c:v', 'libx264', output], check=False)
                                os.remove(temp_video_avi)
                            except:
                                pass
                    
                    progress_cb("done", {"output": output})
                    
                    # Clean up temporary audio file if extracted
                    if use_video_audio and 'temp_audio' in locals() and os.path.exists(temp_audio):
                        try:
                            os.remove(temp_audio)
                        except:
                            pass
                
                self.after(0, lambda: messagebox.showinfo("Completato", f"Video custom creato: {output}"))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda msg=error_msg: messagebox.showerror("Errore", msg))
                # Clean up temporary audio file on error
                if mode == "video" and use_video_audio and 'temp_audio' in locals() and os.path.exists(temp_audio):
                    try:
                        os.remove(temp_audio)
                    except:
                        pass
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
