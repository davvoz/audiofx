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
        self.custom_duration = tk.StringVar(value="")  # Empty = full audio duration
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
        self.effect_floating_text = tk.BooleanVar(value=False)
        
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
        self.intensity_floating_text = tk.DoubleVar(value=1.0)
        
        # Floating text configuration
        self.floating_text_content = tk.StringVar(value="MUSIC")
        self.floating_text_color_scheme = tk.StringVar(value="rainbow")
        self.floating_text_animation = tk.StringVar(value="wave")
        self.floating_text_font_size = tk.IntVar(value=120)
        
        # Effect order management
        self.effect_order = [
            "ColorPulse", "ZoomPulse", "Strobe", "StrobeNegative", "Glitch",
            "ChromaticAberration", "BubbleDistortion", "ScreenShake", "RGBSplit",
            "ElectricArcs", "FashionLightning", "AdvancedGlitch", 
            "DimensionalWarp", "VortexDistortion", "FloatingText"
        ]
        
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
        
        # Duration control
        tk.Label(frame, text="Durata (sec):").grid(row=5, column=2, sticky="e", padx=(20,5))
        duration_entry = tk.Entry(frame, textvariable=self.custom_duration, width=10)
        duration_entry.grid(row=5, column=3, sticky="w", **pad)
        tk.Label(frame, text="(vuoto = durata completa)", font=("Arial", 7)).grid(row=6, column=3, sticky="w", padx=(10,0))
        
        # Initialize visibility based on mode
        self._toggle_custom_mode()
        
        # Separator
        sep = ttk.Separator(frame, orient="horizontal")
        sep.grid(row=7, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 2))
        
        # Thresholds section
        tk.Label(frame, text="Soglie Audio:", font=("Arial", 9, "bold")).grid(row=8, column=0, sticky="w", **pad)
        
        tk.Label(frame, text="Bass:").grid(row=9, column=0, sticky="e", **pad)
        tk.Scale(frame, variable=self.threshold_bass, from_=0.0, to=1.0, resolution=0.01, 
                orient="horizontal", length=150).grid(row=9, column=1, sticky="w", **pad)
        
        tk.Label(frame, text="Mid:").grid(row=9, column=2, sticky="e", padx=(20,5))
        tk.Scale(frame, variable=self.threshold_mid, from_=0.0, to=1.0, resolution=0.01, 
                orient="horizontal", length=150).grid(row=9, column=3, sticky="w", **pad)
        
        tk.Label(frame, text="High:").grid(row=10, column=0, sticky="e", **pad)
        tk.Scale(frame, variable=self.threshold_high, from_=0.0, to=1.0, resolution=0.01, 
                orient="horizontal", length=150).grid(row=10, column=1, sticky="w", **pad)
        
        # Separator
        sep2 = ttk.Separator(frame, orient="horizontal")
        sep2.grid(row=11, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 2))
        
        # Logo overlay section
        tk.Label(frame, text="Logo Overlay (opzionale):", font=("Arial", 9, "bold")).grid(
            row=12, column=0, sticky="w", **pad
        )
        
        tk.Label(frame, text="Logo:").grid(row=13, column=0, sticky="e", **pad)
        tk.Entry(frame, textvariable=self.logo_path, width=35).grid(row=13, column=1, **pad)
        tk.Button(frame, text="Sfoglia", command=self._browse_logo).grid(row=13, column=2, **pad)
        
        tk.Label(frame, text="Posizione:").grid(row=14, column=0, sticky="e", **pad)
        logo_pos_combo = ttk.Combobox(
            frame,
            textvariable=self.logo_position,
            values=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
            state="readonly",
            width=12
        )
        logo_pos_combo.grid(row=14, column=1, sticky="w", **pad)
        
        tk.Label(frame, text="Scala:").grid(row=14, column=2, sticky="e", padx=(20,5))
        tk.Scale(frame, variable=self.logo_scale, from_=0.05, to=0.5, resolution=0.05,
                orient="horizontal", length=100).grid(row=14, column=3, sticky="w", **pad)
        
        tk.Label(frame, text="Opacità:").grid(row=15, column=0, sticky="e", **pad)
        tk.Scale(frame, variable=self.logo_opacity, from_=0.0, to=1.0, resolution=0.1,
                orient="horizontal", length=150).grid(row=15, column=1, sticky="w", **pad)
        
        tk.Label(frame, text="Margine:").grid(row=15, column=2, sticky="e", padx=(20,5))
        tk.Spinbox(frame, from_=0, to=100, textvariable=self.logo_margin, width=10).grid(
            row=15, column=3, sticky="w", **pad
        )
        
        # Separator
        sep3 = ttk.Separator(frame, orient="horizontal")
        sep3.grid(row=16, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 2))
        
        # Effects section
        tk.Label(frame, text="Effetti (seleziona e imposta intensità):", font=("Arial", 9, "bold")).grid(
            row=17, column=0, columnspan=2, sticky="w", **pad
        )
        
        # Create scrollable frame for effects
        effects_frame = tk.Frame(frame)
        effects_frame.grid(row=18, column=0, columnspan=4, sticky="ew", **pad)
        
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
            ("FloatingText", self.effect_floating_text, self.intensity_floating_text),
        ]
        
        for idx, (name, var, intensity_var) in enumerate(effects):
            row_offset = idx // 2
            col_offset = (idx % 2) * 2
            
            tk.Checkbutton(effects_frame, text=name, variable=var, 
                          command=self._refresh_order_listbox).grid(
                row=row_offset, column=col_offset, sticky="w", padx=5, pady=2
            )
            tk.Scale(effects_frame, variable=intensity_var, from_=0.0, to=2.0, resolution=0.1,
                    orient="horizontal", length=100, label="").grid(
                row=row_offset, column=col_offset + 1, sticky="w", padx=5, pady=2
            )
        
        # Separator
        sep4 = ttk.Separator(frame, orient="horizontal")
        sep4.grid(row=17, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 2))
        
        # Floating Text Configuration (shown when FloatingText is enabled)
        tk.Label(frame, text="Configurazione Floating Text:", font=("Arial", 9, "bold")).grid(
            row=18, column=0, columnspan=2, sticky="w", **pad
        )
        
        floating_config_frame = tk.Frame(frame)
        floating_config_frame.grid(row=19, column=0, columnspan=4, sticky="ew", **pad)
        
        tk.Label(floating_config_frame, text="Testo:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        tk.Entry(floating_config_frame, textvariable=self.floating_text_content, width=20).grid(
            row=0, column=1, sticky="w", padx=5, pady=2
        )
        
        tk.Label(floating_config_frame, text="Colori:").grid(row=0, column=2, sticky="e", padx=(20, 5), pady=2)
        color_combo = ttk.Combobox(
            floating_config_frame,
            textvariable=self.floating_text_color_scheme,
            values=["rainbow", "fire", "ice", "neon", "gold", "default"],
            state="readonly",
            width=12
        )
        color_combo.grid(row=0, column=3, sticky="w", padx=5, pady=2)
        
        tk.Label(floating_config_frame, text="Animazione:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        anim_combo = ttk.Combobox(
            floating_config_frame,
            textvariable=self.floating_text_animation,
            values=["wave", "bounce", "spin", "pulse", "glitch"],
            state="readonly",
            width=12
        )
        anim_combo.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        tk.Label(floating_config_frame, text="Dimensione Font:").grid(row=1, column=2, sticky="e", padx=(20, 5), pady=2)
        tk.Spinbox(floating_config_frame, from_=50, to=300, textvariable=self.floating_text_font_size, 
                  width=10).grid(row=1, column=3, sticky="w", padx=5, pady=2)
        
        # Separator
        sep5 = ttk.Separator(frame, orient="horizontal")
        sep5.grid(row=19, column=0, columnspan=4, sticky="ew", padx=10, pady=(8, 2))
        
        # Effect Order Section
        tk.Label(frame, text="Ordine Effetti (importante!):", font=("Arial", 9, "bold")).grid(
            row=20, column=0, columnspan=2, sticky="w", **pad
        )
        tk.Label(frame, text="L'ordine cambia il risultato finale", font=("Arial", 8, "italic"), 
                fg="gray").grid(row=20, column=2, columnspan=2, sticky="w", **pad)
        
        # Frame for effect order management
        order_frame = tk.Frame(frame)
        order_frame.grid(row=21, column=0, columnspan=4, sticky="ew", **pad)
        
        # Listbox to show and reorder effects
        tk.Label(order_frame, text="Ordine corrente:").grid(row=0, column=0, sticky="nw", padx=5)
        
        self.order_listbox = tk.Listbox(order_frame, height=8, width=25)
        self.order_listbox.grid(row=1, column=0, rowspan=4, padx=5, pady=5)
        
        # Nota: la listbox verrà popolata dopo con _refresh_order_listbox()
        # per mostrare solo gli effetti selezionati
        
        # Buttons to reorder
        btn_frame = tk.Frame(order_frame)
        btn_frame.grid(row=1, column=1, padx=5, sticky="n")
        
        tk.Button(btn_frame, text="▲ Su", width=10, command=self._move_effect_up).pack(pady=2)
        tk.Button(btn_frame, text="▼ Giù", width=10, command=self._move_effect_down).pack(pady=2)
        tk.Button(btn_frame, text="⬆ In cima", width=10, command=self._move_effect_top).pack(pady=2)
        tk.Button(btn_frame, text="⬇ In fondo", width=10, command=self._move_effect_bottom).pack(pady=2)
        tk.Button(btn_frame, text="↺ Reset", width=10, command=self._reset_effect_order).pack(pady=2)
        
        # Tips
        tips_frame = tk.Frame(order_frame)
        tips_frame.grid(row=1, column=2, rowspan=4, padx=10, sticky="n")
        
        tk.Label(tips_frame, text="💡 Suggerimenti:", font=("Arial", 8, "bold")).pack(anchor="w")
        tips_text = """
• FloatingText alla fine = testo sopra
• FloatingText all'inizio = effetti sopra testo
• Glitch prima = sfondo distorto
• ColorPulse all'inizio = base colorata
• ScreenShake alla fine = tutto si muove
        """
        tk.Label(tips_frame, text=tips_text, font=("Arial", 7), justify=tk.LEFT, 
                fg="darkblue").pack(anchor="w")
        
        # Progress
        self.custom_progress = ttk.Progressbar(frame, mode="determinate", length=620)
        self.custom_progress.grid(row=22, column=0, columnspan=4, **pad)
        self.custom_status_lbl = tk.Label(frame, text="Pronto")
        self.custom_status_lbl.grid(row=23, column=0, columnspan=4, sticky="w", **pad)
        
        # Actions
        self.custom_run_btn = tk.Button(frame, text="Genera Video Custom", command=self.on_custom_run)
        self.custom_run_btn.grid(row=24, column=2, sticky="e", **pad)
        self.custom_cancel_btn = tk.Button(frame, text="Annulla", command=self.on_cancel, state=tk.DISABLED)
        self.custom_cancel_btn.grid(row=24, column=3, sticky="w", **pad)
        
        # Popola la listbox ordine effetti con gli effetti inizialmente selezionati
        self._refresh_order_listbox()
    
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
    
    def _move_effect_up(self) -> None:
        """Sposta l'effetto selezionato su di una posizione"""
        selection = self.order_listbox.curselection()
        if not selection or selection[0] == 0:
            return
        
        # Ottieni l'effetto selezionato dalla listbox (solo quelli visibili)
        selected_effect = self.order_listbox.get(selection[0])
        
        # Trova la posizione nell'array completo
        idx_in_full_list = self.effect_order.index(selected_effect)
        
        if idx_in_full_list == 0:
            return
        
        # Swap nell'array completo
        self.effect_order[idx_in_full_list], self.effect_order[idx_in_full_list-1] = \
            self.effect_order[idx_in_full_list-1], self.effect_order[idx_in_full_list]
        
        # Aggiorna listbox
        self._refresh_order_listbox()
        
        # Riseleziona l'elemento (se ancora visibile)
        new_idx = max(0, selection[0] - 1)
        if new_idx < self.order_listbox.size():
            self.order_listbox.selection_set(new_idx)
    
    def _move_effect_down(self) -> None:
        """Sposta l'effetto selezionato giù di una posizione"""
        selection = self.order_listbox.curselection()
        if not selection:
            return
        
        # Ottieni l'effetto selezionato
        selected_effect = self.order_listbox.get(selection[0])
        
        # Trova la posizione nell'array completo
        idx_in_full_list = self.effect_order.index(selected_effect)
        
        if idx_in_full_list >= len(self.effect_order) - 1:
            return
        
        # Swap nell'array completo
        self.effect_order[idx_in_full_list], self.effect_order[idx_in_full_list+1] = \
            self.effect_order[idx_in_full_list+1], self.effect_order[idx_in_full_list]
        
        # Aggiorna listbox
        self._refresh_order_listbox()
        
        # Riseleziona l'elemento
        new_idx = min(selection[0] + 1, self.order_listbox.size() - 1)
        if new_idx >= 0:
            self.order_listbox.selection_set(new_idx)
    
    def _move_effect_top(self) -> None:
        """Sposta l'effetto selezionato in cima"""
        selection = self.order_listbox.curselection()
        if not selection:
            return
        
        # Ottieni l'effetto selezionato
        selected_effect = self.order_listbox.get(selection[0])
        
        # Trova e sposta nell'array completo
        idx_in_full_list = self.effect_order.index(selected_effect)
        effect = self.effect_order.pop(idx_in_full_list)
        self.effect_order.insert(0, effect)
        
        # Aggiorna listbox
        self._refresh_order_listbox()
        self.order_listbox.selection_set(0)
    
    def _move_effect_bottom(self) -> None:
        """Sposta l'effetto selezionato in fondo"""
        selection = self.order_listbox.curselection()
        if not selection:
            return
        
        # Ottieni l'effetto selezionato
        selected_effect = self.order_listbox.get(selection[0])
        
        # Trova e sposta nell'array completo
        idx_in_full_list = self.effect_order.index(selected_effect)
        effect = self.effect_order.pop(idx_in_full_list)
        self.effect_order.append(effect)
        
        # Aggiorna listbox
        self._refresh_order_listbox()
        
        # Seleziona l'ultimo elemento visibile
        last_idx = self.order_listbox.size() - 1
        if last_idx >= 0:
            self.order_listbox.selection_set(last_idx)
    
    def _reset_effect_order(self) -> None:
        """Ripristina l'ordine predefinito degli effetti"""
        self.effect_order = [
            "ColorPulse", "ZoomPulse", "Strobe", "StrobeNegative", "Glitch",
            "ChromaticAberration", "BubbleDistortion", "ScreenShake", "RGBSplit",
            "ElectricArcs", "FashionLightning", "AdvancedGlitch", 
            "DimensionalWarp", "VortexDistortion", "FloatingText"
        ]
        self._refresh_order_listbox()
    
    def _refresh_order_listbox(self) -> None:
        """Aggiorna la listbox con l'ordine corrente (solo effetti selezionati)"""
        self.order_listbox.delete(0, tk.END)
        
        # Mappa effetti con le loro checkbox
        effect_enabled_map = {
            "ColorPulse": self.effect_color_pulse,
            "ZoomPulse": self.effect_zoom_pulse,
            "Strobe": self.effect_strobe,
            "StrobeNegative": self.effect_strobe_negative,
            "Glitch": self.effect_glitch,
            "ChromaticAberration": self.effect_chromatic,
            "BubbleDistortion": self.effect_bubble,
            "ScreenShake": self.effect_screen_shake,
            "RGBSplit": self.effect_rgb_split,
            "ElectricArcs": self.effect_electric_arcs,
            "FashionLightning": self.effect_fashion_lightning,
            "AdvancedGlitch": self.effect_advanced_glitch,
            "DimensionalWarp": self.effect_dimensional_warp,
            "VortexDistortion": self.effect_vortex_distortion,
            "FloatingText": self.effect_floating_text,
        }
        
        # Mostra solo gli effetti selezionati
        for effect in self.effect_order:
            if effect in effect_enabled_map and effect_enabled_map[effect].get():
                self.order_listbox.insert(tk.END, effect)
    
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
            self.effect_floating_text.get(),
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
                    VortexDistortionEffect, FloatingText
                )
                from src.factories import EffectFactory
                
                # Build custom effect list
                custom_effects = []

                # Preserve original audio path into a local variable used below.
                # We must not reassign the outer name `audio` inside this worker,
                # otherwise Python treats it as a local and raises UnboundLocalError.
                audio_source = audio
                
                # Get custom duration if specified
                custom_duration = None
                duration_str = self.custom_duration.get().strip()
                if duration_str:
                    try:
                        custom_duration = float(duration_str)
                        if custom_duration <= 0:
                            raise ValueError("La durata deve essere maggiore di 0")
                        progress_cb("status", {"message": f"Usando durata personalizzata: {custom_duration} secondi"})
                    except ValueError as e:
                        self.after(0, lambda msg=str(e): messagebox.showerror("Errore", f"Durata non valida: {msg}"))
                        return
                else:
                    progress_cb("status", {"message": "Usando durata completa dell'audio"})
                
                # Crea dizionario con tutti gli effetti disponibili
                effects_map = {
                    "ColorPulse": lambda: ColorPulseEffect(
                        bass_threshold=self.threshold_bass.get(),
                        mid_threshold=self.threshold_mid.get(),
                        high_threshold=self.threshold_high.get(),
                        intensity=self.intensity_color_pulse.get()
                    ) if self.effect_color_pulse.get() else None,
                    
                    "ZoomPulse": lambda: ZoomPulseEffect(
                        threshold=self.threshold_bass.get(),
                        intensity=self.intensity_zoom_pulse.get()
                    ) if self.effect_zoom_pulse.get() else None,
                    
                    "Strobe": lambda: StrobeEffect(
                        colors=[(1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0)],
                        threshold=0.8,
                        intensity=self.intensity_strobe.get()
                    ) if self.effect_strobe.get() else None,
                    
                    "StrobeNegative": lambda: StrobeNegativeEffect(
                        threshold=0.8,
                        intensity=self.intensity_strobe_negative.get()
                    ) if self.effect_strobe_negative.get() else None,
                    
                    "Glitch": lambda: GlitchEffect(
                        threshold=self.threshold_mid.get(),
                        intensity=self.intensity_glitch.get()
                    ) if self.effect_glitch.get() else None,
                    
                    "ChromaticAberration": lambda: ChromaticAberrationEffect(
                        threshold=self.threshold_high.get(),
                        intensity=self.intensity_chromatic.get()
                    ) if self.effect_chromatic.get() else None,
                    
                    "BubbleDistortion": lambda: BubbleDistortionEffect(
                        threshold=self.threshold_bass.get(),
                        intensity=self.intensity_bubble.get()
                    ) if self.effect_bubble.get() else None,
                    
                    "ScreenShake": lambda: ScreenShakeEffect(
                        threshold=self.threshold_mid.get(),
                        intensity=self.intensity_screen_shake.get()
                    ) if self.effect_screen_shake.get() else None,
                    
                    "RGBSplit": lambda: RGBSplitEffect(
                        threshold=self.threshold_high.get(),
                        intensity=self.intensity_rgb_split.get()
                    ) if self.effect_rgb_split.get() else None,
                    
                    "ElectricArcs": lambda: ElectricArcsEffect(
                        colors=[(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)],
                        threshold=0.7,
                        intensity=self.intensity_electric_arcs.get()
                    ) if self.effect_electric_arcs.get() else None,
                    
                    "FashionLightning": lambda: FashionLightningEffect(
                        colors=[(1.0, 0.0, 0.8), (0.0, 0.9, 1.0), (0.8, 1.0, 0.0)],
                        threshold=0.65,
                        branching_probability=0.6,
                        max_branches=5,
                        segment_length_min=5,
                        segment_length_max=20,
                        intensity=self.intensity_fashion_lightning.get()
                    ) if self.effect_fashion_lightning.get() else None,
                    
                    "AdvancedGlitch": lambda: AdvancedGlitchEffect(
                        threshold=0.5,
                        channel_shift_amount=8,
                        block_size_range=(10, 80),
                        intensity=self.intensity_advanced_glitch.get()
                    ) if self.effect_advanced_glitch.get() else None,
                    
                    "DimensionalWarp": lambda: DimensionalWarpEffect(
                        bass_threshold=self.threshold_bass.get(),
                        mid_threshold=self.threshold_mid.get(),
                        warp_strength=45.0,
                        rotation_speed=0.5,
                        perspective_depth=200.0,
                        wave_frequency=2.0,
                        layer_count=3,
                        intensity=self.intensity_dimensional_warp.get()
                    ) if self.effect_dimensional_warp.get() else None,
                    
                    "VortexDistortion": lambda: VortexDistortionEffect(
                        threshold=0.2,
                        max_angle=35.0,
                        radius_falloff=1.8,
                        rotation_speed=3.0,
                        smoothing=0.3,
                        intensity=self.intensity_vortex_distortion.get()
                    ) if self.effect_vortex_distortion.get() else None,
                    
                    "FloatingText": lambda: FloatingText(
                        text=self.floating_text_content.get(),
                        font_size=self.floating_text_font_size.get(),
                        color_scheme=self.floating_text_color_scheme.get(),
                        animation_style=self.floating_text_animation.get()
                    ) if self.effect_floating_text.get() else None,
                }
                
                # Crea effetti nell'ordine specificato dall'utente
                for effect_name in self.effect_order:
                    if effect_name in effects_map:
                        effect = effects_map[effect_name]()
                        if effect is not None:
                            custom_effects.append(effect)
                
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
                    audio_data = audio_analyzer.load_and_analyze(audio_source, duration=custom_duration, fps=fps)
                    
                    # Debug
                    audio_duration = len(audio_data.audio_signal) / audio_data.sample_rate
                    print(f"[DEBUG IMAGE MODE] custom_duration={custom_duration}, audio_duration={audio_duration:.2f}s, num_frames={len(audio_data.bass_energy)}")
                    
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
                    
                    # Calculate actual video duration based on number of frames
                    video_duration = num_frames / fps
                    
                    # Try ffmpeg first (fastest) - trim audio to match video duration
                    try:
                        result = subprocess.run([
                            'ffmpeg', '-y', '-i', temp_video_avi, '-i', audio_source,
                            '-t', str(video_duration),  # Limit audio duration to match video
                            '-c:v', 'libx264', '-c:a', 'aac', output
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
                            # Trim audio to match video duration
                            audio_clip = audio_clip.subclip(0, video_duration)
                            final_clip = video_clip.set_audio(audio_clip)
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
                    audio_extracted = False
                    if use_video_audio:
                        # Extract audio from video
                        progress_cb("status", {"message": "Estrazione audio dal video..."})
                        temp_audio = tempfile.mktemp(suffix='.wav', dir=os.path.dirname(output))
                        
                        try:
                            # Find ffmpeg
                            ffmpeg_cmd = 'ffmpeg'
                            if os.path.exists(r'C:\ffmpeg\bin\ffmpeg.exe'):
                                ffmpeg_cmd = r'C:\ffmpeg\bin\ffmpeg.exe'
                            
                            # Try ffmpeg first
                            result = subprocess.run([
                                ffmpeg_cmd, '-y', '-i', video, '-vn', '-acodec', 'pcm_s16le', 
                                '-ar', '44100', '-ac', '2', temp_audio
                            ], check=True, capture_output=True, text=True)
                            
                            # Check if audio file was actually created and has content
                            if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 1000:
                                audio_source = temp_audio
                                audio_extracted = True
                                progress_cb("status", {"message": "Audio estratto dal video con successo"})
                            else:
                                progress_cb("status", {"message": "Video senza audio, uso file audio fornito"})
                        except (subprocess.CalledProcessError, FileNotFoundError) as e:
                            # Fallback to moviepy
                            try:
                                try:
                                    from moviepy import VideoFileClip
                                except ImportError:
                                    from moviepy.editor import VideoFileClip
                                
                                video_clip = VideoFileClip(video)
                                
                                # Check if video has audio
                                if video_clip.audio is not None:
                                    video_clip.audio.write_audiofile(temp_audio, logger=None)
                                    audio_source = temp_audio
                                    audio_extracted = True
                                    progress_cb("status", {"message": "Audio estratto dal video con MoviePy"})
                                else:
                                    progress_cb("status", {"message": "Video senza audio, uso file audio fornito"})
                                
                                video_clip.close()
                            except Exception as e2:
                                progress_cb("status", {"message": f"Impossibile estrarre audio: {str(e2)}, uso file audio fornito"})
                        
                        # Clean up temp file if extraction failed
                        if not audio_extracted and os.path.exists(temp_audio):
                            try:
                                os.remove(temp_audio)
                            except:
                                pass
                    
                    # Load and analyze audio (use custom_duration if specified)
                    progress_cb("status", {"message": "Analisi audio..."})
                    audio_analyzer = AudioAnalyzer()
                    audio_data = audio_analyzer.load_and_analyze(audio_source, duration=custom_duration, fps=int(video_fps))
                    audio_duration = len(audio_data.audio_signal) / audio_data.sample_rate
                    
                    # Calculate frame mapping - ALWAYS based on audio_duration (which respects custom_duration)
                    required_frames = int(audio_duration * video_fps)
                    print(f"[DEBUG] audio_duration={audio_duration:.2f}s, video_fps={video_fps}, required_frames={required_frames}, available_frames={available_frames}")
                    if custom_duration:
                        progress_cb("status", {"message": f"Usando durata personalizzata: {custom_duration}s = {required_frames} frame"})
                    
                    if required_frames <= available_frames:
                        frame_indices = list(range(required_frames))
                        print(f"[DEBUG] Case 1: Sequential, frame_indices length={len(frame_indices)}, first 10={frame_indices[:10]}")
                    elif short_mode == "loop":
                        frame_indices = []
                        while len(frame_indices) < required_frames:
                            remaining = required_frames - len(frame_indices)
                            frame_indices.extend(range(min(available_frames, remaining)))
                        print(f"[DEBUG] Case 2: Loop, frame_indices length={len(frame_indices)}, first 10={frame_indices[:10]}, last 10={frame_indices[-10:]}")
                    else:  # stretch
                        stretch_factor = required_frames / available_frames
                        frame_indices = []
                        for i in range(available_frames):
                            repeat_count = int(stretch_factor)
                            if (i * stretch_factor) % 1 >= (1 - stretch_factor % 1):
                                repeat_count += 1
                            frame_indices.extend([i] * repeat_count)
                        frame_indices = frame_indices[:required_frames]
                        print(f"[DEBUG] Case 3: Stretch, stretch_factor={stretch_factor}, frame_indices length={len(frame_indices)}, first 10={frame_indices[:10]}, last 10={frame_indices[-10:]}")
                    
                    # Process video with streaming (frame by frame)
                    total_frames = len(frame_indices)
                    progress_cb("start", {"total_frames": total_frames})
                    progress_cb("status", {"message": f"Processing {total_frames} frame con modalità '{short_mode}'..."})
                    
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
                    
                    # First, load ALL video frames into memory (to avoid seek issues)
                    print(f"[DEBUG] Loading all {available_frames} video frames into memory...")
                    progress_cb("status", {"message": f"Caricamento {available_frames} frame dal video..."})
                    
                    video_frames = []
                    for i in range(available_frames):
                        ret, frame = cap.read()
                        if ret:
                            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        else:
                            print(f"[WARNING] Failed to read frame {i}")
                            break
                    
                    cap.release()
                    print(f"[DEBUG] Loaded {len(video_frames)} frames from video")
                    
                    # Now process frames according to frame_indices
                    print(f"[DEBUG] Processing {len(frame_indices)} frames with effects...")
                    all_frames = []
                    
                    for idx, frame_idx in enumerate(frame_indices):
                        progress_cb("frame", {"index": idx + 1, "total": total_frames})
                        
                        # Get base frame from loaded frames
                        if frame_idx < len(video_frames):
                            base_frame = video_frames[frame_idx].copy()
                        else:
                            # Fallback to last frame
                            base_frame = video_frames[-1].copy() if video_frames else np.zeros((height, width, 3), dtype=np.uint8)
                        
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
                        
                        # Store frame (RGB for MoviePy)
                        all_frames.append(frame_with_effects.copy())
                        
                        # Debug: save some frames
                        if idx in [0, 30, 60, 90, 120]:
                            debug_path = os.path.join(os.path.dirname(output), f"debug_moviepy_frame_{idx}.png")
                            cv2.imwrite(debug_path, cv2.cvtColor(frame_with_effects, cv2.COLOR_RGB2BGR))
                            print(f"[DEBUG] Saved frame {idx} to {debug_path}")
                        
                        # Clear frame to free memory
                        del base_frame
                        del frame_with_effects
                    
                    # Release resources (cap already released)
                    video_frames.clear()
                    
                    # Create video with MoviePy
                    print(f"[DEBUG] Creating video from {len(all_frames)} frames with MoviePy...")
                    progress_cb("status", {"message": "Creazione video con MoviePy..."})
                    
                    try:
                        from moviepy import ImageSequenceClip, AudioFileClip
                    except ImportError:
                        from moviepy.editor import ImageSequenceClip, AudioFileClip
                    
                    video_clip = ImageSequenceClip(all_frames, fps=video_fps)
                    audio_clip = AudioFileClip(audio_source)
                    
                    # Sync video and audio duration
                    if audio_clip.duration > video_clip.duration:
                        audio_clip = audio_clip.subclip(0, video_clip.duration)
                    elif video_clip.duration > audio_clip.duration:
                        video_clip = video_clip.subclip(0, audio_clip.duration)
                    
                    final_clip = video_clip.set_audio(audio_clip)
                    final_clip.write_videofile(output, codec='libx264', audio_codec='aac', fps=video_fps, logger=None)
                    
                    video_clip.close()
                    audio_clip.close()
                    final_clip.close()
                    
                    print(f"[DEBUG] Video created successfully!")
                    
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
