"""
Main view for custom preset configuration.
Handles UI layout and widget creation following separation of concerns.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Callable, Dict
from .components import (
    FileSelector, LabeledEntry, LabeledScale, LabeledSpinbox, 
    LabeledCombobox, EffectControl, SectionHeader, ScrollableFrame
)
from .models import ApplicationState, EffectSettings
from .services import FileTypeService
from .theme import DEFAULT_THEME


class CustomPresetView(ttk.Frame):
    """View for custom preset configuration."""
    
    def __init__(self, parent: tk.Widget, state: ApplicationState, 
                 callbacks: Dict[str, Callable]):
        super().__init__(parent)
        self.state = state
        self.callbacks = callbacks
        self.theme = DEFAULT_THEME
        
        # Create UI variables for two-way binding
        self._create_variables()
        
        # Build UI
        self._build_ui()
    
    def _create_variables(self):
        """Create Tkinter variables for state binding."""
        cfg = self.state.video_config
        
        # Mode and paths
        self.mode_var = tk.StringVar(value=cfg.mode)
        self.audio_path_var = tk.StringVar(value=cfg.audio_path)
        self.image_path_var = tk.StringVar(value=cfg.image_path)
        self.video_path_var = tk.StringVar(value=cfg.video_path)
        self.output_path_var = tk.StringVar(value=cfg.output_path)
        
        # Video settings
        self.fps_var = tk.IntVar(value=cfg.fps)
        self.duration_var = tk.StringVar(value=cfg.duration)
        self.video_mode_var = tk.StringVar(value=cfg.video_mode)
        self.use_video_audio_var = tk.BooleanVar(value=cfg.use_video_audio)
        self.use_native_resolution_var = tk.BooleanVar(value=cfg.use_native_resolution)
        
        # Logo settings
        self.logo_path_var = tk.StringVar(value=cfg.logo_path)
        self.logo_position_var = tk.StringVar(value=cfg.logo_position)
        self.logo_scale_var = tk.DoubleVar(value=cfg.logo_scale)
        self.logo_opacity_var = tk.DoubleVar(value=cfg.logo_opacity)
        self.logo_margin_var = tk.IntVar(value=cfg.logo_margin)
        
        # Audio thresholds
        thresh = self.state.audio_thresholds
        self.bass_threshold_var = tk.DoubleVar(value=thresh.bass)
        self.mid_threshold_var = tk.DoubleVar(value=thresh.mid)
        self.high_threshold_var = tk.DoubleVar(value=thresh.high)
        
        # Effect variables
        effects = self.state.effects_config.effects
        self.effect_vars = {
            name: {
                'enabled': tk.BooleanVar(value=settings.enabled),
                'intensity': tk.DoubleVar(value=settings.intensity)
            }
            for name, settings in effects.items()
        }
        
        # Floating text
        ftc = self.state.effects_config.floating_text_config
        self.floating_text_content_var = tk.StringVar(value=ftc.content)
        self.floating_text_color_var = tk.StringVar(value=ftc.color_scheme)
        self.floating_text_animation_var = tk.StringVar(value=ftc.animation)
        self.floating_text_font_size_var = tk.IntVar(value=ftc.font_size)
    
    def _build_ui(self):
        """Build the complete UI in 2-column layout."""
        # Create scrollable container
        scrollable = ScrollableFrame(self)
        scrollable.pack(fill=tk.BOTH, expand=True)
        
        frame = scrollable.get_frame()
        
        # Header
        header = SectionHeader(frame, "Audio Visual FX Generator - Configurazione")
        header.pack(fill=tk.X, pady=(0, self.theme.spacing.lg), padx=self.theme.spacing.md)
        
        # Create 2-column layout
        columns_frame = ttk.Frame(frame)
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=self.theme.spacing.md)
        
        # Left column
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, 
                        padx=(0, self.theme.spacing.md))
        
        # Right column
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, 
                         padx=(self.theme.spacing.md, 0))
        
        # LEFT COLUMN CONTENT
        # Mode selection
        self._build_mode_section(left_column)
        
        # File inputs
        self._build_file_section(left_column)
        
        # Output settings
        self._build_output_section(left_column)
        
        # Audio thresholds
        self._build_threshold_section(left_column)
        
        # Logo overlay
        self._build_logo_section(left_column)
        
        # Floating text configuration
        self._build_floating_text_section(left_column)
        
        # RIGHT COLUMN CONTENT
        # Effects
        self._build_effects_section(right_column)
        
        # Effect order management
        self._build_effect_order_section(right_column)
        
        # BOTTOM - Full width
        # Progress and controls
        self._build_controls_section(frame)
    
    def _build_mode_section(self, parent):
        """Build mode selection section."""
        section = ttk.Frame(parent)
        section.pack(fill=tk.X, pady=self.theme.spacing.md)
        
        ttk.Label(section, text="Modalità:", 
                 font=(self.theme.typography.font_family, 
                      self.theme.typography.font_size_medium, 
                      self.theme.typography.font_weight_bold)).pack(
            side=tk.LEFT, padx=(0, self.theme.spacing.lg))
        
        ttk.Radiobutton(
            section, text="Audio + Immagine", 
            variable=self.mode_var, value="image",
            command=self.callbacks['on_mode_change']
        ).pack(side=tk.LEFT, padx=self.theme.spacing.md)
        
        ttk.Radiobutton(
            section, text="Audio + Video", 
            variable=self.mode_var, value="video",
            command=self.callbacks['on_mode_change']
        ).pack(side=tk.LEFT, padx=self.theme.spacing.md)
    
    def _build_file_section(self, parent):
        """Build file input section."""
        section = ttk.LabelFrame(parent, text="File di Input", padding=self.theme.spacing.md)
        section.pack(fill=tk.X, pady=self.theme.spacing.md)
        
        # Audio file
        audio_frame = ttk.Frame(section)
        audio_frame.pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        self.audio_selector = FileSelector(
            audio_frame, "Audio:", self.audio_path_var,
            FileTypeService.get_audio_types()
        )
        self.audio_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.use_video_audio_check = ttk.Checkbutton(
            audio_frame, text="Usa audio del video",
            variable=self.use_video_audio_var,
            command=self.callbacks['on_video_audio_toggle']
        )
        self.use_video_audio_check.pack(side=tk.LEFT, padx=self.theme.spacing.md)
        
        # Image file (shown in image mode)
        self.image_selector = FileSelector(
            section, "Immagine:", self.image_path_var,
            FileTypeService.get_image_types()
        )
        self.image_selector.pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        # Video file (shown in video mode)
        self.video_selector = FileSelector(
            section, "Video:", self.video_path_var,
            FileTypeService.get_video_types()
        )
        
        # Video mode selection
        video_mode_frame = ttk.Frame(section)
        self.video_mode_combo = LabeledCombobox(
            video_mode_frame, "Modalità video corto:",
            self.video_mode_var, ["loop", "stretch"]
        )
        
        # Initially hide video controls
        self._toggle_mode_visibility()
    
    def _build_output_section(self, parent):
        """Build output settings section."""
        section = ttk.LabelFrame(parent, text="Impostazioni Output", 
                                padding=self.theme.spacing.md)
        section.pack(fill=tk.X, pady=self.theme.spacing.md)
        
        # Output file
        output_frame = ttk.Frame(section)
        output_frame.pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        ttk.Label(output_frame, text="File Output:", width=15, anchor='e').pack(
            side=tk.LEFT, padx=(0, self.theme.spacing.md))
        
        ttk.Entry(output_frame, textvariable=self.output_path_var, width=45).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, self.theme.spacing.md))
        
        ttk.Button(output_frame, text="Scegli...", 
                  command=self._browse_output).pack(side=tk.LEFT)
        
        # Native resolution checkbox
        ttk.Checkbutton(
            section, text="Usa dimensioni native dell'immagine/video",
            variable=self.use_native_resolution_var
        ).pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        # FPS and Duration
        settings_frame = ttk.Frame(section)
        settings_frame.pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        self.fps_frame = ttk.Frame(settings_frame)
        self.fps_frame.pack(side=tk.LEFT, padx=(0, self.theme.spacing.xl))
        
        self.fps_spinbox = LabeledSpinbox(
            self.fps_frame, "FPS:", self.fps_var, from_=1, to=120
        )
        self.fps_spinbox.pack(side=tk.LEFT)
        
        LabeledEntry(
            settings_frame, "Durata (sec):", self.duration_var, width=10
        ).pack(side=tk.LEFT)
        
        ttk.Label(
            settings_frame, text="(vuoto = durata completa)",
            font=(self.theme.typography.font_family, 
                  self.theme.typography.font_size_small)
        ).pack(side=tk.LEFT, padx=self.theme.spacing.md)
    
    def _build_threshold_section(self, parent):
        """Build audio threshold section."""
        section = ttk.LabelFrame(parent, text="Soglie Audio", 
                                padding=self.theme.spacing.md)
        section.pack(fill=tk.X, pady=self.theme.spacing.md)
        
        # Bass threshold
        LabeledScale(
            section, "Bass:", self.bass_threshold_var, 
            from_=0.0, to=1.0, resolution=0.01
        ).pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        # Mid threshold
        LabeledScale(
            section, "Mid:", self.mid_threshold_var, 
            from_=0.0, to=1.0, resolution=0.01
        ).pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        # High threshold
        LabeledScale(
            section, "High:", self.high_threshold_var, 
            from_=0.0, to=1.0, resolution=0.01
        ).pack(fill=tk.X, pady=self.theme.spacing.sm)
    
    def _build_logo_section(self, parent):
        """Build logo overlay section."""
        section = ttk.LabelFrame(parent, text="Logo Overlay (Opzionale)", 
                                padding=self.theme.spacing.md)
        section.pack(fill=tk.X, pady=self.theme.spacing.md)
        
        # Logo file
        FileSelector(
            section, "Logo:", self.logo_path_var,
            FileTypeService.get_image_types()
        ).pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        # Logo settings
        settings_frame = ttk.Frame(section)
        settings_frame.pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        LabeledCombobox(
            settings_frame, "Posizione:", self.logo_position_var,
            ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
        ).pack(side=tk.LEFT, padx=(0, self.theme.spacing.xl))
        
        LabeledScale(
            settings_frame, "Scala:", self.logo_scale_var,
            from_=0.05, to=0.5, resolution=0.05, length=150
        ).pack(side=tk.LEFT, padx=(0, self.theme.spacing.xl))
        
        LabeledScale(
            settings_frame, "Opacità:", self.logo_opacity_var,
            from_=0.0, to=1.0, resolution=0.1, length=150
        ).pack(side=tk.LEFT)
    
    def _build_effects_section(self, parent):
        """Build effects configuration section."""
        section = ttk.LabelFrame(parent, text="Effetti Visivi", 
                                padding=self.theme.spacing.md)
        section.pack(fill=tk.BOTH, expand=True, pady=self.theme.spacing.md)
        
        # Create single column for effects (since parent is already a column)
        effects_container = ttk.Frame(section)
        effects_container.pack(fill=tk.BOTH, expand=True)
        
        # Create effect controls
        effect_names = list(self.state.effects_config.effects.keys())
        for name in effect_names:
            effect_control = EffectControl(
                effects_container, name,
                self.effect_vars[name]['enabled'],
                self.effect_vars[name]['intensity'],
                on_change=self.callbacks['on_effect_change']
            )
            effect_control.pack(fill=tk.X, pady=self.theme.spacing.xs)
    
    def _build_floating_text_section(self, parent):
        """Build floating text configuration section."""
        section = ttk.LabelFrame(parent, text="Configurazione Floating Text", 
                                padding=self.theme.spacing.md)
        section.pack(fill=tk.X, pady=self.theme.spacing.md)
        
        # Text content
        text_frame = ttk.Frame(section)
        text_frame.pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        LabeledEntry(
            text_frame, "Testo:", self.floating_text_content_var, width=20
        ).pack(fill=tk.X, pady=self.theme.spacing.xs)
        
        # Color and animation
        settings_frame = ttk.Frame(section)
        settings_frame.pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        LabeledCombobox(
            settings_frame, "Colori:", self.floating_text_color_var,
            ["rainbow", "fire", "ice", "neon", "gold", "default"]
        ).pack(fill=tk.X, pady=self.theme.spacing.xs)
        
        LabeledCombobox(
            settings_frame, "Animazione:", self.floating_text_animation_var,
            ["wave", "bounce", "spin", "pulse", "glitch"]
        ).pack(fill=tk.X, pady=self.theme.spacing.xs)
        
        LabeledSpinbox(
            settings_frame, "Font Size:", self.floating_text_font_size_var,
            from_=50, to=300
        ).pack(fill=tk.X, pady=self.theme.spacing.xs)
    
    def _build_effect_order_section(self, parent):
        """Build effect order management section."""
        section = ttk.LabelFrame(parent, text="Ordine Effetti", 
                                padding=self.theme.spacing.md)
        section.pack(fill=tk.BOTH, expand=True, pady=self.theme.spacing.md)
        
        ttk.Label(
            section, text="L'ordine cambia il risultato finale",
            font=(self.theme.typography.font_family, 
                  self.theme.typography.font_size_small),
            foreground="gray"
        ).pack(fill=tk.X, pady=(0, self.theme.spacing.sm))
        
        order_frame = ttk.Frame(section)
        order_frame.pack(fill=tk.BOTH, expand=True)
        
        # Listbox for effect order
        list_frame = ttk.Frame(order_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.order_listbox = tk.Listbox(list_frame, height=10)
        self.order_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, command=self.order_listbox.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.order_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Buttons for reordering
        btn_frame = ttk.Frame(order_frame)
        btn_frame.pack(side=tk.LEFT, padx=self.theme.spacing.md, fill=tk.Y)
        
        ttk.Button(btn_frame, text="▲ Su", width=10,
                  command=self.callbacks['on_move_up']).pack(
            pady=self.theme.spacing.xs)
        ttk.Button(btn_frame, text="▼ Giù", width=10,
                  command=self.callbacks['on_move_down']).pack(
            pady=self.theme.spacing.xs)
        ttk.Button(btn_frame, text="⬆ In cima", width=10,
                  command=self.callbacks['on_move_top']).pack(
            pady=self.theme.spacing.xs)
        ttk.Button(btn_frame, text="⬇ In fondo", width=10,
                  command=self.callbacks['on_move_bottom']).pack(
            pady=self.theme.spacing.xs)
        ttk.Button(btn_frame, text="↺ Reset", width=10,
                  command=self.callbacks['on_reset_order']).pack(
            pady=self.theme.spacing.xs)
        
        # Initialize listbox
        self.refresh_order_listbox()
    
    def _build_controls_section(self, parent):
        """Build progress and control buttons."""
        section = ttk.Frame(parent)
        section.pack(fill=tk.X, pady=self.theme.spacing.lg, padx=self.theme.spacing.md)
        
        # Separator
        ttk.Separator(section, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=self.theme.spacing.md)
        
        # Progress bar
        self.progress = ttk.Progressbar(section, mode="determinate", length=600)
        self.progress.pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        # Status label
        self.status_label = ttk.Label(
            section, text="Pronto", 
            font=(self.theme.typography.font_family, 
                  self.theme.typography.font_size_medium)
        )
        self.status_label.pack(fill=tk.X, pady=self.theme.spacing.sm)
        
        # Control buttons
        btn_frame = ttk.Frame(section)
        btn_frame.pack(fill=tk.X, pady=self.theme.spacing.md)
        
        self.generate_btn = ttk.Button(
            btn_frame, text="🎬 Genera Video Custom", 
            command=self.callbacks['on_generate']
        )
        self.generate_btn.pack(side=tk.RIGHT, padx=self.theme.spacing.sm)
        
        self.cancel_btn = ttk.Button(
            btn_frame, text="✖ Annulla", 
            command=self.callbacks['on_cancel'],
            state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.RIGHT)
    
    def _browse_output(self):
        """Browse for output file."""
        path = filedialog.asksaveasfilename(
            title="Scegli file di output",
            defaultextension=".mp4",
            initialfile="custom_output.mp4",
            filetypes=FileTypeService.get_output_types()
        )
        if path:
            self.output_path_var.set(path)
    
    def _toggle_mode_visibility(self):
        """Toggle visibility of mode-specific controls."""
        mode = self.mode_var.get()
        
        if mode == "image":
            # Show image, hide video
            self.image_selector.pack(fill=tk.X, pady=self.theme.spacing.sm)
            if hasattr(self, 'fps_frame'):
                self.fps_frame.pack(side=tk.LEFT, padx=(0, self.theme.spacing.xl))
            self.video_selector.pack_forget()
            self.use_video_audio_check.pack_forget()
        else:
            # Show video, hide image
            self.video_selector.pack(fill=tk.X, pady=self.theme.spacing.sm)
            self.image_selector.pack_forget()
            if hasattr(self, 'fps_frame'):
                self.fps_frame.pack_forget()
            self.use_video_audio_check.pack(side=tk.LEFT, padx=self.theme.spacing.md)
    
    def refresh_order_listbox(self):
        """Refresh the effect order listbox."""
        self.order_listbox.delete(0, tk.END)
        
        for effect_name in self.state.effects_config.effect_order:
            if effect_name in self.effect_vars:
                if self.effect_vars[effect_name]['enabled'].get():
                    self.order_listbox.insert(tk.END, effect_name)
    
    def update_progress(self, value: int, maximum: int):
        """Update progress bar."""
        self.progress.configure(value=value, maximum=maximum)
    
    def update_status(self, message: str):
        """Update status label."""
        self.status_label.configure(text=message)
    
    def set_processing_state(self, processing: bool):
        """Update UI state during processing."""
        state = tk.DISABLED if processing else tk.NORMAL
        self.generate_btn.configure(state=state)
        self.cancel_btn.configure(state=tk.NORMAL if processing else tk.DISABLED)
    
    def sync_state_from_ui(self):
        """Sync application state from UI variables."""
        # Video config
        cfg = self.state.video_config
        cfg.mode = self.mode_var.get()
        
        # Set audio_path to empty string if using video audio (placeholder text)
        audio_value = self.audio_path_var.get()
        cfg.audio_path = "" if audio_value == "(audio dal video)" else audio_value
        
        cfg.image_path = self.image_path_var.get()
        cfg.video_path = self.video_path_var.get()
        cfg.output_path = self.output_path_var.get()
        cfg.fps = self.fps_var.get()
        cfg.duration = self.duration_var.get()
        cfg.video_mode = self.video_mode_var.get()
        cfg.use_video_audio = self.use_video_audio_var.get()
        cfg.use_native_resolution = self.use_native_resolution_var.get()
        cfg.logo_path = self.logo_path_var.get()
        cfg.logo_position = self.logo_position_var.get()
        cfg.logo_scale = self.logo_scale_var.get()
        cfg.logo_opacity = self.logo_opacity_var.get()
        cfg.logo_margin = self.logo_margin_var.get()
        
        # Thresholds
        thresh = self.state.audio_thresholds
        thresh.bass = self.bass_threshold_var.get()
        thresh.mid = self.mid_threshold_var.get()
        thresh.high = self.high_threshold_var.get()
        
        # Effects
        for name, vars_dict in self.effect_vars.items():
            self.state.effects_config.effects[name].enabled = vars_dict['enabled'].get()
            self.state.effects_config.effects[name].intensity = vars_dict['intensity'].get()
        
        # Floating text
        ftc = self.state.effects_config.floating_text_config
        ftc.content = self.floating_text_content_var.get()
        ftc.color_scheme = self.floating_text_color_var.get()
        ftc.animation = self.floating_text_animation_var.get()
        ftc.font_size = self.floating_text_font_size_var.get()
