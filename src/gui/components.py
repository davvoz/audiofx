"""
Reusable UI components following DRY and Single Responsibility Principle.
Each component handles one specific UI concern.
"""

import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, List, Optional, Tuple
from .theme import DEFAULT_THEME


class LabeledEntry(ttk.Frame):
    """Entry field with label."""
    
    def __init__(self, parent: tk.Widget, label: str, variable: tk.StringVar, 
                 width: int = 45, **kwargs):
        super().__init__(parent, **kwargs)
        self.variable = variable
        
        ttk.Label(self, text=label, width=15, anchor='e').pack(
            side=tk.LEFT, padx=(0, DEFAULT_THEME.spacing.md))
        self.entry = ttk.Entry(self, textvariable=variable, width=width)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def get(self) -> str:
        return self.variable.get()
    
    def set(self, value: str) -> None:
        self.variable.set(value)
    
    def configure_entry(self, **kwargs):
        """Configure the entry widget."""
        self.entry.configure(**kwargs)


class FileSelector(ttk.Frame):
    """File selector with browse button."""
    
    def __init__(self, parent: tk.Widget, label: str, variable: tk.StringVar,
                 file_types: List[Tuple[str, str]], width: int = 45, **kwargs):
        super().__init__(parent, **kwargs)
        self.variable = variable
        self.file_types = file_types
        
        ttk.Label(self, text=label, width=15, anchor='e').pack(
            side=tk.LEFT, padx=(0, DEFAULT_THEME.spacing.md))
        
        self.entry = ttk.Entry(self, textvariable=variable, width=width)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, 
                       padx=(0, DEFAULT_THEME.spacing.md))
        
        ttk.Button(self, text="Sfoglia...", command=self._browse).pack(side=tk.LEFT)
    
    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title=f"Seleziona {self.entry.master.winfo_parent()}",
            filetypes=self.file_types
        )
        if path:
            self.variable.set(path)
    
    def get(self) -> str:
        return self.variable.get()
    
    def configure_entry(self, **kwargs):
        """Configure the entry widget."""
        self.entry.configure(**kwargs)


class LabeledScale(ttk.Frame):
    """Scale with label and value display."""
    
    def __init__(self, parent: tk.Widget, label: str, variable: tk.DoubleVar,
                 from_: float, to: float, resolution: float = 0.1, 
                 length: int = 200, **kwargs):
        super().__init__(parent, **kwargs)
        self.variable = variable
        
        # Label
        ttk.Label(self, text=label, width=15, anchor='e').pack(
            side=tk.LEFT, padx=(0, DEFAULT_THEME.spacing.sm))
        
        # Scale
        self.scale = ttk.Scale(
            self, variable=variable, from_=from_, to=to,
            orient=tk.HORIZONTAL, length=length,
            command=self._on_change
        )
        self.scale.pack(side=tk.LEFT, padx=DEFAULT_THEME.spacing.sm)
        
        # Value label
        self.value_label = ttk.Label(self, text=f"{variable.get():.2f}", width=6)
        self.value_label.pack(side=tk.LEFT)
    
    def _on_change(self, value: str) -> None:
        """Update value label when scale changes."""
        self.value_label.configure(text=f"{float(value):.2f}")
    
    def get(self) -> float:
        return self.variable.get()


class LabeledSpinbox(ttk.Frame):
    """Spinbox with label."""
    
    def __init__(self, parent: tk.Widget, label: str, variable: tk.IntVar,
                 from_: int, to: int, width: int = 10, **kwargs):
        super().__init__(parent, **kwargs)
        self.variable = variable
        
        ttk.Label(self, text=label, width=15, anchor='e').pack(
            side=tk.LEFT, padx=(0, DEFAULT_THEME.spacing.md))
        
        self.spinbox = ttk.Spinbox(
            self, textvariable=variable, from_=from_, to=to, width=width
        )
        self.spinbox.pack(side=tk.LEFT)
    
    def get(self) -> int:
        return self.variable.get()


class LabeledCombobox(ttk.Frame):
    """Combobox with label."""
    
    def __init__(self, parent: tk.Widget, label: str, variable: tk.StringVar,
                 values: List[str], width: int = 15, **kwargs):
        super().__init__(parent, **kwargs)
        self.variable = variable
        
        ttk.Label(self, text=label, width=15, anchor='e').pack(
            side=tk.LEFT, padx=(0, DEFAULT_THEME.spacing.md))
        
        self.combobox = ttk.Combobox(
            self, textvariable=variable, values=values, 
            state='readonly', width=width
        )
        self.combobox.pack(side=tk.LEFT)
        if values:
            self.combobox.current(0)
    
    def get(self) -> str:
        return self.variable.get()


class EffectControl(ttk.Frame):
    """Reusable control for an effect with checkbox and intensity slider."""
    
    def __init__(self, parent: tk.Widget, name: str, 
                 enabled_var: tk.BooleanVar, intensity_var: tk.DoubleVar,
                 on_change: Optional[Callable] = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.name = name
        self.enabled_var = enabled_var
        self.intensity_var = intensity_var
        self.on_change = on_change
        
        # Checkbox with fixed width for alignment
        self.checkbox = ttk.Checkbutton(
            self, text=name, variable=enabled_var,
            command=self._on_checkbox_change, width=20
        )
        self.checkbox.pack(side=tk.LEFT, padx=(0, DEFAULT_THEME.spacing.sm))
        
        # Intensity slider
        self.scale = ttk.Scale(
            self, variable=intensity_var, from_=0.0, to=2.0,
            orient=tk.HORIZONTAL, length=120
        )
        self.scale.pack(side=tk.LEFT, padx=DEFAULT_THEME.spacing.sm)
        
        # Value label
        self.value_label = ttk.Label(self, text=f"{intensity_var.get():.1f}", width=4)
        self.value_label.pack(side=tk.LEFT)
        
        # Bind scale changes
        intensity_var.trace_add('write', self._on_intensity_change)
    
    def _on_checkbox_change(self) -> None:
        """Called when checkbox state changes."""
        if self.on_change:
            self.on_change()
    
    def _on_intensity_change(self, *args) -> None:
        """Update value label when intensity changes."""
        self.value_label.configure(text=f"{self.intensity_var.get():.1f}")
    
    def is_enabled(self) -> bool:
        return self.enabled_var.get()
    
    def get_intensity(self) -> float:
        return self.intensity_var.get()


class SectionHeader(ttk.Frame):
    """Section header with title and optional separator."""
    
    def __init__(self, parent: tk.Widget, title: str, 
                 show_separator: bool = True, **kwargs):
        super().__init__(parent, **kwargs)
        
        ttk.Label(
            self, text=title, 
            font=(DEFAULT_THEME.typography.font_family, 
                  DEFAULT_THEME.typography.font_size_large, 
                  DEFAULT_THEME.typography.font_weight_bold)
        ).pack(side=tk.LEFT, pady=DEFAULT_THEME.spacing.sm)
        
        if show_separator:
            sep = ttk.Separator(self, orient=tk.HORIZONTAL)
            sep.pack(side=tk.LEFT, fill=tk.X, expand=True, 
                    padx=(DEFAULT_THEME.spacing.lg, 0))


class ScrollableFrame(ttk.Frame):
    """Scrollable frame container."""
    
    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create canvas with scrollbar
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def get_frame(self) -> ttk.Frame:
        """Get the scrollable frame for adding widgets."""
        return self.scrollable_frame
