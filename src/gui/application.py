"""
Main application class.
Minimal orchestrator that brings together all components.
"""

import tkinter as tk
from tkinter import ttk
from .models import ApplicationState
from .controller import CustomPresetController
from .theme import DEFAULT_THEME


class AudioVisualFXApp(tk.Tk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("Audio Visual FX Generator - Professional Edition")
        self.geometry("1400x850")
        self.resizable(True, True)
        
        # Center window on screen
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        
        # Apply theme
        self._configure_theme()
        
        # Initialize application state
        self.state = ApplicationState()
        
        # Build UI
        self._build_ui()
    
    def _configure_theme(self):
        """Configure application theme."""
        style = ttk.Style()
        
        # Try to use a modern theme if available
        available_themes = style.theme_names()
        if 'vista' in available_themes:
            style.theme_use('vista')
        elif 'clam' in available_themes:
            style.theme_use('clam')
        
        # Configure custom styles
        style.configure('TButton', padding=DEFAULT_THEME.spacing.sm)
        style.configure('TLabel', padding=DEFAULT_THEME.spacing.xs)
        style.configure('TLabelframe', padding=DEFAULT_THEME.spacing.md)
    
    def _build_ui(self):
        """Build the main UI."""
        # Create main container without tabs
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, 
                           padx=DEFAULT_THEME.spacing.lg, 
                           pady=DEFAULT_THEME.spacing.md)
        
        # Create custom preset view directly
        self.controller = CustomPresetController(main_container, self.state)
        self.controller.get_view().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self._create_status_bar()
    
    def _create_status_bar(self):
        """Create application status bar."""
        status_bar = ttk.Frame(self, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(
            status_bar, 
            text="Pronto | Audio Visual FX Generator v2.0",
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=DEFAULT_THEME.spacing.md, 
                              pady=DEFAULT_THEME.spacing.xs)


def run_application():
    """Run the application."""
    app = AudioVisualFXApp()
    app.mainloop()


if __name__ == "__main__":
    run_application()
