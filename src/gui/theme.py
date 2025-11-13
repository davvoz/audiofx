"""
Theme and design system for the application.
Provides consistent colors, fonts, and styling across the UI.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ColorScheme:
    """Application color scheme."""
    primary: str = "#2563eb"  # Blue
    secondary: str = "#7c3aed"  # Purple
    success: str = "#10b981"  # Green
    warning: str = "#f59e0b"  # Orange
    danger: str = "#ef4444"  # Red
    background: str = "#ffffff"
    surface: str = "#f3f4f6"
    text_primary: str = "#111827"
    text_secondary: str = "#6b7280"
    border: str = "#d1d5db"
    hover: str = "#e5e7eb"


@dataclass
class Typography:
    """Typography system."""
    font_family: str = "Segoe UI"
    font_size_small: int = 8
    font_size_normal: int = 9
    font_size_medium: int = 10
    font_size_large: int = 11
    font_size_xlarge: int = 12
    font_weight_normal: str = "normal"
    font_weight_bold: str = "bold"


@dataclass
class Spacing:
    """Spacing system."""
    xs: int = 2
    sm: int = 4
    md: int = 8
    lg: int = 12
    xl: int = 16
    xxl: int = 24


@dataclass
class Theme:
    """Complete theme configuration."""
    colors: ColorScheme
    typography: Typography
    spacing: Spacing
    
    # Widget specific styles
    button_padding: Dict[str, int]
    entry_padding: Dict[str, int]
    frame_padding: Dict[str, int]
    
    def __init__(self):
        self.colors = ColorScheme()
        self.typography = Typography()
        self.spacing = Spacing()
        self.button_padding = {'padx': self.spacing.md, 'pady': self.spacing.sm}
        self.entry_padding = {'padx': self.spacing.md, 'pady': self.spacing.md}
        self.frame_padding = {'padx': self.spacing.lg, 'pady': self.spacing.md}


# Global theme instance
DEFAULT_THEME = Theme()
