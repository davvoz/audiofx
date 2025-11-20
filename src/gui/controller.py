"""
Controller for custom preset view.
Handles user interactions and coordinates between model and view.
"""

from tkinter import messagebox
from typing import Optional
from .models import ApplicationState
from .views import CustomPresetView
from .services import ValidationService
from .video_generator import VideoGenerationService


class CustomPresetController:
    """Controller for custom preset configuration and generation."""
    
    def __init__(self, parent, state: ApplicationState):
        self.state = state
        self.video_generator: Optional[VideoGenerationService] = None
        
        # Create view with callback bindings
        self.view = CustomPresetView(parent, state, {
            'on_mode_change': self._on_mode_change,
            'on_video_audio_toggle': self._on_video_audio_toggle,
            'on_effect_change': self._on_effect_change,
            'on_move_up': self._on_move_up,
            'on_move_down': self._on_move_down,
            'on_move_top': self._on_move_top,
            'on_move_bottom': self._on_move_bottom,
            'on_reset_order': self._on_reset_order,
            'on_generate': self._on_generate,
            'on_cancel': self._on_cancel
        })
    
    def get_view(self) -> CustomPresetView:
        """Get the view instance."""
        return self.view
    
    def _on_mode_change(self):
        """Handle mode change (image/video)."""
        self.view._toggle_mode_visibility()
    
    def _on_video_audio_toggle(self):
        """Handle video audio checkbox toggle."""
        if self.view.use_video_audio_var.get():
            self.view.audio_selector.configure_entry(state='disabled')
            self.view.audio_path_var.set("(audio dal video)")
        else:
            self.view.audio_selector.configure_entry(state='normal')
            if self.view.audio_path_var.get() == "(audio dal video)":
                self.view.audio_path_var.set("")
    
    def _on_effect_change(self):
        """Handle effect checkbox change."""
        self.view.refresh_order_listbox()
    
    def _on_move_up(self):
        """Move selected effect up in order."""
        selection = self.view.order_listbox.curselection()
        if not selection or selection[0] == 0:
            return
        
        selected_effect = self.view.order_listbox.get(selection[0])
        idx = self.state.effects_config.effect_order.index(selected_effect)
        
        if idx == 0:
            return
        
        # Swap in order
        self.state.effects_config.effect_order[idx], self.state.effects_config.effect_order[idx-1] = \
            self.state.effects_config.effect_order[idx-1], self.state.effects_config.effect_order[idx]
        
        self.view.refresh_order_listbox()
        new_idx = max(0, selection[0] - 1)
        if new_idx < self.view.order_listbox.size():
            self.view.order_listbox.selection_set(new_idx)
    
    def _on_move_down(self):
        """Move selected effect down in order."""
        selection = self.view.order_listbox.curselection()
        if not selection:
            return
        
        selected_effect = self.view.order_listbox.get(selection[0])
        idx = self.state.effects_config.effect_order.index(selected_effect)
        
        if idx >= len(self.state.effects_config.effect_order) - 1:
            return
        
        # Swap in order
        self.state.effects_config.effect_order[idx], self.state.effects_config.effect_order[idx+1] = \
            self.state.effects_config.effect_order[idx+1], self.state.effects_config.effect_order[idx]
        
        self.view.refresh_order_listbox()
        new_idx = min(selection[0] + 1, self.view.order_listbox.size() - 1)
        if new_idx >= 0:
            self.view.order_listbox.selection_set(new_idx)
    
    def _on_move_top(self):
        """Move selected effect to top."""
        selection = self.view.order_listbox.curselection()
        if not selection:
            return
        
        selected_effect = self.view.order_listbox.get(selection[0])
        idx = self.state.effects_config.effect_order.index(selected_effect)
        
        effect = self.state.effects_config.effect_order.pop(idx)
        self.state.effects_config.effect_order.insert(0, effect)
        
        self.view.refresh_order_listbox()
        self.view.order_listbox.selection_set(0)
    
    def _on_move_bottom(self):
        """Move selected effect to bottom."""
        selection = self.view.order_listbox.curselection()
        if not selection:
            return
        
        selected_effect = self.view.order_listbox.get(selection[0])
        idx = self.state.effects_config.effect_order.index(selected_effect)
        
        effect = self.state.effects_config.effect_order.pop(idx)
        self.state.effects_config.effect_order.append(effect)
        
        self.view.refresh_order_listbox()
        last_idx = self.view.order_listbox.size() - 1
        if last_idx >= 0:
            self.view.order_listbox.selection_set(last_idx)
    
    def _on_reset_order(self):
        """Reset effect order to default."""
        self.state.effects_config.effect_order = [
            "ColorPulse", "ZoomPulse", "Strobe", "StrobeNegative", "Glitch",
            "ChromaticAberration", "BubbleDistortion", "ScreenShake", "RGBSplit",
            "ElectricArcs", "FashionLightning", "AdvancedGlitch",
            "DimensionalWarp", "VortexDistortion", "FloatingText", "GhostParticles",
            "TextureStretch"
        ]
        self.view.refresh_order_listbox()
    
    def _on_generate(self):
        """Handle generate video button click."""
        # Sync state from UI
        self.view.sync_state_from_ui()
        
        # Validate configuration
        is_valid, error_msg = ValidationService.validate_video_config(self.state.video_config)
        if not is_valid:
            messagebox.showerror("Errore", error_msg)
            return
        
        is_valid, error_msg = ValidationService.validate_effects(self.state.effects_config)
        if not is_valid:
            messagebox.showerror("Errore", error_msg)
            return
        
        # Update UI state
        self.state.is_processing = True
        self.state.cancel_requested = False
        self.view.set_processing_state(True)
        self.view.update_status("In esecuzione...")
        self.view.update_progress(0, 100)
        
        # Create and start video generator
        self.video_generator = VideoGenerationService(progress_callback=self._on_progress)
        
        try:
            self.video_generator.generate_video(
                self.state.video_config,
                self.state.audio_thresholds,
                self.state.effects_config
            )
        except Exception as e:
            self._on_error(str(e))
    
    def _on_progress(self, event: str, payload: dict):
        """Handle progress events from video generator."""
        if event == "start":
            total = max(1, int(payload.get("total_frames", 100)))
            self.view.after(0, lambda: self.view.update_progress(0, total))
        
        elif event == "frame":
            idx = int(payload.get("index", 0))
            self.view.after(0, lambda i=idx: self.view.update_progress(i, self.view.progress['maximum']))
        
        elif event == "status":
            msg = payload.get("message", "")
            self.view.after(0, lambda m=msg: self.view.update_status(m))
        
        elif event == "done":
            output = payload.get("output", "output.mp4")
            self.view.after(0, lambda o=output: self._on_complete(o))
        
        elif event == "error":
            error = payload.get("message", "Errore sconosciuto")
            self.view.after(0, lambda e=error: self._on_error(e))
    
    def _on_complete(self, output: str):
        """Handle video generation completion."""
        self.state.reset_processing_state()
        self.view.set_processing_state(False)
        self.view.update_status(f"Completato: {output}")
        messagebox.showinfo("Completato", f"Video creato: {output}")
    
    def _on_error(self, error_msg: str):
        """Handle video generation error."""
        self.state.reset_processing_state()
        self.view.set_processing_state(False)
        self.view.update_status(f"Errore: {error_msg}")
        messagebox.showerror("Errore", error_msg)
    
    def _on_cancel(self):
        """Handle cancel button click."""
        if self.video_generator:
            self.video_generator.cancel()
            self.state.cancel_requested = True
            self.view.update_status("Annullamento richiesto...")
            messagebox.showinfo("Annulla", "Annullamento in corso...")
