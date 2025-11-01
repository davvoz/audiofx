"""
Video Audio Synchronization with Audio-Reactive Effects.
Synchronizes a video with an audio track and applies audio-reactive effects.
"""

from typing import Optional, Callable
import numpy as np
import cv2
import librosa
import os
import tempfile
import subprocess

from .models.data_models import EffectStyle, EffectConfig
from .core.audio_analyzer import AudioAnalyzer
from .core.frame_generator import FrameGenerator
from .core.video_exporter import VideoExporter
from .factories.effect_style_manager import EffectStyleManager


class VideoAudioSync:
    """Synchronizes video with audio and applies audio-reactive effects."""
    
    def __init__(
        self,
        audio_file: str,
        video_file: str,
        output_file: str = "synced_output.mp4",
        effect_style: EffectStyle = EffectStyle.STANDARD,
        effect_config: Optional[EffectConfig] = None,
        short_video_mode: str = "loop",
        logo_file: Optional[str] = None,
        logo_position: str = "top-right",
        logo_scale: float = 0.15,
        logo_opacity: float = 1.0,
        logo_margin: int = 12,
        progress_cb: Optional[Callable] = None,
    ):
        """
        Initialize video audio sync.
        
        Args:
            audio_file: Path to audio file
            video_file: Path to input video file
            output_file: Path to output video file
            effect_style: Style of effects to apply
            effect_config: Optional effect configuration
            short_video_mode: "loop" to repeat short videos, "stretch" to slow them down
            logo_file: Optional logo image path
            logo_position: Logo position (top-left, top-right, bottom-left, bottom-right)
            logo_scale: Logo width as fraction of frame width
            logo_opacity: Logo opacity (0.0 to 1.0)
            logo_margin: Logo margin in pixels
            progress_cb: Optional progress callback
        """
        self.audio_file = audio_file
        self.video_file = video_file
        self.output_file = output_file
        self.effect_style = effect_style
        self.effect_config = effect_config or EffectConfig()
        self.short_video_mode = short_video_mode
        self.logo_file = logo_file
        self.logo_position = logo_position
        self.logo_scale = logo_scale
        self.logo_opacity = logo_opacity
        self.logo_margin = logo_margin
        self.progress_cb = progress_cb
        
        # Components
        self.audio_analyzer: Optional[AudioAnalyzer] = None
        self.frame_generator: Optional[FrameGenerator] = None
        self.effect_manager: Optional[EffectStyleManager] = None
        
        # Video properties
        self.video_fps: float = 0
        self.video_resolution: tuple = (0, 0)
        self.video_frames: list = []
        self.audio_duration: float = 0
        
        # Logo cache
        self._logo_rgba: Optional[np.ndarray] = None
    
    def _notify(self, event: str, payload: dict) -> None:
        """Send progress notification."""
        if self.progress_cb:
            self.progress_cb(event, payload)
    
    def _prepare_logo(self, frame_w: int, frame_h: int) -> None:
        """Prepare logo for overlay."""
        if not self.logo_file:
            return
        
        logo = cv2.imread(self.logo_file, cv2.IMREAD_UNCHANGED)
        if logo is None:
            self._logo_rgba = None
            return
        
        # Ensure 4 channels RGBA
        if logo.shape[2] == 3:
            alpha = np.full(logo.shape[:2] + (1,), 255, dtype=np.uint8)
            logo = np.concatenate([logo, alpha], axis=2)
        
        # Convert BGR(A) to RGB(A)
        if logo.shape[2] == 4:
            b, g, r, a = cv2.split(logo)
            logo_rgba = cv2.merge([r, g, b, a])
        else:
            b, g, r = cv2.split(logo)
            logo_rgba = cv2.merge([r, g, b, np.full_like(r, 255)])
        
        # Resize according to frame width
        target_w = max(1, int(frame_w * self.logo_scale))
        h0, w0 = logo_rgba.shape[:2]
        target_h = max(1, int(h0 * (target_w / float(w0))))
        logo_rgba = cv2.resize(logo_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Normalize to 0..1 floats and apply opacity
        logo_rgba = logo_rgba.astype(np.float32) / 255.0
        logo_rgba[:, :, 3] = np.clip(logo_rgba[:, :, 3] * self.logo_opacity, 0.0, 1.0)
        self._logo_rgba = logo_rgba
    
    def _overlay_logo(self, frame: np.ndarray) -> np.ndarray:
        """Overlay logo on frame."""
        if not self.logo_file:
            return frame
        
        h, w = frame.shape[:2]
        if self._logo_rgba is None:
            self._prepare_logo(w, h)
        if self._logo_rgba is None:
            return frame
        
        lh, lw = self._logo_rgba.shape[:2]
        
        # Position
        pos = (self.logo_position or "top-right").lower()
        margin = self.logo_margin
        if pos == "top-left":
            x, y = margin, margin
        elif pos == "bottom-left":
            x, y = margin, max(margin, h - lh - margin)
        elif pos == "bottom-right":
            x, y = max(margin, w - lw - margin), max(margin, h - lh - margin)
        else:  # top-right default
            x, y = max(margin, w - lw - margin), margin
        
        # Clip ROI if needed
        x = max(0, min(x, w - lw))
        y = max(0, min(y, h - lh))
        
        roi = frame[y : y + lh, x : x + lw].astype(np.float32) / 255.0
        logo_rgb = self._logo_rgba[:, :, :3]
        alpha = self._logo_rgba[:, :, 3:4]  # HxWx1
        comp = logo_rgb * alpha + roi * (1.0 - alpha)
        frame[y : y + lh, x : x + lw] = np.clip(comp * 255.0, 0, 255).astype(np.uint8)
        
        return frame
    
    def _load_video(self) -> None:
        """Load video and extract frames."""
        self._notify("status", {"message": "Caricamento video..."})
        
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_file}")
        
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_resolution = (width, height)
        
        # Get frame count without loading frames
        self.total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if self.total_video_frames == 0:
            raise ValueError("No frames in video file")
        
        self._notify("status", {
            "message": f"Video analizzato: {self.total_video_frames} frames @ {self.video_fps:.2f} fps"
        })
    
    def _sync_video_to_audio(self) -> list:
        """
        Synchronize video frames to audio duration.
        
        Returns:
            List of frame indices to use for output
        """
        video_duration = self.total_video_frames / self.video_fps
        
        self._notify("status", {
            "message": f"Sincronizzazione: audio={self.audio_duration:.2f}s, video={video_duration:.2f}s"
        })
        
        # Calculate required frames
        required_frames = int(self.audio_duration * self.video_fps)
        available_frames = self.total_video_frames
        
        if required_frames <= available_frames:
            # Video is longer or equal - just trim
            self._notify("status", {"message": "Video più lungo: taglio frames"})
            return list(range(required_frames))
        else:
            # Video is shorter - loop or stretch
            if self.short_video_mode == "loop":
                # Repeat video frames to match audio duration
                self._notify("status", {"message": "Video più corto: loop frames"})
                frame_indices = []
                while len(frame_indices) < required_frames:
                    remaining = required_frames - len(frame_indices)
                    frame_indices.extend(range(min(available_frames, remaining)))
                return frame_indices
            else:  # stretch
                # Slow down video by repeating frames
                self._notify("status", {"message": "Video più corto: stretch (rallentamento)"})
                stretch_factor = required_frames / available_frames
                frame_indices = []
                for i in range(available_frames):
                    repeat_count = int(stretch_factor)
                    if (i * stretch_factor) % 1 >= (1 - stretch_factor % 1):
                        repeat_count += 1
                    frame_indices.extend([i] * repeat_count)
                # Adjust to exact required length
                return frame_indices[:required_frames]
    
    def sync(self) -> None:
        """Execute video audio synchronization with effects."""
        # Load video first to get FPS
        self._load_video()
        
        # Load and analyze audio with correct FPS
        self._notify("status", {"message": "Analisi audio..."})
        self.audio_analyzer = AudioAnalyzer()
        audio_data = self.audio_analyzer.load_and_analyze(self.audio_file, fps=int(self.video_fps))
        self.audio_duration = len(audio_data.audio_signal) / audio_data.sample_rate
        
        # Synchronize frame indices
        frame_indices = self._sync_video_to_audio()
        
        # Setup effect manager
        self.effect_config.resolution = self.video_resolution
        self.effect_manager = EffectStyleManager(self.effect_config)
        pipeline = self.effect_manager.get_pipeline(self.effect_style)
        
        # Generate frames with effects using streaming
        total_frames = len(frame_indices)
        self._notify("start", {"total_frames": total_frames})
        self._notify("status", {"message": "Processing video con streaming..."})
        
        # Open video for streaming
        cap = cv2.VideoCapture(self.video_file)
        width, height = self.video_resolution
        
        # Write directly to temporary video file (use AVI temp)
        import tempfile
        import os
        temp_video_avi = tempfile.mktemp(suffix='_temp.avi', dir=os.path.dirname(self.output_file))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(temp_video_avi, fourcc, int(self.video_fps), (width, height))
        
        # Cache for loop mode
        frame_cache = {}
        last_frame_idx = -1
        cached_frame = None
        
        for idx, frame_idx in enumerate(frame_indices):
            self._notify("frame", {"index": idx + 1, "total": total_frames})
            
            # Load frame only if needed (optimize for sequential and loop access)
            if frame_idx == last_frame_idx and cached_frame is not None:
                base_frame = cached_frame.copy()
            elif frame_idx in frame_cache:
                base_frame = frame_cache[frame_idx].copy()
            else:
                # Seek and read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    if cached_frame is not None:
                        base_frame = cached_frame.copy()
                    else:
                        base_frame = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    base_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Cache frame if in loop mode and not too many cached
                    if self.short_video_mode == "loop" and len(frame_cache) < 300:
                        frame_cache[frame_idx] = base_frame.copy()
            
            cached_frame = base_frame.copy()
            last_frame_idx = frame_idx
            
            # Setup frame generator for this frame
            frame_generator = FrameGenerator(
                base_image=base_frame,
                effect_pipeline=pipeline,
                fps=int(self.video_fps)
            )
            
            # Generate frame with effects
            frame_with_effects = frame_generator.generate_frame(
                frame_index=idx,
                audio_analysis=audio_data,
                color_index=0
            )
            
            # Apply logo if configured
            if self.logo_file:
                frame_with_effects = self._overlay_logo(frame_with_effects)
            
            # Write directly to video
            writer.write(cv2.cvtColor(frame_with_effects, cv2.COLOR_RGB2BGR))
            
            # Free memory
            del base_frame
            del frame_with_effects
        
        # Release resources
        cap.release()
        writer.release()
        frame_cache.clear()
        
        # Add audio to video
        self._notify("status", {"message": "Aggiunta audio al video..."})
        import subprocess
        import os
        import shutil
        
        audio_added = False
        
        # Try ffmpeg first (fastest)
        try:
            result = subprocess.run([
                'ffmpeg', '-y', '-i', temp_video_avi, '-i', self.audio_file,
                '-c:v', 'libx264', '-c:a', 'aac', '-shortest', self.output_file
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
                
                self._notify("status", {"message": "Aggiunta audio con moviepy..."})
                video_clip = VideoFileClip(temp_video_avi)
                audio_clip = AudioFileClip(self.audio_file)
                final_clip = video_clip.with_audio(audio_clip)
                final_clip.write_videofile(self.output_file, codec='libx264', audio_codec='aac', logger=None)
                video_clip.close()
                audio_clip.close()
                final_clip.close()
                os.remove(temp_video_avi)
                audio_added = True
            except Exception as e2:
                # Last resort: convert video without audio
                try:
                    subprocess.run(['ffmpeg', '-y', '-i', temp_video_avi, '-c:v', 'libx264', self.output_file], check=False)
                    os.remove(temp_video_avi)
                except:
                    pass
        
        self._notify("done", {"output": self.output_file})
