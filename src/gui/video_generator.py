"""
Video generation service.
Handles the complex video generation logic separately from UI.
"""

import os
import threading
from typing import Callable, Optional
from .models import VideoGenerationConfig, AudioThresholds, EffectsConfiguration
from .services import EffectFactoryService


class VideoGenerationService:
    """Service for generating videos from configuration."""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self._worker_thread: Optional[threading.Thread] = None
        self._cancel_requested = False
    
    def generate_video(self, config: VideoGenerationConfig, 
                      thresholds: AudioThresholds,
                      effects_config: EffectsConfiguration) -> None:
        """
        Start video generation in background thread.
        
        Args:
            config: Video generation configuration
            thresholds: Audio thresholds
            effects_config: Effects configuration
        """
        self._cancel_requested = False
        
        def worker():
            try:
                self._generate_video_internal(config, thresholds, effects_config)
            except Exception as e:
                self._emit_event('error', {'message': str(e)})
        
        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
    
    def _generate_video_internal(self, config: VideoGenerationConfig,
                                 thresholds: AudioThresholds,
                                 effects_config: EffectsConfiguration):
        """Internal video generation logic."""
        from src.factories import EffectFactory
        
        # Get custom duration if specified
        custom_duration = None
        if config.duration:
            try:
                custom_duration = float(config.duration)
                if custom_duration <= 0:
                    raise ValueError("La durata deve essere maggiore di 0")
                self._emit_event('status', {'message': f"Usando durata: {custom_duration}s"})
            except ValueError as e:
                raise ValueError(f"Durata non valida: {e}")
        
        # Create custom effects
        custom_effects = EffectFactoryService.create_effects_from_config(
            effects_config, thresholds
        )
        
        # Create custom pipeline
        custom_pipeline = EffectFactory.create_custom_pipeline(custom_effects)
        
        if config.mode == "image":
            self._generate_from_image(config, custom_pipeline, custom_duration)
        else:
            self._generate_from_video(config, custom_pipeline, custom_duration)
    
    def _generate_from_image(self, config: VideoGenerationConfig,
                            custom_pipeline, custom_duration: Optional[float]):
        """Generate video from image."""
        from src.core.audio_analyzer import AudioAnalyzer
        from src.core.frame_generator import FrameGenerator
        from src.utils.image_loader import ImageLoader
        from src.utils.logo_overlay import load_logo, apply_logo_to_frame
        import cv2
        import subprocess
        import tempfile
        
        # Analyze audio
        self._emit_event('status', {'message': "Analisi audio..."})
        audio_analyzer = AudioAnalyzer()
        audio_data = audio_analyzer.load_and_analyze(
            config.audio_path, duration=custom_duration, fps=config.fps
        )
        
        # Load image
        self._emit_event('status', {'message': "Caricamento immagine..."})
        target_res = None if config.use_native_resolution else (720, 720)
        base_image = ImageLoader.load_and_prepare(config.image_path, target_resolution=target_res)
        height, width = base_image.shape[:2]
        self._emit_event('status', {'message': f"Risoluzione: {width}x{height}"})
        
        # Create frame generator
        self._emit_event('status', {'message': "Setup effetti..."})
        frame_generator = FrameGenerator(
            base_image=base_image,
            effect_pipeline=custom_pipeline,
            fps=config.fps
        )
        
        # Generate frames using STREAMING (memory-efficient)
        num_frames = len(audio_data.bass_energy)
        self._emit_event('start', {'total_frames': num_frames})
        self._emit_event('status', {'message': "Generazione frame (streaming mode)..."})
        
        # Load logo if provided
        logo_img = None
        if config.logo_path and os.path.exists(config.logo_path):
            try:
                logo_img = load_logo(config.logo_path)
                self._emit_event('status', {'message': f"Logo caricato"})
            except Exception as e:
                self._emit_event('status', {'message': f"Errore logo: {e}"})
        
        # Use PARALLEL rendering for speed (much faster!)
        import multiprocessing as mp
        
        num_workers = max(1, mp.cpu_count() - 1)
        self._emit_event('status', {'message': f"Rendering con {num_workers} workers paralleli..."})
        
        # Generate frames in parallel
        frames = frame_generator.generate_frames_parallel(
            audio_analysis=audio_data,
            num_frames=num_frames,
            progress_cb=self._emit_event,
            num_workers=num_workers,
            batch_size=10
        )
        
        # Now write to video with streaming
        from src.core.streaming_video_writer import StreamingVideoWriter
        output_dir = os.path.dirname(os.path.abspath(config.output_path))
        temp_video_mp4 = os.path.join(output_dir, 'temp_video_no_audio.mp4')
        
        self._emit_event('status', {'message': f"Scrivendo {num_frames} frames su video..."})
        
        with StreamingVideoWriter(temp_video_mp4, config.fps, (width, height)) as writer:
            for idx, frame in enumerate(frames):
                if self._cancel_requested:
                    if os.path.exists(temp_video_mp4):
                        os.remove(temp_video_mp4)
                    return
                
                # Progress every 30 frames
                if idx % 30 == 0:
                    self._emit_event('status', {'message': f"Scrittura frame {idx}/{num_frames}"})
                
                # Apply logo
                if logo_img is not None:
                    frame = apply_logo_to_frame(
                        frame, logo_img, config.logo_position, 
                        config.logo_scale, config.logo_opacity, config.logo_margin
                    )
                
                writer.write_frame(frame)
        
        # Verify temp file exists
        if not os.path.exists(temp_video_mp4):
            raise RuntimeError(f"File temp non creato: {temp_video_mp4}")
        
        self._emit_event('status', {'message': f"Video temp creato: {os.path.getsize(temp_video_mp4)} bytes"})
        
        # Add audio with ffmpeg
        self._emit_event('status', {'message': "Aggiunta audio con FFmpeg..."})
        video_duration = num_frames / config.fps
        
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_video_mp4, '-i', config.audio_path,
                '-t', str(video_duration),
                '-c:v', 'libx264', '-c:a', 'aac', config.output_path
            ], check=True, capture_output=True)
            os.remove(temp_video_mp4)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to moviepy
            try:
                from moviepy import VideoFileClip, AudioFileClip
            except ImportError:
                from moviepy.editor import VideoFileClip, AudioFileClip
            
            video_clip = VideoFileClip(temp_video_mp4)
            audio_clip = AudioFileClip(config.audio_path).subclip(0, video_duration)
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(config.output_path, codec='libx264', 
                                      audio_codec='aac', logger=None)
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            os.remove(temp_video_mp4)
        
        self._emit_event('done', {'output': config.output_path})
    
    def _generate_from_video(self, config: VideoGenerationConfig,
                            custom_pipeline, custom_duration: Optional[float]):
        """Generate video from video."""
        import cv2
        import numpy as np
        import subprocess
        import tempfile
        from src.core.audio_analyzer import AudioAnalyzer
        from src.models.data_models import FrameContext
        from src.utils.logo_overlay import load_logo, apply_logo_to_frame
        
        # Get video metadata
        self._emit_event('status', {'message': "Analisi video..."})
        cap = cv2.VideoCapture(config.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        available_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize audio source
        audio_source = config.audio_path
        temp_audio = None
        
        if config.use_video_audio:
            # Extract audio from video
            self._emit_event('status', {'message': "Estrazione audio dal video..."})
            temp_audio = tempfile.mktemp(suffix='.wav', dir=os.path.dirname(config.output_path))
            
            try:
                # Find ffmpeg
                ffmpeg_cmd = 'ffmpeg'
                if os.path.exists(r'C:\ffmpeg\bin\ffmpeg.exe'):
                    ffmpeg_cmd = r'C:\ffmpeg\bin\ffmpeg.exe'
                
                subprocess.run([
                    ffmpeg_cmd, '-y', '-i', config.video_path, '-vn', 
                    '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', temp_audio
                ], check=True, capture_output=True, text=True)
                
                if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 1000:
                    audio_source = temp_audio
                    self._emit_event('status', {'message': "Audio estratto dal video con successo"})
                else:
                    self._emit_event('status', {'message': "Video senza audio, uso file audio fornito"})
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                # Fallback to moviepy
                try:
                    from moviepy.editor import VideoFileClip
                    
                    video_clip = VideoFileClip(config.video_path)
                    
                    if video_clip.audio is not None:
                        video_clip.audio.write_audiofile(temp_audio, logger=None)
                        audio_source = temp_audio
                        self._emit_event('status', {'message': "Audio estratto dal video con MoviePy"})
                    else:
                        self._emit_event('status', {'message': "Video senza audio, uso file audio fornito"})
                    
                    video_clip.close()
                except Exception as e2:
                    self._emit_event('status', {'message': f"Impossibile estrarre audio: {str(e2)}, uso file audio fornito"})
        
        # Analyze audio
        self._emit_event('status', {'message': "Analisi audio..."})
        audio_analyzer = AudioAnalyzer()
        audio_data = audio_analyzer.load_and_analyze(
            audio_source, duration=custom_duration, fps=int(video_fps)
        )
        audio_duration = len(audio_data.audio_signal) / audio_data.sample_rate
        
        # Calculate frame mapping
        required_frames = int(audio_duration * video_fps)
        
        if required_frames <= available_frames:
            frame_indices = list(range(required_frames))
        elif config.video_mode == "loop":
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
        
        total_frames = len(frame_indices)
        self._emit_event('start', {'total_frames': total_frames})
        
        # Load logo
        logo_img = None
        if config.logo_path and os.path.exists(config.logo_path):
            try:
                logo_img = load_logo(config.logo_path)
            except Exception:
                pass
        
        # Load all video frames
        self._emit_event('status', {'message': f"Caricamento {available_frames} frame..."})
        video_frames = []
        for i in range(available_frames):
            ret, frame = cap.read()
            if ret:
                video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        # Process frames
        self._emit_event('status', {'message': "Elaborazione frame..."})
        all_frames = []
        
        for idx, frame_idx in enumerate(frame_indices):
            if self._cancel_requested:
                break
            
            self._emit_event('frame', {'index': idx + 1, 'total': total_frames})
            
            base_frame = video_frames[frame_idx].copy() if frame_idx < len(video_frames) else video_frames[-1].copy()
            
            # Apply effects
            time = idx / video_fps
            bass = audio_data.bass_energy[idx] if idx < len(audio_data.bass_energy) else 0.0
            mid = audio_data.mid_energy[idx] if idx < len(audio_data.mid_energy) else 0.0
            treble = audio_data.treble_energy[idx] if idx < len(audio_data.treble_energy) else 0.0
            
            beat_intensity = 0.0
            for beat_time in audio_data.beat_times:
                if abs(beat_time - time) < 0.1:
                    beat_intensity = 1.0
                    break
            
            frame_context = FrameContext(
                frame=base_frame, time=time, frame_index=idx,
                bass=bass, mid=mid, treble=treble,
                beat_intensity=beat_intensity, color_index=0
            )
            
            frame_with_effects = custom_pipeline.apply(frame_context)
            
            # Apply logo
            if logo_img is not None:
                frame_with_effects = apply_logo_to_frame(
                    frame_with_effects, logo_img, config.logo_position,
                    config.logo_scale, config.logo_opacity, config.logo_margin
                )
            
            all_frames.append(frame_with_effects.copy())
            del base_frame, frame_with_effects
        
        if self._cancel_requested:
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
            return
        
        # Create final video
        self._emit_event('status', {'message': "Creazione video finale..."})
        
        try:
            from moviepy import ImageSequenceClip, AudioFileClip
        except ImportError:
            from moviepy.editor import ImageSequenceClip, AudioFileClip
        
        video_clip = ImageSequenceClip(all_frames, fps=video_fps)
        audio_clip = AudioFileClip(audio_source)
        
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        elif video_clip.duration > audio_clip.duration:
            video_clip = video_clip.subclip(0, audio_clip.duration)
        
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(config.output_path, codec='libx264', 
                                   audio_codec='aac', fps=video_fps, logger=None)
        
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        # Cleanup
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        self._emit_event('done', {'output': config.output_path})
    
    def _emit_event(self, event: str, payload: dict):
        """Emit a progress event."""
        if self.progress_callback:
            self.progress_callback(event, payload)
    
    def cancel(self):
        """Request cancellation of current operation."""
        self._cancel_requested = True
