"""
Streaming video writer for memory-efficient video export.
Professional-grade implementation that writes frames directly to disk.
"""

import os
import subprocess
import numpy as np
import cv2
from typing import Optional, Tuple
from ..models.data_models import ProgressCallback


class StreamingVideoWriter:
    """
    Memory-efficient video writer that streams frames directly to disk.
    
    This avoids the memory explosion of accumulating all frames in RAM.
    Professional video software (Premiere, DaVinci) work this way.
    """
    
    def __init__(
        self,
        output_file: str,
        fps: int,
        resolution: Tuple[int, int],
        codec: str = 'libx264',
        crf: int = 18,
        preset: str = 'medium'
    ):
        """
        Initialize streaming video writer.
        
        Args:
            output_file: Output video file path
            fps: Frames per second
            resolution: Video resolution (width, height)
            codec: Video codec (default: libx264 for H.264)
            crf: Quality (0-51, lower is better, 18 is visually lossless)
            preset: Encoding speed (ultrafast, fast, medium, slow, veryslow)
        """
        self.output_file = output_file
        self.fps = fps
        self.width, self.height = resolution
        self.codec = codec
        self.crf = crf
        self.preset = preset
        
        # Write directly to output_file (caller handles temp naming)
        self.temp_video = output_file
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        
    def __enter__(self):
        """Context manager entry - opens video writer."""
        # Ensure output directory exists
        output_dir = os.path.dirname(self.temp_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Temporary codec
        self.writer = cv2.VideoWriter(
            self.temp_video,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open: {self.temp_video}. Check path and codec.")
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to video (streaming mode).
        
        Args:
            frame: RGB frame to write
        """
        if self.writer is None:
            raise RuntimeError("Writer not opened. Use context manager.")
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Ensure correct size
        if frame_bgr.shape[:2] != (self.height, self.width):
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        
        self.writer.write(frame_bgr)
        self.frame_count += 1
    
    def finalize_with_audio(
        self,
        audio_file: str,
        progress_cb: Optional[ProgressCallback] = None
    ) -> None:
        """
        Finalize video by adding audio using FFmpeg.
        
        This is much faster than MoviePy's approach.
        
        Args:
            audio_file: Path to audio file
            progress_cb: Optional progress callback
        """
        if progress_cb:
            progress_cb("status", {"message": "Adding audio with FFmpeg..."})
        
        # Close the writer first
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        # Use FFmpeg to re-encode with better codec and add audio
        # This is what professional software does under the hood
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', self.temp_video,  # Input video (no audio)
            '-i', audio_file,  # Input audio
            '-c:v', self.codec,  # Video codec
            '-crf', str(self.crf),  # Quality
            '-preset', self.preset,  # Speed/compression tradeoff
            '-c:a', 'aac',  # Audio codec
            '-b:a', '192k',  # Audio bitrate
            '-shortest',  # Match shortest stream
            '-movflags', '+faststart',  # Enable web streaming
            self.output_file
        ]
        
        try:
            # Run FFmpeg
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Clean up temp file
            if os.path.exists(self.temp_video):
                os.remove(self.temp_video)
            
            if progress_cb:
                progress_cb("done", {"output": self.output_file})
                
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg failed: {e.stderr}"
            if progress_cb:
                progress_cb("error", {"message": error_msg})
            raise RuntimeError(error_msg)
        except FileNotFoundError:
            # FFmpeg not installed - fallback to simpler method
            if progress_cb:
                progress_cb("status", {
                    "message": "FFmpeg not found, using fallback method..."
                })
            self._fallback_audio_merge(audio_file, progress_cb)
    
    def _fallback_audio_merge(
        self,
        audio_file: str,
        progress_cb: Optional[ProgressCallback] = None
    ) -> None:
        """
        Fallback audio merge using MoviePy (slower but no FFmpeg dependency).
        
        Args:
            audio_file: Path to audio file
            progress_cb: Optional progress callback
        """
        try:
            from moviepy import VideoFileClip, AudioFileClip
            
            video = VideoFileClip(self.temp_video)
            audio = AudioFileClip(audio_file)
            
            final = video.with_audio(audio)
            final.write_videofile(
                self.output_file,
                codec=self.codec,
                audio_codec='aac',
                logger=None
            )
            
            video.close()
            audio.close()
            final.close()
            
            # Clean up
            if os.path.exists(self.temp_video):
                os.remove(self.temp_video)
            
            if progress_cb:
                progress_cb("done", {"output": self.output_file})
                
        except ImportError:
            if progress_cb:
                progress_cb("error", {
                    "message": "Neither FFmpeg nor MoviePy available for audio merge"
                })
            raise RuntimeError(
                "Cannot merge audio: Install FFmpeg or MoviePy 2.x"
            )


class ChunkedVideoProcessor:
    """
    Process video in chunks to avoid memory overflow for long videos.
    
    Professional approach: process N seconds at a time, write to disk, repeat.
    """
    
    def __init__(self, chunk_duration: float = 30.0):
        """
        Initialize chunked processor.
        
        Args:
            chunk_duration: Duration of each chunk in seconds (default: 30s)
        """
        self.chunk_duration = chunk_duration
    
    def get_chunk_ranges(
        self,
        total_frames: int,
        fps: int
    ) -> list[Tuple[int, int]]:
        """
        Calculate frame ranges for chunked processing.
        
        Args:
            total_frames: Total number of frames
            fps: Frames per second
            
        Returns:
            List of (start_frame, end_frame) tuples
        """
        frames_per_chunk = int(self.chunk_duration * fps)
        chunks = []
        
        for start in range(0, total_frames, frames_per_chunk):
            end = min(start + frames_per_chunk, total_frames)
            chunks.append((start, end))
        
        return chunks
