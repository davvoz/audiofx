"""
Video export component.
"""

import os
import shutil
from typing import List, Optional
import numpy as np
import cv2

try:
    # MoviePy 2.x
    from moviepy import AudioFileClip, ImageClip
    from moviepy import concatenate_videoclips
    MOVIEPY_VERSION = 2
except ImportError:
    try:
        # MoviePy 1.x
        from moviepy.editor import AudioFileClip, ImageSequenceClip # type: ignore
        MOVIEPY_VERSION = 1
    except ImportError:
        from moviepy.editor import AudioFileClip, VideoFileClip # type: ignore
        ImageSequenceClip = None
        MOVIEPY_VERSION = 1

from ..models.data_models import ProgressCallback


class VideoExporter:
    """Handles video export operations."""
    
    def __init__(self, output_file: str, fps: int = 30):
        """
        Initialize video exporter.
        
        Args:
            output_file: Path for output video file
            fps: Frames per second
        """
        self.output_file = output_file
        self.fps = fps
    
    def export(
        self,
        frames: List[np.ndarray],
        audio_file: str,
        progress_cb: ProgressCallback = None
    ) -> None:
        """
        Export frames to video with audio.
        
        Args:
            frames: List of video frames
            audio_file: Path to audio file
            progress_cb: Optional progress callback
        """
        if progress_cb:
            progress_cb("status", {"message": "Creating video file..."})
        
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save frames as temporary images
            frame_paths = []
            for i, frame in enumerate(frames):
                path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                # Convert RGB to BGR for OpenCV
                cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(path)
                
                if progress_cb and i % 30 == 0:
                    progress_cb("progress", {
                        "message": f"Saving frames: {i}/{len(frames)}",
                        "percent": (i / len(frames)) * 50
                    })
            
            # Create video clip
            if progress_cb:
                progress_cb("status", {"message": "Encoding video..."})
            
            if MOVIEPY_VERSION == 2:
                # MoviePy 2.x: Create clips from individual frames
                clips = []
                for frame_path in frame_paths:
                    img_clip = ImageClip(frame_path, duration=1/self.fps)
                    clips.append(img_clip)
                
                video_clip = concatenate_videoclips(clips, method="compose")
                audio = AudioFileClip(audio_file)
                final_clip = video_clip.with_audio(audio)
            else:
                # MoviePy 1.x: Use ImageSequenceClip
                clip = ImageSequenceClip(frame_paths, fps=self.fps)
                audio = AudioFileClip(audio_file)
                final_clip = clip.set_audio(audio)
            
            # Write video file
            if MOVIEPY_VERSION == 2:
                # MoviePy 2.x: no verbose parameter
                final_clip.write_videofile(
                    self.output_file,
                    codec='libx264',
                    audio_codec='aac',
                    fps=self.fps,
                    logger=None
                )
            else:
                # MoviePy 1.x: supports verbose
                final_clip.write_videofile(
                    self.output_file,
                    codec='libx264',
                    audio_codec='aac',
                    fps=self.fps,
                    verbose=False,
                    logger=None
                )
            
            if progress_cb:
                progress_cb("done", {"output": self.output_file})
        
        finally:
            # Cleanup temporary files
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def export_frames_only(
        self,
        frames: List[np.ndarray],
        output_dir: str,
        progress_cb: ProgressCallback = None
    ) -> None:
        """
        Export frames without creating video.
        
        Args:
            frames: List of video frames
            output_dir: Directory to save frames
            progress_cb: Optional progress callback
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            path = os.path.join(output_dir, f"frame_{i:06d}.png")
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            if progress_cb and i % 30 == 0:
                progress_cb("progress", {
                    "current": i,
                    "total": len(frames),
                    "percent": (i / len(frames)) * 100
                })
        
        if progress_cb:
            progress_cb("done", {"output": output_dir})
