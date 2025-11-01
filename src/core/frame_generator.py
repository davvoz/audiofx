"""
Frame generation component.
"""

import numpy as np

from ..models.data_models import AudioAnalysis, FrameContext
from ..effects.effect_pipeline import EffectPipeline
from .beat_detector import BeatDetector


class FrameGenerator:
    """Generates individual frames with effects applied."""
    
    def __init__(
        self,
        base_image: np.ndarray,
        effect_pipeline: EffectPipeline,
        fps: int = 30
    ):
        """
        Initialize frame generator.
        
        Args:
            base_image: Base image to apply effects to
            effect_pipeline: Pipeline of effects to apply
            fps: Frames per second
        """
        self.base_image = base_image
        self.effect_pipeline = effect_pipeline
        self.fps = fps
    
    def generate_frame(
        self,
        frame_index: int,
        audio_analysis: AudioAnalysis,
        color_index: int = 0
    ) -> np.ndarray:
        """
        Generate a single frame with effects.
        
        Args:
            frame_index: Index of frame to generate
            audio_analysis: Audio analysis results
            color_index: Current color palette index
            
        Returns:
            Processed frame with effects applied
        """
        frame = self.base_image.copy()
        current_time = frame_index / self.fps
        
        # Ensure frame_index is within bounds
        bass_idx = min(frame_index, len(audio_analysis.bass_energy) - 1)
        mid_idx = min(frame_index, len(audio_analysis.mid_energy) - 1)
        treble_idx = min(frame_index, len(audio_analysis.treble_energy) - 1)
        
        # Create frame context with audio data
        context = FrameContext(
            frame=frame,
            time=current_time,
            frame_index=frame_index,
            bass=float(audio_analysis.bass_energy[bass_idx]),
            mid=float(audio_analysis.mid_energy[mid_idx]),
            treble=float(audio_analysis.treble_energy[treble_idx]),
            beat_intensity=self._calculate_beat_intensity(
                current_time, 
                audio_analysis.beat_times
            ),
            color_index=color_index
        )
        
        # Apply effect pipeline
        return self.effect_pipeline.apply(context)
    
    def generate_frames(
        self,
        audio_analysis: AudioAnalysis,
        num_frames: int,
        progress_cb=None
    ) -> list:
        """
        Generate all frames for video.
        
        Args:
            audio_analysis: Audio analysis results
            num_frames: Number of frames to generate
            progress_cb: Optional progress callback
            
        Returns:
            List of generated frames
        """
        frames = []
        
        for i in range(num_frames):
            # Calculate color index based on audio energy
            bass_idx = min(i, len(audio_analysis.bass_energy) - 1)
            mid_idx = min(i, len(audio_analysis.mid_energy) - 1)
            treble_idx = min(i, len(audio_analysis.treble_energy) - 1)
            
            total_energy = (
                audio_analysis.bass_energy[bass_idx] +
                audio_analysis.mid_energy[mid_idx] +
                audio_analysis.treble_energy[treble_idx]
            )
            color_index = int(total_energy * 10)
            
            # Generate frame
            frame = self.generate_frame(i, audio_analysis, color_index)
            frames.append(frame)
            
            # Progress callback
            if progress_cb and i % 30 == 0:
                progress_cb("progress", {
                    "current": i,
                    "total": num_frames,
                    "percent": (i / num_frames) * 100
                })
        
        return frames
    
    @staticmethod
    def _calculate_beat_intensity(current_time: float, beat_times: np.ndarray) -> float:
        """
        Calculate beat intensity at given time using BeatDetector.
        
        Args:
            current_time: Current time in seconds
            beat_times: Array of beat timestamps
            
        Returns:
            Beat intensity value (0.0-1.0)
        """
        return BeatDetector.calculate_beat_intensity(current_time, beat_times)
