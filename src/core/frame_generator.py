"""
Frame generation component.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import pickle

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
        current_time = frame_index / self.fps
        
        # Ensure frame_index is within bounds
        bass_idx = min(frame_index, len(audio_analysis.bass_energy) - 1)
        mid_idx = min(frame_index, len(audio_analysis.mid_energy) - 1)
        treble_idx = min(frame_index, len(audio_analysis.treble_energy) - 1)
        
        # Create frame context with audio data (without copying frame yet)
        context = FrameContext(
            frame=self.base_image,  # Reference, not copy - effects will copy when needed
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
        
        # Apply effect pipeline (will make copies as needed)
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
        import time
        frames = []
        start_time = time.time()
        
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
            
            # Progress callback with performance metrics
            if progress_cb and i % 30 == 0:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                remaining_frames = num_frames - i
                eta_seconds = remaining_frames / fps if fps > 0 else 0
                eta_minutes = int(eta_seconds / 60)
                
                progress_cb("progress", {
                    "current": i,
                    "total": num_frames,
                    "percent": (i / num_frames) * 100,
                    "message": f"Frame {i}/{num_frames} - {fps:.1f} FPS - ETA: {eta_minutes}m {int(eta_seconds % 60)}s"
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
    
    @staticmethod
    def _process_frame_batch(batch_data):
        """
        Process a batch of frames in parallel (static method for pickling).
        
        Args:
            batch_data: Tuple of (frame_indices, base_image, effect_pipeline, audio_analysis, fps, color_indices)
            
        Returns:
            List of processed frames for this batch
        """
        frame_indices, base_image, effect_pipeline_state, audio_data, fps = batch_data
        
        # Reconstruct effect pipeline from pickled state
        effect_pipeline = pickle.loads(effect_pipeline_state)
        
        # Unpack audio data
        bass_energy, mid_energy, treble_energy, beat_times = audio_data
        
        batch_frames = []
        for i in frame_indices:
            current_time = i / fps
            
            # Ensure indices are within bounds
            bass_idx = min(i, len(bass_energy) - 1)
            mid_idx = min(i, len(mid_energy) - 1)
            treble_idx = min(i, len(treble_energy) - 1)
            
            # Calculate color index
            total_energy = (
                bass_energy[bass_idx] +
                mid_energy[mid_idx] +
                treble_energy[treble_idx]
            )
            color_index = int(total_energy * 10)
            
            # Create frame context
            context = FrameContext(
                frame=base_image,
                time=current_time,
                frame_index=i,
                bass=float(bass_energy[bass_idx]),
                mid=float(mid_energy[mid_idx]),
                treble=float(treble_energy[treble_idx]),
                beat_intensity=BeatDetector.calculate_beat_intensity(current_time, beat_times),
                color_index=color_index
            )
            
            # Apply effects and add to batch
            frame = effect_pipeline.apply(context)
            batch_frames.append((i, frame))
        
        return batch_frames
    
    def generate_frames_parallel(
        self,
        audio_analysis: AudioAnalysis,
        num_frames: int,
        progress_cb=None,
        num_workers: int = None,
        batch_size: int = 10
    ) -> list:
        """
        Generate all frames for video using parallel processing.
        
        Args:
            audio_analysis: Audio analysis results
            num_frames: Number of frames to generate
            progress_cb: Optional progress callback
            num_workers: Number of worker processes (default: CPU count)
            batch_size: Frames per batch (default: 10)
            
        Returns:
            List of generated frames in correct order
        """
        import time
        
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
        
        start_time = time.time()
        
        # Prepare data for parallel processing
        # Pickle the effect pipeline for transfer to workers
        effect_pipeline_state = pickle.dumps(self.effect_pipeline)
        
        # Pack audio data (arrays are automatically shared efficiently by multiprocessing)
        audio_data = (
            audio_analysis.bass_energy,
            audio_analysis.mid_energy,
            audio_analysis.treble_energy,
            audio_analysis.beat_times
        )
        
        # Create batches of frame indices
        batches = []
        for i in range(0, num_frames, batch_size):
            batch_indices = list(range(i, min(i + batch_size, num_frames)))
            batches.append((batch_indices, self.base_image, effect_pipeline_state, audio_data, self.fps))
        
        if progress_cb:
            progress_cb("status", {"message": f"Rendering with {num_workers} parallel workers..."})
        
        # Process batches in parallel
        results = []
        processed_frames = 0
        
        with Pool(processes=num_workers) as pool:
            # Use imap for iterative results with progress tracking
            for batch_result in pool.imap(self._process_frame_batch, batches):
                results.extend(batch_result)
                processed_frames += len(batch_result)
                
                # Progress callback
                if progress_cb:
                    elapsed = time.time() - start_time
                    fps = processed_frames / elapsed if elapsed > 0 else 0
                    remaining_frames = num_frames - processed_frames
                    eta_seconds = remaining_frames / fps if fps > 0 else 0
                    eta_minutes = int(eta_seconds / 60)
                    
                    progress_cb("progress", {
                        "current": processed_frames,
                        "total": num_frames,
                        "percent": (processed_frames / num_frames) * 100,
                        "message": f"Frame {processed_frames}/{num_frames} - {fps:.1f} FPS ({num_workers} workers) - ETA: {eta_minutes}m {int(eta_seconds % 60)}s"
                    })
        
        # Sort results by frame index to ensure correct order
        results.sort(key=lambda x: x[0])
        frames = [frame for _, frame in results]
        
        total_time = time.time() - start_time
        avg_fps = num_frames / total_time if total_time > 0 else 0
        
        if progress_cb:
            progress_cb("status", {
                "message": f"Rendering complete! Average: {avg_fps:.1f} FPS ({total_time:.1f}s total)"
            })
        
        return frames
