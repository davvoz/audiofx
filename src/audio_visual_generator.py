"""
Main audio-visual generator class.
"""

from typing import Optional, Tuple, List
import numpy as np

from .models.data_models import (
    EffectConfig,
    EffectStyle,
    ProgressCallback,
    AudioAnalysis,
)
from .effects import BaseEffect
from .core import AudioAnalyzer, FrameGenerator, VideoExporter
from .factories import EffectStyleManager
from .utils import ImageLoader


class AudioVisualGenerator:
    """
    Main class for generating audio-reactive visuals using OOP architecture.
    
    This is the main facade that orchestrates all components.
    """
    
    def __init__(
        self,
        audio_file: str,
        image_file: str,
        output_file: str = "output.mp4",
        fps: int = 30,
        duration: Optional[float] = None,
        effect_config: Optional[EffectConfig] = None,
        effect_style: EffectStyle = EffectStyle.STANDARD,
        target_resolution: Optional[Tuple[int, int]] = (720, 720),
        progress_cb: ProgressCallback = None,
        use_multiprocessing: bool = True,
        num_workers: int = None
    ):
        """
        Initialize audio-visual generator.
        
        Args:
            audio_file: Path to audio file
            image_file: Path to background image
            output_file: Path for output video
            fps: Frames per second
            duration: Optional duration limit
            effect_config: Optional effect configuration
            effect_style: Effect style preset
            target_resolution: Target video resolution (width, height).
                             If None, uses native image resolution.
            progress_cb: Optional progress callback function
            use_multiprocessing: Enable parallel frame rendering (default: True)
            num_workers: Number of worker processes (default: CPU count - 1)
        """
        self.audio_file = audio_file
        self.image_file = image_file
        self.output_file = output_file
        self.fps = fps
        self.duration = duration
        self.target_resolution = target_resolution
        self.progress_cb = progress_cb
        self.effect_style = effect_style
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers
        
        # Set default effect config
        if effect_config is None:
            default_colors = [
                (0.8, 0.0, 0.8), (0.0, 0.8, 0.8), (0.8, 0.3, 0.0),
                (0.5, 0.0, 0.8), (0.8, 0.0, 0.3), (0.0, 0.8, 0.3)
            ]
            effect_config = EffectConfig(colors=default_colors)
        
        self.effect_config = effect_config
        
        # Initialize OOP components (dependency injection)
        self.audio_analyzer = AudioAnalyzer()
        self.style_manager = EffectStyleManager(effect_config)
        self.video_exporter = VideoExporter(output_file, fps)
        self.frame_generator: Optional[FrameGenerator] = None
    
    def generate(self) -> None:
        """Generate the audio-reactive video using OOP architecture."""
        if self.progress_cb:
            self.progress_cb("status", {"message": "Starting generation..."})
        
        # Step 1: Analyze audio
        if self.progress_cb:
            self.progress_cb("status", {"message": "Analyzing audio..."})
        analysis = self.audio_analyzer.load_and_analyze(
            self.audio_file,
            self.duration,
            self.fps
        )
        
        # Step 2: Load and prepare image
        if self.progress_cb:
            self.progress_cb("status", {"message": "Loading image..."})
        img = self._load_image()
        
        # Step 3: Setup frame generator with appropriate pipeline
        if self.progress_cb:
            self.progress_cb("status", {"message": "Setting up effects..."})
        effect_pipeline = self.style_manager.get_pipeline(self.effect_style)
        self.frame_generator = FrameGenerator(img, effect_pipeline, self.fps)
        
        # Step 4: Generate frames
        if self.progress_cb:
            self.progress_cb("status", {"message": "Generating frames..."})
        frames = self._generate_frames(analysis)
        
        # Step 5: Export video
        self.video_exporter.export(frames, self.audio_file, self.progress_cb)
    
    def _load_image(self) -> np.ndarray:
        """
        Load and prepare base image using ImageLoader.
        
        Returns:
            Processed image array
            
        Raises:
            FileNotFoundError: If image file not found
        """
        return ImageLoader.load_and_prepare(self.image_file, self.target_resolution)
    
    def _generate_frames(self, analysis: AudioAnalysis) -> List[np.ndarray]:
        """
        Generate all video frames using OOP architecture.
        
        Args:
            analysis: Audio analysis results
            
        Returns:
            List of generated frames
            
        Raises:
            RuntimeError: If frame generator not initialized
        """
        if self.frame_generator is None:
            raise RuntimeError("Frame generator not initialized")
        
        num_frames = len(analysis.bass_energy)
        
        if self.progress_cb:
            self.progress_cb("start", {"total_frames": num_frames})
        
        # Use parallel processing if enabled
        if self.use_multiprocessing:
            if self.progress_cb:
                self.progress_cb("status", {
                    "message": f"Using parallel rendering (multiprocessing enabled)..."
                })
            return self.frame_generator.generate_frames_parallel(
                analysis, 
                num_frames, 
                self.progress_cb,
                num_workers=self.num_workers
            )
        else:
            # Fallback to sequential processing
            if self.progress_cb:
                self.progress_cb("status", {"message": "Using sequential rendering..."})
            return self.frame_generator.generate_frames(
                analysis, 
                num_frames, 
                self.progress_cb
            )
    
    def add_custom_effect(self, effect: BaseEffect) -> 'AudioVisualGenerator':
        """
        Add a custom effect to the current pipeline.
        
        Args:
            effect: Effect instance to add
            
        Returns:
            Self for method chaining
        """
        pipeline = self.style_manager.get_pipeline(self.effect_style)
        pipeline.add_effect(effect)
        return self
    
    def remove_effect(self, effect_type: type) -> 'AudioVisualGenerator':
        """
        Remove effects of given type from pipeline.
        
        Args:
            effect_type: Type of effect to remove
            
        Returns:
            Self for method chaining
        """
        pipeline = self.style_manager.get_pipeline(self.effect_style)
        pipeline.remove_effect(effect_type)
        return self
    
    def set_effect_style(self, style: EffectStyle) -> 'AudioVisualGenerator':
        """
        Change effect style.
        
        Args:
            style: New effect style
            
        Returns:
            Self for method chaining
        """
        self.effect_style = style
        return self
    
    def get_pipeline(self):
        """
        Get current effect pipeline.
        
        Returns:
            Current EffectPipeline
        """
        return self.style_manager.get_pipeline(self.effect_style)
