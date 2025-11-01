"""
Beat detection utilities.
"""

import numpy as np
import librosa


class BeatDetector:
    """Handles beat detection in audio signals."""
    
    @staticmethod
    def detect_beats(y: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect beat timestamps in audio.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Array of beat times in seconds
        """
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
            return beats
        except Exception:
            return np.array([])
    
    @staticmethod
    def calculate_beat_intensity(current_time: float, beat_times: np.ndarray) -> float:
        """
        Calculate beat intensity at given time.
        
        Args:
            current_time: Current time in seconds
            beat_times: Array of beat timestamps
            
        Returns:
            Beat intensity value (0.0-1.0)
        """
        if len(beat_times) == 0:
            return 0.0
        
        # Find closest beat
        time_diffs = np.abs(beat_times - current_time)
        min_diff = np.min(time_diffs)
        
        # Intensity decays with distance from beat
        if min_diff < 0.1:
            return 1.0 - min_diff * 10
        
        return 0.0
