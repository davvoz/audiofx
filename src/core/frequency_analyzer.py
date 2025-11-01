"""
Frequency band extraction utilities.
"""

from typing import Tuple
import numpy as np


class FrequencyAnalyzer:
    """Handles frequency band extraction from STFT magnitude."""
    
    @staticmethod
    def extract_bands(
        magnitude: np.ndarray,
        frequencies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract bass, mid, and treble frequency bands.
        
        Args:
            magnitude: STFT magnitude array
            frequencies: Frequency bins
            
        Returns:
            Tuple of (bass, mid, treble) energy arrays
        """
        # Define frequency ranges
        bass_idx = np.where((frequencies >= 20) & (frequencies <= 200))[0]
        mid_idx = np.where((frequencies >= 200) & (frequencies <= 2000))[0]
        treble_idx = np.where((frequencies >= 2000) & (frequencies <= 8000))[0]
        
        # Sum energy in each band
        bass = np.sum(magnitude[bass_idx, :], axis=0)
        mid = np.sum(magnitude[mid_idx, :], axis=0)
        treble = np.sum(magnitude[treble_idx, :], axis=0)
        
        # Normalize each band
        return (
            FrequencyAnalyzer._normalize(bass),
            FrequencyAnalyzer._normalize(mid),
            FrequencyAnalyzer._normalize(treble)
        )
    
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        """Normalize array to 0-1 range."""
        vmax = np.max(v)
        return v / vmax if vmax > 0 else v
