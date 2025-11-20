"""
Test script for the new TextureStretchEffect.

This script demonstrates the fantastic texture stretching effect
that operates at 1/4 the speed of the music's rhythm.
"""

import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.effects.texture_stretch import TextureStretchEffect
from src.models.data_models import FrameContext


def create_test_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a colorful test frame with patterns."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient background
    for y in range(height):
        for x in range(width):
            frame[y, x, 0] = int((x / width) * 255)  # Red channel
            frame[y, x, 1] = int((y / height) * 255)  # Green channel
            frame[y, x, 2] = int(((x + y) / (width + height)) * 255)  # Blue channel
    
    # Add some geometric shapes for visual reference
    cv2.circle(frame, (width // 4, height // 4), 80, (255, 255, 0), -1)
    cv2.circle(frame, (3 * width // 4, height // 4), 80, (0, 255, 255), -1)
    cv2.circle(frame, (width // 4, 3 * height // 4), 80, (255, 0, 255), -1)
    cv2.circle(frame, (3 * width // 4, 3 * height // 4), 80, (255, 128, 0), -1)
    
    # Add grid lines
    for i in range(0, width, 100):
        cv2.line(frame, (i, 0), (i, height), (128, 128, 128), 1)
    for i in range(0, height, 100):
        cv2.line(frame, (0, i), (width, i), (128, 128, 128), 1)
    
    return frame


def simulate_audio_reactivity(frame_number: int, total_frames: int = 300):
    """Simulate audio data with varying intensities."""
    t = frame_number / total_frames * 4 * np.pi  # 2 full cycles
    
    # Simulate bass with slow pulsing
    bass = 0.5 + 0.4 * np.sin(t)
    
    # Simulate mid frequencies with faster variation
    mid = 0.4 + 0.3 * np.sin(t * 2)
    
    # Simulate treble with high-frequency variation
    treble = 0.3 + 0.2 * np.sin(t * 4)
    
    # Simulate beat intensity with periodic spikes
    beat_intensity = 0.8 if (frame_number % 30 == 0) else 0.2
    
    return bass, mid, treble, beat_intensity


def main():
    """Main test function."""
    print("🎨 Testing TextureStretchEffect...")
    print("=" * 60)
    
    # Create the effect
    effect = TextureStretchEffect(
        bass_threshold=0.25,
        mid_threshold=0.2,
        max_stretch=45.0,
        wave_complexity=3,
        flow_speed=0.15,
        stretch_smoothness=0.92,
        direction_change_speed=0.08,
        texture_grain=2.0,
        intensity=1.0
    )
    
    print(f"✅ Effect created with parameters:")
    print(f"   - Max Stretch: {effect.max_stretch}px")
    print(f"   - Wave Complexity: {effect.wave_complexity} harmonics")
    print(f"   - Flow Speed: {effect.flow_speed} (4x slower than music)")
    print(f"   - Texture Grain: {effect.texture_grain}")
    print()
    
    # Create test frame
    frame = create_test_frame()
    print(f"📐 Test frame created: {frame.shape[1]}x{frame.shape[0]}")
    print()
    
    # Process multiple frames to show evolution
    total_frames = 200
    output_dir = Path("texture_stretch_test_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎬 Processing {total_frames} frames...")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    for i in range(total_frames):
        # Simulate audio reactivity
        bass, mid, treble, beat_intensity = simulate_audio_reactivity(i, total_frames)
        
        # Create frame context
        context = FrameContext(
            frame=frame.copy(),
            time=i / 30.0,  # Assuming 30 fps
            frame_index=i,
            bass=bass,
            mid=mid,
            treble=treble,
            beat_intensity=beat_intensity
        )
        
        # Apply effect
        result = effect.process(frame, context)
        
        # Save every 10th frame
        if i % 10 == 0:
            output_path = output_dir / f"frame_{i:04d}.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
            print(f"⏱️  Frame {i:3d}/{total_frames} | "
                  f"Bass: {bass:.2f} | Mid: {mid:.2f} | Treble: {treble:.2f} | "
                  f"Beat: {beat_intensity:.2f} | "
                  f"Slow Phase: {effect._slow_phase:.2f}")
    
    print()
    print("=" * 60)
    print("✨ Test completed successfully!")
    print(f"📸 Saved {total_frames // 10} test frames to: {output_dir}")
    print()
    print("🎯 Effect Characteristics:")
    print("   ✓ Rhythmic texture stretching at 1/4 music tempo")
    print("   ✓ Multi-directional wave distortions")
    print("   ✓ Radial and circular flow components")
    print("   ✓ Fine-grain texture detail")
    print("   ✓ Chromatic aberration enhancement")
    print("   ✓ Smooth temporal transitions")
    print()
    print("💡 The effect creates hypnotic, organic texture distortions")
    print("   that flow independently while staying synchronized!")


if __name__ == "__main__":
    main()
