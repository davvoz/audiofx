#!/usr/bin/env python3
"""
Script di test per verificare le ottimizzazioni di performance.
Confronta vecchio metodo (generate) vs nuovo (generate_streaming).
"""

import time
import tracemalloc
from pathlib import Path
from src.audio_visual_generator import AudioVisualGenerator
from src.models.data_models import EffectStyle, EffectConfig


def format_bytes(bytes_value):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def test_old_method(audio_file, image_file, output_file, duration=None):
    """Test old method (accumulates frames in RAM)."""
    print("\n" + "="*60)
    print("🔴 TESTING OLD METHOD (generate)")
    print("="*60)
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()
    
    try:
        config = EffectConfig(
            colors=[
                (0.8, 0.0, 0.8),
                (0.0, 0.8, 0.8),
                (0.8, 0.3, 0.0)
            ]
        )
        
        def progress_callback(event_type, data):
            if event_type == "progress" and data.get('current', 0) % 90 == 0:
                current, peak = tracemalloc.get_traced_memory()
                print(f"  Progress: {data.get('percent', 0):.1f}% - RAM: {format_bytes(current)} (peak: {format_bytes(peak)})")
        
        generator = AudioVisualGenerator(
            audio_file=audio_file,
            image_file=image_file,
            output_file=output_file,
            fps=30,
            duration=duration,
            effect_config=config,
            effect_style=EffectStyle.STANDARD,
            target_resolution=(720, 720),
            progress_cb=progress_callback,
            use_multiprocessing=False  # Disable for fair comparison
        )
        
        # OLD METHOD
        generator.generate()
        
        elapsed_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        
        print("\n📊 OLD METHOD RESULTS:")
        print(f"  ⏱️  Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"  💾 RAM Peak: {format_bytes(peak)}")
        print(f"  💾 RAM Final: {format_bytes(current)}")
        
        tracemalloc.stop()
        return elapsed_time, peak
        
    except Exception as e:
        print(f"\n❌ ERROR in old method: {e}")
        tracemalloc.stop()
        return None, None


def test_new_method(audio_file, image_file, output_file, duration=None):
    """Test new streaming method (memory-efficient)."""
    print("\n" + "="*60)
    print("🟢 TESTING NEW METHOD (generate_streaming)")
    print("="*60)
    
    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()
    
    try:
        config = EffectConfig(
            colors=[
                (0.8, 0.0, 0.8),
                (0.0, 0.8, 0.8),
                (0.8, 0.3, 0.0)
            ]
        )
        
        def progress_callback(event_type, data):
            if event_type == "progress" and data.get('current', 0) % 90 == 0:
                current, peak = tracemalloc.get_traced_memory()
                print(f"  Progress: {data.get('percent', 0):.1f}% - RAM: {format_bytes(current)} (peak: {format_bytes(peak)})")
        
        generator = AudioVisualGenerator(
            audio_file=audio_file,
            image_file=image_file,
            output_file=output_file,
            fps=30,
            duration=duration,
            effect_config=config,
            effect_style=EffectStyle.STANDARD,
            target_resolution=(720, 720),
            progress_cb=progress_callback,
            use_multiprocessing=False  # Disable for fair comparison
        )
        
        # NEW METHOD
        generator.generate_streaming()
        
        elapsed_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        
        print("\n📊 NEW METHOD RESULTS:")
        print(f"  ⏱️  Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"  💾 RAM Peak: {format_bytes(peak)}")
        print(f"  💾 RAM Final: {format_bytes(current)}")
        
        tracemalloc.stop()
        return elapsed_time, peak
        
    except Exception as e:
        print(f"\n❌ ERROR in new method: {e}")
        tracemalloc.stop()
        return None, None


def compare_methods(audio_file, image_file, duration=None):
    """Compare both methods and show improvements."""
    print("\n" + "="*60)
    print("🎬 PERFORMANCE COMPARISON TEST")
    print("="*60)
    print(f"Audio: {audio_file}")
    print(f"Image: {image_file}")
    print(f"Duration: {duration}s" if duration else "Duration: Full audio")
    print()
    
    # Ensure files exist
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return
    if not Path(image_file).exists():
        print(f"❌ Image file not found: {image_file}")
        return
    
    # Test old method
    old_time, old_ram = test_old_method(
        audio_file, 
        image_file, 
        "test_output_old.mp4",
        duration
    )
    
    # Test new method
    new_time, new_ram = test_new_method(
        audio_file, 
        image_file, 
        "test_output_new.mp4",
        duration
    )
    
    # Compare results
    if old_time and new_time and old_ram and new_ram:
        print("\n" + "="*60)
        print("📈 IMPROVEMENT SUMMARY")
        print("="*60)
        
        time_improvement = ((old_time - new_time) / old_time) * 100
        ram_improvement = ((old_ram - new_ram) / old_ram) * 100
        
        print(f"\n⏱️  SPEED:")
        print(f"  Old method: {old_time:.2f}s")
        print(f"  New method: {new_time:.2f}s")
        print(f"  Improvement: {time_improvement:+.1f}% ({'FASTER' if time_improvement > 0 else 'SLOWER'})")
        
        print(f"\n💾 MEMORY:")
        print(f"  Old method: {format_bytes(old_ram)}")
        print(f"  New method: {format_bytes(new_ram)}")
        print(f"  Improvement: {ram_improvement:+.1f}% ({'LESS' if ram_improvement > 0 else 'MORE'})")
        
        print(f"\n🎯 SPEEDUP: {old_time/new_time:.2f}x faster")
        print(f"🎯 RAM REDUCTION: {old_ram/new_ram:.2f}x less memory")
        
        print("\n" + "="*60)


def quick_test():
    """Quick test with sample files."""
    print("🔍 Looking for test files...")
    
    # Try to find audio and image files in workspace
    audio_candidates = list(Path('.').rglob('*.mp3')) + list(Path('.').rglob('*.wav'))
    image_candidates = list(Path('.').rglob('*.jpg')) + list(Path('.').rglob('*.png'))
    
    if audio_candidates and image_candidates:
        audio_file = str(audio_candidates[0])
        image_file = str(image_candidates[0])
        print(f"✅ Found audio: {audio_file}")
        print(f"✅ Found image: {image_file}")
        
        # Test with just 10 seconds for quick comparison
        compare_methods(audio_file, image_file, duration=10.0)
    else:
        print("❌ No test files found. Please provide:")
        print("  python test_performance.py <audio_file> <image_file> [duration_seconds]")
        print("\nExample:")
        print("  python test_performance.py song.mp3 background.jpg 30")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        audio_file = sys.argv[1]
        image_file = sys.argv[2]
        duration = float(sys.argv[3]) if len(sys.argv) >= 4 else None
        
        compare_methods(audio_file, image_file, duration)
    else:
        quick_test()
