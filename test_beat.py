"""Test per capire quale parametro causa l'errore in librosa.beat.beat_track"""
import librosa
import numpy as np

# Crea audio di test
y = np.random.randn(22050 * 5)
sr = 22050

print("Test 1: Chiamata base")
try:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print(f"✓ OK - Tempo: {tempo}, Beats: {len(beats)}")
except Exception as e:
    print(f"✗ ERRORE: {e}")

print("\nTest 2: Con hop_length")
try:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    print(f"✓ OK - Tempo: {tempo}, Beats: {len(beats)}")
except Exception as e:
    print(f"✗ ERRORE: {e}")

print("\nTest 3: Con start_bpm")
try:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120.0)
    print(f"✓ OK - Tempo: {tempo}, Beats: {len(beats)}")
except Exception as e:
    print(f"✗ ERRORE: {e}")

print("\nTest 4: Con units='frames'")
try:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    print(f"✓ OK - Tempo: {tempo}, Beats: {len(beats)}")
except Exception as e:
    print(f"✗ ERRORE: {e}")

print("\nTest 5: Tutti i parametri insieme")
try:
    tempo, beats = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=512, start_bpm=120.0, units='frames'
    )
    print(f"✓ OK - Tempo: {tempo}, Beats: {len(beats)}")
except Exception as e:
    print(f"✗ ERRORE: {e}")

print("\nTest 6: Con trim=False")
try:
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    print(f"✓ OK - Tempo: {tempo}, Beats: {len(beats)}")
except Exception as e:
    print(f"✗ ERRORE: {e}")
