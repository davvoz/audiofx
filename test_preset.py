"""Test per verificare l'errore del preset"""
from config import get_preset_config
from video_generator import AudioVisualFX
import os

# Test con file di esempio
audio_file = r"C:/Users/Utente/Music/automatic.wav"
image_file = r"C:/Users/Utente/Pictures/triGXhucfDpTaT2GU5Ql--.jpg"
output_file = "test_output.mp4"

# Carica preset Acid House
preset = get_preset_config('acid_house')
print(f"Preset: {preset['name']}")
print(f"Thresholds: {preset['thresholds']}")

colors = preset.get('colors')
th = preset.get('thresholds', {})
thresholds = (
    float(th.get('bass', 0.3)),
    float(th.get('mid', 0.2)),
    float(th.get('high', 0.15)),
)

print(f"Thresholds tuple: {thresholds}")
print(f"Bass: {thresholds[0]}, Mid: {thresholds[1]}, High: {thresholds[2]}")

if thresholds[2] <= 0:
    print("ERROR: high <= 0!")
else:
    print("OK: high > 0")

# Prova a creare l'oggetto
try:
    fx = AudioVisualFX(
        audio_file=audio_file,
        image_file=image_file,
        output_file=output_file,
        fps=30,
        colors=colors,
        thresholds=thresholds,
    )
    print("AudioVisualFX object created successfully!")
    print(f"High threshold set to: {fx.high_threshold}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
