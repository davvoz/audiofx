# Audio Visual FX Generator

Create audio‑reactive videos by animating a still image based on the input audio’s frequency content and beats. Includes a simple CLI and a Tkinter GUI with presets.

## Features

- Audio‑reactive pipeline: color pulse, strobe, distortion, and glitch driven by bass/mid/treble energy and beat detection (librosa)
- Beat‑synced flashes and color cycling
- Optional logo overlay with position, scale, opacity, and margin controls
- GUI with visual presets and interactive controls (Tkinter)
- H.264 video + AAC audio via ffmpeg, faststart for better web playback
- Temporary frames auto‑cleanup

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`
- ffmpeg runtime
	- The app first tries to use `imageio-ffmpeg`’s bundled ffmpeg when installed
	- Otherwise, a system ffmpeg must be available on PATH

Windows ffmpeg options:
- Recommended: `pip install imageio-ffmpeg` (already listed in `requirements.txt`)
- Or install a static build from https://www.gyan.dev/ffmpeg/builds/ and add the `bin` folder to PATH

## Setup (Windows PowerShell)

It’s best to use a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional quick checks:

```powershell
python -c "from moviepy import AudioFileClip, ImageSequenceClip; print('MoviePy imports OK!')"
python -c "import imageio_ffmpeg, os; print('ffmpeg at:', imageio_ffmpeg.get_ffmpeg_exe())"
```

## Command‑line usage

Basic:

```powershell
python audio_visual_fx.py --audio path\to\track.wav --image path\to\image.jpg --output out.mp4 --fps 30
```

Options:

- `--audio, -a` (required) input audio file
- `--image, -i` (required) input image file
- `--output, -o` output video file (default: `dark_techno_fx.mp4`)
- `--fps` frames per second (default: 30)
- `--duration, -d` limit output length in seconds
- Logo overlay
	- `--logo` path to a logo image (PNG recommended)
	- `--logo-position` one of `top-left | top-right | bottom-left | bottom-right` (default: `top-right`)
	- `--logo-scale` logo width relative to frame, 0–1 (default: `0.15`)
	- `--logo-opacity` 0–1 (default: `1.0`)
	- `--logo-margin` pixel margin from frame edges (default: `12`)

Examples:

```powershell
# Full track at 30 fps
python audio_visual_fx.py --audio .\music\tec.wav --image .\covers\art.jpg --output .\renders\video.mp4 --fps 30

# First 30 seconds only
python audio_visual_fx.py -a tec.wav -i example_image.jpg -o short.mp4 --fps 30 -d 30

# With a semi‑transparent logo bottom‑right
python audio_visual_fx.py -a tec.wav -i cover.jpg -o with_logo.mp4 --fps 30 \
	--logo .\brand\logo.png --logo-position bottom-right --logo-scale 0.2 --logo-opacity 0.8 --logo-margin 16
```

## GUI

Launch the GUI and pick audio/image/FPS/output interactively. Includes presets and logo controls:

```powershell
python gui.py
```

Presets (from `config.py`): Dark Techno, Cyberpunk, Industrial, Acid House. Presets mainly change the color palette and thresholds that drive the FX.

## Batch helper (Windows)

`run.bat` provides a quick start menu:

- Option 1: run a quick example (if present)
- Option 2: prompt for your audio/image and render
- Option 3: install dependencies only

Run it by double‑clicking in Explorer or from PowerShell:

```powershell
./run.bat
```

## Output details

- Resolution: 720×720 (square) by default
- Video: H.264 (`libx264`), pixel format `yuv420p` for broad compatibility
- Audio: AAC 192 kbps
- Container: MP4 with `+faststart` for better streaming
- Temporary frames are written to `temp_frames` and deleted automatically

## Configuration and presets

Advanced color/threshold presets live in `config.py`. The GUI maps user‑friendly names to config keys and passes colors/thresholds to the generator. You can add your own preset or use `create_custom_config(...)` to craft a palette and thresholds.

Note: The core generator (`video_generator.AudioVisualFX`) currently targets 720×720; change `target_resolution` in the constructor if you want another size programmatically.

## Troubleshooting

- ffmpeg not found / render fails
	- Ensure `imageio-ffmpeg` is installed: `python -m pip install imageio-ffmpeg`
	- Or install a system ffmpeg and make sure `ffmpeg -version` works in PowerShell
- OpenCV errors on Windows
	- Ensure you have a recent Python and up‑to‑date `opencv-python`
	- If you don’t need any GUI from OpenCV, `opencv-python-headless` is an alternative
- Audio length vs. frames mismatch
	- The tool pads the last frame so the video length matches audio (or `--duration` if provided)
	- If you want a shorter preview, use `--duration`
- Performance
	- Lower `--fps` or use shorter `--duration`
	- Close other heavy processes; large source images will increase processing time

## Credits

Built with: MoviePy, Librosa, OpenCV, NumPy, SciPy, `imageio-ffmpeg`.

