# Audio Visual FX Generator

Create audio‑reactive videos by animating a still image based on the input audio’s frequency content and beats. Includes a simple CLI and a Tkinter GUI with presets.


## Demo

Check out this example video:

[![Demo Video](https://img.youtube.com/vi/7xh1b1njanA/0.jpg)](https://www.youtube.com/shorts/7xh1b1njanA)

## Features

- Audio‑reactive pipeline: color pulse, strobe, distortion, and glitch driven by bass/mid/treble energy and beat detection (librosa)
- Beat‑synced flashes and color cycling
- **Video + Audio Sync**: Synchronize any video with audio - the audio duration commands, videos are automatically trimmed or looped
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

### Audio + Image (Audio-reactive visuals)

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

### Audio + Video Sync

Synchronize a video with an audio track - **audio duration commands**:

```powershell
python video_sync_cli.py --audio path\to\track.mp3 --video path\to\video.mp4 --output synced.mp4
```

Options:

- `--audio, -a` (required) input audio file
- `--video, -v` (required) input video file
- `--output, -o` output video file (default: `synced_video.mp4`)
- `--short-video-mode` how to handle videos shorter than audio:
  - `loop` (default): repeats the video from the beginning
  - `stretch`: slows down the video to match audio duration
- Logo overlay (same options as above)

Examples:

```powershell
# Basic sync with loop (default)
python video_sync_cli.py --audio song.mp3 --video clip.mp4 --output final.mp4

# Stretch mode instead of loop
python video_sync_cli.py --audio song.mp3 --video clip.mp4 --output final.mp4 --short-video-mode stretch

# With logo
python video_sync_cli.py -a song.mp3 -v clip.mp4 -o final.mp4 \
	--logo logo.png --logo-position top-left --logo-scale 0.2
```

**How it works:**
- **Audio analysis**: Extracts bass, mid, and treble frequencies + beat detection
- **Video adaptation**: 
  - If video is **longer** than audio → trimmed
  - If video is **shorter** than audio → looped (repeats) or stretched (slowed down)
- **Effects application**: Every frame gets audio-reactive effects (color pulse, strobe, distortion, glitch)
- **Audio replacement**: Original video audio is removed and replaced with your track
- Original video FPS is preserved

**Mode comparison:**
- **Loop**: Natural repetition, good for short clips or background loops
- **Stretch**: Slow motion effect, good when you want to avoid visible loops

**Note**: This process is more computationally intensive than basic sync as it applies effects to each frame.

## GUI

Launch the GUI and pick audio/image/FPS/output interactively. Includes presets and logo controls:

```powershell
python gui.py
```

The GUI has two tabs:

### 1. Audio + Image
Create audio-reactive visuals from a still image. 
- Presets (from `config.py`): Dark Techno, Cyberpunk, Industrial, Acid House
- Presets mainly change the color palette and thresholds that drive the FX
- Full control over FPS, logo positioning, and effects

### 2. Audio + Video Sync
Synchronize a video with an audio track and apply audio-reactive effects. **The audio duration is the master**:
- **Original video audio is removed** and replaced with your audio track
- **Audio-reactive effects** (color pulse, strobe, distortion, glitch) are applied to every frame
- **Video too long?** It will be automatically trimmed to match the audio
- **Video too short?** Choose between:
  - **Loop mode** (default): Seamlessly repeats the video from the beginning
  - **Stretch mode**: Slows down the video to match the audio duration
- **Preset support**: Choose from Dark Techno, Cyberpunk, Industrial, or Acid House visual styles
- Perfect for creating music videos, lyric videos, or syncing existing footage to new soundtracks
- Logo overlay supported on all frames

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

