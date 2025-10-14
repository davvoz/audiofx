"""
Reusable core for generating audio-reactive visuals.
Exposes AudioVisualFX with optional progress callbacks for CLI/GUI use.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

# Third-party libs; import guards maintained as in original
try:
    from moviepy import AudioFileClip  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    from moviepy.editor import AudioFileClip  # type: ignore

import librosa


# Type for progress callback: (event, payload)
# Events:
# - "start": {total_frames:int}
# - "frame": {index:int, total:int}
# - "status": {message:str}
# - "done": {output:str}
ProgressCallback = Optional[Callable[[str, dict], None]]


class AudioVisualFX:
    def __init__(
        self,
        audio_file: str,
        image_file: str,
        output_file: str = "dark_techno_fx.mp4",
        fps: int = 30,
        duration: Optional[float] = None,
        *,
        progress_cb: ProgressCallback = None,
        colors: Optional[List[Tuple[float, float, float]]] = None,
        thresholds: Optional[Tuple[float, float, float]] = None,
        target_resolution: Tuple[int, int] = (720, 720),
        # Logo options
        logo_file: Optional[str] = None,
        logo_position: str = "top-right",
        logo_scale: float = 0.15,
        logo_opacity: float = 1.0,
        logo_margin: int = 12,
    ) -> None:
        self.audio_file = audio_file
        self.image_file = image_file
        self.output_file = output_file
        self.fps = fps
        self.duration = duration
        self.progress_cb = progress_cb

        # thresholds: (bass, mid, high)
        if thresholds is None:
            self.bass_threshold = 0.3
            self.mid_threshold = 0.2
            self.high_threshold = 0.15
        else:
            self.bass_threshold, self.mid_threshold, self.high_threshold = thresholds

        # color palette
        self.dark_colors: List[Tuple[float, float, float]] = colors or [
            (0.8, 0.0, 0.8),  # Magenta scuro
            (0.0, 0.8, 0.8),  # Cyan scuro
            (0.8, 0.3, 0.0),  # Arancione scuro
            (0.5, 0.0, 0.8),  # Viola
            (0.8, 0.0, 0.3),  # Rosso scuro
            (0.0, 0.8, 0.3),  # Verde acido
        ]

        self.target_resolution = target_resolution

        # logo configuration
        self.logo_file = logo_file
        self.logo_position = logo_position
        self.logo_scale = float(max(0.01, min(1.0, logo_scale)))
        self.logo_opacity = float(max(0.0, min(1.0, logo_opacity)))
        self.logo_margin = max(0, int(logo_margin))
        # normalized float [0..1]
        self._logo_rgba = None

    # ---------------------------- Audio analysis ---------------------------- #
    def _load_and_analyze_audio(
        self,
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        if self.progress_cb:
            self.progress_cb("status", {"message": "Analisi audio..."})

        y, sr = librosa.load(self.audio_file, duration=self.duration)

        hop_length = max(1, sr // self.fps)
        n_fft = 2048
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return y, sr, frequencies, magnitude

    @staticmethod
    def _extract_frequency_bands(
        magnitude: np.ndarray, frequencies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bass_range = (20, 200)
        mid_range = (200, 2000)
        treble_range = (2000, 8000)

        bass_idx = np.where((frequencies >= bass_range[0]) & (frequencies <= bass_range[1]))[0]
        mid_idx = np.where((frequencies >= mid_range[0]) & (frequencies <= mid_range[1]))[0]
        treble_idx = np.where((frequencies >= treble_range[0]) & (frequencies <= treble_range[1]))[0]

        bass_energy = np.sum(magnitude[bass_idx, :], axis=0)
        mid_energy = np.sum(magnitude[mid_idx, :], axis=0)
        treble_energy = np.sum(magnitude[treble_idx, :], axis=0)

        def _norm(v: np.ndarray) -> np.ndarray:
            m = np.max(v)
            return v / m if m > 0 else v

        return _norm(bass_energy), _norm(mid_energy), _norm(treble_energy)

    @staticmethod
    def _detect_beats(y: np.ndarray, sr: int) -> np.ndarray:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        return librosa.frames_to_time(beat_frames, sr=sr)

    # ------------------------------- FX passes ------------------------------ #
    def _color_pulse(
        self,
        img: np.ndarray,
        bass: float,
        mid: float,
        treble: float,
        beat_intensity: float = 0.0,
    ) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        if bass > self.bass_threshold:
            bass_factor = (bass - self.bass_threshold) * 3
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + bass_factor), 0, 255)
        if mid > self.mid_threshold:
            mid_factor = (mid - self.mid_threshold) * 2
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + mid_factor), 0, 255)
        if treble > self.high_threshold:
            treble_factor = (treble - self.high_threshold) * 50
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + treble_factor, 0, 179)
        if beat_intensity > 0.7:
            flash_intensity = (beat_intensity - 0.7) * 100
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + flash_intensity, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if bass > 0.5 or treble > 0.3:
            noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
            result = cv2.addWeighted(result, 0.9, noise, 0.1, 0)
        return result

    def _strobe(self, img: np.ndarray, intensity: float, color_index: int) -> np.ndarray:
        if intensity < 0.8:
            return img
        color = self.dark_colors[color_index % len(self.dark_colors)]
        overlay = np.full(img.shape, color, dtype=np.float32) * 255
        strobe_factor = (intensity - 0.8) * 5
        result = cv2.addWeighted(
            img.astype(np.float32), 1 - strobe_factor * 0.3, overlay, strobe_factor * 0.3, 0
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _distort(img: np.ndarray, intensity: float) -> np.ndarray:
        if intensity < 0.6:
            return img
        h, w = img.shape[:2]
        factor = (intensity - 0.6) * 20
        ys, xs = np.meshgrid(
            np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
        )
        map_x = xs + factor * np.sin(ys * 0.1) * np.cos(xs * 0.1)
        map_y = ys + factor * np.cos(ys * 0.1) * np.sin(xs * 0.1)
        return cv2.remap(
            img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

    @staticmethod
    def _glitch(img: np.ndarray, intensity: float) -> np.ndarray:
        if intensity < 0.4:
            return img
        result = img.copy()
        h, w = img.shape[:2]
        gf = (intensity - 0.4) * 2
        if np.random.random() < gf:
            shift = int(10 * gf)
            result[:, :, 0] = np.roll(img[:, :, 0], shift, axis=1)
            result[:, :, 2] = np.roll(img[:, :, 2], -shift, axis=1)
        if intensity > 0.6:
            for i in range(0, h, 4):
                result[i : i + 2, :] = (result[i : i + 2, :] * 0.7).astype(result.dtype)
        if intensity > 0.7:
            num_blocks = int(20 * gf)
            for _ in range(num_blocks):
                bh = np.random.randint(10, 50)
                bw = np.random.randint(20, 100)
                y = np.random.randint(0, h - bh)
                x = np.random.randint(0, w - bw)
                sy = np.random.randint(0, h - bh)
                sx = np.random.randint(0, w - bw)
                result[y : y + bh, x : x + bw] = result[sy : sy + bh, sx : sx + bw]
        if intensity > 0.8:
            num_lines = int(15 * gf)
            for _ in range(num_lines):
                ly = np.random.randint(0, h)
                lh = np.random.randint(1, 5)
                shift = np.random.randint(-50, 50)
                if ly + lh < h:
                    result[ly : ly + lh, :] = np.roll(result[ly : ly + lh, :], shift, axis=1)
        return result

    # ------------------------------ Logo overlay --------------------------- #
    def _prepare_logo(self, frame_w: int, frame_h: int) -> None:
        if not self.logo_file:
            return
        logo = cv2.imread(self.logo_file, cv2.IMREAD_UNCHANGED)
        if logo is None:
            # fail softly: just skip logo
            self._logo_rgba = None
            return
        # Ensure 4 channels RGBA
        if logo.shape[2] == 3:
            alpha = np.full(logo.shape[:2] + (1,), 255, dtype=np.uint8)
            logo = np.concatenate([logo, alpha], axis=2)
        # Convert BGR(A) to RGB(A)
        if logo.shape[2] == 4:
            b, g, r, a = cv2.split(logo)
            logo_rgba = cv2.merge([r, g, b, a])
        else:
            b, g, r = cv2.split(logo)
            logo_rgba = cv2.merge([r, g, b, np.full_like(r, 255)])
        # Resize according to frame width
        target_w = max(1, int(frame_w * self.logo_scale))
        h0, w0 = logo_rgba.shape[:2]
        target_h = max(1, int(h0 * (target_w / float(w0))))
        logo_rgba = cv2.resize(logo_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # Normalize to 0..1 floats and apply opacity
        logo_rgba = logo_rgba.astype(np.float32) / 255.0
        logo_rgba[:, :, 3] = np.clip(logo_rgba[:, :, 3] * self.logo_opacity, 0.0, 1.0)
        self._logo_rgba = logo_rgba

    def _overlay_logo(self, frame: np.ndarray) -> np.ndarray:
        if not self.logo_file:
            return frame
        h, w = frame.shape[:2]
        if self._logo_rgba is None:
            self._prepare_logo(w, h)
        if self._logo_rgba is None:
            return frame
        lh, lw = self._logo_rgba.shape[:2]
        # Position
        pos = (self.logo_position or "top-right").lower()
        margin = self.logo_margin
        if pos == "top-left":
            x, y = margin, margin
        elif pos == "bottom-left":
            x, y = margin, max(margin, h - lh - margin)
        elif pos == "bottom-right":
            x, y = max(margin, w - lw - margin), max(margin, h - lh - margin)
        else:  # top-right default
            x, y = max(margin, w - lw - margin), margin
        # Clip ROI if needed
        x = max(0, min(x, w - lw))
        y = max(0, min(y, h - lh))
        roi = frame[y : y + lh, x : x + lw].astype(np.float32) / 255.0
        logo_rgb = self._logo_rgba[:, :, :3]
        alpha = self._logo_rgba[:, :, 3:4]  # HxWx1
        comp = logo_rgb * alpha + roi * (1.0 - alpha)
        frame[y : y + lh, x : x + lw] = np.clip(comp * 255.0, 0, 255).astype(np.uint8)
        return frame

    # ----------------------------- Frame pipeline --------------------------- #
    def _generate_frames(self) -> List[np.ndarray]:
        y, sr, frequencies, magnitude = self._load_and_analyze_audio()
        bass, mid, treble = self._extract_frequency_bands(magnitude, frequencies)
        beat_times = self._detect_beats(y, sr)

        img = cv2.imread(self.image_file)
        if img is None:
            raise FileNotFoundError(f"Impossibile caricare immagine: {self.image_file}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_resolution)

        frames: List[np.ndarray] = []
        num_frames = len(bass)
        if self.progress_cb:
            self.progress_cb("start", {"total_frames": num_frames})

        for i in range(num_frames):
            current_time = i / self.fps
            beat_intensity = 0.0
            # quick proximity check
            # since beat_times is sorted, a simple any is enough at this scale
            for bt in beat_times:
                if abs(current_time - bt) < (1 / self.fps):
                    beat_intensity = 1.0
                    break

            frame = self._color_pulse(img, bass[i], mid[i], treble[i], beat_intensity)
            total_intensity = (bass[i] + mid[i] + treble[i]) / 3
            color_index = int(current_time * 2) % len(self.dark_colors)
            frame = self._strobe(frame, total_intensity, color_index)
            frame = self._distort(frame, total_intensity)
            glitch_intensity = bass[i] * 0.5 + treble[i] * 0.5
            frame = self._glitch(frame, glitch_intensity)
            # Overlay logo last to keep it stable
            frame = self._overlay_logo(frame)

            frames.append(frame)
            if self.progress_cb and (i % 5 == 0 or i == num_frames - 1):
                self.progress_cb("frame", {"index": i + 1, "total": num_frames})

        return frames

    # ------------------------------- Public API ----------------------------- #
    def create_video(self) -> None:
        if self.progress_cb:
            self.progress_cb("status", {"message": "Generazione frame..."})
        frames = self._generate_frames()

        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        frame_files: List[str] = []
        try:
            if self.progress_cb:
                self.progress_cb("status", {"message": "Salvataggio frame temporanei..."})
            for i, frame in enumerate(frames):
                path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_files.append(path)

            # audio duration
            try:
                a_clip = AudioFileClip(self.audio_file)
                audio_duration = a_clip.duration if not self.duration else min(self.duration, a_clip.duration)
            except Exception:  # pragma: no cover
                import soundfile as sf  # lazy import

                info = sf.info(self.audio_file)
                audio_duration = info.frames / info.samplerate
                if self.duration:
                    audio_duration = min(audio_duration, self.duration)

            target_total = int(np.ceil(audio_duration * self.fps))
            if len(frame_files) < target_total and frame_files:
                last = frame_files[-1]
                for i in range(len(frame_files), target_total):
                    pad = os.path.join(temp_dir, f"frame_{i:06d}.png")
                    shutil.copy(last, pad)
                    frame_files.append(pad)

            if self.progress_cb:
                self.progress_cb("status", {"message": "Mux con ffmpeg..."})
            try:
                from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore

                ffmpeg_path = get_ffmpeg_exe()
            except Exception:  # pragma: no cover
                ffmpeg_path = "ffmpeg"

            pattern = os.path.join(temp_dir, "frame_%06d.png")
            cmd = [
                ffmpeg_path,
                "-y",
                "-framerate",
                str(self.fps),
                "-i",
                pattern,
                "-i",
                self.audio_file,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-r",
                str(self.fps),
                "-movflags",
                "+faststart",
                self.output_file,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")
            if self.progress_cb:
                self.progress_cb("done", {"output": self.output_file})
        finally:
            # cleanup
            for f in frame_files:
                try:
                    os.remove(f)
                except Exception:
                    pass
            try:
                os.rmdir("temp_frames")
            except Exception:
                pass


def generate_video(
    audio: str,
    image: str,
    output: str,
    fps: int = 30,
    duration: Optional[float] = None,
    progress_cb: ProgressCallback = None,
    *,
    colors: Optional[List[Tuple[float, float, float]]] = None,
    thresholds: Optional[Tuple[float, float, float]] = None,
    # Logo options
    logo: Optional[str] = None,
    logo_position: str = "top-right",
    logo_scale: float = 0.15,
    logo_opacity: float = 1.0,
    logo_margin: int = 12,
) -> None:
    """Functional façade over AudioVisualFX for simple callers."""
    fx = AudioVisualFX(
        audio_file=audio,
        image_file=image,
        output_file=output,
        fps=fps,
        duration=duration,
        progress_cb=progress_cb,
        colors=colors,
        thresholds=thresholds,
        logo_file=logo,
        logo_position=logo_position,
        logo_scale=logo_scale,
        logo_opacity=logo_opacity,
        logo_margin=logo_margin,
    )
    fx.create_video()
