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
        target_resolution: Optional[Tuple[int, int]] = (720, 720),
        effect_style: str = "standard",  # "standard" o "extreme"
        # Logo options
        logo_file: Optional[str] = None,
        logo_position: str = "top-right",
        logo_scale: float = 0.15,
        logo_opacity: float = 1.0,
        logo_margin: int = 12,
        config: Optional[dict] = None,  # Full preset configuration
    ) -> None:
        self.audio_file = audio_file
        self.image_file = image_file
        self.output_file = output_file
        self.fps = fps
        self.duration = duration
        self.progress_cb = progress_cb
        
        # Store full config and read transition_duration if available
        self._config = config or {}
        effects_config = self._config.get('effects', {})
        self._transition_duration_from_config = effects_config.get('transition_duration', 3.0)  # Default 3.0s ULTRA SMOOTH

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

        self.target_resolution = target_resolution  # None = use native resolution
        
        # effect style
        self.effect_style = effect_style
        
        # frame counter for time-based effects
        self._frame_counter = 0
        
        # Transition system for smooth section changes
        self._previous_section = None
        self._section_start_time = 0.0
        # Use config value if provided, otherwise default to 3.0 (ULTRA SMOOTH)
        self._transition_duration = self._transition_duration_from_config
        self._previous_color_index = 0  # Track color for smooth interpolation
        self._color_transition_progress = 0.0

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

        try:
            y, sr = librosa.load(self.audio_file, duration=self.duration)
            print(f"DEBUG: Audio caricato - Sample rate: {sr}, Durata: {len(y)/sr:.2f}s")
        except Exception as e:
            raise RuntimeError(f"Errore caricamento audio: {e}")

        hop_length = max(1, sr // self.fps)
        n_fft = 2048
        print(f"DEBUG: hop_length={hop_length}, n_fft={n_fft}, fps={self.fps}")
        
        try:
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            print(f"DEBUG: STFT completato - magnitude shape: {magnitude.shape}")
            return y, sr, frequencies, magnitude
        except Exception as e:
            raise RuntimeError(f"Errore analisi spettrale: {e}")

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
        try:
            print(f"DEBUG _detect_beats: y.shape={y.shape}, sr={sr}")
            # Usa parametri più robusti per evitare errori
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, 
                sr=sr,
                hop_length=512,
                start_bpm=120.0,
                units='frames'
            )
            print(f"DEBUG _detect_beats: Successo! tempo={tempo}, beat_frames={len(beat_frames)}")
            return librosa.frames_to_time(beat_frames, sr=sr)
        except Exception as e:
            print(f"WARNING _detect_beats: Beat detection failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])
    
    @staticmethod
    def _analyze_track_sections(
        y: np.ndarray,
        sr: int,
        bass_energy: np.ndarray,
        mid_energy: np.ndarray,
        treble_energy: np.ndarray,
        transition_duration: float = 3.0
    ) -> List[dict]:
        """
        Analizza la traccia audio per riconoscere le diverse sezioni musicali.
        Restituisce una lista di sezioni con tipo, start, end e caratteristiche.
        
        Tipi di sezione:
        - 'intro': Introduzione (bassa energia, graduale incremento)
        - 'buildup': Crescita tensione (energia crescente, spesso con riser)
        - 'drop': Climax energetico (impatto improvviso, alta energia)
        - 'breakdown': Calo energia (transizione, spesso melodica)
        - 'outro': Conclusione (decadimento energia)
        - 'steady': Sezione stabile (energia costante)
        """
        print("\n🧠 Analisi intelligente sezioni traccia...")
        
        # Calcola energia totale per frame
        total_energy = (bass_energy + mid_energy + treble_energy) / 3.0
        
        # Parametri analisi
        window_size = int(sr / 512 * 2)  # ~2 secondi
        
        # Smooth energy per ridurre variazioni rapide
        from scipy.ndimage import gaussian_filter1d
        smooth_energy = gaussian_filter1d(total_energy, sigma=window_size // 4)
        
        # Calcola derivata (rate of change) dell'energia
        energy_diff = np.diff(smooth_energy, prepend=smooth_energy[0])
        
        # Normalizza
        energy_diff_norm = energy_diff / (np.max(np.abs(energy_diff)) + 1e-10)
        
        sections = []
        current_section = None
        
        # Analizza frame per frame
        for i in range(len(total_energy)):
            time = i / len(total_energy) * len(y) / sr
            energy = smooth_energy[i]
            slope = energy_diff_norm[i] if i < len(energy_diff_norm) else 0
            
            # Determina tipo sezione
            section_type = None
            
            # Intro: bassa energia all'inizio
            if time < 15 and energy < 0.35:
                section_type = 'intro'
            
            # Buildup: energia crescente rapidamente
            elif slope > 0.15 and energy < 0.70:
                section_type = 'buildup'
            
            # Drop: alta energia improvvisa
            elif energy > 0.75 or (slope > 0.25 and energy > 0.60):
                section_type = 'drop'
            
            # Breakdown: calo energia dopo drop
            elif slope < -0.10 and energy < 0.50:
                section_type = 'breakdown'
            
            # Outro: decadimento finale
            elif time > len(y) / sr - 20 and energy < 0.30:
                section_type = 'outro'
            
            # Steady: energia stabile
            else:
                section_type = 'steady'
            
            # Gestione sezioni
            if current_section is None:
                current_section = {
                    'type': section_type,
                    'start_frame': i,
                    'start_time': time,
                    'avg_energy': energy,
                    'avg_bass': bass_energy[i],
                    'avg_mid': mid_energy[i],
                    'avg_treble': treble_energy[i]
                }
            elif section_type != current_section['type']:
                # Completa sezione corrente
                current_section['end_frame'] = i - 1
                current_section['end_time'] = time
                current_section['duration'] = time - current_section['start_time']
                sections.append(current_section)
                
                # Inizia nuova sezione
                current_section = {
                    'type': section_type,
                    'start_frame': i,
                    'start_time': time,
                    'avg_energy': energy,
                    'avg_bass': bass_energy[i],
                    'avg_mid': mid_energy[i],
                    'avg_treble': treble_energy[i]
                }
        
        # Completa ultima sezione
        if current_section:
            current_section['end_frame'] = len(total_energy) - 1
            current_section['end_time'] = len(y) / sr
            current_section['duration'] = current_section['end_time'] - current_section['start_time']
            sections.append(current_section)
        
        # Merge sezioni troppo corte per garantire transizioni complete
        # La soglia deve essere > transition_duration per evitare transizioni incomplete
        min_section_duration = transition_duration + 0.5  # +0.5s buffer di sicurezza
        merged_sections = []
        for sec in sections:
            if sec['duration'] >= min_section_duration or not merged_sections:
                merged_sections.append(sec)
            else:
                # Merge con precedente
                merged_sections[-1]['end_frame'] = sec['end_frame']
                merged_sections[-1]['end_time'] = sec['end_time']
                merged_sections[-1]['duration'] = merged_sections[-1]['end_time'] - merged_sections[-1]['start_time']
        
        # Log sezioni rilevate
        print(f"\n📊 Sezioni rilevate: {len(merged_sections)} (min_duration={min_section_duration:.1f}s)")
        for i, sec in enumerate(merged_sections, 1):
            emoji = {
                'intro': '🎵', 'buildup': '📈', 'drop': '💥',
                'breakdown': '🎹', 'outro': '🌅', 'steady': '🔄'
            }.get(sec['type'], '❓')
            print(f"  {i}. {emoji} {sec['type'].upper():12} | "
                  f"{sec['start_time']:6.1f}s - {sec['end_time']:6.1f}s | "
                  f"Durata: {sec['duration']:5.1f}s | "
                  f"Energia: {sec['avg_energy']:.2f}")
        
        return merged_sections

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

    def _strobe(self, img: np.ndarray, intensity: float, color_index: int, transition_factor: float = 1.0) -> np.ndarray:
        """Strobe effect con transizione smooth dei colori."""
        if intensity < 0.8:
            return img
        
        # Usa colore interpolato smooth
        color = self._get_smooth_color(color_index, transition_factor)
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
    
    @staticmethod
    def _chromatic_aberration(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto aberrazione cromatica cyberpunk - separa i canali RGB."""
        if intensity < 0.3:
            return img
        result = img.copy()
        h, w = img.shape[:2]
        shift_amount = int((intensity - 0.3) * 15)
        
        # Separa canali RGB e shifta in direzioni diverse
        result[:, :, 0] = np.roll(img[:, :, 0], -shift_amount, axis=1)  # Rosso a sinistra
        result[:, :, 2] = np.roll(img[:, :, 2], shift_amount, axis=1)   # Blu a destra
        
        return result
    
    @staticmethod
    def _electric_arcs(img: np.ndarray, intensity: float, color: Tuple[float, float, float]) -> np.ndarray:
        """Effetto scariche elettriche - linee luminose casuali."""
        if intensity < 0.7:
            return img
        result = img.copy()
        h, w = img.shape[:2]
        
        num_arcs = int((intensity - 0.7) * 20)
        for _ in range(num_arcs):
            # Genera punti casuali per l'arco
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            
            # Colore elettrico brillante
            arc_color = tuple(int(c * 255) for c in color)
            thickness = np.random.randint(1, 4)
            
            # Disegna linea con glow
            cv2.line(result, (x1, y1), (x2, y2), arc_color, thickness)
            
            # Aggiungi glow
            if intensity > 0.85:
                cv2.line(result, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # Blending per effetto glow
        result = cv2.addWeighted(img, 0.7, result, 0.3, 0)
        return result
    
    @staticmethod
    def _bubble_distortion(img: np.ndarray, bass_intensity: float) -> np.ndarray:
        """Effetto bolla sui bassi - deformazione fluida circolare."""
        if bass_intensity < 0.4:
            return img
        
        h, w = img.shape[:2]
        factor = (bass_intensity - 0.4) * 30
        
        # Centro della bolla (al centro dell'immagine)
        cx, cy = w // 2, h // 2
        
        # Crea griglia di coordinate
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        
        # Calcola distanza dal centro
        dx = x_coords - cx
        dy = y_coords - cy
        distance = np.sqrt(dx**2 + dy**2)
        
        # Effetto bolla: espansione dal centro
        max_distance = np.sqrt(cx**2 + cy**2)
        normalized_distance = distance / max_distance
        
        # Funzione di deformazione (più forte al centro)
        deformation = factor * np.sin(normalized_distance * np.pi) * (1 - normalized_distance)
        
        # Applica deformazione
        angle = np.arctan2(dy, dx)
        map_x = x_coords + deformation * np.cos(angle)
        map_y = y_coords + deformation * np.sin(angle)
        
        # Remap
        result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return result
    
    @staticmethod
    def _zoom_pulse(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto zoom pulsante - ingrandisce e rimpicciolisce."""
        if intensity < 0.3:
            return img
        
        h, w = img.shape[:2]
        zoom_factor = 1.0 + (intensity - 0.3) * 0.5  # Max 1.35x zoom
        
        # Calcola nuove dimensioni
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        
        # Zoom in
        zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop al centro per tornare alle dimensioni originali
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        result = zoomed[start_y:start_y+h, start_x:start_x+w]
        
        return result
    
    @staticmethod
    def _screen_shake(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto shake dello schermo - sposta l'immagine casualmente."""
        if intensity < 0.5:
            return img
        
        h, w = img.shape[:2]
        shake_amount = int((intensity - 0.5) * 30)
        
        # Se shake_amount è 0, nessun shake
        if shake_amount <= 0:
            return img
        
        # Offset casuali
        offset_x = np.random.randint(-shake_amount, shake_amount + 1)
        offset_y = np.random.randint(-shake_amount, shake_amount + 1)
        
        # Matrice di traslazione
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    @staticmethod
    def _rgb_split(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto split RGB estremo - separa drasticamente i canali."""
        if intensity < 0.6:
            return img
        
        result = img.copy()
        split_amount = int((intensity - 0.6) * 25)
        
        # Split verticale
        result[:, :, 0] = np.roll(img[:, :, 0], split_amount, axis=0)    # Rosso su
        result[:, :, 1] = img[:, :, 1]                                    # Verde fermo
        result[:, :, 2] = np.roll(img[:, :, 2], -split_amount, axis=0)   # Blu giù
        
        return result
    
    @staticmethod
    def _scan_lines(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto scan lines CRT - linee orizzontali."""
        if intensity < 0.4:
            return img
        
        result = img.copy().astype(np.float32)
        h = img.shape[0]
        
        # Crea pattern di scan lines
        line_intensity = (intensity - 0.4) * 0.6
        for y in range(0, h, 3):
            result[y:y+1, :] = result[y:y+1, :] * (1.0 - line_intensity)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _vhs_distortion(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto distorsione VHS - shift orizzontali casuali."""
        if intensity < 0.5:
            return img
        
        result = img.copy()
        h = img.shape[0]
        
        # Numero di bande distorte
        num_bands = int((intensity - 0.5) * 15)
        
        for _ in range(num_bands):
            y = np.random.randint(0, h - 10)
            band_height = np.random.randint(2, 8)
            shift = np.random.randint(-30, 30)
            
            if y + band_height < h:
                result[y:y+band_height, :] = np.roll(result[y:y+band_height, :], shift, axis=1)
        
        return result
    
    @staticmethod
    def _psychedelic_refraction(img: np.ndarray, intensity: float, frame_time: float) -> np.ndarray:
        """
        Effetto rifrazione psichedelica intelligente - simula rifrazione prismatica
        con distorsioni fluide e dispersione cromatica dei pixel.
        """
        if intensity < 0.2:
            return img
        
        h, w = img.shape[:2]
        result = img.copy()
        
        # Parametri di rifrazione basati sull'intensità
        refraction_strength = (intensity - 0.2) * 40
        wave_speed = frame_time * 2.0  # Animazione fluida
        
        # Crea griglia di coordinate
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        
        # ===== RIFRAZIONE ONDULATA MULTI-DIREZIONALE =====
        # Onde sinusoidali multiple per effetto prismatico
        wave1_x = refraction_strength * np.sin((y_coords / 30.0) + wave_speed) * np.cos((x_coords / 40.0) + wave_speed * 0.7)
        wave1_y = refraction_strength * np.cos((x_coords / 35.0) + wave_speed) * np.sin((y_coords / 45.0) + wave_speed * 0.5)
        
        wave2_x = refraction_strength * 0.6 * np.sin((y_coords / 20.0) - wave_speed * 0.8) * np.cos((x_coords / 25.0))
        wave2_y = refraction_strength * 0.6 * np.cos((x_coords / 22.0) - wave_speed * 0.6) * np.sin((y_coords / 28.0))
        
        # Combina onde per rifrazione complessa
        map_x = x_coords + wave1_x + wave2_x
        map_y = y_coords + wave1_y + wave2_y
        
        # ===== DISPERSIONE CROMATICA (PRISMATICA) =====
        # Separa i canali RGB con offset diversi per simulare rifrazione della luce
        dispersion_amount = int(refraction_strength * 0.3)
        
        # Canale Rosso - rifrazione verso l'esterno
        map_x_r = np.clip(map_x + dispersion_amount, 0, w - 1)
        map_y_r = np.clip(map_y, 0, h - 1)
        r_channel = cv2.remap(img[:, :, 0], map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Canale Verde - rifrazione minima (centro)
        map_x_g = np.clip(map_x, 0, w - 1)
        map_y_g = np.clip(map_y, 0, h - 1)
        g_channel = cv2.remap(img[:, :, 1], map_x_g, map_y_g, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Canale Blu - rifrazione verso l'interno
        map_x_b = np.clip(map_x - dispersion_amount, 0, w - 1)
        map_y_b = np.clip(map_y, 0, h - 1)
        b_channel = cv2.remap(img[:, :, 2], map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Ricomponi con dispersione cromatica
        result = np.stack([r_channel, g_channel, b_channel], axis=2)
        
        # ===== EFFETTO CRISTALLO (PIXEL SHIFT INTELLIGENTE) =====
        if intensity > 0.5:
            # Crea zone di rifrazione locale (simula facce di cristallo)
            crystal_intensity = (intensity - 0.5) * 2.0
            num_crystals = int(crystal_intensity * 8)
            
            for _ in range(num_crystals):
                # Centro casuale della "faccia cristallina"
                cx = np.random.randint(w // 4, 3 * w // 4)
                cy = np.random.randint(h // 4, 3 * h // 4)
                radius = np.random.randint(30, 80)
                
                # Calcola distanza da centro cristallo
                dx = x_coords - cx
                dy = y_coords - cy
                distance = np.sqrt(dx**2 + dy**2)
                
                # Maschera circolare con fade
                mask = np.exp(-(distance / radius)**2)
                mask = mask[:, :, np.newaxis]  # Aggiungi dimensione per broadcasting
                
                # Angolo di rifrazione basato sulla posizione
                angle = np.arctan2(dy, dx)
                shift_strength = crystal_intensity * 15
                
                # Shift radiale (allontanamento dal centro)
                shift_x = (shift_strength * np.cos(angle) * mask[:, :, 0]).astype(np.float32)
                shift_y = (shift_strength * np.sin(angle) * mask[:, :, 0]).astype(np.float32)
                
                # Applica shift locale
                local_map_x = np.clip(x_coords + shift_x, 0, w - 1)
                local_map_y = np.clip(y_coords + shift_y, 0, h - 1)
                
                refracted_region = cv2.remap(result, local_map_x, local_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                
                # Blending con maschera
                result = (result * (1 - mask) + refracted_region * mask).astype(np.uint8)
        
        # ===== KALEIDOSCOPE EFFECT (ALTA INTENSITÀ) =====
        if intensity > 0.7:
            kaleidoscope_strength = (intensity - 0.7) * 3.3
            segments = 6  # Segmenti del caleidoscopio
            
            # Crea coordinate polari dal centro
            cx, cy = w // 2, h // 2
            dx = x_coords - cx
            dy = y_coords - cy
            radius = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # Mirror pattern per segmenti
            theta_segment = (theta % (2 * np.pi / segments)) * segments
            
            # Coordinate cartesiane riflesse
            new_x = cx + radius * np.cos(theta_segment)
            new_y = cy + radius * np.sin(theta_segment)
            
            # Clamp alle dimensioni
            new_x = np.clip(new_x, 0, w - 1).astype(np.float32)
            new_y = np.clip(new_y, 0, h - 1).astype(np.float32)
            
            kaleidoscope_img = cv2.remap(result, new_x, new_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # Blending con immagine originale
            result = cv2.addWeighted(result, 1 - kaleidoscope_strength * 0.4, kaleidoscope_img, kaleidoscope_strength * 0.4, 0)
        
        return result
    
    @staticmethod
    def _liquid_flow(img: np.ndarray, intensity: float, frame_time: float) -> np.ndarray:
        """
        Effetto flusso liquido - distorsione fluida che simula rifrazione attraverso liquido.
        """
        if intensity < 0.3:
            return img
        
        h, w = img.shape[:2]
        flow_strength = (intensity - 0.3) * 25
        
        # Crea griglia coordinate
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        
        # Pattern di flusso fluido con movimento
        flow_time = frame_time * 1.5
        
        # Onde multiple per simulare turbolenza
        flow_x = flow_strength * (
            np.sin((y_coords / 25.0) + flow_time) * 0.5 +
            np.cos((x_coords / 30.0) + flow_time * 0.7) * 0.3 +
            np.sin((y_coords / 15.0) - flow_time * 0.5) * 0.2
        )
        
        flow_y = flow_strength * (
            np.cos((x_coords / 28.0) + flow_time) * 0.5 +
            np.sin((y_coords / 22.0) + flow_time * 0.6) * 0.3 +
            np.cos((x_coords / 18.0) - flow_time * 0.4) * 0.2
        )
        
        # Applica distorsione
        map_x = np.clip(x_coords + flow_x, 0, w - 1).astype(np.float32)
        map_y = np.clip(y_coords + flow_y, 0, h - 1).astype(np.float32)
        
        result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    def _interpolate_color(
        self, 
        color1: Tuple[float, float, float], 
        color2: Tuple[float, float, float], 
        factor: float
    ) -> Tuple[float, float, float]:
        """
        Interpola smooth tra due colori RGB.
        
        Args:
            color1: Colore iniziale (R, G, B) in range 0.0-1.0
            color2: Colore finale (R, G, B) in range 0.0-1.0
            factor: Fattore di interpolazione 0.0-1.0
        
        Returns:
            Colore interpolato (R, G, B)
        """
        r = color1[0] * (1 - factor) + color2[0] * factor
        g = color1[1] * (1 - factor) + color2[1] * factor
        b = color1[2] * (1 - factor) + color2[2] * factor
        return (r, g, b)
    
    def _get_smooth_color(self, target_color_index: int, transition_factor: float) -> Tuple[float, float, float]:
        """
        Ottiene un colore con transizione smooth dal colore precedente.
        
        Args:
            target_color_index: Indice del colore target nella palette
            transition_factor: Fattore di transizione 0.0-1.0
        
        Returns:
            Colore interpolato smooth
        """
        # Se il colore è cambiato, interpola dal precedente
        if target_color_index != self._previous_color_index:
            # Interpola tra colore precedente e nuovo
            prev_color = self.dark_colors[self._previous_color_index % len(self.dark_colors)]
            new_color = self.dark_colors[target_color_index % len(self.dark_colors)]
            
            # Usa transition_factor per interpolazione smooth
            interpolated = self._interpolate_color(prev_color, new_color, transition_factor)
            
            # Aggiorna colore precedente quando transizione è quasi completa
            if transition_factor > 0.9:
                self._previous_color_index = target_color_index
            
            return interpolated
        else:
            # Nessun cambio, ritorna colore diretto
            return self.dark_colors[target_color_index % len(self.dark_colors)]
    
    def _get_transition_factor(self, current_time: float, section_type: str) -> float:
        """
        Calcola il fattore di transizione smooth (0.0 a 1.0) tra sezioni.
        Ritorna 1.0 se siamo completamente nella nuova sezione,
        valori più bassi se siamo ancora in transizione dalla precedente.
        Usa curva SMOOTHERSTEP per fluidità perfetta e uniforme.
        """
        # Se la sezione è cambiata, aggiorna il tracking
        if self._previous_section != section_type:
            self._previous_section = section_type
            self._section_start_time = current_time
        
        # Calcola quanto tempo è passato dall'inizio della nuova sezione
        time_in_section = current_time - self._section_start_time
        
        # Leggi durata transizione dalla config (default 3.0s per extra smoothness)
        transition_duration = self._transition_duration_from_config if hasattr(self, '_transition_duration_from_config') else self._transition_duration
        
        # Smooth transition usando una curva ease-in-out
        if time_in_section >= transition_duration:
            return 1.0
        
        # SMOOTHERSTEP (Ken Perlin) - PERFETTAMENTE SMOOTH in tutto il range!
        # Formula: 6x^5 - 15x^4 + 10x^3
        # Questa è LA curva standard per interpolazioni smooth in computer graphics
        # Derivata prima e seconda sono 0 agli estremi = massima fluidità
        progress = time_in_section / transition_duration
        return progress * progress * progress * (progress * (progress * 6 - 15) + 10)
    
    def _apply_section_effects(
        self,
        frame: np.ndarray,
        section_type: str,
        bass: float,
        mid: float,
        treble: float,
        beat_intensity: float,
        current_time: float,
        color_index: int
    ) -> np.ndarray:
        """
        Applica effetti specifici basati sul tipo di sezione musicale.
        Include i nuovi effetti (gradient, B&W, negative, triangular, geometric).
        Con transizioni smooth tra sezioni.
        """
        # Ottieni fattore di transizione smooth
        transition = self._get_transition_factor(current_time, section_type)
        
        if section_type == 'intro':
            # ===== INTRO: Effetti minimalisti e graduali =====
            # NUOVO: Gradient blend radiale per focus centrale
            frame = self._gradient_blend(frame, 0.4 * transition, 'radial')
            
            # NUOVO: Black & White parziale per atmosfera cinematografica
            frame = self._black_and_white(frame, 0.6 * transition)
            
            # NUOVO: Geometric distortion sottile (pincushion)
            frame = self._geometric_distortion(frame, 0.3 * transition, 'pincushion')
            
            # Color pulse soft
            frame = self._color_pulse(frame, bass * 0.6 * transition, mid * 0.5 * transition, treble * 0.5 * transition, beat_intensity * 0.3)
            
            # Zoom molto leggero
            if bass > 0.3:
                frame = self._zoom_pulse(frame, bass * 0.4 * transition)
            
            # Liquid flow dolce
            frame = self._liquid_flow(frame, bass * 0.5 * transition, current_time)
            
            # Strobe minimo con colori smooth
            total_intensity = (bass + mid + treble) / 3
            if total_intensity > 0.5:
                frame = self._strobe(frame, total_intensity * 0.5 * transition, color_index, transition)
        
        elif section_type == 'buildup':
            # ===== BUILDUP: Crescita tensione, effetti progressivi =====
            # Intensità crescente nel tempo (simula buildup)
            buildup_factor = 1.0 + (bass + mid) * 0.5 * transition
            
            # NUOVO: Gradient diagonale per dinamica
            frame = self._gradient_blend(frame, mid * 0.6 * transition, 'diagonal')
            
            # NUOVO: Triangular distortion crescente sui mid
            if mid > 0.4:
                frame = self._triangular_distortion(frame, mid * 1.3 * transition, current_time)
            
            # NUOVO: Geometric pinch per tensione
            frame = self._geometric_distortion(frame, bass * 0.7 * transition, 'pinch')
            
            # Color pulse intenso
            frame = self._color_pulse(frame, bass * 1.2 * transition, mid * 1.1 * transition, treble * transition, beat_intensity * 0.7)
            
            # Zoom crescente
            frame = self._zoom_pulse(frame, bass * buildup_factor * 0.9)
            
            # Distorsione crescente
            total_intensity = (bass * 1.3 + mid * 1.2 + treble) / 3
            frame = self._distort(frame, total_intensity * 0.8 * transition)
            
            # Chromatic aberration
            frame = self._chromatic_aberration(frame, treble * 0.9 * transition)
            
            # Screen shake crescente
            frame = self._screen_shake(frame, mid * buildup_factor * 0.7)
            
            # Strobe crescente con colori smooth
            frame = self._strobe(frame, total_intensity * 0.85 * transition, color_index, transition)
        
        elif section_type == 'drop':
            # ===== DROP: Massima energia, tutti gli effetti al top =====
            # NUOVO: Negative flash sui beat forti
            if beat_intensity > 0.75:
                frame = self._negative(frame, beat_intensity * 0.6 * transition)
            
            # NUOVO: Geometric swirl per caos
            total_intensity = (bass * 1.5 + mid * 1.3 + treble * 1.2) / 3
            frame = self._geometric_distortion(frame, total_intensity * 0.85 * transition, 'swirl')
            
            # NUOVO: Gradient horizontal per dinamica
            frame = self._gradient_blend(frame, bass * 0.5 * transition, 'horizontal')
            
            # NUOVO: Triangular distortion massima
            if mid > 0.5:
                frame = self._triangular_distortion(frame, mid * 1.2 * transition, current_time)
            
            # Color pulse estremo (sempre pieno per energia drop)
            frame = self._color_pulse(frame, bass * 1.4, mid * 1.3, treble * 1.2, beat_intensity)
            
            # Zoom esplosivo (graduale)
            frame = self._zoom_pulse(frame, bass * 1.3 * (0.3 + 0.7 * transition))
            
            # Bubble distortion sui bassi forti
            if bass > 0.5:
                frame = self._bubble_distortion(frame, bass * 1.2 * transition)
            
            # Screen shake intenso
            frame = self._screen_shake(frame, (mid + beat_intensity) * 0.8 * transition)
            
            # Distorsione massima
            frame = self._distort(frame, total_intensity * (0.5 + 0.5 * transition))
            
            # Chromatic aberration forte
            frame = self._chromatic_aberration(frame, treble * 1.2 * transition)
            
            # RGB split
            frame = self._rgb_split(frame, treble * 1.1 * transition)
            
            # Strobe massimo con colori smooth
            frame = self._strobe(frame, total_intensity * (0.6 + 0.4 * transition), color_index, transition)
            
            # Glitch sui beat
            if beat_intensity > 0.7:
                frame = self._glitch(frame, beat_intensity * 0.8 * transition)
            
            # Scariche elettriche con colori smooth
            if beat_intensity > 0.6:
                current_color = self._get_smooth_color(color_index, transition)
                frame = self._electric_arcs(frame, beat_intensity * transition, current_color)
        
        elif section_type == 'breakdown':
            # ===== BREAKDOWN: Effetti melodici e fluidi =====
            # NUOVO: Black & White forte per atmosfera melodica
            frame = self._black_and_white(frame, 0.7 * transition)
            
            # NUOVO: Gradient vertical per profondità
            frame = self._gradient_blend(frame, 0.5 * transition, 'vertical')
            
            # NUOVO: Geometric barrel per espansione
            frame = self._geometric_distortion(frame, 0.5 * transition, 'barrel')
            
            # Color pulse melodico
            frame = self._color_pulse(frame, bass * 0.7 * transition, mid * 0.9 * transition, treble * 1.1 * transition, beat_intensity * 0.5)
            
            # Liquid flow prominente
            frame = self._liquid_flow(frame, (bass + mid) * 0.7 * transition, current_time)
            
            # Prism split
            frame = self._prism_split(frame, treble * 0.9 * transition, current_time)
            
            # Distorsione leggera
            total_intensity = (bass * 0.7 + mid * 0.8 + treble) / 3
            frame = self._distort(frame, total_intensity * 0.5 * transition)
            
            # Chromatic aberration soft
            frame = self._chromatic_aberration(frame, treble * 0.6 * transition)
            
            # Scan lines
            frame = self._scan_lines(frame, mid * 0.6 * transition)
        
        elif section_type == 'outro':
            # ===== OUTRO: Effetti che decadono, atmosferici =====
            # NUOVO: Black & White progressivo (fade to B&W)
            # Calcola il progresso dell'outro (da 0 a 1)
            # Assumiamo che current_time aumenta, quindi usiamo energia decrescente come proxy
            total_energy = (bass + mid + treble) / 3
            bw_intensity = (1.0 - total_energy) * transition  # Più energia bassa, più B&W
            frame = self._black_and_white(frame, min(bw_intensity, 0.9 * transition))
            
            # NUOVO: Gradient radiale per fade concentrico
            frame = self._gradient_blend(frame, 0.7 * transition, 'radial')
            
            # NUOVO: Geometric pincushion per chiusura
            frame = self._geometric_distortion(frame, 0.4 * transition, 'pincushion')
            
            # Color pulse decrescente
            frame = self._color_pulse(frame, bass * 0.5 * transition, mid * 0.4 * transition, treble * 0.6 * transition, beat_intensity * 0.2)
            
            # Liquid flow lento
            frame = self._liquid_flow(frame, bass * 0.4 * transition, current_time * 0.5)
            
            # Zoom minimo
            if bass > 0.3:
                frame = self._zoom_pulse(frame, bass * 0.3 * transition)
            
            # Distorsione minima
            total_intensity = (bass + mid + treble) / 3
            frame = self._distort(frame, total_intensity * 0.3 * transition)
            
            # VHS distortion occasionale (effetto "fine cassetta")
            if np.random.random() < 0.3:
                frame = self._vhs_distortion(frame, treble * 0.5 * transition)
        
        else:  # 'steady'
            # ===== STEADY: Effetti bilanciati e stabili =====
            # NUOVO: Triangular distortion sui mid (synth/lead)
            if mid > 0.4:
                frame = self._triangular_distortion(frame, mid * 0.8 * transition, current_time)
            
            # NUOVO: Geometric adaptive (cambia occasionalmente)
            if bass > 0.5:
                frame = self._geometric_distortion(frame, bass * 0.6 * transition, 'pinch')
            
            # Color pulse normale
            frame = self._color_pulse(frame, bass * transition, mid * transition, treble * transition, beat_intensity * 0.6)
            
            # Zoom moderato
            frame = self._zoom_pulse(frame, bass * 0.7 * transition)
            
            # Distorsione media
            total_intensity = (bass + mid + treble) / 3
            frame = self._distort(frame, total_intensity * 0.6 * transition)
            
            # Chromatic aberration
            frame = self._chromatic_aberration(frame, treble * 0.7 * transition)
            
            # Strobe medio con colori smooth
            frame = self._strobe(frame, total_intensity * 0.7 * transition, color_index, transition)
            
            # Glitch occasionale
            if np.random.random() < 0.1:
                frame = self._glitch(frame, (bass + treble) * 0.4)
        
        return frame
    
    @staticmethod
    def _prism_split(img: np.ndarray, intensity: float, frame_time: float) -> np.ndarray:
        """
        Effetto split prismatico avanzato - separa RGB con pattern complesso.
        """
        if intensity < 0.4:
            return img
        
        result = img.copy()
        h, w = img.shape[:2]
        
        split_amount = int((intensity - 0.4) * 30)
        
        # Pattern sinusoidale per split non uniforme
        y_coords = np.arange(h)
        wave_offset = (np.sin(y_coords / 20.0 + frame_time * 2) * split_amount / 2).astype(int)
        
        # Split RGB con pattern ondulato
        for y in range(h):
            offset = wave_offset[y]
            result[y, :, 0] = np.roll(img[y, :, 0], -split_amount - offset)  # Rosso
            result[y, :, 2] = np.roll(img[y, :, 2], split_amount + offset)   # Blu
        
        # Aggiungi split verticale per intensità alte
        if intensity > 0.6:
            x_coords = np.arange(w)
            wave_offset_v = (np.cos(x_coords / 25.0 + frame_time * 1.5) * split_amount / 3).astype(int)
            
            for x in range(w):
                offset_v = wave_offset_v[x]
                result[:, x, 0] = np.roll(result[:, x, 0], offset_v)
                result[:, x, 2] = np.roll(result[:, x, 2], -offset_v)
        
        return result
    
    @staticmethod
    def _gradient_blend(img: np.ndarray, intensity: float, direction: str = "horizontal") -> np.ndarray:
        """
        Effetto passaggi sfumati - applica gradienti di colore con transizioni smooth.
        
        Args:
            img: Immagine input
            intensity: Intensità dell'effetto (0-1)
            direction: Direzione del gradiente ("horizontal", "vertical", "radial", "diagonal")
        """
        if intensity < 0.2:
            return img
        
        h, w = img.shape[:2]
        result = img.copy().astype(np.float32)
        
        # Crea maschera di gradiente
        if direction == "horizontal":
            gradient = np.linspace(0, 1, w)
            gradient = np.tile(gradient, (h, 1))
        elif direction == "vertical":
            gradient = np.linspace(0, 1, h)
            gradient = np.tile(gradient[:, np.newaxis], (1, w))
        elif direction == "radial":
            cx, cy = w // 2, h // 2
            y_coords, x_coords = np.indices((h, w))
            dx = x_coords - cx
            dy = y_coords - cy
            distance = np.sqrt(dx**2 + dy**2)
            max_dist = np.sqrt(cx**2 + cy**2)
            gradient = distance / max_dist
        else:  # diagonal
            y_coords, x_coords = np.indices((h, w))
            gradient = (x_coords + y_coords) / (w + h)
        
        # Applica sfumatura con intensità
        gradient = gradient[:, :, np.newaxis]  # Aggiungi dimensione per broadcasting
        blend_factor = (intensity - 0.2) * 1.2
        
        # Inverti colori seguendo il gradiente
        inverted = 255 - result
        result = result * (1 - gradient * blend_factor) + inverted * (gradient * blend_factor)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _black_and_white(img: np.ndarray, intensity: float) -> np.ndarray:
        """
        Effetto bianco e nero con controllo di intensità.
        
        Args:
            img: Immagine input RGB
            intensity: Intensità dell'effetto (0-1). 0 = colore originale, 1 = completamente B&W
        """
        if intensity < 0.1:
            return img
        
        # Converti a scala di grigi usando formula luminance standard
        # Y = 0.299*R + 0.587*G + 0.114*B
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        
        # Converti gray da 1 canale a 3 canali RGB
        gray_rgb = np.stack([gray, gray, gray], axis=2).astype(np.uint8)
        
        # Blending tra immagine originale e B&W
        blend_factor = min(intensity, 1.0)
        result = cv2.addWeighted(img, 1 - blend_factor, gray_rgb, blend_factor, 0)
        
        return result
    
    @staticmethod
    def _negative(img: np.ndarray, intensity: float) -> np.ndarray:
        """
        Effetto negativo - inverte i colori dell'immagine.
        
        Args:
            img: Immagine input RGB
            intensity: Intensità dell'effetto (0-1). 0 = colore originale, 1 = completamente negativo
        """
        if intensity < 0.1:
            return img
        
        # Inverti i colori
        inverted = 255 - img
        
        # Blending tra immagine originale e negativo
        blend_factor = min(intensity, 1.0)
        result = cv2.addWeighted(img, 1 - blend_factor, inverted, blend_factor, 0)
        
        return result
    
    @staticmethod
    def _triangular_distortion(img: np.ndarray, intensity: float, frame_time: float) -> np.ndarray:
        """
        Effetto distorsione triangolare - crea pattern triangolari di distorsione.
        
        Args:
            img: Immagine input
            intensity: Intensità dell'effetto (0-1)
            frame_time: Tempo corrente per animazione
        """
        if intensity < 0.3:
            return img
        
        h, w = img.shape[:2]
        distortion_strength = (intensity - 0.3) * 40
        
        # Crea griglia di coordinate
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        
        # Pattern triangolare (onda triangolare invece di sinusoidale)
        # Frequenza triangolare lungo x e y
        freq_x = 0.05
        freq_y = 0.04
        
        # Onda triangolare: 2*abs(x - floor(x + 0.5))
        tri_wave_x = 2 * np.abs((x_coords * freq_x + frame_time) - np.floor((x_coords * freq_x + frame_time) + 0.5))
        tri_wave_y = 2 * np.abs((y_coords * freq_y + frame_time * 0.7) - np.floor((y_coords * freq_y + frame_time * 0.7) + 0.5))
        
        # Combina onde triangolari per distorsione
        distort_x = distortion_strength * (tri_wave_y - 0.5)
        distort_y = distortion_strength * (tri_wave_x - 0.5)
        
        # Applica distorsione
        map_x = np.clip(x_coords + distort_x, 0, w - 1).astype(np.float32)
        map_y = np.clip(y_coords + distort_y, 0, h - 1).astype(np.float32)
        
        result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    @staticmethod
    def _geometric_distortion(img: np.ndarray, intensity: float, mode: str = "pinch") -> np.ndarray:
        """
        Effetto distorsione geometrica - applica distorsioni basate su forme geometriche.
        
        Args:
            img: Immagine input
            intensity: Intensità dell'effetto (0-1)
            mode: Tipo di distorsione ("pinch", "barrel", "pincushion", "swirl")
        """
        if intensity < 0.2:
            return img
        
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Crea griglia di coordinate normalizzate
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        
        # Coordinate relative al centro
        dx = (x_coords - cx) / cx
        dy = (y_coords - cy) / cy
        
        # Distanza dal centro
        distance = np.sqrt(dx**2 + dy**2)
        
        # Parametro di intensità
        strength = (intensity - 0.2) * 1.5
        
        if mode == "pinch":
            # Pinch effect - comprime verso il centro
            factor = 1.0 - strength * (1.0 - distance) ** 2
            factor = np.maximum(factor, 0.1)  # Evita divisione per zero
            new_dx = dx * factor
            new_dy = dy * factor
        
        elif mode == "barrel":
            # Barrel distortion - rigonfiamento
            r2 = dx**2 + dy**2
            distortion = 1.0 + strength * r2
            new_dx = dx * distortion
            new_dy = dy * distortion
        
        elif mode == "pincushion":
            # Pincushion distortion - compressione bordi
            r2 = dx**2 + dy**2
            distortion = 1.0 - strength * r2
            new_dx = dx * distortion
            new_dy = dy * distortion
        
        else:  # swirl
            # Swirl effect - rotazione crescente dal centro
            angle = np.arctan2(dy, dx)
            swirl_amount = strength * (1.0 - distance)
            new_angle = angle + swirl_amount
            new_dx = distance * np.cos(new_angle) * np.sign(dx)
            new_dy = distance * np.sin(new_angle) * np.sign(dy)
        
        # Converti coordinate normalizzate a pixel
        map_x = np.clip(cx + new_dx * cx, 0, w - 1).astype(np.float32)
        map_y = np.clip(cy + new_dy * cy, 0, h - 1).astype(np.float32)
        
        result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    @staticmethod
    def _rotating_3d_solids(
        img: np.ndarray, 
        intensity: float, 
        frame_time: float,
        beat_intensity: float = 0.0,
        bass: float = 0.0,
        mid: float = 0.0,
        treble: float = 0.0
    ) -> np.ndarray:
        """
        Effetto solidi 3D rotanti - distorce l'immagine in forme geometriche 3D che ruotano,
        pulsano e ESPLODONO in particelle psichedeliche a tempo di musica.
        
        Args:
            img: Immagine input
            intensity: Intensità globale dell'effetto (0-1)
            frame_time: Tempo corrente per animazione
            beat_intensity: Intensità del beat (0-1) per collisioni/esplosioni
            bass: Intensità bassi (0-1) per pulsazioni
            mid: Intensità medi (0-1) per velocità rotazione
            treble: Intensità alti (0-1) per frammentazione
        """
        if intensity < 0.3:
            return img
        
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Crea griglia di coordinate
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        
        # Coordinate relative al centro (-1 a 1)
        dx = (x_coords - cx) / (w / 2)
        dy = (y_coords - cy) / (h / 2)
        
        # Distanza dal centro
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # ========== PARAMETRI AUDIO-REATTIVI (ULTRA RALLENTATI) ==========
        # Velocità rotazione ULTRA LENTA - controllata dai mid
        rotation_speed = 0.15 + mid * 0.4  # DRASTICAMENTE rallentato! Era 0.5 + mid * 1.2
        
        # Pulsazione controllata dai bassi (kick drum) - MOLTO PIÙ LENTA
        pulse_amount = 0.1 + bass * 0.3
        pulse_wave = np.sin(frame_time * 2.0) * pulse_amount  # LENTISSIMA - era 4.0
        
        # Numero di solidi FISSO per semplicità - pochi solidi grandi e chiari
        num_solids = 4  # FISSO a 4 per non confondere
        
        # ========== ESPLOSIONE SEMPLIFICATA ==========
        # Esplosione sui beat forti - MOLTO PIÙ SEMPLICE E LENTA
        explosion_intensity = 0.0
        particle_dispersion = 0.0
        
        if beat_intensity > 0.7:  # Soglia più alta - solo sui beat forti
            # Intensità esplosione RIDOTTA per non confondere
            explosion_intensity = (beat_intensity - 0.7) * 1.2  # Era 2.5 - MOLTO ridotto!
            
            # Dispersione particelle RALLENTATA
            explosion_wave = np.sin(distance * 4.0 - frame_time * 3.0) * explosion_intensity  # Era 8.0 e 12.0
            particle_dispersion = explosion_wave * (1.0 - distance * 0.5)  # Meno intenso
        
        # ========== PROIEZIONE 3D (RALLENTATA) ==========
        # Simula rotazione 3D con proiezione prospettica
        # Dividi l'immagine in segmenti radiali (solidi)
        segment_angle = (2 * np.pi) / num_solids
        segment_id = np.floor(angle / segment_angle).astype(int)
        
        # Ogni segmento ruota indipendentemente (PIÙ LENTO)
        rotation_offset = segment_id * (np.pi / num_solids)
        current_rotation = frame_time * rotation_speed + rotation_offset
        
        # Proiezione pseudo-3D SEMPLIFICATA
        # Simula rotazione attorno all'asse Y (orizzontale) - MOLTO PIÙ LENTA
        depth = 0.4 + 0.6 * np.cos(current_rotation * 0.5 + angle * num_solids * 0.3)  # RALLENTATO TUTTO
        depth = np.clip(depth, 0.3, 1.0)  # Range ridotto per meno variazione
        
        # Scala basata sulla profondità (prospettiva) - PIÙ STABILE
        perspective_scale = 0.8 + 0.2 * depth  # Era 0.7 + 0.3 - meno variazione!
        
        # Pulsazione basata sui bassi - RIDOTTA
        pulse_scale = 1.0 + pulse_wave * 0.3 * (1.0 - distance * 0.3)  # Molto più sottile
        
        # ESPLOSIONE PARTICELLARE sui beat - MOLTO PIÙ DELICATA
        explosion_scale = 1.0
        if beat_intensity > 0.7:  # Soglia più alta
            # Esplosione MINIMA per non confondere
            explosion_scale = 1.0 + explosion_intensity * 0.4 * (1.1 - distance * 0.5)  # Era 1.2 - RIDOTTO!
        
        # Scala finale combinata
        final_scale = perspective_scale * pulse_scale * explosion_scale
        
        # ========== DISTORSIONE DIMENSIONALE (ULTRA SEMPLIFICATA) ==========
        # Warp MINIMO - solo un tocco di liquidità
        warp_intensity = (intensity - 0.3) * 0.6  # Era 1.5 - DRASTICAMENTE ridotto!
        
        # Onda sinusoidale LENTISSIMA e SOTTILE
        dimensional_wave_x = np.sin(angle * num_solids * 0.3 + current_rotation * 0.5) * warp_intensity * 0.08  # Era 0.7 e 1.0
        dimensional_wave_y = np.cos(angle * num_solids * 0.3 - current_rotation * 0.4) * warp_intensity * 0.08  # Era 0.7 e 0.8
        
        # ========== EFFETTO PARTICELLE (MOLTO RIDOTTO) ==========
        # Solo un leggero disturbo durante esplosioni forti
        if beat_intensity > 0.75:  # Solo beat MOLTO forti
            # Particelle MINIME - solo un tocco
            particle_angle = angle + np.sin(distance * 8.0 + frame_time * 3.0) * explosion_intensity * 0.3  # RALLENTATO
            particle_chaos_x = np.cos(particle_angle) * particle_dispersion * 0.1  # Era 0.3 - ridotto!
            particle_chaos_y = np.sin(particle_angle) * particle_dispersion * 0.1
            
            dimensional_wave_x += particle_chaos_x
            dimensional_wave_y += particle_chaos_y
        
        # ========== ROTAZIONE SOLIDI (RALLENTATA) ==========
        # Ruota ogni segmento attorno al proprio centro PIÙ LENTAMENTE
        segment_center_angle = segment_id * segment_angle + segment_angle / 2
        segment_cx = cx + np.cos(segment_center_angle) * (w * 0.15)
        segment_cy = cy + np.sin(segment_center_angle) * (h * 0.15)
        
        # Coordinate relative al centro del segmento
        seg_dx = x_coords - segment_cx
        seg_dy = y_coords - segment_cy
        seg_distance = np.sqrt(seg_dx**2 + seg_dy**2)
        seg_angle = np.arctan2(seg_dy, seg_dx)
        
        # Rotazione locale del segmento ULTRA LENTA
        local_rotation = current_rotation * 0.5  # Era 1.0 - ANCORA PIÙ LENTA!
        rotated_angle = seg_angle + local_rotation
        
        # ========== RICOMPOSIZIONE COORDINATE (SEMPLIFICATA) ==========
        # Applica tutte le trasformazioni
        new_distance = distance * final_scale
        
        # Mix tra rotazione globale e locale - PIÙ GLOBALE (meno caotico)
        mix_factor = 0.4  # Era 0.6 - meno mix per movimento più fluido
        final_angle = angle * (1 - mix_factor) + rotated_angle * mix_factor
        
        # Converti coordinate polari in cartesiane
        new_x = cx + new_distance * np.cos(final_angle) * (w / 2)
        new_y = cy + new_distance * np.sin(final_angle) * (h / 2)
        
        # Aggiungi warp dimensionale
        new_x += dimensional_wave_x * w * 0.1
        new_y += dimensional_wave_y * h * 0.1
        
        # ========== EFFETTO ESPLOSIONE (MOLTO RIDOTTO) ==========
        if beat_intensity > 0.75:  # Solo beat MOLTO forti
            # INTERFERENZE MINIME tra solidi
            interference = np.sin(angle * num_solids + frame_time * 2.5) * beat_intensity * 0.2  # Era 0.5 e 6
            new_x += np.cos(angle) * interference * w * 0.04  # Era 0.08 - dimezzato!
            new_y += np.sin(angle) * interference * h * 0.04
            
            # ESPLOSIONE RADIALE MINIMA
            explosion_push = explosion_intensity * distance * 0.15  # Era 0.4 - molto ridotto!
            new_x += np.cos(angle) * explosion_push * w
            new_y += np.sin(angle) * explosion_push * h
        
        # Clipping e rimapping
        map_x = np.clip(new_x, 0, w - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h - 1).astype(np.float32)
        
        result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        
        # ========== POST-PROCESSING: EFFETTI SEMPLIFICATI ==========
        
        # 1. BORDI LUMINOSI SOTTILI sui solidi
        if intensity > 0.55:
            # Identifica bordi tra segmenti - linee SOTTILI e pulite
            segment_boundary = np.abs(np.sin(angle * num_solids)) < 0.04  # Era 0.08 - PIÙ SOTTILE!
            edge_brightness = (intensity - 0.55) * 2.0  # Era 3.5 - MOLTO ridotto!
            
            # Colori FISSI più sobri (meno psichedelici)
            edge_color = np.array([200, 150, 255])  # Viola/blu fisso - meno caotico
            
            # Applica glow DELICATO sui bordi
            for i in range(3):
                result[:, :, i] = np.where(
                    segment_boundary,
                    np.clip(result[:, :, i] + edge_brightness * edge_color[i], 0, 255),
                    result[:, :, i]
                )
        
        # 2. FLASH DELICATO sui beat forti (MOLTO ridotto)
        if beat_intensity > 0.8:  # Solo beat FORTISSIMI
            # Flash MINIMO dal centro
            flash_mask = np.exp(-distance * 3.0) * explosion_intensity * 0.2  # Era 2.0 e 0.6 - molto ridotto!
            flash_mask = flash_mask[:, :, np.newaxis]
            
            # Colore fisso bianco/blu invece di ciclico
            flash_color = np.array([180, 200, 255])  # Blu chiaro fisso
            
            # Applica flash LEGGERO
            result = result.astype(np.float32)
            result += flash_mask * flash_color * beat_intensity * 0.5  # Ridotto al 50%
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result.astype(np.uint8)

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
        print(f"DEBUG: Estrazione bande di frequenza...")
        bass, mid, treble = self._extract_frequency_bands(magnitude, frequencies)
        print(f"DEBUG: Bande estratte - bass: {len(bass)}, mid: {len(mid)}, treble: {len(treble)}")
        print(f"DEBUG: Rilevamento beat...")
        beat_times = self._detect_beats(y, sr)
        print(f"DEBUG: Beat rilevati: {len(beat_times)}")
        
        # Analisi sezioni intelligente (se richiesto)
        sections = None
        if self.effect_style == "intelligent":
            sections = self._analyze_track_sections(y, sr, bass, mid, treble, self._transition_duration_from_config)
            print(f"🧠 Modalità intelligente attivata con {len(sections)} sezioni")

        img = cv2.imread(self.image_file)
        if img is None:
            raise FileNotFoundError(f"Impossibile caricare immagine: {self.image_file}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Use native resolution if target_resolution is None
        if self.target_resolution is not None:
            img = cv2.resize(img, self.target_resolution)
        else:
            # Log native resolution
            h, w = img.shape[:2]
            if self.progress_cb:
                self.progress_cb("status", {"message": f"Usando risoluzione nativa: {w}x{h}"})

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

            # Base frame
            frame = img.copy()
            
            total_intensity = (bass[i] + mid[i] + treble[i]) / 3
            color_index = int(current_time * 2) % len(self.dark_colors)
            
            if self.effect_style == "intelligent":
                # ========== INTELLIGENT ADAPTIVE: Effetti basati su analisi sezioni ==========
                # Trova sezione corrente
                current_section_type = 'steady'  # default
                if sections:
                    for section in sections:
                        # Converti tempo frame in tempo audio
                        audio_time = (i / num_frames) * (len(y) / sr)
                        if section['start_time'] <= audio_time <= section['end_time']:
                            current_section_type = section['type']
                            break
                
                # Applica effetti specifici per la sezione
                frame = self._apply_section_effects(
                    frame, 
                    current_section_type,
                    bass[i], 
                    mid[i], 
                    treble[i],
                    beat_intensity,
                    current_time,
                    color_index
                )
            
            elif self.effect_style == "psychedelic":
                # ========== PSYCHEDELIC REFRACTION: EFFETTI DI RIFRAZIONE INTELLIGENTE ==========
                
                # RIFRAZIONE PRINCIPALE - Effetto prismatico fluido
                refraction_intensity = bass[i] * 0.6 + mid[i] * 0.3 + treble[i] * 0.3
                frame = self._psychedelic_refraction(frame, refraction_intensity, current_time)
                
                # FLUSSO LIQUIDO - Distorsione fluida sui bassi
                frame = self._liquid_flow(frame, bass[i] * 1.1, current_time)
                
                # SPLIT PRISMATICO - Dispersione RGB complessa
                frame = self._prism_split(frame, treble[i] * 1.15, current_time)
                
                # Effetti colore base potenziati
                frame = self._color_pulse(frame, bass[i] * 1.2, mid[i] * 1.1, treble[i] * 1.1, beat_intensity)
                
                # Zoom pulsante leggero sui bassi
                if bass[i] > 0.4:
                    frame = self._zoom_pulse(frame, bass[i] * 0.8)
                
                # Effetto bolla sui bassi forti
                if bass[i] > 0.5:
                    frame = self._bubble_distortion(frame, bass[i] * 0.9)
                
                # Strobe moderato
                frame = self._strobe(frame, total_intensity * 0.85, color_index)
                
                # Distorsione fluida
                frame = self._distort(frame, total_intensity * 0.7)
                
                # Aberrazione cromatica psichedelica
                frame = self._chromatic_aberration(frame, treble[i] * 0.9)
                
                # Kaleidoscope sui beat intensi
                if beat_intensity > 0.8:
                    # Effetto già integrato in _psychedelic_refraction con intensità alta
                    pass
                
                # Glitch leggero per texture
                if np.random.random() < 0.1:
                    glitch_intensity = (bass[i] + treble[i]) * 0.3
                    frame = self._glitch(frame, glitch_intensity)
            
            elif self.effect_style == "extreme":
                # ========== EXTREME VIBRANT: EFFETTI ELETTRICI E DISTORSIVI ==========
                # Zoom pulsante sui bassi
                frame = self._zoom_pulse(frame, bass[i] * 1.2)
                
                # Effetto bolla sui bassi
                frame = self._bubble_distortion(frame, bass[i])
                
                # Screen shake su snare/mid (vibrazioni intense)
                frame = self._screen_shake(frame, mid[i] * 1.3)
                
                # Effetti colore base
                frame = self._color_pulse(frame, bass[i], mid[i], treble[i], beat_intensity)
                
                # Strobe
                frame = self._strobe(frame, total_intensity, color_index)
                
                # Distorsione base
                frame = self._distort(frame, total_intensity)
                
                # Aberrazione cromatica
                frame = self._chromatic_aberration(frame, treble[i] * 1.1)
                
                # RGB Split estremo su alti
                frame = self._rgb_split(frame, treble[i] * 1.2)
                
                # Glitch base
                glitch_intensity = bass[i] * 0.5 + treble[i] * 0.5
                frame = self._glitch(frame, glitch_intensity)
                
                # Scariche elettriche sui beat
                if beat_intensity > 0.5:
                    current_color = self.dark_colors[color_index]
                    frame = self._electric_arcs(frame, beat_intensity, current_color)
                
                # Scan lines CRT
                frame = self._scan_lines(frame, mid[i] * 0.8)
                
                # VHS distortion casuale
                if np.random.random() < treble[i] * 0.3:
                    frame = self._vhs_distortion(frame, treble[i])
            
            elif self.effect_style == "3d_solids":
                # ========== 3D ROTATING SOLIDS: SOLIDI GEOMETRICI TRIDIMENSIONALI ==========
                
                # EFFETTO PRINCIPALE: 3D Rotating Solids
                frame = self._rotating_3d_solids(
                    frame,
                    intensity=total_intensity * 0.85,
                    frame_time=current_time,
                    beat_intensity=beat_intensity,
                    bass=bass[i],
                    mid=mid[i],
                    treble=treble[i]
                )
                
                # Effetti complementari che enfatizzano il 3D
                
                # Aberrazione cromatica per profondità
                frame = self._chromatic_aberration(frame, treble[i] * 0.9)
                
                # Zoom leggero per enfasi prospettiva
                if bass[i] > 0.5:
                    frame = self._zoom_pulse(frame, bass[i] * 0.4)
                
                # Color pulse per vividezza
                frame = self._color_pulse(frame, bass[i] * 1.1, mid[i] * 1.0, treble[i] * 0.9, beat_intensity)
                
                # Geometric distortion complementare sui mid alti
                if mid[i] > 0.6:
                    frame = self._geometric_distortion(frame, mid[i] * 0.5, 'swirl')
                
                # Glitch leggero per texture techno
                if np.random.random() < 0.15:
                    glitch_intensity = (bass[i] + treble[i]) * 0.3
                    frame = self._glitch(frame, glitch_intensity)
                
                # Bordi elettrici sui beat forti
                if beat_intensity > 0.7:
                    current_color = self.dark_colors[color_index]
                    frame = self._electric_arcs(frame, beat_intensity * 0.6, current_color)
                
                # Strobe moderato
                frame = self._strobe(frame, total_intensity * 0.75, color_index)
            
            else:
                # ========== STANDARD: EFFETTI CLASSICI ==========
                frame = self._color_pulse(frame, bass[i], mid[i], treble[i], beat_intensity)
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
        
        try:
            frames = self._generate_frames()
        except Exception as e:
            print(f"ERROR in _generate_frames: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

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


class VideoAudioFX:
    """
    Classe per creare video sincronizzati con audio.
    Il video viene tagliato, messo in loop o allungato per adattarsi alla durata dell'audio.
    L'audio originale del video viene rimosso e vengono applicati effetti audio-reattivi.
    """
    
    def __init__(
        self,
        audio_file: str,
        video_file: str,
        output_file: str = "synced_video.mp4",
        *,
        progress_cb: ProgressCallback = None,
        colors: Optional[List[Tuple[float, float, float]]] = None,
        thresholds: Optional[Tuple[float, float, float]] = None,
        short_video_mode: str = "loop",  # "loop" or "stretch"
        # Logo options
        logo_file: Optional[str] = None,
        logo_position: str = "top-right",
        logo_scale: float = 0.15,
        logo_opacity: float = 1.0,
        logo_margin: int = 12,
    ) -> None:
        self.audio_file = audio_file
        self.video_file = video_file
        self.output_file = output_file
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
        
        # video adaptation mode
        if short_video_mode not in ("loop", "stretch"):
            raise ValueError(f"short_video_mode deve essere 'loop' o 'stretch', ricevuto: {short_video_mode}")
        self.short_video_mode = short_video_mode
        
        # logo configuration
        self.logo_file = logo_file
        self.logo_position = logo_position
        self.logo_scale = float(max(0.01, min(1.0, logo_scale)))
        self.logo_opacity = float(max(0.0, min(1.0, logo_opacity)))
        self.logo_margin = max(0, int(logo_margin))
        self._logo_rgba = None
    
    def _prepare_logo(self, frame_w: int, frame_h: int) -> None:
        """Prepara il logo per l'overlay."""
        if not self.logo_file:
            return
        logo = cv2.imread(self.logo_file, cv2.IMREAD_UNCHANGED)
        if logo is None:
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
        """Sovrappone il logo al frame."""
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
    
    # ------------------------------- Audio Analysis ----------------------------- #
    def _load_and_analyze_audio(
        self,
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """Carica e analizza l'audio."""
        if self.progress_cb:
            self.progress_cb("status", {"message": "Analisi audio..."})

        y, sr = librosa.load(self.audio_file)

        hop_length = max(1, sr // 30)  # Fixed to 30 fps for analysis
        n_fft = 2048
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        return y, sr, frequencies, magnitude

    @staticmethod
    def _extract_frequency_bands(
        magnitude: np.ndarray, frequencies: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estrae bande di frequenza (bass, mid, treble)."""
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
        """Rileva i beat nell'audio."""
        try:
            # Usa parametri più robusti per evitare errori
            tempo, beat_frames = librosa.beat.beat_track(
                y=y, 
                sr=sr,
                hop_length=512,
                start_bpm=120.0,
                units='frames'
            )
            return librosa.frames_to_time(beat_frames, sr=sr)
        except Exception as e:
            print(f"Warning: Beat detection failed: {e}. Returning empty beat array.")
            return np.array([])
    
    # ------------------------------- FX passes ------------------------------ #
    def _color_pulse(
        self,
        img: np.ndarray,
        bass: float,
        mid: float,
        treble: float,
        beat_intensity: float = 0.0,
    ) -> np.ndarray:
        """Applica effetto color pulse reattivo all'audio."""
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
        """Applica effetto strobe."""
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
        """Applica effetto distorsione."""
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
        """Applica effetto glitch."""
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
    
    @staticmethod
    def _chromatic_aberration(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto aberrazione cromatica cyberpunk."""
        if intensity < 0.3:
            return img
        result = img.copy()
        shift_amount = int((intensity - 0.3) * 15)
        result[:, :, 0] = np.roll(img[:, :, 0], -shift_amount, axis=1)
        result[:, :, 2] = np.roll(img[:, :, 2], shift_amount, axis=1)
        return result
    
    @staticmethod
    def _electric_arcs(img: np.ndarray, intensity: float, color: Tuple[float, float, float]) -> np.ndarray:
        """Effetto scariche elettriche."""
        if intensity < 0.7:
            return img
        result = img.copy()
        h, w = img.shape[:2]
        num_arcs = int((intensity - 0.7) * 20)
        for _ in range(num_arcs):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            arc_color = tuple(int(c * 255) for c in color)
            thickness = np.random.randint(1, 4)
            cv2.line(result, (x1, y1), (x2, y2), arc_color, thickness)
            if intensity > 0.85:
                cv2.line(result, (x1, y1), (x2, y2), (255, 255, 255), 1)
        result = cv2.addWeighted(img, 0.7, result, 0.3, 0)
        return result
    
    @staticmethod
    def _bubble_distortion(img: np.ndarray, bass_intensity: float) -> np.ndarray:
        """Effetto bolla sui bassi."""
        if bass_intensity < 0.4:
            return img
        h, w = img.shape[:2]
        factor = (bass_intensity - 0.4) * 30
        cx, cy = w // 2, h // 2
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)
        dx = x_coords - cx
        dy = y_coords - cy
        distance = np.sqrt(dx**2 + dy**2)
        max_distance = np.sqrt(cx**2 + cy**2)
        normalized_distance = distance / max_distance
        deformation = factor * np.sin(normalized_distance * np.pi) * (1 - normalized_distance)
        angle = np.arctan2(dy, dx)
        map_x = x_coords + deformation * np.cos(angle)
        map_y = y_coords + deformation * np.sin(angle)
        result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return result
    
    @staticmethod
    def _zoom_pulse(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto zoom pulsante."""
        if intensity < 0.3:
            return img
        h, w = img.shape[:2]
        zoom_factor = 1.0 + (intensity - 0.3) * 0.5
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        result = zoomed[start_y:start_y+h, start_x:start_x+w]
        return result
    
    @staticmethod
    def _screen_shake(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto shake dello schermo."""
        if intensity < 0.5:
            return img
        h, w = img.shape[:2]
        shake_amount = int((intensity - 0.5) * 30)
        
        # Se shake_amount è 0, nessun shake
        if shake_amount <= 0:
            return img
        
        offset_x = np.random.randint(-shake_amount, shake_amount + 1)
        offset_y = np.random.randint(-shake_amount, shake_amount + 1)
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return result
    
    @staticmethod
    def _rgb_split(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto split RGB estremo."""
        if intensity < 0.6:
            return img
        result = img.copy()
        split_amount = int((intensity - 0.6) * 25)
        result[:, :, 0] = np.roll(img[:, :, 0], split_amount, axis=0)
        result[:, :, 1] = img[:, :, 1]
        result[:, :, 2] = np.roll(img[:, :, 2], -split_amount, axis=0)
        return result
    
    @staticmethod
    def _scan_lines(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto scan lines CRT."""
        if intensity < 0.4:
            return img
        result = img.copy().astype(np.float32)
        h = img.shape[0]
        line_intensity = (intensity - 0.4) * 0.6
        for y in range(0, h, 3):
            result[y:y+1, :] = result[y:y+1, :] * (1.0 - line_intensity)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def _vhs_distortion(img: np.ndarray, intensity: float) -> np.ndarray:
        """Effetto distorsione VHS."""
        if intensity < 0.5:
            return img
        result = img.copy()
        h = img.shape[0]
        num_bands = int((intensity - 0.5) * 15)
        for _ in range(num_bands):
            y = np.random.randint(0, h - 10)
            band_height = np.random.randint(2, 8)
            shift = np.random.randint(-30, 30)
            if y + band_height < h:
                result[y:y+band_height, :] = np.roll(result[y:y+band_height, :], shift, axis=1)
        return result
    
    def create_video(self) -> None:
        """
        Crea il video sincronizzato con l'audio.
        Il video viene tagliato o messo in loop per adattarsi alla durata dell'audio.
        Applica effetti audio-reattivi a ogni frame.
        """
        if self.progress_cb:
            self.progress_cb("status", {"message": "Caricamento e analisi audio..."})
        
        # Analizza audio
        y, sr, frequencies, magnitude = self._load_and_analyze_audio()
        bass, mid, treble = self._extract_frequency_bands(magnitude, frequencies)
        beat_times = self._detect_beats(y, sr)
        
        # Ottieni durata audio
        try:
            audio_clip = AudioFileClip(self.audio_file)
            audio_duration = audio_clip.duration
        except Exception:
            import soundfile as sf
            info = sf.info(self.audio_file)
            audio_duration = info.frames / info.samplerate
        
        # Apri video e ottieni info
        if self.progress_cb:
            self.progress_cb("status", {"message": "Caricamento video..."})
            
        cap = cv2.VideoCapture(self.video_file)
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossibile aprire il video: {self.video_file}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_video_frames / video_fps if video_fps > 0 else 0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.progress_cb:
            self.progress_cb("status", {
                "message": f"Audio: {audio_duration:.2f}s | Video: {video_duration:.2f}s | FPS: {video_fps}"
            })
        
        # Calcola quanti frame totali servono
        target_total_frames = int(np.ceil(audio_duration * video_fps))
        
        # Prepara directory temporanea
        temp_dir = "temp_video_frames"
        os.makedirs(temp_dir, exist_ok=True)
        frame_files: List[str] = []
        
        try:
            if self.progress_cb:
                self.progress_cb("start", {"total_frames": target_total_frames})
            
            # Leggi tutti i frame del video originale
            video_frames: List[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Converti BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_frames.append(frame_rgb)
            
            cap.release()
            
            if not video_frames:
                raise ValueError("Il video non contiene frame validi")
            
            if self.progress_cb:
                self.progress_cb("status", {
                    "message": f"Frame caricati: {len(video_frames)} | Necessari: {target_total_frames}"
                })
            
            # Prepara frame base adattando la durata (trim, loop o stretch)
            base_frames: List[np.ndarray] = []
            
            if len(video_frames) >= target_total_frames:
                # Video troppo lungo: tagliamo
                if self.progress_cb:
                    self.progress_cb("status", {"message": "Taglio del video..."})
                base_frames = video_frames[:target_total_frames]
            else:
                # Video troppo corto: loop o stretch
                if self.short_video_mode == "loop":
                    # Loop: ripete il video
                    if self.progress_cb:
                        self.progress_cb("status", {"message": "Loop del video..."})
                    num_loops = (target_total_frames // len(video_frames)) + 1
                    for loop_idx in range(num_loops):
                        for frame in video_frames:
                            if len(base_frames) >= target_total_frames:
                                break
                            base_frames.append(frame.copy())
                        if len(base_frames) >= target_total_frames:
                            break
                else:
                    # Stretch: rallenta il video distribuendo i frame
                    if self.progress_cb:
                        self.progress_cb("status", {"message": "Stretch del video (rallentamento)..."})
                    # Calcola quante volte ripetere ogni frame
                    stretch_factor = target_total_frames / len(video_frames)
                    
                    for i, frame in enumerate(video_frames):
                        # Calcola quante copie di questo frame servono
                        start_idx = int(i * stretch_factor)
                        end_idx = int((i + 1) * stretch_factor)
                        num_repeats = end_idx - start_idx
                        
                        for _ in range(num_repeats):
                            if len(base_frames) >= target_total_frames:
                                break
                            base_frames.append(frame.copy())
                        
                        if len(base_frames) >= target_total_frames:
                            break
                    
                    # Assicurati di avere esattamente il numero corretto di frame
                    while len(base_frames) < target_total_frames:
                        base_frames.append(video_frames[-1].copy())
            
            # Applica effetti audio-reattivi a ogni frame
            if self.progress_cb:
                self.progress_cb("status", {"message": "Applicazione effetti audio-reattivi..."})
            
            output_frames: List[np.ndarray] = []
            
            # Calcola mapping tra frame e analisi audio
            # bass, mid, treble hanno lunghezza basata su hop_length dell'analisi
            audio_frames_count = len(bass)
            
            for i in range(len(base_frames)):
                # Mappa frame video a frame audio
                audio_idx = int((i / len(base_frames)) * audio_frames_count)
                audio_idx = min(audio_idx, audio_frames_count - 1)
                
                current_time = i / video_fps
                
                # Rileva beat
                beat_intensity = 0.0
                for bt in beat_times:
                    if abs(current_time - bt) < (1 / video_fps):
                        beat_intensity = 1.0
                        break
                
                # Applica effetti
                frame = base_frames[i]
                
                # NUOVI EFFETTI ELETTRICI E DISTORSIVI
                # Zoom pulsante sui bassi
                frame = self._zoom_pulse(frame, bass[audio_idx] * 1.2)
                
                # Effetto bolla sui bassi
                frame = self._bubble_distortion(frame, bass[audio_idx])
                
                # Screen shake su snare/mid (vibrazioni intense)
                frame = self._screen_shake(frame, mid[audio_idx] * 1.3)
                
                # Effetti colore base
                frame = self._color_pulse(frame, bass[audio_idx], mid[audio_idx], treble[audio_idx], beat_intensity)
                
                total_intensity = (bass[audio_idx] + mid[audio_idx] + treble[audio_idx]) / 3
                color_index = int(current_time * 2) % len(self.dark_colors)
                
                # Strobe
                frame = self._strobe(frame, total_intensity, color_index)
                
                # Distorsione base
                frame = self._distort(frame, total_intensity)
                
                # EFFETTI CYBERPUNK
                # Aberrazione cromatica
                frame = self._chromatic_aberration(frame, treble[audio_idx] * 1.1)
                
                # RGB Split estremo su alti
                frame = self._rgb_split(frame, treble[audio_idx] * 1.2)
                
                # Glitch base
                glitch_intensity = bass[audio_idx] * 0.5 + treble[audio_idx] * 0.5
                frame = self._glitch(frame, glitch_intensity)
                
                # Scariche elettriche sui beat
                if beat_intensity > 0.5:
                    current_color = self.dark_colors[color_index]
                    frame = self._electric_arcs(frame, beat_intensity, current_color)
                
                # Scan lines CRT
                frame = self._scan_lines(frame, mid[audio_idx] * 0.8)
                
                # VHS distortion casuale
                if np.random.random() < treble[audio_idx] * 0.3:
                    frame = self._vhs_distortion(frame, treble[audio_idx])
                
                # Applica logo
                frame = self._overlay_logo(frame)
                
                output_frames.append(frame)
                
                if self.progress_cb and (i % 10 == 0 or i == len(base_frames) - 1):
                    self.progress_cb("frame", {"index": i + 1, "total": len(base_frames)})
            
            # Salva frame temporanei
            if self.progress_cb:
                self.progress_cb("status", {"message": "Salvataggio frame temporanei..."})
            
            for i, frame in enumerate(output_frames):
                path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_files.append(path)
                if self.progress_cb and (i % 50 == 0 or i == len(output_frames) - 1):
                    self.progress_cb("frame", {"index": i + 1, "total": len(output_frames)})
            
            # Mux con ffmpeg
            if self.progress_cb:
                self.progress_cb("status", {"message": "Encoding finale con ffmpeg..."})
            
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                ffmpeg_path = get_ffmpeg_exe()
            except Exception:
                ffmpeg_path = "ffmpeg"
            
            pattern = os.path.join(temp_dir, "frame_%06d.png")
            cmd = [
                ffmpeg_path,
                "-y",
                "-framerate", str(video_fps),
                "-i", pattern,
                "-i", self.audio_file,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-r", str(video_fps),
                "-movflags", "+faststart",
                self.output_file,
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")
            
            if self.progress_cb:
                self.progress_cb("done", {"output": self.output_file})
        
        finally:
            # Cleanup
            for f in frame_files:
                try:
                    os.remove(f)
                except Exception:
                    pass
            try:
                os.rmdir(temp_dir)
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
    effect_style: str = "standard",  # "standard" o "extreme"
    # Logo options
    logo: Optional[str] = None,
    logo_position: str = "top-right",
    logo_scale: float = 0.15,
    logo_opacity: float = 1.0,
    logo_margin: int = 12,
    config: Optional[dict] = None,  # Full preset configuration
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
        effect_style=effect_style,
        logo_file=logo,
        logo_position=logo_position,
        logo_scale=logo_scale,
        logo_opacity=logo_opacity,
        logo_margin=logo_margin,
        config=config,
    )
    fx.create_video()


def sync_video_with_audio(
    audio: str,
    video: str,
    output: str,
    progress_cb: ProgressCallback = None,
    *,
    colors: Optional[List[Tuple[float, float, float]]] = None,
    thresholds: Optional[Tuple[float, float, float]] = None,
    short_video_mode: str = "loop",
    logo: Optional[str] = None,
    logo_position: str = "top-right",
    logo_scale: float = 0.15,
    logo_opacity: float = 1.0,
    logo_margin: int = 12,
) -> None:
    """
    Functional façade over VideoAudioFX for simple callers.
    
    Args:
        short_video_mode: Come gestire video più corti dell'audio.
            - "loop": ripete il video dall'inizio (default)
            - "stretch": rallenta il video per matchare la durata
    """
    fx = VideoAudioFX(
        audio_file=audio,
        video_file=video,
        output_file=output,
        progress_cb=progress_cb,
        colors=colors,
        thresholds=thresholds,
        short_video_mode=short_video_mode,
        logo_file=logo,
        logo_position=logo_position,
        logo_scale=logo_scale,
        logo_opacity=logo_opacity,
        logo_margin=logo_margin,
    )
    fx.create_video()
