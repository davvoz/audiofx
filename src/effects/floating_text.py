"""
Floating Text Effect - Testo fluttuante animato a tempo di musica
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
from typing import Optional, List, Tuple
from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class FloatingText(BaseEffect):
    """
    Effetto di testo fluttuante con animazioni multiple:
    - Movimento fluido nello spazio
    - Scala dinamica basata sui bassi
    - Rotazione sincronizzata con i medi
    - Colore che cambia con gli alti
    - Glow pulsante
    - Ondulazione del testo
    """
    
    def __init__(self, 
                 text: str = "MUSIC",
                 font_size: int = 120,
                 font_path: Optional[str] = None,
                 color_scheme: str = "rainbow",
                 animation_style: str = "wave",
                 intensity: float = 1.0,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None):
        """
        Args:
            text: Testo da visualizzare
            font_size: Dimensione base del font
            font_path: Path al font personalizzato (opzionale)
            color_scheme: Schema colori ('rainbow', 'fire', 'ice', 'neon', 'gold')
            animation_style: Stile animazione ('wave', 'bounce', 'spin', 'pulse', 'glitch')
            intensity: Intensità effetto (0.0-2.0)
            start_time: Tempo in secondi quando il testo inizia ad apparire (None = dall'inizio)
            end_time: Tempo in secondi quando il testo scompare (None = fino alla fine)
        """
        super().__init__(intensity=intensity)
        self.text = text.upper()
        self.base_font_size = font_size
        self.font_path = font_path
        self.color_scheme = color_scheme
        self.animation_style = animation_style
        self.start_time = start_time
        self.end_time = end_time
        
        # Variabili di stato per animazioni fluide
        self.position_x = 0.5  # Inizia al centro
        self.position_y = 0.5  # Inizia al centro
        self.rotation = 0
        self.scale = 1.0
        self.wave_phase = 0
        self.glow_intensity = 0
        
        # Velocità di movimento target per smoothing
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        
        # Carica il font
        self._load_font()
        
    def _load_font(self):
        """Carica il font da usare"""
        try:
            if self.font_path and os.path.exists(self.font_path):
                self.font = ImageFont.truetype(self.font_path, self.base_font_size)
            else:
                # Prova font di sistema comuni
                common_fonts = [
                    "C:/Windows/Fonts/impact.ttf",
                    "C:/Windows/Fonts/arial.ttf",
                    "C:/Windows/Fonts/arialbd.ttf",
                ]
                
                font_loaded = False
                for font_path in common_fonts:
                    if os.path.exists(font_path):
                        self.font = ImageFont.truetype(font_path, self.base_font_size)
                        font_loaded = True
                        break
                
                if not font_loaded:
                    self.font = ImageFont.load_default()
        except Exception as e:
            print(f"Errore caricamento font: {e}, uso font default")
            self.font = ImageFont.load_default()
    
    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """
        Spezza il testo in linee che stanno nella larghezza massima.
        
        Args:
            text: Testo da spezzare
            font: Font da utilizzare
            max_width: Larghezza massima in pixel
            
        Returns:
            Lista di linee
        """
        words = text.split()
        if not words:
            return [text]
        
        lines = []
        current_line = words[0]
        
        for word in words[1:]:
            # Prova ad aggiungere la parola alla linea corrente
            test_line = current_line + " " + word
            try:
                bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), test_line, font=font)
                test_width = bbox[2] - bbox[0]
            except:
                # Fallback per font che non supportano textbbox
                test_width = len(test_line) * (font.size if hasattr(font, 'size') else 12) * 0.6
            
            if test_width <= max_width:
                current_line = test_line
            else:
                # La linea è troppo lunga, salva quella corrente e inizia una nuova
                lines.append(current_line)
                current_line = word
        
        # Aggiungi l'ultima linea
        lines.append(current_line)
        return lines
    
    def _calculate_optimal_font_size(self, text: str, width: int, height: int, 
                                     initial_size: int) -> Tuple[int, List[str]]:
        """
        Calcola la dimensione font ottimale e le linee per far stare il testo nel frame.
        
        Args:
            text: Testo da visualizzare
            width: Larghezza disponibile
            height: Altezza disponibile
            initial_size: Dimensione iniziale del font
            
        Returns:
            (font_size, lines) - Dimensione finale e linee di testo
        """
        # Area di sicurezza: usa max 80% della larghezza e 60% dell'altezza
        max_width = int(width * 0.8)
        max_height = int(height * 0.6)
        
        font_size = initial_size
        min_font_size = 30  # Dimensione minima leggibile
        
        while font_size >= min_font_size:
            # Crea font con la dimensione corrente
            try:
                if hasattr(self.font, 'path'):
                    test_font = ImageFont.truetype(self.font.path, font_size)
                else:
                    test_font = ImageFont.load_default()
            except:
                test_font = ImageFont.load_default()
            
            # Prova a wrappare il testo
            lines = self._wrap_text(text, test_font, max_width)
            
            # Calcola altezza totale
            try:
                dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
                line_height = 0
                total_width = 0
                
                for line in lines:
                    bbox = dummy_draw.textbbox((0, 0), line, font=test_font)
                    line_height = max(line_height, bbox[3] - bbox[1])
                    total_width = max(total_width, bbox[2] - bbox[0])
                
                total_height = line_height * len(lines) * 1.2  # 1.2 per spacing tra linee
            except:
                # Fallback
                line_height = font_size
                total_height = line_height * len(lines) * 1.2
                total_width = max_width
            
            # Verifica se sta tutto
            if total_height <= max_height and total_width <= max_width:
                return font_size, lines
            
            # Riduci font size del 10%
            font_size = int(font_size * 0.9)
        
        # Se arriviamo qui, usa la dimensione minima
        try:
            if hasattr(self.font, 'path'):
                test_font = ImageFont.truetype(self.font.path, min_font_size)
            else:
                test_font = ImageFont.load_default()
        except:
            test_font = ImageFont.load_default()
        
        lines = self._wrap_text(text, test_font, max_width)
        return min_font_size, lines
    
    def _get_color(self, progress: float, bass: float, mid: float, high: float) -> tuple:
        """
        Calcola il colore in base allo schema e ai parametri audio
        
        Args:
            progress: Progresso temporale (0-1)
            bass: Intensità bassi (0-1)
            mid: Intensità medi (0-1)
            high: Intensità alti (0-1)
        """
        if self.color_scheme == "rainbow":
            hue = (progress + high * 0.3) % 1.0
            r = int(255 * (0.5 + 0.5 * np.sin(hue * np.pi * 2)))
            g = int(255 * (0.5 + 0.5 * np.sin((hue + 0.33) * np.pi * 2)))
            b = int(255 * (0.5 + 0.5 * np.sin((hue + 0.66) * np.pi * 2)))
            
        elif self.color_scheme == "fire":
            r = int(255 * (0.7 + 0.3 * bass))
            g = int(180 * (0.5 + 0.5 * mid))
            b = int(50 * high)
            
        elif self.color_scheme == "ice":
            r = int(100 * (0.5 + 0.5 * high))
            g = int(200 * (0.7 + 0.3 * mid))
            b = int(255 * (0.8 + 0.2 * bass))
            
        elif self.color_scheme == "neon":
            if progress % 0.5 < 0.25:
                r, g, b = 255, int(50 + 205 * bass), 255
            else:
                r, g, b = int(50 + 205 * mid), 255, int(50 + 205 * high)
                
        elif self.color_scheme == "gold":
            r = int(255 * (0.9 + 0.1 * bass))
            g = int(215 * (0.8 + 0.2 * mid))
            b = int(50 * (0.3 + 0.7 * high))
            
        else:  # default bianco
            intensity = int(200 + 55 * (bass + mid + high) / 3)
            r = g = b = intensity
        
        return (r, g, b)
    
    def _apply_wave_distortion(self, img: Image.Image, intensity: float) -> Image.Image:
        """Applica distorsione a onda al testo"""
        if intensity < 0.01:
            return img
            
        width, height = img.size
        pixels = np.array(img)
        
        # Crea mesh di coordinate
        x = np.arange(width)
        y = np.arange(height)
        
        # Applica onda sinusoidale
        wave_offset = intensity * 20 * np.sin(self.wave_phase + x / 30.0)
        
        # Shifta i pixel
        output = np.zeros_like(pixels)
        for i in range(width):
            offset = int(wave_offset[i])
            if 0 <= offset < height:
                output[:, i] = np.roll(pixels[:, i], offset, axis=0)
        
        return Image.fromarray(output)
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """Applica l'effetto di testo fluttuante"""
        # Controlla se il testo deve essere visibile in questo momento
        current_time = context.time
        
        # Se start_time è specificato e non abbiamo ancora raggiunto quel tempo, non mostrare il testo
        if self.start_time is not None and current_time < self.start_time:
            return frame
        
        # Se end_time è specificato e lo abbiamo superato, non mostrare il testo
        if self.end_time is not None and current_time > self.end_time:
            return frame
        
        # Estrai features dal context
        bass = context.bass * self.intensity
        mid = context.mid * self.intensity
        high = context.treble * self.intensity
        beat = context.beat_intensity
        progress = (context.time % 10.0) / 10.0  # Ciclo ogni 10 secondi
        
        height, width = frame.shape[:2]
        
        # === AGGIORNA PARAMETRI ANIMAZIONE ===
        
        # Movimento fluido MOLTO più lento e armonico con inerzia
        # Calcola target position (movimento molto ridotto)
        time_factor = progress * np.pi * 0.3  # Movimento lentissimo
        target_x = 0.5 + np.sin(time_factor + mid * 0.2) * 0.08  # Max 8% di spostamento dal centro
        target_y = 0.5 + np.cos(time_factor * 0.7 + bass * 0.15) * 0.06  # Max 6% verticale
        
        # Applica smoothing con inerzia (movimento fluido senza scatti)
        smoothing = 0.95  # Più alto = più smooth (0.0-1.0)
        self.velocity_x = self.velocity_x * smoothing + (target_x - self.position_x) * (1 - smoothing)
        self.velocity_y = self.velocity_y * smoothing + (target_y - self.position_y) * (1 - smoothing)
        
        self.position_x += self.velocity_x
        self.position_y += self.velocity_y
        
        # Limita il movimento per sicurezza
        self.position_x = np.clip(self.position_x, 0.35, 0.65)
        self.position_y = np.clip(self.position_y, 0.35, 0.65)
        
        # Scala basata sui bassi - movimento più morbido
        target_scale = 1.0 + bass * 0.3 + beat * 0.2  # Ridotto da 0.5 e 0.3
        self.scale = self.scale * 0.85 + target_scale * 0.15  # Transizione più morbida (era 0.8/0.2)
        
        # Rotazione basata sui medi - più lenta
        if self.animation_style in ["spin", "glitch"]:
            rotation_speed = mid * 2 + beat * 3  # Ridotto da mid * 5 + beat * 10
            self.rotation += rotation_speed
            self.rotation %= 360
        
        # Fase onda - rallentata
        self.wave_phase += 0.05 + high * 0.1  # Ridotto da 0.1 + high * 0.2
        
        # Intensità glow
        target_glow = bass * 0.7 + beat * 0.3
        self.glow_intensity = self.glow_intensity * 0.7 + target_glow * 0.3
        
        # === CALCOLA DIMENSIONE FONT DINAMICA E LINEE DI TESTO ===
        dynamic_font_size = int(self.base_font_size * self.scale)
        dynamic_font_size = max(20, min(300, dynamic_font_size))
        
        # Calcola dimensione ottimale e linee per far stare tutto nel frame
        optimal_size, text_lines = self._calculate_optimal_font_size(
            self.text, width, height, dynamic_font_size
        )
        
        # Ricrea font con dimensione ottimale
        try:
            if hasattr(self.font, 'path'):
                current_font = ImageFont.truetype(self.font.path, optimal_size)
            else:
                current_font = ImageFont.load_default()
        except:
            current_font = self.font
        
        # === CREA IMMAGINE TESTO ===
        
        # Crea immagine temporanea grande per il testo
        text_img = Image.new('RGBA', (width * 2, height * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        
        # Calcola dimensioni totali del testo multi-linea
        try:
            line_heights = []
            line_widths = []
            for line in text_lines:
                bbox = draw.textbbox((0, 0), line, font=current_font)
                line_widths.append(bbox[2] - bbox[0])
                line_heights.append(bbox[3] - bbox[1])
            
            max_line_width = max(line_widths) if line_widths else 0
            line_height = max(line_heights) if line_heights else optimal_size
            total_height = int(line_height * len(text_lines) * 1.2)  # 1.2 per spacing
        except:
            max_line_width = len(max(text_lines, key=len)) * optimal_size * 0.6 if text_lines else 0
            line_height = optimal_size
            total_height = int(line_height * len(text_lines) * 1.2)
        
        # Posizione iniziale centrata
        text_x = width - max_line_width // 2
        text_y = height - total_height // 2
        
        # Colore dinamico
        color = self._get_color(progress, bass, mid, high)
        
        # === DISEGNA TESTO MULTI-LINEA CON STILI DIVERSI ===
        
        current_y = text_y
        
        if self.animation_style == "wave":
            # Disegna ogni linea, poi ogni lettera con offset verticale
            for line_idx, line in enumerate(text_lines):
                x_offset = 0
                # Centra ogni linea orizzontalmente
                try:
                    line_bbox = draw.textbbox((0, 0), line, font=current_font)
                    line_w = line_bbox[2] - line_bbox[0]
                except:
                    line_w = len(line) * optimal_size * 0.6
                
                line_x = width - line_w // 2
                
                for i, char in enumerate(line):
                    wave_offset = int(15 * np.sin(self.wave_phase + i * 0.5 + line_idx * 0.3))
                    char_y = current_y + wave_offset
                    draw.text((line_x + x_offset, char_y), char, font=current_font, fill=color)
                    try:
                        char_bbox = draw.textbbox((0, 0), char, font=current_font)
                        x_offset += char_bbox[2] - char_bbox[0]
                    except:
                        x_offset += optimal_size * 0.6
                
                current_y += int(line_height * 1.2)
                    
        elif self.animation_style == "bounce":
            # Lettere che rimbalzano per ogni linea
            for line_idx, line in enumerate(text_lines):
                x_offset = 0
                try:
                    line_bbox = draw.textbbox((0, 0), line, font=current_font)
                    line_w = line_bbox[2] - line_bbox[0]
                except:
                    line_w = len(line) * optimal_size * 0.6
                
                line_x = width - line_w // 2
                
                for i, char in enumerate(line):
                    bounce = int(20 * abs(np.sin(self.wave_phase + i * 0.8 + line_idx * 0.4)))
                    draw.text((line_x + x_offset, current_y - bounce), char, font=current_font, fill=color)
                    try:
                        char_bbox = draw.textbbox((0, 0), char, font=current_font)
                        x_offset += char_bbox[2] - char_bbox[0]
                    except:
                        x_offset += optimal_size * 0.6
                
                current_y += int(line_height * 1.2)
                    
        elif self.animation_style == "glitch":
            # Effetto glitch con offset RGB per ogni linea
            for line in text_lines:
                try:
                    line_bbox = draw.textbbox((0, 0), line, font=current_font)
                    line_w = line_bbox[2] - line_bbox[0]
                except:
                    line_w = len(line) * optimal_size * 0.6
                
                line_x = width - line_w // 2
                
                for offset, rgb_color in [((-2, 0), (255, 0, 0)), ((2, 0), (0, 255, 255)), ((0, 0), color)]:
                    alpha = 200 if offset != (0, 0) else 255
                    glitch_color = rgb_color[:3] + (alpha,)
                    draw.text((line_x + offset[0], current_y + offset[1]), 
                             line, font=current_font, fill=glitch_color)
                
                current_y += int(line_height * 1.2)
        else:
            # Standard e pulse - disegna ogni linea centrata
            for line in text_lines:
                try:
                    line_bbox = draw.textbbox((0, 0), line, font=current_font)
                    line_w = line_bbox[2] - line_bbox[0]
                except:
                    line_w = len(line) * optimal_size * 0.6
                
                line_x = width - line_w // 2
                draw.text((line_x, current_y), line, font=current_font, fill=color)
                current_y += int(line_height * 1.2)
        
        # === APPLICA GLOW ===
        if self.glow_intensity > 0.1:
            # Crea layer di glow
            glow_img = text_img.copy()
            blur_amount = int(10 + self.glow_intensity * 20)
            glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=blur_amount))
            
            # Componi glow + testo
            text_img = Image.alpha_composite(glow_img, text_img)
        
        # === APPLICA ROTAZIONE ===
        if abs(self.rotation) > 0.1:
            text_img = text_img.rotate(self.rotation, expand=False, resample=Image.BICUBIC)
        
        # === APPLICA DISTORSIONE ONDA ===
        if self.animation_style == "wave" and mid > 0.3:
            text_img = self._apply_wave_distortion(text_img, mid)
        
        # === POSIZIONA SUL FRAME ===
        
        # Calcola posizione finale
        final_x = int(self.position_x * width - width)
        final_y = int(self.position_y * height - height)
        
        # Converti frame a PIL
        frame_pil = Image.fromarray(frame).convert('RGBA')
        
        # Componi
        frame_pil.paste(text_img, (final_x, final_y), text_img)
        
        # Converti back a numpy
        result = np.array(frame_pil.convert('RGB'))
        
        return result
    
    def reset(self):
        """Reset dello stato dell'effetto"""
        self.position_x = 0.5
        self.position_y = 0.5
        self.rotation = 0
        self.scale = 1.0
        self.wave_phase = 0
        self.glow_intensity = 0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
