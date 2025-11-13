"""
Ghost particles effect - phantasmagoric particles that explode and mutate with music.
Refactored following OOP, SOLID principles and SonarQube best practices.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from enum import Enum
import numpy as np
import cv2

from .base_effect import BaseEffect
from ..models.data_models import FrameContext


class ParticleType(Enum):
    """Enumeration of particle types."""
    NORMAL = "normal"
    STREAK = "streak"
    BLOB = "blob"
    SPARKLE = "sparkle"
    SMOKE = "smoke"


class ParticlePhysics:
    """Handles particle physics calculations (Single Responsibility)."""
    
    @staticmethod
    def apply_wave_motion(vx: float, vy: float, phase: float, intensity: float = 0.2) -> Tuple[float, float]:
        """Apply wave-based motion to velocity."""
        wave = np.sin(phase * 0.5) * intensity
        return vx + wave, vy + wave * 0.5
    
    @staticmethod
    def apply_drag(vx: float, vy: float, drag_coefficient: float) -> Tuple[float, float]:
        """Apply drag/friction to velocity."""
        return vx * drag_coefficient, vy * drag_coefficient
    
    @staticmethod
    def calculate_fade_alpha(life_ratio: float, fade_in_threshold: float = 0.8, 
                            fade_out_threshold: float = 0.2) -> float:
        """Calculate alpha based on particle life ratio."""
        if life_ratio > fade_in_threshold:
            return (1.0 - life_ratio) / (1.0 - fade_in_threshold)
        if life_ratio < fade_out_threshold:
            return life_ratio / fade_out_threshold
        return 1.0


class ParticleMovementStrategy(ABC):
    """Abstract base class for particle movement strategies (Strategy Pattern)."""
    
    @abstractmethod
    def update_position(self, particle: 'Particle', beat_intensity: float, dt: float) -> None:
        """Update particle position based on movement strategy."""


class NormalMovement(ParticleMovementStrategy):
    """Normal flowing particle movement - stays localized."""
    
    def update_position(self, particle: 'Particle', beat_intensity: float, dt: float) -> None:
        # Gentle orbital movement around spawn point
        vx, vy = ParticlePhysics.apply_wave_motion(
            particle.vx, particle.vy, particle.mutation_phase, 0.15
        )
        particle.x += vx * dt
        particle.y += vy * dt
        # Strong drag to keep particles localized
        particle.vx, particle.vy = ParticlePhysics.apply_drag(vx, vy, 0.93)


class StreakMovement(ParticleMovementStrategy):
    """Fast linear streak movement."""
    
    def update_position(self, particle: 'Particle', beat_intensity: float, dt: float) -> None:
        particle.x += particle.vx * dt * 1.5
        particle.y += particle.vy * dt * 1.5
        particle.vx, particle.vy = ParticlePhysics.apply_drag(particle.vx, particle.vy, 0.99)


class BlobMovement(ParticleMovementStrategy):
    """Slow floating blob movement - minimal drift."""
    
    def update_position(self, particle: 'Particle', beat_intensity: float, dt: float) -> None:
        # Circular floating motion
        wave_x = np.sin(particle.mutation_phase * 0.3) * 0.4
        wave_y = np.cos(particle.mutation_phase * 0.3) * 0.4
        particle.x += particle.vx * dt * 0.3 + wave_x
        particle.y += particle.vy * dt * 0.3 + wave_y
        # Very strong drag - blobs stay in place
        particle.vx, particle.vy = ParticlePhysics.apply_drag(particle.vx, particle.vy, 0.90)


class SparkleMovement(ParticleMovementStrategy):
    """Erratic twinkling movement - stays near origin."""
    
    def update_position(self, particle: 'Particle', beat_intensity: float, dt: float) -> None:
        # Small random jitter
        jitter_x = (np.random.random() - 0.5) * beat_intensity * 0.5
        jitter_y = (np.random.random() - 0.5) * beat_intensity * 0.5
        particle.x += (particle.vx * 0.5 + jitter_x) * dt
        particle.y += (particle.vy * 0.5 + jitter_y) * dt
        # Strong drag to prevent drift
        particle.vx, particle.vy = ParticlePhysics.apply_drag(particle.vx, particle.vy, 0.92)


class SmokeMovement(ParticleMovementStrategy):
    """Gentle floating smoke - very slow rise."""
    
    def update_position(self, particle: 'Particle', beat_intensity: float, dt: float) -> None:
        # Very gentle upward drift
        particle.y -= 0.2 * dt
        # Minimal horizontal expansion
        life_ratio = particle.life / particle.max_life
        expansion = (1.0 - life_ratio) * 0.05
        particle.x += particle.vx * dt * 0.3 + expansion
        # Ignore initial vy to prevent downward drift
        particle.vx, particle.vy = ParticlePhysics.apply_drag(particle.vx, 0, 0.94)


class ParticleMovementFactory:
    """Factory for creating movement strategies (Factory Pattern)."""
    
    _STRATEGIES = {
        ParticleType.NORMAL: NormalMovement(),
        ParticleType.STREAK: StreakMovement(),
        ParticleType.BLOB: BlobMovement(),
        ParticleType.SPARKLE: SparkleMovement(),
        ParticleType.SMOKE: SmokeMovement(),
    }
    
    @classmethod
    def get_strategy(cls, particle_type: ParticleType) -> ParticleMovementStrategy:
        """Get movement strategy for particle type."""
        return cls._STRATEGIES.get(particle_type, cls._STRATEGIES[ParticleType.NORMAL])


class ParticleRenderer(ABC):
    """Abstract base class for particle rendering strategies (Strategy Pattern)."""
    
    @abstractmethod
    def render(self, overlay: np.ndarray, particle: 'Particle', 
              x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        """Render particle on overlay."""


class NormalRenderer(ParticleRenderer):
    """Renderer for normal particles with glow."""
    
    def render(self, overlay: np.ndarray, particle: 'Particle', 
              x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        alpha = particle.alpha
        outer_color = tuple(int(c * 0.3 * alpha) for c in color)
        cv2.circle(overlay, (x, y), size + 6, outer_color, -1, lineType=cv2.LINE_AA)
        mid_color = tuple(int(c * 0.6 * alpha) for c in color)
        cv2.circle(overlay, (x, y), size + 3, mid_color, -1, lineType=cv2.LINE_AA)
        main_color = tuple(int(c * alpha) for c in color)
        cv2.circle(overlay, (x, y), size, main_color, -1, lineType=cv2.LINE_AA)
        core_color = tuple(min(255, int(c * 1.3 * alpha)) for c in color)
        cv2.circle(overlay, (x, y), max(1, size // 2), core_color, -1, lineType=cv2.LINE_AA)


class StreakRenderer(ParticleRenderer):
    """Renderer for streak particles with motion trail."""
    
    def render(self, overlay: np.ndarray, particle: 'Particle', 
              x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        h, w = overlay.shape[:2]
        for t in range(particle.trail_length):
            trail_x = int(x - particle.vx * t * 0.5)
            trail_y = int(y - particle.vy * t * 0.5)
            if 0 <= trail_x < w and 0 <= trail_y < h:
                trail_alpha = particle.alpha * (1.0 - t / particle.trail_length)
                trail_size = max(1, int(size * (1.0 - t / particle.trail_length * 0.5)))
                trail_color = tuple(int(c * trail_alpha) for c in color)
                cv2.circle(overlay, (trail_x, trail_y), trail_size, trail_color, -1, lineType=cv2.LINE_AA)


class BlobRenderer(ParticleRenderer):
    """Renderer for blob particles with multiple layers."""
    
    def render(self, overlay: np.ndarray, particle: 'Particle', 
              x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        alpha = particle.alpha
        for layer in range(3):
            layer_size = size + (3 - layer) * 3
            layer_alpha = alpha * (0.3 + 0.2 * layer)
            layer_color = tuple(int(c * layer_alpha) for c in color)
            cv2.circle(overlay, (x, y), layer_size, layer_color, -1, lineType=cv2.LINE_AA)
        core_color = tuple(min(255, int(c * 1.5 * alpha)) for c in color)
        cv2.circle(overlay, (x, y), max(1, size // 2), core_color, -1, lineType=cv2.LINE_AA)


class SparkleRenderer(ParticleRenderer):
    """Renderer for sparkle particles with star rays."""
    
    def render(self, overlay: np.ndarray, particle: 'Particle', 
              x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        alpha = particle.alpha
        bright_color = tuple(min(255, int(c * 1.8 * alpha)) for c in color)
        cv2.circle(overlay, (x, y), size, bright_color, -1, lineType=cv2.LINE_AA)
        ray_length = size * 2
        glow_color = tuple(int(c * 0.8 * alpha) for c in color)
        cv2.line(overlay, (x - ray_length, y), (x + ray_length, y), glow_color, 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (x, y - ray_length), (x, y + ray_length), glow_color, 1, lineType=cv2.LINE_AA)
        diag = int(ray_length * 0.7)
        cv2.line(overlay, (x - diag, y - diag), (x + diag, y + diag), glow_color, 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (x - diag, y + diag), (x + diag, y - diag), glow_color, 1, lineType=cv2.LINE_AA)


class SmokeRenderer(ParticleRenderer):
    """Renderer for smoke particles with diffuse effect."""
    
    def render(self, overlay: np.ndarray, particle: 'Particle', 
              x: int, y: int, size: int, color: Tuple[int, int, int]) -> None:
        alpha = particle.alpha
        for layer in range(4):
            smoke_size = size + layer * 2
            smoke_alpha = alpha * (0.15 * (4 - layer))
            smoke_color = tuple(int(c * smoke_alpha) for c in color)
            cv2.circle(overlay, (x, y), smoke_size, smoke_color, -1, lineType=cv2.LINE_AA)


class ParticleRendererFactory:
    """Factory for creating renderer strategies (Factory Pattern)."""
    
    _RENDERERS = {
        ParticleType.NORMAL: NormalRenderer(),
        ParticleType.STREAK: StreakRenderer(),
        ParticleType.BLOB: BlobRenderer(),
        ParticleType.SPARKLE: SparkleRenderer(),
        ParticleType.SMOKE: SmokeRenderer(),
    }
    
    @classmethod
    def get_renderer(cls, particle_type: ParticleType) -> ParticleRenderer:
        """Get renderer for particle type."""
        return cls._RENDERERS.get(particle_type, cls._RENDERERS[ParticleType.NORMAL])


class Particle:
    """Represents a single ghost particle with type-specific behavior."""
    
    FADE_IN_THRESHOLD = 0.8
    FADE_OUT_THRESHOLD = 0.2
    TWINKLE_MULTIPLIER = 3.0
    SMOKE_BASE_ALPHA = 0.4
    SMOKE_EXPANSION_FACTOR = 1.5
    
    def __init__(self, x: float, y: float, vx: float, vy: float, 
                 color: Tuple[int, int, int], size: float, life: float, 
                 particle_type: ParticleType = ParticleType.NORMAL):
        """Initialize particle with position, velocity, appearance and type."""
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        self.alpha = 1.0
        self.mutation_phase = 0.0
        self.particle_type = particle_type
        self.rotation = np.random.random() * 360
        self.rotation_speed = np.random.uniform(-5, 5)
        self.pulse_speed = np.random.uniform(0.5, 2.0)
        self.trail_length = np.random.randint(3, 8) if particle_type == ParticleType.STREAK else 0
        self._movement_strategy = ParticleMovementFactory.get_strategy(particle_type)
        self._renderer = ParticleRendererFactory.get_renderer(particle_type)
    
    def update(self, beat_intensity: float, dt: float = 1.0) -> None:
        """Update particle state."""
        self.rotation += self.rotation_speed * dt
        self._movement_strategy.update_position(self, beat_intensity, dt)
        self.life -= dt
        self._update_alpha()
        self.mutation_phase += beat_intensity * 0.4 * self.pulse_speed
    
    def _update_alpha(self) -> None:
        """Update particle transparency based on type and life."""
        life_ratio = self.life / self.max_life
        
        if self.particle_type == ParticleType.SPARKLE:
            twinkle = 0.5 + 0.5 * np.sin(self.mutation_phase * self.pulse_speed * self.TWINKLE_MULTIPLIER)
            self.alpha = twinkle * (0.3 + life_ratio * 0.7)
        elif self.particle_type == ParticleType.SMOKE:
            self.alpha = life_ratio * self.SMOKE_BASE_ALPHA
        else:
            self.alpha = ParticlePhysics.calculate_fade_alpha(
                life_ratio, self.FADE_IN_THRESHOLD, self.FADE_OUT_THRESHOLD
            )
    
    def is_alive(self) -> bool:
        """Check if particle is still alive."""
        return self.life > 0
    
    def get_mutated_size(self) -> float:
        """Get size with type-specific mutation effect."""
        if self.particle_type == ParticleType.BLOB:
            pulse = 1.0 + 0.6 * np.sin(self.mutation_phase * self.pulse_speed)
            return self.size * pulse
        if self.particle_type == ParticleType.SPARKLE:
            pulse = 1.0 + 0.8 * abs(np.sin(self.mutation_phase * self.pulse_speed * 2))
            return self.size * pulse
        if self.particle_type == ParticleType.SMOKE:
            expansion = 1.0 + (1.0 - self.life / self.max_life) * self.SMOKE_EXPANSION_FACTOR
            return self.size * expansion
        if self.particle_type == ParticleType.STREAK:
            return self.size * 1.2
        pulse = 1.0 + 0.3 * np.sin(self.mutation_phase * self.pulse_speed)
        return self.size * pulse
    
    def get_mutated_color(self) -> Tuple[int, int, int]:
        """Get color with mutation effect."""
        wave = np.sin(self.mutation_phase * 2.0)
        shift = int(wave * 30)
        b, g, r = self.color
        return (
            max(0, min(255, b + shift)),
            max(0, min(255, g - shift)),
            max(0, min(255, r + shift))
        )
    
    def render(self, overlay: np.ndarray, x: int, y: int) -> None:
        """Render particle using its specific renderer."""
        size = max(1, int(self.get_mutated_size()))
        color = self.get_mutated_color()
        self._renderer.render(overlay, self, x, y, size, color)


class ColorAnalyzer:
    """Analyzes pixel colors for frequency mapping (Single Responsibility)."""
    
    MIN_BRIGHTNESS = 50
    MIN_SATURATION = 0.15
    MIN_SATURATION_ALT = 120
    WARMTH_THRESHOLD = 0.15
    
    @staticmethod
    def is_valid_pixel(r: int, g: int, b: int) -> bool:
        """Check if pixel is valid for particle generation."""
        brightness = (r + g + b) / 3.0
        if brightness < ColorAnalyzer.MIN_BRIGHTNESS:
            return False
        
        max_channel = max(r, g, b)
        min_channel = min(r, g, b)
        saturation = (max_channel - min_channel) / max(1, max_channel)
        
        return saturation >= ColorAnalyzer.MIN_SATURATION or brightness >= ColorAnalyzer.MIN_SATURATION_ALT
    
    @staticmethod
    def calculate_frequency_weight(r: int, g: int, b: int, context: FrameContext) -> float:
        """Calculate frequency weight based on color temperature."""
        brightness = (r + g + b) / 3.0
        warmth = (r - b) / 255.0
        
        if warmth > ColorAnalyzer.WARMTH_THRESHOLD:
            return context.bass * (brightness / 255.0) * 1.2
        if warmth < -ColorAnalyzer.WARMTH_THRESHOLD:
            return context.treble * (brightness / 255.0) * 1.2
        return context.mid * (brightness / 255.0) * 1.2


class ParticleFactory:
    """Factory for creating particles with different configurations (Factory Pattern)."""
    
    # Reduced speed ranges to keep particles more localized
    TYPE_DISTRIBUTION = [
        (0.3, ParticleType.NORMAL, (0.8, 2.0), (0.8, 1.2)),
        (0.5, ParticleType.STREAK, (2.5, 4.0), (0.4, 0.8)),
        (0.65, ParticleType.BLOB, (0.2, 0.8), (1.0, 1.5)),
        (0.85, ParticleType.SPARKLE, (0.5, 1.5), (0.6, 1.0)),
        (1.0, ParticleType.SMOKE, (0.2, 0.8), (1.2, 1.8)),
    ]
    
    SIZE_RANGES = {
        ParticleType.NORMAL: (3, 7),
        ParticleType.STREAK: (2, 4),
        ParticleType.BLOB: (5, 12),
        ParticleType.SPARKLE: (2, 5),
        ParticleType.SMOKE: (4, 8),
    }
    
    @staticmethod
    def create_particle(x: float, y: float, color: Tuple[int, int, int], 
                       intensity: float, base_lifetime: float) -> Particle:
        """Create a particle with random type and properties."""
        angle = np.random.random() * 2 * np.pi
        rand = np.random.random()
        
        for threshold, ptype, speed_range, life_mult_range in ParticleFactory.TYPE_DISTRIBUTION:
            if rand < threshold:
                speed = np.random.uniform(*speed_range) * intensity
                vx = np.cos(angle) * speed
                vy = np.sin(angle) * speed
                
                size = ParticleFactory._get_size_for_type(ptype, intensity)
                life = base_lifetime * np.random.uniform(*life_mult_range)
                ghost_color = ParticleFactory._add_color_variation(color)
                
                return Particle(x, y, vx, vy, ghost_color, size * intensity, life, ptype)
        
        return ParticleFactory._create_normal_particle(x, y, color, intensity, base_lifetime)
    
    @staticmethod
    def _get_size_for_type(ptype: ParticleType, intensity: float) -> float:
        """Get appropriate size for particle type."""
        min_size, max_size = ParticleFactory.SIZE_RANGES.get(ptype, (3, 7))
        return np.random.uniform(min_size, max_size)
    
    @staticmethod
    def _add_color_variation(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Add variation to color."""
        b, g, r = color
        variation = 0.3
        return (
            max(0, min(255, int(b * np.random.uniform(1 - variation, 1 + variation)))),
            max(0, min(255, int(g * np.random.uniform(1 - variation, 1 + variation)))),
            max(0, min(255, int(r * np.random.uniform(1 - variation, 1 + variation))))
        )
    
    @staticmethod
    def _create_normal_particle(x: float, y: float, color: Tuple[int, int, int],
                               intensity: float, base_lifetime: float) -> Particle:
        """Create a default normal particle."""
        angle = np.random.random() * 2 * np.pi
        speed = np.random.uniform(0.8, 2.0) * intensity
        return Particle(
            x, y,
            np.cos(angle) * speed,
            np.sin(angle) * speed,
            color, 5.0 * intensity, base_lifetime,
            ParticleType.NORMAL
        )


class GhostParticlesEffect(BaseEffect):
    """Ghost particles effect - generates phantasmagoric particles from pixels based on frequencies."""
    
    DEFAULT_SAMPLE_DENSITY = 18
    DEFAULT_EXPLOSION_THRESHOLD = 0.5
    DEFAULT_PARTICLE_LIFETIME = 70.0
    DEFAULT_MAX_PARTICLES = 600
    COOLDOWN_FRAMES = 5
    SAMPLING_PROBABILITY = 0.6
    THRESHOLD_MULTIPLIER = 0.25
    BEAT_BOOST_FACTOR = 0.5
    PARTICLE_COUNT_MULTIPLIER = 0.8
    MIN_PARTICLES_PER_EXPLOSION = 8
    BLEND_ALPHA = 0.85
    
    def __init__(self, 
                 sample_density: int = DEFAULT_SAMPLE_DENSITY,
                 explosion_threshold: float = DEFAULT_EXPLOSION_THRESHOLD,
                 particle_lifetime: float = DEFAULT_PARTICLE_LIFETIME,
                 max_particles: int = DEFAULT_MAX_PARTICLES,
                 **kwargs):
        """Initialize ghost particles effect."""
        super().__init__(**kwargs)
        self.sample_density = sample_density
        self.explosion_threshold = explosion_threshold
        self.particle_lifetime = particle_lifetime
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        self.last_explosion_frame = -100
    
    def _sample_pixels_by_frequency(self, frame: np.ndarray, context: FrameContext) -> List[Tuple[int, int, Tuple[int, int, int]]]:
        """Sample pixels uniformly across entire frame with vertical distribution boost."""
        h, w = frame.shape[:2]
        samples = []
        step = max(15, self.sample_density)
        
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                pixel_bgr = frame[y, x]
                b, g, r = int(pixel_bgr[0]), int(pixel_bgr[1]), int(pixel_bgr[2])
                
                if not ColorAnalyzer.is_valid_pixel(r, g, b):
                    continue
                
                frequency_weight = ColorAnalyzer.calculate_frequency_weight(r, g, b, context)
                beat_boost = 1.0 + context.beat_intensity * self.BEAT_BOOST_FACTOR
                frequency_weight *= beat_boost
                
                # Add vertical position boost - favor top half
                vertical_position = y / h  # 0.0 at top, 1.0 at bottom
                if vertical_position < 0.5:
                    # Boost top half significantly
                    vertical_boost = 1.5
                else:
                    # Normal weight for bottom half
                    vertical_boost = 1.0
                
                frequency_weight *= vertical_boost
                
                threshold = self.THRESHOLD_MULTIPLIER * self.intensity
                if frequency_weight > threshold and np.random.random() < self.SAMPLING_PROBABILITY:
                    samples.append((x, y, (b, g, r)))
        
        return samples
    
    def _create_explosion(self, x: float, y: float, color: Tuple[int, int, int], 
                         intensity: float, num_particles: int) -> None:
        """Create an explosion of diverse particles from a point."""
        actual_num = max(self.MIN_PARTICLES_PER_EXPLOSION, int(num_particles * self.PARTICLE_COUNT_MULTIPLIER))
        
        for _ in range(actual_num):
            particle = ParticleFactory.create_particle(x, y, color, intensity, self.particle_lifetime)
            # Give particles a slight upward bias to counteract any downward tendency
            particle.vy -= 0.3 * intensity
            self.particles.append(particle)
    
    def _update_particles(self, context: FrameContext) -> None:
        """Update all particles and remove dead ones."""
        for particle in self.particles:
            particle.update(context.beat_intensity)
        
        self.particles = [p for p in self.particles if p.is_alive()]
        
        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]
    
    def _render_particles(self, frame: np.ndarray) -> np.ndarray:
        """Render all particles with diverse visual styles."""
        if not self.particles:
            return frame
        
        result = frame.copy()
        h, w = result.shape[:2]
        overlay = np.zeros_like(result, dtype=np.float32)
        
        for particle in self.particles:
            x, y = int(particle.x), int(particle.y)
            
            if 0 <= x < w and 0 <= y < h:
                particle.render(overlay, x, y)
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return cv2.addWeighted(result, 1.0, overlay, self.BLEND_ALPHA, 0)
    
    def _should_trigger_explosion(self, context: FrameContext) -> bool:
        """Determine if explosion should be triggered."""
        return (
            context.beat_intensity > self.explosion_threshold and
            context.frame_index - self.last_explosion_frame > self.COOLDOWN_FRAMES
        )
    
    def process(self, frame: np.ndarray, context: FrameContext) -> np.ndarray:
        """Apply ghost particles effect."""
        if self._should_trigger_explosion(context):
            sampled_pixels = self._sample_pixels_by_frequency(frame, context)
            explosion_intensity = context.beat_intensity * self.intensity
            particles_per_explosion = max(5, int(15 * explosion_intensity))
            
            for x, y, color in sampled_pixels:
                self._create_explosion(x, y, color, explosion_intensity, particles_per_explosion)
            
            self.last_explosion_frame = context.frame_index
        
        self._update_particles(context)
        return self._render_particles(frame)
    
    def reset(self) -> None:
        """Reset effect state (clear all particles)."""
        self.particles.clear()
        self.last_explosion_frame = -100
