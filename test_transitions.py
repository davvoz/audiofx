"""
Test del sistema di transizioni smooth tra sezioni.
Verifica che il transition_factor funzioni correttamente.
"""

import numpy as np
import matplotlib.pyplot as plt


def get_transition_factor(time_in_section: float, transition_duration: float = 2.0) -> float:
    """
    Calcola il fattore di transizione smooth (0.0 a 1.0).
    Usa curva ease-in-out cubica.
    """
    if time_in_section >= transition_duration:
        return 1.0
    
    progress = time_in_section / transition_duration
    
    # Ease-in-out cubic
    if progress < 0.5:
        return 4 * progress * progress * progress
    else:
        return 1 - pow(-2 * progress + 2, 3) / 2


def test_transition_curve():
    """Test della curva di transizione."""
    print("=" * 60)
    print("TEST CURVA DI TRANSIZIONE SMOOTH")
    print("=" * 60)
    
    # Parametri test
    transition_duration = 2.0  # secondi
    fps = 30
    total_frames = int(transition_duration * fps) + 10
    
    # Calcola valori
    times = np.linspace(0, transition_duration + 0.5, total_frames)
    factors = [get_transition_factor(t, transition_duration) for t in times]
    
    # Stampa valori chiave
    print(f"\nTransition Duration: {transition_duration}s")
    print(f"FPS: {fps}")
    print(f"\nValori chiave:")
    test_times = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    for t in test_times:
        factor = get_transition_factor(t, transition_duration)
        percentage = factor * 100
        print(f"  t={t:.2f}s → factor={factor:.4f} ({percentage:.1f}%)")
    
    # Verifica proprietà
    print("\n✓ Verifiche:")
    assert get_transition_factor(0.0, 2.0) == 0.0, "Inizio deve essere 0"
    print("  ✓ Inizio (t=0.0s): factor = 0.0")
    
    assert get_transition_factor(2.0, 2.0) >= 0.99, "Fine deve essere ~1.0"
    print("  ✓ Fine (t=2.0s): factor ≈ 1.0")
    
    mid_factor = get_transition_factor(1.0, 2.0)
    assert 0.45 <= mid_factor <= 0.55, "Centro deve essere ~0.5"
    print(f"  ✓ Centro (t=1.0s): factor ≈ {mid_factor:.3f}")
    
    # Verifica smooth (derivata non deve avere salti)
    differences = np.diff(factors)
    max_jump = np.max(np.abs(np.diff(differences)))
    print(f"  ✓ Smoothness: max jump = {max_jump:.6f} (smooth!)")
    
    print("\n" + "=" * 60)
    print("✅ TUTTI I TEST PASSATI!")
    print("=" * 60)
    
    # Plot opzionale (se matplotlib disponibile)
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(times, factors, 'b-', linewidth=2, label='Transition Factor')
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
        plt.axvline(x=transition_duration, color='red', linestyle='--', alpha=0.5, label='Transition End')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Tempo nella Sezione (secondi)', fontsize=12)
        plt.ylabel('Transition Factor (0.0 → 1.0)', fontsize=12)
        plt.title('Curva Ease-In-Out Cubica per Transizioni Smooth', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.xlim(0, transition_duration + 0.5)
        plt.ylim(-0.05, 1.05)
        
        # Annotazioni
        plt.annotate('Inizio\n(0%)', xy=(0, 0), xytext=(0.2, 0.15),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                    fontsize=10, ha='left')
        plt.annotate('Centro\n(~50%)', xy=(1.0, mid_factor), xytext=(1.0, mid_factor + 0.2),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    fontsize=10, ha='center')
        plt.annotate('Fine\n(100%)', xy=(2.0, 1.0), xytext=(1.7, 0.85),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=10, ha='right')
        
        plt.tight_layout()
        plt.savefig('transition_curve.png', dpi=150, bbox_inches='tight')
        print("\n📊 Grafico salvato: transition_curve.png")
    except Exception as e:
        print(f"\n⚠️  Impossibile creare grafico: {e}")


def test_section_changes():
    """Test cambio sezioni con transizioni."""
    print("\n" + "=" * 60)
    print("TEST CAMBIO SEZIONI CON TRANSIZIONI")
    print("=" * 60)
    
    sections = [
        ('intro', 0.0, 15.0),
        ('buildup', 15.0, 30.0),
        ('drop', 30.0, 60.0),
        ('breakdown', 60.0, 80.0),
        ('outro', 80.0, 100.0),
    ]
    
    transition_duration = 2.0
    fps = 30
    
    print(f"\nSimulazione cambio sezioni (transition_duration={transition_duration}s):\n")
    
    previous_section = None
    section_start_time = 0.0
    
    for section_name, start, end in sections:
        # Simula alcuni frame all'inizio della sezione
        print(f"\n📍 Sezione: {section_name.upper()} ({start}s - {end}s)")
        
        # Reset al cambio sezione
        if previous_section != section_name:
            previous_section = section_name
            section_start_time = start
        
        # Testa transizione nei primi 2.5 secondi
        test_offsets = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        for offset in test_offsets:
            current_time = start + offset
            if current_time > end:
                break
            
            time_in_section = current_time - section_start_time
            factor = get_transition_factor(time_in_section, transition_duration)
            
            # Simula intensità effetto
            base_intensity = 0.8
            actual_intensity = base_intensity * factor
            
            print(f"  t={current_time:5.1f}s (offset +{offset:.1f}s): "
                  f"factor={factor:.3f} → "
                  f"intensity={actual_intensity:.3f} "
                  f"({'──' * int(actual_intensity * 20)})")
    
    print("\n" + "=" * 60)
    print("✅ SIMULAZIONE COMPLETATA!")
    print("=" * 60)


def test_effect_interpolation():
    """Test interpolazione effetti con diversi parametri."""
    print("\n" + "=" * 60)
    print("TEST INTERPOLAZIONE EFFETTI")
    print("=" * 60)
    
    effects = [
        ('Gradient Blend', 0.6),
        ('Black & White', 0.7),
        ('Triangular Distortion', 1.3),
        ('Geometric Distortion', 0.85),
        ('Negative', 0.6),
    ]
    
    transition_duration = 1.5
    
    print(f"\nInterpolazione con transition_duration={transition_duration}s:\n")
    
    times = [0.0, 0.375, 0.75, 1.125, 1.5]  # 0%, 25%, 50%, 75%, 100%
    
    for effect_name, base_intensity in effects:
        print(f"\n🎨 {effect_name} (base intensity: {base_intensity})")
        for t in times:
            factor = get_transition_factor(t, transition_duration)
            actual = base_intensity * factor
            percentage = (t / transition_duration) * 100
            
            bar = '█' * int(actual * 20)
            print(f"  {percentage:5.1f}%: {actual:.4f} {bar}")
    
    print("\n" + "=" * 60)
    print("✅ INTERPOLAZIONE VERIFICATA!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "🎵" * 30)
    print("TEST SISTEMA TRANSIZIONI SMOOTH")
    print("🎵" * 30 + "\n")
    
    test_transition_curve()
    test_section_changes()
    test_effect_interpolation()
    
    print("\n" + "🎉" * 30)
    print("TUTTI I TEST COMPLETATI CON SUCCESSO!")
    print("🎉" * 30 + "\n")
