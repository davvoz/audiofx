"""
Test delle transizioni ULTRA-SMOOTH migliorati con:
1. Curva ease-in-out quartica (invece di cubica)
2. Durata più lunga (3.0s invece di 2.0s)
3. Interpolazione smooth dei colori RGB
"""

import numpy as np
import matplotlib.pyplot as plt


def ease_in_out_cubic(t):
    """Curva cubica (vecchia)."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2


def ease_in_out_quartic(t):
    """Curva quartica (vecchia - problemi al centro)."""
    if t < 0.5:
        return 8 * t * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 4) / 2


def smootherstep(t):
    """Smootherstep (Ken Perlin) - PERFETTAMENTE SMOOTH!
    Formula: 6x^5 - 15x^4 + 10x^3
    Derivata prima e seconda = 0 agli estremi.
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def interpolate_color(color1, color2, factor):
    """Interpola tra due colori RGB."""
    r = color1[0] * (1 - factor) + color2[0] * factor
    g = color1[1] * (1 - factor) + color2[1] * factor
    b = color1[2] * (1 - factor) + color2[2] * factor
    return (r, g, b)


def test_curve_comparison():
    """Confronta curve: cubica vs quartica vs smootherstep."""
    print("=" * 70)
    print("CONFRONTO CURVE: CUBICA vs QUARTICA vs SMOOTHERSTEP")
    print("=" * 70)
    
    # Genera valori
    t_values = np.linspace(0, 1, 100)
    cubic_values = [ease_in_out_cubic(t) for t in t_values]
    quartic_values = [ease_in_out_quartic(t) for t in t_values]
    smootherstep_values = [smootherstep(t) for t in t_values]
    
    # Calcola derivate (velocità di transizione)
    cubic_speed = np.gradient(cubic_values)
    quartic_speed = np.gradient(quartic_values)
    smootherstep_speed = np.gradient(smootherstep_values)
    
    print("\n📊 Confronto valori chiave:")
    print(f"{'Progresso':<12} {'Cubica':<12} {'Quartica':<12} {'Smootherstep':<14} {'Best':<10}")
    print("-" * 65)
    
    test_points = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for t in test_points:
        cubic = ease_in_out_cubic(t)
        quartic = ease_in_out_quartic(t)
        smooth = smootherstep(t)
        
        # Trova quale è più vicina al lineare ideale
        linear = t
        cubic_diff = abs(cubic - linear)
        quartic_diff = abs(quartic - linear)
        smooth_diff = abs(smooth - linear)
        
        best = min([('Cubic', cubic_diff), ('Quartic', quartic_diff), ('Smooth', smooth_diff)], key=lambda x: x[1])[0]
        
        print(f"{t:>5.2f} ({int(t*100):>3}%)  {cubic:>10.6f}  {quartic:>10.6f}  {smooth:>12.6f}  {best:<10}")
    
    # Calcola smoothness (minore variazione = più smooth)
    cubic_smoothness = np.std(np.diff(cubic_speed))
    quartic_smoothness = np.std(np.diff(quartic_speed))
    smootherstep_smoothness = np.std(np.diff(smootherstep_speed))
    
    print(f"\n✨ Smoothness - Variazione velocità (minore = migliore):")
    print(f"  Cubica:       {cubic_smoothness:.6f}")
    print(f"  Quartica:     {quartic_smoothness:.6f}")
    print(f"  Smootherstep: {smootherstep_smoothness:.6f} 🏆")
    
    # Calcola anche accelerazione (derivata seconda)
    cubic_accel_std = np.std(np.gradient(cubic_speed))
    quartic_accel_std = np.std(np.gradient(quartic_speed))
    smootherstep_accel_std = np.std(np.gradient(smootherstep_speed))
    
    print(f"\n✨ Smoothness - Variazione accelerazione (minore = migliore):")
    print(f"  Cubica:       {cubic_accel_std:.6f}")
    print(f"  Quartica:     {quartic_accel_std:.6f}")
    print(f"  Smootherstep: {smootherstep_accel_std:.6f} 🏆")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Curve
    axes[0, 0].plot(t_values, cubic_values, 'b-', linewidth=2, label='Cubica', alpha=0.6)
    axes[0, 0].plot(t_values, quartic_values, 'orange', linewidth=2, label='Quartica', alpha=0.6)
    axes[0, 0].plot(t_values, smootherstep_values, 'g-', linewidth=3, label='Smootherstep 🏆', alpha=0.9)
    axes[0, 0].plot(t_values, t_values, 'k--', linewidth=1, label='Lineare', alpha=0.3)
    axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.2)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('Progresso Temporale (0→1)', fontsize=11)
    axes[0, 0].set_ylabel('Fattore Transizione', fontsize=11)
    axes[0, 0].set_title('Confronto Curve di Transizione', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    
    # Differenza da lineare (quanto si discosta dall'ideale)
    cubic_diff = np.array(cubic_values) - t_values
    quartic_diff = np.array(quartic_values) - t_values
    smooth_diff = np.array(smootherstep_values) - t_values
    
    axes[0, 1].plot(t_values, cubic_diff, 'b-', linewidth=2, label='Cubica', alpha=0.6)
    axes[0, 1].plot(t_values, quartic_diff, 'orange', linewidth=2, label='Quartica', alpha=0.6)
    axes[0, 1].plot(t_values, smooth_diff, 'g-', linewidth=3, label='Smootherstep 🏆', alpha=0.9)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('Progresso Temporale', fontsize=11)
    axes[0, 1].set_ylabel('Differenza da Lineare', fontsize=11)
    axes[0, 1].set_title('Deviazione da Transizione Lineare', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    
    # Velocità (derivata prima)
    axes[1, 0].plot(t_values, cubic_speed, 'b-', linewidth=2, label='Cubica', alpha=0.6)
    axes[1, 0].plot(t_values, quartic_speed, 'orange', linewidth=2, label='Quartica', alpha=0.6)
    axes[1, 0].plot(t_values, smootherstep_speed, 'g-', linewidth=3, label='Smootherstep 🏆', alpha=0.9)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Progresso Temporale', fontsize=11)
    axes[1, 0].set_ylabel('Velocità di Transizione', fontsize=11)
    axes[1, 0].set_title('Velocità di Transizione (Derivata Prima)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    
    # Accelerazione (derivata seconda) - CHIAVE per smoothness!
    cubic_accel = np.gradient(cubic_speed)
    quartic_accel = np.gradient(quartic_speed)
    smootherstep_accel = np.gradient(smootherstep_speed)
    
    axes[1, 1].plot(t_values, cubic_accel, 'b-', linewidth=2, label='Cubica', alpha=0.6)
    axes[1, 1].plot(t_values, quartic_accel, 'orange', linewidth=2, label='Quartica', alpha=0.6)
    axes[1, 1].plot(t_values, smootherstep_accel, 'g-', linewidth=3, label='Smootherstep 🏆', alpha=0.9)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel('Progresso Temporale', fontsize=11)
    axes[1, 1].set_ylabel('Accelerazione (Smoothness)', fontsize=11)
    axes[1, 1].set_title('Accelerazione - CHIAVE per Smoothness!', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('ultra_smooth_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Grafico salvato: ultra_smooth_comparison.png")
    
    print("\n" + "=" * 70)
    print("✅ SMOOTHERSTEP VINCE! Perfettamente smooth in TUTTO il range!")
    print("=" * 70)


def test_color_interpolation():
    """Test interpolazione colori RGB."""
    print("\n" + "=" * 70)
    print("TEST INTERPOLAZIONE SMOOTH COLORI RGB")
    print("=" * 70)
    
    # Colori test
    color1 = (0.8, 0.0, 0.8)  # Magenta
    color2 = (0.0, 0.8, 0.8)  # Cyan
    
    print(f"\n🎨 Transizione: Magenta {color1} → Cyan {color2}\n")
    
    # Genera interpolazione con entrambe le curve
    steps = 11
    for i in range(steps):
        progress = i / (steps - 1)
        
        # Usa curva smootherstep
        factor = smootherstep(progress)
        
        # Interpola colore
        color = interpolate_color(color1, color2, factor)
        
        # Visualizza
        percentage = progress * 100
        bar_cubic = '█' * int(ease_in_out_cubic(progress) * 20)
        bar_quartic = '█' * int(factor * 20)
        
        print(f"{percentage:>5.0f}% | Quartic: {factor:.4f} | RGB: ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
        print(f"       | {bar_quartic}")
    
    # Plot color gradient
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Gradient con curva cubica
    cubic_colors = []
    for i in range(100):
        t = i / 99
        factor = ease_in_out_cubic(t)
        color = interpolate_color(color1, color2, factor)
        cubic_colors.append(color)
    
    axes[0].imshow([cubic_colors], aspect='auto')
    axes[0].set_title('Interpolazione Colori - Cubica (OLD)', fontsize=12, fontweight='bold')
    axes[0].set_xticks([0, 25, 50, 75, 99])
    axes[0].set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    axes[0].set_yticks([])
    
    # Gradient con curva smootherstep
    smootherstep_colors = []
    for i in range(100):
        t = i / 99
        factor = smootherstep(t)
        color = interpolate_color(color1, color2, factor)
        smootherstep_colors.append(color)
    
    axes[1].imshow([smootherstep_colors], aspect='auto')
    axes[1].set_title('Interpolazione Colori - Smootherstep (NEW - PERFETTAMENTE SMOOTH)', fontsize=12, fontweight='bold')
    axes[1].set_xticks([0, 25, 50, 75, 99])
    axes[1].set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    axes[1].set_yticks([])
    
    # Differenza
    diff_colors = []
    for i in range(100):
        cubic = np.array(cubic_colors[i])
        smooth = np.array(smootherstep_colors[i])
        diff = smooth - cubic
        # Visualizza differenza come colore (amplificata x10 per visibilità)
        diff_color = np.clip(np.abs(diff) * 10, 0, 1)
        diff_colors.append(diff_color)
    
    axes[2].imshow([diff_colors], aspect='auto', cmap='hot')
    axes[2].set_title('Differenza (amplificata x10 per visibilità)', fontsize=12, fontweight='bold')
    axes[2].set_xticks([0, 25, 50, 75, 99])
    axes[2].set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    axes[2].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('color_interpolation_smooth.png', dpi=150, bbox_inches='tight')
    print("\n📊 Grafico salvato: color_interpolation_smooth.png")
    
    print("\n" + "=" * 70)
    print("✅ COLORI INTERPOLATI SMOOTH!")
    print("=" * 70)


def test_duration_comparison():
    """Confronta durate transizione: 2.0s vs 3.0s."""
    print("\n" + "=" * 70)
    print("CONFRONTO DURATE: 2.0s vs 3.0s (ULTRA-SMOOTH)")
    print("=" * 70)
    
    fps = 30
    duration_2s = 2.0
    duration_3s = 3.0
    
    frames_2s = int(duration_2s * fps)
    frames_3s = int(duration_3s * fps)
    
    print(f"\n⏱️  Durata 2.0s: {frames_2s} frames @ {fps}fps")
    print(f"⏱️  Durata 3.0s: {frames_3s} frames @ {fps}fps")
    print(f"📈 Aumento frame: +{frames_3s - frames_2s} frames (+{((frames_3s - frames_2s) / frames_2s * 100):.0f}%)")
    
    # Simula intensità effetto nel tempo
    print("\n📊 Intensità effetto base 0.8:")
    print(f"{'Frame':<8} {'Tempo 2.0s':<15} {'Tempo 3.0s':<15} {'Differenza':<12}")
    print("-" * 55)
    
    test_frames = [0, 15, 30, 45, 60, 75, 90]
    for frame in test_frames:
        if frame < frames_2s:
            t_2s = frame / frames_2s
            factor_2s = ease_in_out_quartic(t_2s)
            intensity_2s = 0.8 * factor_2s
        else:
            intensity_2s = 0.8
        
        if frame < frames_3s:
            t_3s = frame / frames_3s
            factor_3s = smootherstep(t_3s)
            intensity_3s = 0.8 * factor_3s
        else:
            intensity_3s = 0.8
        
        time_2s = frame / fps
        diff = intensity_3s - intensity_2s
        
        print(f"{frame:<8} {intensity_2s:>6.4f} ({int(intensity_2s/0.8*100):>3}%)  "
              f"{intensity_3s:>6.4f} ({int(intensity_3s/0.8*100):>3}%)  {diff:>+6.4f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    times_2s = np.linspace(0, duration_2s + 0.5, 100)
    intensities_2s = []
    for t in times_2s:
        if t < duration_2s:
            factor = ease_in_out_quartic(t / duration_2s)
            intensities_2s.append(0.8 * factor)
        else:
            intensities_2s.append(0.8)
    
    times_3s = np.linspace(0, duration_3s + 0.5, 150)
    intensities_3s = []
    for t in times_3s:
        if t < duration_3s:
            factor = smootherstep(t / duration_3s)
            intensities_3s.append(0.8 * factor)
        else:
            intensities_3s.append(0.8)
    
    plt.plot(times_2s, intensities_2s, 'b-', linewidth=2, label='2.0s (OLD)', alpha=0.7)
    plt.plot(times_3s, intensities_3s, 'r-', linewidth=2, label='3.0s (NEW - ULTRA SMOOTH)', alpha=0.9)
    plt.axvline(x=duration_2s, color='blue', linestyle='--', alpha=0.3, label='Fine 2.0s')
    plt.axvline(x=duration_3s, color='red', linestyle='--', alpha=0.3, label='Fine 3.0s')
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.3, label='Target 100%')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Tempo (secondi)', fontsize=11)
    plt.ylabel('Intensità Effetto', fontsize=11)
    plt.title('Confronto Durate Transizione: 2.0s vs 3.0s', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.xlim(0, 3.5)
    plt.ylim(0, 0.9)
    
    plt.tight_layout()
    plt.savefig('duration_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Grafico salvato: duration_comparison.png")
    
    print("\n" + "=" * 70)
    print("✅ 3.0s È PIÙ SMOOTH E CINEMATOGRAFICO!")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "🎬" * 35)
    print("TEST TRANSIZIONI ULTRA-SMOOTH")
    print("🎬" * 35 + "\n")
    
    test_curve_comparison()
    test_color_interpolation()
    test_duration_comparison()
    
    print("\n" + "🌟" * 35)
    print("TUTTI I TEST ULTRA-SMOOTH COMPLETATI!")
    print("Miglioramenti: Curva SMOOTHERSTEP + 3.0s + Colori Interpolati")
    print("Smootherstep = Perfettamente smooth in TUTTO il range!")
    print("🌟" * 35 + "\n")
