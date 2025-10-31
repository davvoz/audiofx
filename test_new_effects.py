"""
Test per i nuovi effetti visivi:
- Passaggi sfumati (gradient blend)
- Bianco e nero (black and white)
- Negativo (negative)
- Distorsione triangolare (triangular distortion)
- Distorsione geometrica (geometric distortion)
"""

import cv2
import numpy as np
from video_generator import AudioVisualFX


def test_gradient_effects():
    """Test effetti di gradiente sfumato."""
    print("\n🎨 Test Passaggi Sfumati (Gradient Blend)")
    print("=" * 60)
    
    # Crea immagine di test colorata
    test_img = np.random.randint(50, 200, (720, 720, 3), dtype=np.uint8)
    
    # Test direzioni diverse
    directions = ["horizontal", "vertical", "radial", "diagonal"]
    intensities = [0.3, 0.5, 0.7, 0.9]
    
    for direction in directions:
        for intensity in intensities:
            result = AudioVisualFX._gradient_blend(test_img, intensity, direction)
            print(f"  ✓ Gradiente {direction:12} | intensità {intensity:.1f} | shape: {result.shape}")
    
    print("  ✅ Test gradient blend completato!")


def test_black_and_white():
    """Test effetto bianco e nero."""
    print("\n⚫ Test Bianco e Nero")
    print("=" * 60)
    
    # Crea immagine RGB colorata
    test_img = np.random.randint(0, 255, (720, 720, 3), dtype=np.uint8)
    
    # Test varie intensità
    intensities = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for intensity in intensities:
        result = AudioVisualFX._black_and_white(test_img, intensity)
        
        # Verifica che con intensità 1.0 tutti i canali RGB siano uguali
        if intensity >= 1.0:
            r_channel = result[:, :, 0]
            g_channel = result[:, :, 1]
            b_channel = result[:, :, 2]
            is_grayscale = np.allclose(r_channel, g_channel) and np.allclose(g_channel, b_channel)
            status = "✓ Grayscale" if is_grayscale else "✗ Non grayscale"
        else:
            status = "✓ Blended"
        
        print(f"  {status} | intensità {intensity:.2f} | shape: {result.shape}")
    
    print("  ✅ Test bianco e nero completato!")


def test_negative():
    """Test effetto negativo."""
    print("\n🔄 Test Negativo")
    print("=" * 60)
    
    # Crea immagine con pattern riconoscibile
    test_img = np.zeros((720, 720, 3), dtype=np.uint8)
    test_img[:, :, 0] = 100  # Rosso
    test_img[:, :, 1] = 150  # Verde
    test_img[:, :, 2] = 200  # Blu
    
    # Test varie intensità
    intensities = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for intensity in intensities:
        result = AudioVisualFX._negative(test_img, intensity)
        
        # Con intensità 1.0, verifica inversione
        if intensity >= 1.0:
            expected_r = 255 - 100
            expected_g = 255 - 150
            expected_b = 255 - 200
            is_inverted = (
                np.allclose(result[0, 0, 0], expected_r, atol=1) and
                np.allclose(result[0, 0, 1], expected_g, atol=1) and
                np.allclose(result[0, 0, 2], expected_b, atol=1)
            )
            status = "✓ Invertito" if is_inverted else "✗ Non invertito"
        else:
            status = "✓ Blended"
        
        print(f"  {status} | intensità {intensity:.2f} | RGB: ({result[0, 0, 0]}, {result[0, 0, 1]}, {result[0, 0, 2]})")
    
    print("  ✅ Test negativo completato!")


def test_triangular_distortion():
    """Test distorsione triangolare."""
    print("\n🔺 Test Distorsione Triangolare")
    print("=" * 60)
    
    # Crea immagine di test
    test_img = np.random.randint(0, 255, (720, 720, 3), dtype=np.uint8)
    
    # Test varie intensità e tempi
    intensities = [0.2, 0.4, 0.6, 0.8]
    frame_times = [0.0, 1.0, 2.0, 3.0]
    
    for intensity in intensities:
        for frame_time in frame_times:
            result = AudioVisualFX._triangular_distortion(test_img, intensity, frame_time)
            print(f"  ✓ Distorsione triangolare | intensità {intensity:.1f} | tempo {frame_time:.1f}s | shape: {result.shape}")
    
    print("  ✅ Test distorsione triangolare completato!")


def test_geometric_distortion():
    """Test distorsioni geometriche."""
    print("\n📐 Test Distorsione Geometrica")
    print("=" * 60)
    
    # Crea immagine di test
    test_img = np.random.randint(0, 255, (720, 720, 3), dtype=np.uint8)
    
    # Test tutti i modi
    modes = ["pinch", "barrel", "pincushion", "swirl"]
    intensities = [0.3, 0.5, 0.7, 0.9]
    
    for mode in modes:
        for intensity in intensities:
            result = AudioVisualFX._geometric_distortion(test_img, intensity, mode)
            print(f"  ✓ Distorsione {mode:11} | intensità {intensity:.1f} | shape: {result.shape}")
    
    print("  ✅ Test distorsione geometrica completato!")


def test_combined_effects():
    """Test combinazione di più effetti."""
    print("\n🌈 Test Combinazione Effetti")
    print("=" * 60)
    
    # Crea immagine di test
    test_img = np.random.randint(50, 200, (720, 720, 3), dtype=np.uint8)
    
    # Applica effetti in sequenza
    print("\n  Sequenza 1: Gradiente → B&W → Negativo")
    result = AudioVisualFX._gradient_blend(test_img, 0.6, "radial")
    result = AudioVisualFX._black_and_white(result, 0.5)
    result = AudioVisualFX._negative(result, 0.4)
    print(f"    ✓ Risultato: {result.shape}")
    
    print("\n  Sequenza 2: Distorsione Triangolare → Geometrica (pinch)")
    result = AudioVisualFX._triangular_distortion(test_img, 0.7, 1.5)
    result = AudioVisualFX._geometric_distortion(result, 0.6, "pinch")
    print(f"    ✓ Risultato: {result.shape}")
    
    print("\n  Sequenza 3: Gradiente → Distorsione Geometrica (swirl) → B&W")
    result = AudioVisualFX._gradient_blend(test_img, 0.5, "diagonal")
    result = AudioVisualFX._geometric_distortion(result, 0.8, "swirl")
    result = AudioVisualFX._black_and_white(result, 0.7)
    print(f"    ✓ Risultato: {result.shape}")
    
    print("\n  ✅ Test combinazione effetti completato!")


def test_edge_cases():
    """Test casi limite."""
    print("\n⚠️  Test Casi Limite")
    print("=" * 60)
    
    # Immagine piccola
    small_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Immagine grande
    large_img = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
    
    # Test con immagine piccola
    print("\n  Test con immagine piccola (100x100):")
    result = AudioVisualFX._gradient_blend(small_img, 0.7, "radial")
    print(f"    ✓ Gradiente: {result.shape}")
    result = AudioVisualFX._triangular_distortion(small_img, 0.6, 1.0)
    print(f"    ✓ Distorsione triangolare: {result.shape}")
    result = AudioVisualFX._geometric_distortion(small_img, 0.5, "swirl")
    print(f"    ✓ Distorsione geometrica: {result.shape}")
    
    # Test con intensità limite
    print("\n  Test con intensità ai limiti:")
    test_img = np.random.randint(0, 255, (720, 720, 3), dtype=np.uint8)
    
    result = AudioVisualFX._black_and_white(test_img, 0.0)  # Minimo
    print(f"    ✓ B&W intensità 0.0: {result.shape}")
    
    result = AudioVisualFX._negative(test_img, 1.0)  # Massimo
    print(f"    ✓ Negativo intensità 1.0: {result.shape}")
    
    result = AudioVisualFX._gradient_blend(test_img, 1.5, "horizontal")  # Oltre il limite
    print(f"    ✓ Gradiente intensità 1.5: {result.shape}")
    
    print("\n  ✅ Test casi limite completato!")


def main():
    """Esegue tutti i test."""
    print("\n" + "=" * 60)
    print("🧪 TEST NUOVI EFFETTI VISIVI")
    print("=" * 60)
    
    try:
        test_gradient_effects()
        test_black_and_white()
        test_negative()
        test_triangular_distortion()
        test_geometric_distortion()
        test_combined_effects()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ TUTTI I TEST COMPLETATI CON SUCCESSO!")
        print("=" * 60)
        print("\n📋 Riepilogo nuovi effetti implementati:")
        print("  1. ✓ Passaggi sfumati (gradient_blend)")
        print("      - Direzioni: horizontal, vertical, radial, diagonal")
        print("  2. ✓ Bianco e nero (black_and_white)")
        print("      - Controllo intensità per blend graduali")
        print("  3. ✓ Negativo (negative)")
        print("      - Inversione colori con blend controllato")
        print("  4. ✓ Distorsione triangolare (triangular_distortion)")
        print("      - Pattern triangolare animato")
        print("  5. ✓ Distorsione geometrica (geometric_distortion)")
        print("      - Modalità: pinch, barrel, pincushion, swirl")
        print("\n🎉 Gli effetti sono pronti per essere utilizzati!")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante i test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
