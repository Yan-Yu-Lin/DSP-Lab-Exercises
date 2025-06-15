import numpy as np
import matplotlib.pyplot as plt

"""
******************************************************************
[Practice 3-1] Implement the DTFT without using functions exp(), abs(), and angle(), but instead,
using the following expressions.

X(e^{jw}) = \sum_{n=-\infty}^{\infty} x[n]e^{-jwn} = \sum_{n=-\infty}^{\infty} \{x[n]\cos(wn) - jx[n]\sin(wn)\} = X_R(e^{jw}) + jX_I(e^{jw}), i.e., real and

imaginary parts, and its magnitude and phase are \sqrt{X_R^2(e^{jw}) + X_I^2(e^{jw})} and \tan^{-1}\left[\frac{X_I(e^{jw})}{X_R(e^{jw})}\right],

respectively.
******************************************************************

Complete Problem Statement:
Implement the Discrete-Time Fourier Transform (DTFT) without using exp(), abs(), or angle() functions.

Instead, use the decomposition:
    X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n]e^{-j\omega n}
                   = \sum_{n=-\infty}^{\infty} \{x[n]\cos(\omega n) - jx[n]\sin(\omega n)\}
                   = X_R(e^{j\omega}) + jX_I(e^{j\omega})

where:
- Real part: X_R(e^{j\omega}) = \sum x[n]\cos(\omega n)
- Imaginary part: X_I(e^{j\omega}) = -\sum x[n]\sin(\omega n)
- Magnitude: |X(e^{j\omega})| = \sqrt{X_R^2(e^{j\omega}) + X_I^2(e^{j\omega})}
- Phase: \angle X(e^{j\omega}) = \tan^{-1}\left[\frac{X_I(e^{j\omega})}{X_R(e^{j\omega})}\right]

Note: Use atan2 instead of atan to handle all quadrants correctly.
"""

def dtft_manual(x, w):
    """
    Compute DTFT at frequencies w without using exp(), abs(), or angle()
    
    X(e^jw) = Σ x[n]e^(-jwn) = Σ x[n](cos(wn) - j*sin(wn))
    
    Parameters:
    x: input sequence
    w: array of frequencies to evaluate DTFT at
    
    Returns:
    X_R: real part
    X_I: imaginary part
    magnitude: |X(e^jw)|
    phase: angle of X(e^jw)
    """
    N = len(x)
    n = np.arange(N)
    
    # Initialize arrays for results
    X_R = np.zeros(len(w))  # Real part
    X_I = np.zeros(len(w))  # Imaginary part
    
    # Compute DTFT for each frequency
    for i, freq in enumerate(w):
        # Real part: X_R = Σ x[n]cos(wn)
        X_R[i] = np.sum(x * np.cos(freq * n))
        
        # Imaginary part: X_I = -Σ x[n]sin(wn)
        # Note: negative because e^(-jwn) = cos(wn) - j*sin(wn)
        X_I[i] = -np.sum(x * np.sin(freq * n))
    
    # Compute magnitude: |X| = sqrt(X_R^2 + X_I^2)
    magnitude = np.sqrt(X_R**2 + X_I**2)
    
    # Compute phase: angle = atan2(X_I, X_R)
    # Using atan2 instead of atan to handle all quadrants correctly
    phase = np.arctan2(X_I, X_R)
    
    return X_R, X_I, magnitude, phase

# Test signal 1: Simple rectangular pulse
x1 = np.array([1, 1, 1, 1, 1])  # 5-point rectangular pulse

# Frequency points (0 to 2π)
w = np.linspace(0, 2*np.pi, 500)

# Compute DTFT manually
X_R1, X_I1, mag1, phase1 = dtft_manual(x1, w)

# Plotting
plt.figure(figsize=(12, 10))

# Plot original signal
plt.subplot(4, 1, 1)
plt.stem(range(len(x1)), x1, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Input Signal: 5-point Rectangular Pulse')
plt.grid(True, alpha=0.3)

# Plot magnitude spectrum
plt.subplot(4, 1, 2)
plt.plot(w, mag1, 'b-', linewidth=2)
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('|X(e^jω)|')
plt.title('Magnitude Spectrum')
plt.grid(True, alpha=0.3)
plt.xlim([0, 2*np.pi])

# Plot phase spectrum
plt.subplot(4, 1, 3)
plt.plot(w, phase1, 'r-', linewidth=2)
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('∠X(e^jω) (rad)')
plt.title('Phase Spectrum')
plt.grid(True, alpha=0.3)
plt.xlim([0, 2*np.pi])
plt.ylim([-np.pi, np.pi])

# Plot real and imaginary parts
plt.subplot(4, 1, 4)
plt.plot(w, X_R1, 'g-', linewidth=2, label='Real part')
plt.plot(w, X_I1, 'm-', linewidth=2, label='Imaginary part')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Amplitude')
plt.title('Real and Imaginary Parts of X(e^jω)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 2*np.pi])

plt.tight_layout()
plt.show()

# Test signal 2: Exponentially decaying sequence
n2 = np.arange(10)
a = 0.8
x2 = a**n2  # x[n] = a^n for n >= 0

# Compute DTFT
X_R2, X_I2, mag2, phase2 = dtft_manual(x2, w)

# Plotting for signal 2
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.stem(n2, x2, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title(f'Input Signal: Exponential Decay x[n] = {a}^n')
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 2)
plt.plot(w, mag2, 'b-', linewidth=2)
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('|X(e^jω)|')
plt.title('Magnitude Spectrum')
plt.grid(True, alpha=0.3)
plt.xlim([0, 2*np.pi])

plt.subplot(4, 1, 3)
plt.plot(w, phase2, 'r-', linewidth=2)
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('∠X(e^jω) (rad)')
plt.title('Phase Spectrum')
plt.grid(True, alpha=0.3)
plt.xlim([0, 2*np.pi])
plt.ylim([-np.pi, np.pi])

plt.subplot(4, 1, 4)
plt.plot(w, X_R2, 'g-', linewidth=2, label='Real part')
plt.plot(w, X_I2, 'm-', linewidth=2, label='Imaginary part')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Amplitude')
plt.title('Real and Imaginary Parts of X(e^jω)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 2*np.pi])

plt.tight_layout()
plt.show()

# Verification: Compare with numpy's FFT (which uses exp internally)
print("Verification against NumPy's FFT:")
print("=" * 50)

# For rectangular pulse
N_fft = 512
X_fft = np.fft.fft(x1, N_fft)
w_fft = 2 * np.pi * np.arange(N_fft) / N_fft

# Find closest frequency points
idx = [np.argmin(np.abs(w_fft - freq)) for freq in [np.pi/2, np.pi, 3*np.pi/2]]

print("\nRectangular Pulse at specific frequencies:")
for i, freq in enumerate([np.pi/2, np.pi, 3*np.pi/2]):
    idx_manual = np.argmin(np.abs(w - freq))
    print(f"\nω = {freq:.2f} rad/sample:")
    print(f"Manual DTFT magnitude: {mag1[idx_manual]:.4f}")
    print(f"NumPy FFT magnitude:   {np.abs(X_fft[idx[i]]):.4f}")
    print(f"Difference:            {np.abs(mag1[idx_manual] - np.abs(X_fft[idx[i]])):.2e}")

# Show the formulas used
print("\n\nFormulas used in implementation:")
print("=" * 50)
print("Real part:      X_R(e^jω) = Σ x[n]cos(ωn)")
print("Imaginary part: X_I(e^jω) = -Σ x[n]sin(ωn)")
print("Magnitude:      |X(e^jω)| = √(X_R² + X_I²)")
print("Phase:          ∠X(e^jω) = atan2(X_I, X_R)")
print("\nNote: atan2 handles all quadrants correctly!")

# Demonstrate manual calculation for a specific frequency
print("\n\nManual calculation example:")
print("=" * 50)
freq_example = np.pi/4
print(f"Computing DTFT of x = [1, 1, 1, 1, 1] at ω = π/4:")
print("\nReal part:")
for n in range(len(x1)):
    print(f"  x[{n}]×cos({n}π/4) = 1×{np.cos(n*np.pi/4):.4f} = {x1[n]*np.cos(n*np.pi/4):.4f}")
print(f"  Sum = {np.sum(x1 * np.cos(freq_example * np.arange(len(x1)))):.4f}")

print("\nImaginary part:")
for n in range(len(x1)):
    print(f"  -x[{n}]×sin({n}π/4) = -1×{np.sin(n*np.pi/4):.4f} = {-x1[n]*np.sin(n*np.pi/4):.4f}")
print(f"  Sum = {-np.sum(x1 * np.sin(freq_example * np.arange(len(x1)))):.4f}")

"""
==================== RUN RESULTS ====================
Verification against NumPy's FFT:
==================================================

Rectangular Pulse at specific frequencies:

ω = 1.57 rad/sample:
Manual DTFT magnitude: 1.0063
NumPy FFT magnitude:   1.0000
Difference:            6.26e-03

ω = 3.14 rad/sample:
Manual DTFT magnitude: 0.9999
NumPy FFT magnitude:   1.0000
Difference:            1.19e-04

ω = 4.71 rad/sample:
Manual DTFT magnitude: 1.0063
NumPy FFT magnitude:   1.0000
Difference:            6.26e-03


Formulas used in implementation:
==================================================
Real part:      X_R(e^jω) = Σ x[n]cos(ωn)
Imaginary part: X_I(e^jω) = -Σ x[n]sin(ωn)
Magnitude:      |X(e^jω)| = √(X_R² + X_I²)
Phase:          ∠X(e^jω) = atan2(X_I, X_R)

Note: atan2 handles all quadrants correctly!


Manual calculation example:
==================================================
Computing DTFT of x = [1, 1, 1, 1, 1] at ω = π/4:

Real part:
  x[0]×cos(0π/4) = 1×1.0000 = 1.0000
  x[1]×cos(1π/4) = 1×0.7071 = 0.7071
  x[2]×cos(2π/4) = 1×0.0000 = 0.0000
  x[3]×cos(3π/4) = 1×-0.7071 = -0.7071
  x[4]×cos(4π/4) = 1×-1.0000 = -1.0000
  Sum = 0.0000

Imaginary part:
  -x[0]×sin(0π/4) = -1×0.0000 = -0.0000
  -x[1]×sin(1π/4) = -1×0.7071 = -0.7071
  -x[2]×sin(2π/4) = -1×1.0000 = -1.0000
  -x[3]×sin(3π/4) = -1×0.7071 = -0.7071
  -x[4]×sin(4π/4) = -1×0.0000 = -0.0000
  Sum = -2.4142

KEY INSIGHTS:
1. DTFT computed without complex exponentials - only sin/cos needed
2. Rectangular pulse produces sinc-like frequency response
3. Zero crossings occur at multiples of 2π/N (N=5 for 5-point pulse)
4. Exponential decay sequence has smooth frequency response
5. Manual calculation matches theory: decompose e^{-jωn} = cos(ωn) - j·sin(ωn)
6. Small differences from FFT due to continuous vs discrete frequency sampling
7. atan2 correctly handles phase in all four quadrants

The implementation demonstrates the fundamental nature of the Fourier transform
as a decomposition into sinusoidal components.
==================== END OF RESULTS ====================
"""