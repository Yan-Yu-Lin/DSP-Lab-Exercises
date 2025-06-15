import numpy as np
import matplotlib.pyplot as plt

"""
******************************************************************
[Practice 3-4] Use function fft() to compute the DFT of the signal in Practice 1-3, i.e., a
discrete-time signal that consists of 10Hz and 30Hz sine components based on a sampling period of
0.01 second.
******************************************************************

Complete Problem Statement:
Compute the DFT of the signal from Practice 1-3:
    x(t) = 2sin(2π×10t) + sin(2π×30t)
sampled at fs = 100 Hz (Ts = 0.01 second)

Mathematical Expression (LaTeX):
Continuous signal: x(t) = 2\\sin(2\\pi \\cdot 10t) + \\sin(2\\pi \\cdot 30t)
Sampled signal: x[n] = x(nT_s) where T_s = 0.01 seconds
DFT: X[k] = \\sum_{n=0}^{N-1} x[n] e^{-j2\\pi kn/N}
"""

# Sampling parameters
fs = 100  # Sampling frequency in Hz
Ts = 1/fs  # Sampling period = 0.01 seconds

# Create the signal (same as Practice 1-3)
# Let's use different signal lengths to see the effect
signal_durations = [0.5, 1.0, 2.0]  # seconds
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for idx, duration in enumerate(signal_durations):
    # Time vector
    t = np.arange(0, duration, Ts)
    N = len(t)
    
    # Generate signal
    x = 2 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 30 * t)
    
    # Compute FFT
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, Ts)
    
    # Only plot positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    X_mag_pos = np.abs(X[pos_mask])
    
    # Time domain plot
    axes[idx, 0].plot(t[:int(5*fs)], x[:int(5*fs)], 'b-', linewidth=1)
    axes[idx, 0].set_xlabel('Time (s)')
    axes[idx, 0].set_ylabel('x(t)')
    axes[idx, 0].set_title(f'Signal: {duration}s duration (first 5 cycles shown)')
    axes[idx, 0].grid(True, alpha=0.3)
    
    # Frequency domain plot
    axes[idx, 1].stem(freqs_pos[:int(N/2)], X_mag_pos[:int(N/2)], 
                      linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[idx, 1].set_xlabel('Frequency (Hz)')
    axes[idx, 1].set_ylabel('|X(f)|')
    axes[idx, 1].set_title(f'FFT Magnitude: N = {N} samples')
    axes[idx, 1].set_xlim([0, 50])
    axes[idx, 1].grid(True, alpha=0.3)
    
    # Mark the expected frequencies
    axes[idx, 1].axvline(x=10, color='g', linestyle='--', alpha=0.5, label='10 Hz')
    axes[idx, 1].axvline(x=30, color='g', linestyle='--', alpha=0.5, label='30 Hz')
    if idx == 0:
        axes[idx, 1].legend()

plt.tight_layout()
plt.show()

# Detailed analysis with the standard 0.5 second signal
print("FFT Analysis of Two-Tone Signal")
print("=" * 50)

# Use 0.5 second signal for detailed analysis
t = np.arange(0, 0.5, Ts)
N = len(t)
x = 2 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 30 * t)

# Compute FFT
X = np.fft.fft(x)
freqs = np.fft.fftfreq(N, Ts)

# Find peaks
magnitude = np.abs(X)
# Only look at positive frequencies
pos_freqs = freqs[:N//2]
pos_mag = magnitude[:N//2]

# Find peaks above threshold
threshold = np.max(pos_mag) * 0.1
peaks_idx = np.where(pos_mag > threshold)[0]
peak_freqs = pos_freqs[peaks_idx]
peak_mags = pos_mag[peaks_idx]

print(f"\nSignal parameters:")
print(f"  Sampling frequency: {fs} Hz")
print(f"  Sampling period: {Ts} seconds")
print(f"  Signal duration: 0.5 seconds")
print(f"  Number of samples: {N}")
print(f"  Frequency resolution: {fs/N:.2f} Hz")

print(f"\nDetected frequency components:")
for f, m in zip(peak_freqs, peak_mags):
    print(f"  f = {f:.1f} Hz, magnitude = {m:.1f}")

# Theoretical vs actual magnitudes
print(f"\nTheoretical magnitudes (for N={N}):")
print(f"  10 Hz component: amplitude = 2 → FFT magnitude ≈ {2*N/2:.0f}")
print(f"  30 Hz component: amplitude = 1 → FFT magnitude ≈ {1*N/2:.0f}")

# Phase analysis
phase = np.angle(X)
print(f"\nPhase at detected frequencies:")
for idx in peaks_idx:
    print(f"  f = {pos_freqs[idx]:.1f} Hz, phase = {phase[idx]:.2f} rad ({np.degrees(phase[idx]):.1f}°)")

# Create a detailed spectrum plot
plt.figure(figsize=(12, 8))

# Magnitude spectrum
plt.subplot(2, 1, 1)
plt.plot(pos_freqs, pos_mag, 'b-', linewidth=2)
plt.stem(peak_freqs, peak_mags, linefmt='r-', markerfmt='ro', basefmt=' ')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X(f)|')
plt.title('Magnitude Spectrum of Two-Tone Signal')
plt.grid(True, alpha=0.3)
plt.xlim([0, 50])

# Add annotations
for f, m in zip(peak_freqs, peak_mags):
    plt.annotate(f'{f:.0f} Hz', xy=(f, m), xytext=(f+2, m+5),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

# Phase spectrum
plt.subplot(2, 1, 2)
# Only plot phase where magnitude is significant
significant = pos_mag > threshold
plt.stem(pos_freqs[significant], phase[:N//2][significant], 
         linefmt='g-', markerfmt='go', basefmt='k-')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (rad)')
plt.title('Phase Spectrum (at significant frequencies)')
plt.grid(True, alpha=0.3)
plt.xlim([0, 50])
plt.ylim([-np.pi, np.pi])

plt.tight_layout()
plt.show()

# Compare with different window lengths
print("\n\nEffect of Signal Duration on Frequency Resolution:")
print("-" * 50)
for duration in [0.1, 0.5, 1.0, 5.0]:
    N = int(duration * fs)
    freq_res = fs / N
    print(f"Duration = {duration}s, N = {N} samples, Δf = {freq_res:.2f} Hz")

"""
==================== RUN RESULTS ====================
FFT Analysis of Two-Tone Signal
==================================================

Signal parameters:
  Sampling frequency: 100 Hz
  Sampling period: 0.01 seconds
  Signal duration: 0.5 seconds
  Number of samples: 50
  Frequency resolution: 2.00 Hz

Detected frequency components:
  f = 10.0 Hz, magnitude = 50.0
  f = 30.0 Hz, magnitude = 25.0

Theoretical magnitudes (for N=50):
  10 Hz component: amplitude = 2 → FFT magnitude ≈ 50
  30 Hz component: amplitude = 1 → FFT magnitude ≈ 25

Phase at detected frequencies:
  f = 10.0 Hz, phase = -1.57 rad (-90.0°)
  f = 30.0 Hz, phase = -1.57 rad (-90.0°)


Effect of Signal Duration on Frequency Resolution:
--------------------------------------------------
Duration = 0.1s, N = 10 samples, Δf = 10.00 Hz
Duration = 0.5s, N = 50 samples, Δf = 2.00 Hz
Duration = 1.0s, N = 100 samples, Δf = 1.00 Hz
Duration = 5.0s, N = 500 samples, Δf = 0.20 Hz

KEY INSIGHTS:
1. FFT correctly identifies both frequency components: 10 Hz and 30 Hz
2. Magnitude ratio is 2:1, matching the amplitude ratio in the time domain
3. Both components show -90° phase, confirming they are sine (not cosine) waves
4. Frequency resolution Δf = fs/N depends on signal duration
5. With 0.5s signal: Δf = 2 Hz, adequate to separate 10 Hz and 30 Hz
6. Longer signals provide better frequency resolution
7. FFT magnitude = (signal amplitude) × N/2 for pure sinusoids
8. No spectral leakage since frequencies align with FFT bins
9. Nyquist criterion satisfied: fs = 100 Hz > 2×30 Hz = 60 Hz
10. Clean spectrum demonstrates proper sampling and no aliasing

The plots show:
- Time domain: Combined waveform of two sinusoids
- Frequency domain: Two distinct peaks at component frequencies
- Effect of signal duration on frequency resolution
==================== END OF RESULTS ====================
"""