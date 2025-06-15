import numpy as np
import matplotlib.pyplot as plt

"""
******************************************************************
[Practice 3-5] Sketch a discrete-time signal that consists of 10Hz and 30Hz sine components based
on a sampling period of 0.02 second. Then, use function fft() to compute its DFT, and compare the
differences between this result and that in Practice 3-4.
******************************************************************

Complete Problem Statement:
Create a signal with the same frequency components as Practice 3-4:
    x(t) = 2sin(2π×10t) + sin(2π×30t)
but sample at fs = 50 Hz (Ts = 0.02 seconds) instead of 100 Hz.

Compare with Practice 3-4 which used fs = 100 Hz (Ts = 0.01 seconds).

Mathematical Expression (LaTeX):
Continuous signal: x(t) = 2\\sin(2\\pi \\cdot 10t) + \\sin(2\\pi \\cdot 30t)
Sampled signal: x[n] = x(nT_s) where T_s = 0.02 seconds
Sampling frequency: f_s = 50 Hz
Nyquist frequency: f_{Nyquist} = f_s/2 = 25 Hz

Note: 30 Hz component is above Nyquist frequency!
"""

# Sampling parameters
fs_new = 50   # New sampling frequency in Hz
fs_old = 100  # Old sampling frequency from Practice 3-4
Ts_new = 1/fs_new  # Sampling period = 0.02 seconds
Ts_old = 1/fs_old  # Sampling period = 0.01 seconds

# Signal duration
duration = 1.0  # seconds

# Time vectors
t_new = np.arange(0, duration, Ts_new)
t_old = np.arange(0, duration, Ts_old)
N_new = len(t_new)
N_old = len(t_old)

# Generate signals
x_new = 2 * np.sin(2 * np.pi * 10 * t_new) + np.sin(2 * np.pi * 30 * t_new)
x_old = 2 * np.sin(2 * np.pi * 10 * t_old) + np.sin(2 * np.pi * 30 * t_old)

# For reference: continuous signal
t_cont = np.linspace(0, 0.1, 1000)  # First 0.1 seconds
x_cont = 2 * np.sin(2 * np.pi * 10 * t_cont) + np.sin(2 * np.pi * 30 * t_cont)

# Compute FFTs
X_new = np.fft.fft(x_new)
X_old = np.fft.fft(x_old)
freqs_new = np.fft.fftfreq(N_new, Ts_new)
freqs_old = np.fft.fftfreq(N_old, Ts_old)

# Create comprehensive comparison figure
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Time domain comparison
axes[0, 0].plot(t_cont, x_cont, 'b-', alpha=0.5, label='Continuous')
axes[0, 0].stem(t_new[:5], x_new[:5], linefmt='r-', markerfmt='ro', 
                basefmt='k-', label=f'fs = {fs_new} Hz')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('x(t)')
axes[0, 0].set_title('Time Domain: fs = 50 Hz (Ts = 0.02s)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t_cont, x_cont, 'b-', alpha=0.5, label='Continuous')
axes[0, 1].stem(t_old[:10], x_old[:10], linefmt='g-', markerfmt='go', 
                basefmt='k-', label=f'fs = {fs_old} Hz')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('x(t)')
axes[0, 1].set_title('Time Domain: fs = 100 Hz (Ts = 0.01s)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Frequency domain - fs = 50 Hz
pos_mask_new = freqs_new >= 0
axes[1, 0].stem(freqs_new[pos_mask_new][:N_new//2], 
                np.abs(X_new[pos_mask_new])[:N_new//2],
                linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('|X(f)|')
axes[1, 0].set_title(f'FFT Magnitude: fs = {fs_new} Hz')
axes[1, 0].set_xlim([0, fs_new/2])
axes[1, 0].axvline(x=fs_new/2, color='k', linestyle='--', alpha=0.5, label='Nyquist')
axes[1, 0].axvline(x=10, color='g', linestyle='--', alpha=0.3)
axes[1, 0].axvline(x=30, color='g', linestyle='--', alpha=0.3)
axes[1, 0].text(10, axes[1, 0].get_ylim()[1]*0.9, '10 Hz', ha='center')
axes[1, 0].text(20, axes[1, 0].get_ylim()[1]*0.8, '30 Hz→20 Hz\n(aliased!)', 
                ha='center', color='red')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Frequency domain - fs = 100 Hz
pos_mask_old = freqs_old >= 0
axes[1, 1].stem(freqs_old[pos_mask_old][:N_old//2], 
                np.abs(X_old[pos_mask_old])[:N_old//2],
                linefmt='g-', markerfmt='go', basefmt='k-')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('|X(f)|')
axes[1, 1].set_title(f'FFT Magnitude: fs = {fs_old} Hz')
axes[1, 1].set_xlim([0, fs_old/2])
axes[1, 1].axvline(x=fs_old/2, color='k', linestyle='--', alpha=0.5, label='Nyquist')
axes[1, 1].axvline(x=10, color='g', linestyle='--', alpha=0.3)
axes[1, 1].axvline(x=30, color='g', linestyle='--', alpha=0.3)
axes[1, 1].text(10, axes[1, 1].get_ylim()[1]*0.9, '10 Hz', ha='center')
axes[1, 1].text(30, axes[1, 1].get_ylim()[1]*0.9, '30 Hz', ha='center')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Direct comparison in frequency domain
axes[2, 0].stem(freqs_new[pos_mask_new][:N_new//2], 
                np.abs(X_new[pos_mask_new])[:N_new//2],
                linefmt='r-', markerfmt='ro', basefmt='k-', label='fs = 50 Hz')
axes[2, 0].set_xlabel('Frequency (Hz)')
axes[2, 0].set_ylabel('|X(f)|')
axes[2, 0].set_title('Aliasing Effect: 30 Hz → 20 Hz')
axes[2, 0].set_xlim([0, 30])
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Phase comparison
phase_new = np.angle(X_new)
phase_old = np.angle(X_old)
# Only plot significant phases
sig_new = np.abs(X_new) > np.max(np.abs(X_new)) * 0.1
sig_old = np.abs(X_old) > np.max(np.abs(X_old)) * 0.1

axes[2, 1].stem(freqs_new[sig_new & pos_mask_new][:10], 
                phase_new[sig_new & pos_mask_new][:10],
                linefmt='r-', markerfmt='ro', basefmt='k-', label='fs = 50 Hz')
axes[2, 1].set_xlabel('Frequency (Hz)')
axes[2, 1].set_ylabel('Phase (rad)')
axes[2, 1].set_title('Phase Spectrum Comparison')
axes[2, 1].set_xlim([0, 30])
axes[2, 1].set_ylim([-np.pi, np.pi])
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Detailed analysis
print("Comparison: fs = 50 Hz vs fs = 100 Hz")
print("=" * 50)

print(f"\nSampling Parameters:")
print(f"  Practice 3-5: fs = {fs_new} Hz, Ts = {Ts_new} s, Nyquist = {fs_new/2} Hz")
print(f"  Practice 3-4: fs = {fs_old} Hz, Ts = {Ts_old} s, Nyquist = {fs_old/2} Hz")

print(f"\nSignal Components:")
print(f"  Component 1: 10 Hz (below both Nyquist frequencies)")
print(f"  Component 2: 30 Hz")
print(f"    - fs = 100 Hz: 30 Hz < 50 Hz Nyquist ✓ No aliasing")
print(f"    - fs = 50 Hz:  30 Hz > 25 Hz Nyquist ✗ ALIASING!")

# Find peaks in both spectra
def find_peaks(X, freqs, threshold=0.1):
    mag = np.abs(X)
    pos_mask = freqs >= 0
    pos_freqs = freqs[pos_mask][:len(X)//2]
    pos_mag = mag[pos_mask][:len(X)//2]
    peaks = np.where(pos_mag > np.max(pos_mag) * threshold)[0]
    return pos_freqs[peaks], pos_mag[peaks]

peaks_new_f, peaks_new_m = find_peaks(X_new, freqs_new)
peaks_old_f, peaks_old_m = find_peaks(X_old, freqs_old)

print(f"\nDetected Frequencies:")
print(f"  fs = 50 Hz:  {peaks_new_f} Hz")
print(f"  fs = 100 Hz: {peaks_old_f} Hz")

print(f"\nAliasing Analysis:")
print(f"  30 Hz sampled at 50 Hz appears at: {fs_new - 30} Hz")
print(f"  This is because: 30 Hz = 50 Hz - 20 Hz")
print(f"  The 30 Hz component is 'folded' back into the spectrum")

# Demonstrate the aliasing formula
print(f"\nAliasing Formula:")
print(f"  For a frequency f > fs/2:")
print(f"  Apparent frequency = |f - k*fs| where k is chosen so result is in [0, fs/2]")
print(f"  30 Hz: |30 - 1*50| = 20 Hz ✓")

# Show reconstructed signals
t_recon = np.linspace(0, 0.2, 1000)
# True signal
x_true = 2 * np.sin(2 * np.pi * 10 * t_recon) + np.sin(2 * np.pi * 30 * t_recon)
# What fs=50Hz "thinks" the signal is
x_aliased = 2 * np.sin(2 * np.pi * 10 * t_recon) + np.sin(2 * np.pi * 20 * t_recon)

plt.figure(figsize=(10, 6))
plt.plot(t_recon, x_true, 'b-', label='True signal (10 Hz + 30 Hz)', linewidth=2)
plt.plot(t_recon, x_aliased, 'r--', label='Aliased interpretation (10 Hz + 20 Hz)', linewidth=2)
plt.plot(t_new[:10], x_new[:10], 'ko', markersize=8, label='fs = 50 Hz samples')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Aliasing Effect: True Signal vs Aliased Interpretation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 0.2])
plt.show()

"""
==================== RUN RESULTS ====================
Comparison: fs = 50 Hz vs fs = 100 Hz
==================================================

Sampling Parameters:
  Practice 3-5: fs = 50 Hz, Ts = 0.02 s, Nyquist = 25.0 Hz
  Practice 3-4: fs = 100 Hz, Ts = 0.01 s, Nyquist = 50.0 Hz

Signal Components:
  Component 1: 10 Hz (below both Nyquist frequencies)
  Component 2: 30 Hz
    - fs = 100 Hz: 30 Hz < 50 Hz Nyquist ✓ No aliasing
    - fs = 50 Hz:  30 Hz > 25 Hz Nyquist ✗ ALIASING!

Detected Frequencies:
  fs = 50 Hz:  [10. 20.] Hz
  fs = 100 Hz: [10. 30.] Hz

Aliasing Analysis:
  30 Hz sampled at 50 Hz appears at: 20 Hz
  This is because: 30 Hz = 50 Hz - 20 Hz
  The 30 Hz component is 'folded' back into the spectrum

Aliasing Formula:
  For a frequency f > fs/2:
  Apparent frequency = |f - k*fs| where k is chosen so result is in [0, fs/2]
  30 Hz: |30 - 1*50| = 20 Hz ✓

KEY INSIGHTS:
1. CRITICAL: fs = 50 Hz violates Nyquist criterion for 30 Hz component
2. The 30 Hz component aliases to 20 Hz in the sampled signal
3. The 10 Hz component is correctly represented in both cases
4. Aliasing makes the signal indistinguishable from 10 Hz + 20 Hz combination
5. The DFT cannot detect that aliasing has occurred
6. Original signal: 2sin(2π×10t) + sin(2π×30t)
7. Aliased interpretation: 2sin(2π×10t) + sin(2π×20t)
8. This is why anti-aliasing filters are crucial before sampling
9. To properly sample this signal, fs must be > 60 Hz
10. The time domain plots show how samples can fit multiple interpretations

Comparison with Practice 3-4:
- Practice 3-4: fs = 100 Hz → Both components correctly identified
- Practice 3-5: fs = 50 Hz → 30 Hz component misidentified as 20 Hz
- This demonstrates the importance of proper sampling rate selection
==================== END OF RESULTS ====================
"""