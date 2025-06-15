import numpy as np
import matplotlib.pyplot as plt

# Practice 1-3: Let x(t) = 2sin(2π×10t) + sin(2π×30t)
# (a) Sketch x(t) for 0 ≤ t ≤ 0.5 sec
# (b) Sample x(t) at fs = 100 Hz, sketch x[n] for 0 ≤ n ≤ 50

# Define the continuous-time signal
def x_continuous(t):
    return 2 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 30 * t)

# Part (a): Continuous signal
t_continuous = np.linspace(0, 0.5, 1000)  # 1000 points for smooth curve
x_t = x_continuous(t_continuous)

# Part (b): Sampled signal
fs = 100  # Sampling frequency in Hz
Ts = 1/fs  # Sampling period
n = np.arange(0, 51)  # n from 0 to 50
t_sampled = n * Ts  # Actual time values for samples
x_n = x_continuous(t_sampled)

# Create figure with subplots
plt.figure(figsize=(14, 10))

# Plot 1: Continuous signal
plt.subplot(3, 1, 1)
plt.plot(t_continuous, x_t, 'b-', linewidth=1.5)
plt.xlabel('Time (seconds)')
plt.ylabel('x(t)')
plt.title('(a) Continuous Signal: x(t) = 2sin(2π×10t) + sin(2π×30t)')
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.5)

# Plot 2: Sampled signal (stem plot)
plt.subplot(3, 1, 2)
plt.stem(t_sampled, x_n, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.xlabel('Time (seconds)')
plt.ylabel('x[n]')
plt.title(f'(b) Sampled Signal at fs = {fs} Hz')
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.5)

# Plot 3: Both signals overlaid for comparison
plt.subplot(3, 1, 3)
plt.plot(t_continuous, x_t, 'b-', linewidth=1.5, alpha=0.7, label='Continuous x(t)')
plt.stem(t_sampled, x_n, linefmt='r-', markerfmt='ro', basefmt='k-', label='Sampled x[n]')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Comparison: Continuous vs Sampled Signal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.5)

plt.tight_layout()
plt.show()

# Analysis
print("Signal Analysis:")
print(f"Component 1: 10 Hz sinusoid with amplitude 2")
print(f"Component 2: 30 Hz sinusoid with amplitude 1")
print(f"Sampling frequency: {fs} Hz")
print(f"Sampling period: {Ts:.4f} seconds")
print(f"Nyquist frequency: {fs/2} Hz")
print("\nSampling adequacy:")
print(f"Highest frequency component: 30 Hz")
print(f"Nyquist criterion satisfied? {30 < fs/2} (30 < {fs/2})")
print(f"\nNumber of samples in 0.5 seconds: {int(0.5 * fs)}")

# Additional analysis: Show frequency content
plt.figure(figsize=(10, 6))

# Compute FFT of the sampled signal
# Extend the signal to get better frequency resolution
n_extended = np.arange(0, 200)
t_extended = n_extended * Ts
x_extended = x_continuous(t_extended)

# FFT
fft_result = np.fft.fft(x_extended)
frequencies = np.fft.fftfreq(len(x_extended), Ts)

# Only plot positive frequencies
positive_freq_idx = frequencies >= 0
frequencies_positive = frequencies[positive_freq_idx]
magnitude = np.abs(fft_result[positive_freq_idx])

plt.stem(frequencies_positive[:50], magnitude[:50], linefmt='g-', markerfmt='go', basefmt='k-')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Sampled Signal')
plt.grid(True, alpha=0.3)
plt.xlim(0, 50)
plt.axvline(x=10, color='r', linestyle='--', alpha=0.5, label='10 Hz component')
plt.axvline(x=30, color='r', linestyle='--', alpha=0.5, label='30 Hz component')
plt.legend()

plt.tight_layout()
plt.show()

"""
==================== RUN RESULTS ====================
Signal Analysis:
Component 1: 10 Hz sinusoid with amplitude 2
Component 2: 30 Hz sinusoid with amplitude 1
Sampling frequency: 100 Hz
Sampling period: 0.0100 seconds
Nyquist frequency: 50.0 Hz

Sampling adequacy:
Highest frequency component: 30 Hz
Nyquist criterion satisfied? True (30 < 50.0)

Number of samples in 0.5 seconds: 50

KEY INSIGHTS:
1. The signal contains two frequency components: 10 Hz and 30 Hz
2. Sampling at 100 Hz satisfies Nyquist criterion (fs > 2*fmax = 60 Hz)
3. No aliasing occurs since all frequencies are below Nyquist frequency
4. The combined signal shows beat-like pattern due to frequency interaction
5. FFT clearly shows the two frequency peaks at 10 Hz and 30 Hz
6. The sampled signal faithfully represents the continuous signal

The plots demonstrate:
- Continuous signal: smooth combination of two sinusoids
- Sampled signal: discrete points capturing the waveform
- Frequency spectrum: clear peaks at component frequencies
==================== END OF RESULTS ====================
"""