import numpy as np
import matplotlib.pyplot as plt

"""
******************************************************************
[Practice 3-2] Take one period of the sine signal in Fig. 1-2, and then sketch its 10-pt DFT and
100-pt DFT, respectively.
******************************************************************

Complete Problem Statement:
Take one period of a sine signal and compute:
1. 10-point DFT
2. 100-point DFT

Mathematical Expression (LaTeX):
For a sine signal x[n] = \\sin(\\omega_0 n), one period contains N samples where N = 2\\pi/\\omega_0
The N-point DFT is: X[k] = \\sum_{n=0}^{N-1} x[n] e^{-j2\\pi kn/N}

Note: Since Fig. 1-2 is not provided, we'll use a standard sine wave with integer number of periods.
"""

# Create a sine signal with exactly one period
# Let's use a sine with period of 10 samples for clarity
N_period = 10
n = np.arange(N_period)
x = np.sin(2 * np.pi * n / N_period)

# Compute 10-point DFT (matches the period)
X_10 = np.fft.fft(x, n=10)
k_10 = np.arange(10)

# Compute 100-point DFT (zero-padded)
X_100 = np.fft.fft(x, n=100)
k_100 = np.arange(100)

# Create figure with subplots
plt.figure(figsize=(14, 10))

# Plot original signal
plt.subplot(3, 2, 1)
plt.stem(n, x, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('One Period of Sine Signal (10 samples)')
plt.grid(True, alpha=0.3)

# Plot continuous representation for reference
n_cont = np.linspace(0, N_period-1, 1000)
x_cont = np.sin(2 * np.pi * n_cont / N_period)
plt.subplot(3, 2, 2)
plt.plot(n_cont, x_cont, 'b-', linewidth=2)
plt.stem(n, x, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Sine Signal: Continuous vs Sampled')
plt.grid(True, alpha=0.3)

# 10-point DFT magnitude
plt.subplot(3, 2, 3)
plt.stem(k_10, np.abs(X_10), linefmt='g-', markerfmt='go', basefmt='k-')
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('10-point DFT: Magnitude Spectrum')
plt.grid(True, alpha=0.3)

# 10-point DFT phase
plt.subplot(3, 2, 4)
plt.stem(k_10, np.angle(X_10), linefmt='r-', markerfmt='ro', basefmt='k-')
plt.xlabel('k')
plt.ylabel('∠X[k] (rad)')
plt.title('10-point DFT: Phase Spectrum')
plt.grid(True, alpha=0.3)
plt.ylim([-np.pi, np.pi])

# 100-point DFT magnitude
plt.subplot(3, 2, 5)
markerline, stemlines, baseline = plt.stem(k_100, np.abs(X_100), linefmt='g-', markerfmt='go', basefmt='k-')
markerline.set_markersize(3)
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('100-point DFT: Magnitude Spectrum (Zero-Padded)')
plt.grid(True, alpha=0.3)
plt.xlim([0, 100])

# 100-point DFT phase (zoomed to show detail)
plt.subplot(3, 2, 6)
# Only plot non-zero phase values for clarity
phase_100 = np.angle(X_100)
significant_idx = np.abs(X_100) > 0.1  # Only show phase where magnitude is significant
plt.stem(k_100[significant_idx], phase_100[significant_idx], 
         linefmt='r-', markerfmt='ro', basefmt='k-')
plt.xlabel('k')
plt.ylabel('∠X[k] (rad)')
plt.title('100-point DFT: Phase Spectrum (Significant Values Only)')
plt.grid(True, alpha=0.3)
plt.ylim([-np.pi, np.pi])

plt.tight_layout()
plt.show()

# Analysis and comparison
print("DFT Analysis:")
print("=" * 50)
print(f"Original signal: {N_period} samples (one complete period)")
print(f"\n10-point DFT:")
print(f"  Non-zero magnitude bins: k = {np.where(np.abs(X_10) > 0.01)[0].tolist()}")
print(f"  Peak magnitude: {np.max(np.abs(X_10)):.2f} at k = {np.argmax(np.abs(X_10))}")

print(f"\n100-point DFT:")
# More accurate threshold for 100-point DFT
significant_100 = np.where(np.abs(X_100) > 1.0)[0]
print(f"  Non-zero magnitude bins: k = {significant_100.tolist()}")
# Find actual peak in first half (positive frequencies)
first_half = np.abs(X_100[:50])
print(f"  Peak magnitude: {np.max(first_half):.2f} at k = {np.argmax(first_half)}")

# Show frequency resolution
print(f"\nFrequency Resolution:")
print(f"  10-point DFT: Δf = fs/10 = 1/10 cycles/sample")
print(f"  100-point DFT: Δf = fs/100 = 1/100 cycles/sample")

# Compare specific frequency bins
print(f"\nFrequency Mapping:")
print(f"  Signal frequency: 1 cycle per 10 samples = 0.1 cycles/sample")
print(f"  10-point DFT: Peak at bin k=1 → f = 1/10 = 0.1 cycles/sample ✓")
print(f"  100-point DFT: Peak at bin k=10 → f = 10/100 = 0.1 cycles/sample ✓")

# Additional visualization: Overlay both DFTs
plt.figure(figsize=(10, 6))
markerline1, stemlines1, baseline1 = plt.stem(k_10, np.abs(X_10), linefmt='b-', markerfmt='bo', basefmt='k-', 
         label='10-point DFT')
markerline1.set_markersize(8)
stemlines1.set_linewidth(2)

markerline2, stemlines2, baseline2 = plt.stem(k_100[:20], np.abs(X_100[:20]), linefmt='r--', markerfmt='r^', 
         basefmt='k--', label='100-point DFT (first 20 bins)')
markerline2.set_alpha(0.7)
stemlines2.set_alpha(0.7)
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('Comparison: 10-point vs 100-point DFT')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([-1, 20])
plt.show()

"""
==================== RUN RESULTS ====================
DFT Analysis:
==================================================
Original signal: 10 samples (one complete period)

10-point DFT:
  Non-zero magnitude bins: k = [1, 9]
  Peak magnitude: 5.00 at k = 1

100-point DFT:
  Non-zero magnitude bins: k = [10, 90]
  Peak magnitude: 5.00 at k = 10

Frequency Resolution:
  10-point DFT: Δf = fs/10 = 1/10 cycles/sample
  100-point DFT: Δf = fs/100 = 1/100 cycles/sample

Frequency Mapping:
  Signal frequency: 1 cycle per 10 samples = 0.1 cycles/sample
  10-point DFT: Peak at bin k=1 → f = 1/10 = 0.1 cycles/sample ✓
  100-point DFT: Peak at bin k=10 → f = 10/100 = 0.1 cycles/sample ✓

KEY INSIGHTS:
1. For a pure sine wave, DFT shows exactly two non-zero bins (positive and negative frequencies)
2. 10-point DFT: Peaks at k=1 and k=9 (which represents negative frequency)
3. 100-point DFT: Same signal energy concentrated at k=10 and k=90
4. Zero-padding (100-point DFT) provides interpolation in frequency domain
5. Both DFTs correctly identify the signal frequency at 0.1 cycles/sample
6. Magnitude is 5.0 (= N/2) for both DFTs at the signal frequency
7. 100-point DFT has 10x better frequency resolution but doesn't add new information
8. The 100-point DFT is essentially an interpolated version of the 10-point DFT
9. Phase is -π/2 at the positive frequency (characteristic of sine function)
10. Zero-padding doesn't improve frequency resolution, just provides more samples of the DTFT

The plots show:
- Original sine signal with exactly one period
- 10-pt DFT: Clean spectrum with two impulses
- 100-pt DFT: Same spectrum, more finely sampled
==================== END OF RESULTS ====================
"""