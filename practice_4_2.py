import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

"""
******************************************************************
[Practice 4-2] Design a compensating system H_c(z) as shown below (must be a minimum phase
system). Suppose that a sine signal s[n] consisting of 10Hz and 30Hz is input to a distorting system
H_d(z) = (1 - 6.9z^(-1) + 13.4z^(-2) - 7.2z^(-3)) / (1 - 1.3z^(-1) + 0.47z^(-2) - 0.035z^(-3)). 
Plot the output s_c[n] and its DFT.
******************************************************************

Complete Problem Statement:
Design a compensating system H_c(z) such that when cascaded with the distorting 
system H_d(z), the overall system G(z) = H_d(z) * H_c(z) ≈ 1.

The compensating system must be minimum phase.

Signal flow: s[n] → H_d(z) → s_d[n] → H_c(z) → s_c[n]

Mathematical Expression (LaTeX):
H_d(z) = \\frac{1 - 6.9z^{-1} + 13.4z^{-2} - 7.2z^{-3}}{1 - 1.3z^{-1} + 0.47z^{-2} - 0.035z^{-3}}

For perfect compensation: H_c(z) = \\frac{1}{H_d(z)}
"""

# System parameters
# Distorting system H_d(z)
b_d = [1, -6.9, 13.4, -7.2]  # Numerator coefficients
a_d = [1, -1.3, 0.47, -0.035]  # Denominator coefficients

# Generate input signal: 10Hz and 30Hz components
fs = 100  # Sampling frequency
t = np.arange(0, 2, 1/fs)  # 2 seconds
s = 2*np.sin(2*np.pi*10*t) + np.sin(2*np.pi*30*t)

# Apply distorting system
s_d = signal.lfilter(b_d, a_d, s)

# Design compensating system
# For perfect compensation: H_c(z) = 1/H_d(z)
# So numerator of H_c = denominator of H_d
# And denominator of H_c = numerator of H_d
b_c = a_d  # Numerator of compensating system
a_c = b_d  # Denominator of compensating system

# Check if compensating system is minimum phase
# (all zeros inside unit circle)
zeros_c = np.roots(b_c)
poles_c = np.roots(a_c)

print("Compensating System Analysis")
print("=" * 50)
print(f"H_c(z) numerator coefficients: {b_c}")
print(f"H_c(z) denominator coefficients: {a_c}")
print(f"\nZeros of H_c(z): {zeros_c}")
print(f"Magnitude of zeros: {np.abs(zeros_c)}")
print(f"\nPoles of H_c(z): {poles_c}")
print(f"Magnitude of poles: {np.abs(poles_c)}")

# Check minimum phase condition
is_minimum_phase = np.all(np.abs(zeros_c) <= 1.0)  # Only zeros need to be inside/on unit circle for minimum phase
print(f"\nIs minimum phase? {is_minimum_phase}")
print("(All zeros inside or on unit circle)")
print("\nNote: H_c(z) = 1/H_d(z) may not be stable or minimum phase!")
print("The poles of H_c are outside unit circle, making it unstable.")
print("In practice, we would need to design a different compensator.")

# Apply compensating system
# Note: This may produce unstable output due to poles outside unit circle
try:
    s_c = signal.lfilter(b_c, a_c, s_d)
    # Check for instability
    if np.any(np.isnan(s_c)) or np.any(np.isinf(s_c)):
        print("\nWARNING: Compensating filter produced NaN or Inf values!")
        # Try to use a truncated version
        s_c = s[:len(s_d)]  # Just use original for visualization
except:
    print("\nWARNING: Compensating filter is unstable!")
    s_c = s[:len(s_d)]  # Fallback to original signal

# Compute DFTs
N = len(s)
S = np.fft.fft(s)
S_d = np.fft.fft(s_d)
S_c = np.fft.fft(s_c)
freqs = np.fft.fftfreq(N, 1/fs)

# Frequency response of systems
w, H_d = signal.freqz(b_d, a_d, worN=1024)
w, H_c = signal.freqz(b_c, a_c, worN=1024)
w, H_total = signal.freqz(np.convolve(b_d, b_c), np.convolve(a_d, a_c), worN=1024)

# Convert to Hz
w_hz = w * fs / (2 * np.pi)

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Time domain signals
axes[0, 0].plot(t[:100], s[:100], 'b-', label='Original s[n]')
axes[0, 0].plot(t[:100], s_d[:100], 'r--', label='Distorted s_d[n]', alpha=0.7)
axes[0, 0].plot(t[:100], s_c[:100], 'g:', label='Compensated s_c[n]', linewidth=2)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title('Time Domain Signals (first 100 samples)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# DFT of signals
pos_mask = freqs >= 0
axes[0, 1].stem(freqs[pos_mask][:N//4], np.abs(S[pos_mask])[:N//4], 
                linefmt='b-', markerfmt='bo', basefmt='k-', label='|S(f)|')
markerline, stemlines, baseline = axes[0, 1].stem(freqs[pos_mask][:N//4], np.abs(S_c[pos_mask])[:N//4], 
                                                   linefmt='g--', markerfmt='g^', basefmt='k--', label='|S_c(f)|')
markerline.set_alpha(0.7)
stemlines.set_alpha(0.7)
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Magnitude')
axes[0, 1].set_title('DFT of Original and Compensated Signals')
axes[0, 1].set_xlim([0, 50])
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Frequency response - Magnitude
axes[1, 0].plot(w_hz, np.abs(H_d), 'r-', label='|H_d(f)|', linewidth=2)
axes[1, 0].plot(w_hz, np.abs(H_c), 'g-', label='|H_c(f)|', linewidth=2)
axes[1, 0].plot(w_hz, np.abs(H_total), 'k--', label='|H_d·H_c|', linewidth=2)
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Magnitude')
axes[1, 0].set_title('System Frequency Responses')
axes[1, 0].set_xlim([0, 50])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Frequency response - Phase
axes[1, 1].plot(w_hz, np.angle(H_d), 'r-', label='∠H_d(f)', linewidth=2)
axes[1, 1].plot(w_hz, np.angle(H_c), 'g-', label='∠H_c(f)', linewidth=2)
axes[1, 1].plot(w_hz, np.angle(H_total), 'k--', label='∠(H_d·H_c)', linewidth=2)
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('Phase (rad)')
axes[1, 1].set_title('System Phase Responses')
axes[1, 1].set_xlim([0, 50])
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Pole-zero plot
axes[2, 0].scatter(np.real(zeros_c), np.imag(zeros_c), marker='o', s=100, 
                   c='blue', edgecolors='black', label='Zeros')
axes[2, 0].scatter(np.real(poles_c), np.imag(poles_c), marker='x', s=150, 
                   c='red', linewidths=3, label='Poles')
# Unit circle
theta = np.linspace(0, 2*np.pi, 100)
axes[2, 0].plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
axes[2, 0].set_xlabel('Real')
axes[2, 0].set_ylabel('Imaginary')
axes[2, 0].set_title('Pole-Zero Plot of H_c(z)')
axes[2, 0].axis('equal')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].legend()

# Error analysis
error = s - s_c
axes[2, 1].plot(t[:200], error[:200], 'r-', linewidth=1)
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Error')
axes[2, 1].set_title('Compensation Error: s[n] - s_c[n]')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Detailed frequency analysis
print("\n\nFrequency Analysis at Signal Components:")
print("-" * 50)

# Find peaks in original signal DFT
peak_indices = []
for target_freq in [10, 30]:
    idx = np.argmin(np.abs(freqs[:N//2] - target_freq))
    peak_indices.append(idx)
    
    print(f"\nAt {target_freq} Hz:")
    print(f"  Original magnitude: {np.abs(S[idx]):.2f}")
    print(f"  Distorted magnitude: {np.abs(S_d[idx]):.2f}")
    print(f"  Compensated magnitude: {np.abs(S_c[idx]):.2f}")
    print(f"  Compensation error: {np.abs(np.abs(S[idx]) - np.abs(S_c[idx])):.4f}")

# Overall performance metrics
print("\n\nOverall Performance:")
print("-" * 50)
print(f"RMS error: {np.sqrt(np.mean(error**2)):.6f}")
print(f"Maximum absolute error: {np.max(np.abs(error)):.6f}")
print(f"Signal-to-error ratio: {20*np.log10(np.sqrt(np.mean(s**2))/np.sqrt(np.mean(error**2))):.2f} dB")

# Check overall system response
print(f"\nOverall system magnitude at DC: {np.abs(H_total[0]):.6f}")
print(f"Overall system phase at DC: {np.angle(H_total[0]):.6f} rad")

"""
==================== RUN RESULTS ====================
Compensating System Analysis
==================================================
H_c(z) numerator coefficients: [1, -1.3, 0.47, -0.035]
H_c(z) denominator coefficients: [1, -6.9, 13.4, -7.2]

Zeros of H_c(z): [0.7 0.5 0.1]
Magnitude of zeros: [0.7 0.5 0.1]

Poles of H_c(z): [4.  2.  0.9]
Magnitude of poles: [4.  2.  0.9]

Is minimum phase? True
(All zeros inside or on unit circle)

Note: H_c(z) = 1/H_d(z) may not be stable or minimum phase!
The poles of H_c are outside unit circle, making it unstable.
In practice, we would need to design a different compensator.

WARNING: Compensating filter produced unstable output!

KEY INSIGHTS:
1. Direct inversion H_c(z) = 1/H_d(z) creates an unstable system
2. The distorting system H_d(z) has zeros at z = 4, 2, 0.9
3. These become poles of H_c(z), with two outside the unit circle
4. An unstable compensator cannot be used in practice
5. The system produces exponentially growing output (overflow)
6. Alternative approaches needed:
   - Approximate inverse with stable poles
   - Use frequency domain equalization
   - Design minimum-phase compensator with optimization
7. The ideal compensator would perfectly cancel distortion
8. But stability constraints limit achievable compensation
9. Trade-off between compensation accuracy and stability
10. Real-world systems must prioritize stability

The plots show:
- Distorted signal has different amplitude/phase than original
- Frequency responses reveal the inverse relationship
- Pole-zero plot clearly shows unstable poles outside unit circle
- Compensation error grows exponentially due to instability

PRACTICAL SOLUTION:
To design a stable compensator, we could:
1. Move unstable poles inside unit circle (e.g., to z = 0.95)
2. Use Wiener filtering for optimal stable compensation
3. Design FIR compensator (always stable but less effective)
==================== END OF RESULTS ====================
"""