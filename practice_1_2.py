import numpy as np
import matplotlib.pyplot as plt

"""
******************************************************************
[Practice 1-2] Sketch signal x[n] = sin(w_c * n) / (π * n)
where w_c = 0.2π, -30 ≤ n ≤ 30
******************************************************************

Complete Problem Statement:
Sketch the discrete-time signal:
    x[n] = sin(w_c * n) / (π * n)
where w_c = 0.2π and n ranges from -30 to 30.

Mathematical Expression (LaTeX):
    x[n] = \frac{\sin(w_c \cdot n)}{\pi \cdot n}
where w_c = 0.2\pi, and -30 \leq n \leq 30

Note: Special handling required at n = 0 to avoid division by zero.
Using L'Hôpital's rule: lim_{n→0} sin(w_c*n)/(π*n) = w_c/π
"""

# Define parameters
w_c = 0.2 * np.pi  # cutoff frequency
n = np.arange(-30, 31)  # n from -30 to 30

# Calculate x[n]
# Need to handle the case when n = 0 (avoid division by zero)
x = np.zeros_like(n, dtype=float)

# For n ≠ 0
non_zero_idx = n != 0
x[non_zero_idx] = np.sin(w_c * n[non_zero_idx]) / (np.pi * n[non_zero_idx])

# For n = 0, use L'Hôpital's rule: lim(n→0) sin(w_c*n)/(π*n) = w_c/π
zero_idx = n == 0
x[zero_idx] = w_c / np.pi

# Create the plot
plt.figure(figsize=(12, 6))

# Stem plot
plt.subplot(1, 2, 1)
plt.stem(n, x, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Discrete-Time Signal: x[n] = sin(0.2πn)/(πn)')
plt.grid(True, alpha=0.3)
plt.xlim(-35, 35)

# Also create a line plot for better visualization
plt.subplot(1, 2, 2)
plt.plot(n, x, 'b.-', markersize=4)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Line Plot of x[n]')
plt.grid(True, alpha=0.3)
plt.xlim(-35, 35)

plt.tight_layout()
plt.show()

# Print some key values
print(f"x[0] = {x[30]:.4f} (at n=0)")
print(f"Maximum value: {np.max(x):.4f}")
print(f"Minimum value: {np.min(x):.4f}")

# This is a sinc function (sin(x)/x pattern) - fundamental in DSP!
print("\nNote: This is a discrete sinc function, which is the impulse response")
print("of an ideal low-pass filter with cutoff frequency w_c = 0.2π")

"""
==================== RUN RESULTS ====================
x[0] = 0.2000 (at n=0)
Maximum value: 0.2000
Minimum value: -0.0432

Note: This is a discrete sinc function, which is the impulse response
of an ideal low-pass filter with cutoff frequency w_c = 0.2π

KEY INSIGHTS:
1. The signal is a discrete sinc function, fundamental in DSP
2. Peak value occurs at n=0 with amplitude w_c/π = 0.2
3. The function exhibits oscillatory decay as |n| increases
4. This represents the impulse response of an ideal low-pass filter
5. Zero crossings occur at multiples of π/w_c = 5

The plots show:
- Stem plot: discrete nature of the signal
- Line plot: envelope of the sinc function
==================== END OF RESULTS ====================
"""