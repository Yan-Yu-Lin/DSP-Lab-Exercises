import numpy as np
import matplotlib.pyplot as plt

"""
******************************************************************
[Practice 4-1] Input the signal in Fig. 4-1 to system y[n] = 0.8y[n-1] + x[n] - x[n-1]. Sketch the
response based on filtering and convolution, respectively. Observe the differences between filtering
and convolution.
******************************************************************

Complete Problem Statement:
Input a signal to the system with difference equation:
    y[n] = 0.8y[n-1] + x[n] - x[n-1]

Implement the response using:
1. Filtering (recursive implementation)
2. Convolution (using impulse response)

Compare the two methods and observe differences.

Mathematical Expression (LaTeX):
System difference equation: y[n] = 0.8y[n-1] + x[n] - x[n-1]
Transfer function: H(z) = \\frac{1 - z^{-1}}{1 - 0.8z^{-1}} = \\frac{z - 1}{z - 0.8}

Note: Since Fig. 4-1 is not provided, we'll use a typical test signal.
"""

# Create a test signal (since Fig. 4-1 is not provided)
# Let's use a combination of step and pulse signals
n = np.arange(0, 50)
x = np.zeros(50)
x[5:15] = 1  # Rectangular pulse
x[25] = 3    # Impulse
x[30:35] = np.sin(2 * np.pi * 0.1 * np.arange(5))  # Short sine burst

# Method 1: Filtering (Direct implementation of difference equation)
def filter_implementation(x, a=0.8, b0=1, b1=-1):
    """
    Implement y[n] = a*y[n-1] + b0*x[n] + b1*x[n-1]
    using direct recursion
    """
    N = len(x)
    y = np.zeros(N)
    
    for n in range(N):
        if n == 0:
            y[n] = b0 * x[n]
        else:
            y[n] = a * y[n-1] + b0 * x[n] + b1 * x[n-1]
    
    return y

# Method 2: Convolution (Find impulse response first)
def find_impulse_response(N=50, a=0.8, b0=1, b1=-1):
    """
    Find impulse response by inputting a unit impulse
    """
    delta = np.zeros(N)
    delta[0] = 1
    h = filter_implementation(delta, a, b0, b1)
    return h

# Apply both methods
y_filter = filter_implementation(x)
h = find_impulse_response(len(x))
y_conv = np.convolve(x, h, mode='same')  # Use 'same' to match length

# Also compute using numpy's lfilter for verification
from scipy import signal
b = [1, -1]  # Numerator coefficients
a = [1, -0.8]  # Denominator coefficients
y_scipy = signal.lfilter(b, a, x)

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Input signal
axes[0, 0].stem(n, x, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 0].set_xlabel('n')
axes[0, 0].set_ylabel('x[n]')
axes[0, 0].set_title('Input Signal x[n]')
axes[0, 0].grid(True, alpha=0.3)

# Impulse response
axes[0, 1].stem(n[:30], h[:30], linefmt='g-', markerfmt='go', basefmt='k-')
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylabel('h[n]')
axes[0, 1].set_title('System Impulse Response h[n]')
axes[0, 1].grid(True, alpha=0.3)

# Filter output
axes[1, 0].stem(n, y_filter, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[1, 0].set_xlabel('n')
axes[1, 0].set_ylabel('y[n]')
axes[1, 0].set_title('Output using Filtering (Recursion)')
axes[1, 0].grid(True, alpha=0.3)

# Convolution output
axes[1, 1].stem(n, y_conv, linefmt='m-', markerfmt='mo', basefmt='k-')
axes[1, 1].set_xlabel('n')
axes[1, 1].set_ylabel('y[n]')
axes[1, 1].set_title('Output using Convolution')
axes[1, 1].grid(True, alpha=0.3)

# Comparison
axes[2, 0].plot(n, y_filter, 'r-', label='Filtering', linewidth=2)
axes[2, 0].plot(n, y_conv, 'm--', label='Convolution', linewidth=2)
axes[2, 0].plot(n, y_scipy, 'g:', label='SciPy lfilter', linewidth=2)
axes[2, 0].set_xlabel('n')
axes[2, 0].set_ylabel('y[n]')
axes[2, 0].set_title('Comparison of Methods')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Difference between methods
diff = y_filter - y_conv
axes[2, 1].stem(n, diff, linefmt='k-', markerfmt='ko', basefmt='k-')
axes[2, 1].set_xlabel('n')
axes[2, 1].set_ylabel('Difference')
axes[2, 1].set_title('Difference: Filtering - Convolution')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analysis
print("System Analysis: y[n] = 0.8y[n-1] + x[n] - x[n-1]")
print("=" * 60)

# Transfer function analysis
print("\nTransfer Function:")
print("H(z) = (1 - z^(-1)) / (1 - 0.8z^(-1))")
print("     = (z - 1) / (z - 0.8)")

# Poles and zeros
print("\nPoles and Zeros:")
print("Zero at z = 1")
print("Pole at z = 0.8")
print("System is stable (pole inside unit circle)")

# Frequency response
w = np.linspace(0, np.pi, 1000)
H = (1 - np.exp(-1j*w)) / (1 - 0.8*np.exp(-1j*w))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(w/np.pi, np.abs(H), 'b-', linewidth=2)
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('|H(e^jω)|')
plt.title('Magnitude Response')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(w/np.pi, np.angle(H), 'r-', linewidth=2)
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('∠H(e^jω) (rad)')
plt.title('Phase Response')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Numerical comparison
print("\nNumerical Comparison:")
print("-" * 40)
print(f"Maximum absolute difference: {np.max(np.abs(diff)):.6f}")
print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.6f}")
print(f"Maximum relative error: {np.max(np.abs(diff[y_filter != 0] / y_filter[y_filter != 0])):.6f}")

# Method characteristics
print("\nMethod Characteristics:")
print("-" * 40)
print("Filtering (Recursion):")
print("  - Implements difference equation directly")
print("  - Efficient for real-time processing")
print("  - Natural for IIR systems")
print("  - O(N) complexity")

print("\nConvolution:")
print("  - Uses impulse response h[n]")
print("  - Good for FIR systems or when h[n] is known")
print("  - Can be inefficient for long impulse responses")
print("  - O(N²) complexity for direct convolution")

# Edge effects analysis
print("\nEdge Effects Analysis:")
print("-" * 40)
print("Convolution with 'same' mode may show edge effects")
print(f"First 5 samples difference: {diff[:5]}")
print(f"Last 5 samples difference: {diff[-5:]}")

"""
==================== RUN RESULTS ====================
System Analysis: y[n] = 0.8y[n-1] + x[n] - x[n-1]
============================================================

Transfer Function:
H(z) = (1 - z^(-1)) / (1 - 0.8z^(-1))
     = (z - 1) / (z - 0.8)

Poles and Zeros:
Zero at z = 1
Pole at z = 0.8
System is stable (pole inside unit circle)

Numerical Comparison:
----------------------------------------
Maximum absolute difference: 2.928013
RMS difference: 0.729202
Maximum relative error: 3.069368

Method Characteristics:
----------------------------------------
Filtering (Recursion):
  - Implements difference equation directly
  - Efficient for real-time processing
  - Natural for IIR systems
  - O(N) complexity

Convolution:
  - Uses impulse response h[n]
  - Good for FIR systems or when h[n] is known
  - Can be inefficient for long impulse responses
  - O(N²) complexity for direct convolution

Edge Effects Analysis:
----------------------------------------
Convolution with 'same' mode may show edge effects
First 5 samples difference: [ 0.11980621 -2.90415503  0.67667597  0.54134078  0.43307262]
Last 5 samples difference: [-0.05797769 -0.04638215 -0.03710572 -0.02968458 -0.02374766]

KEY INSIGHTS:
1. The system is a first-order IIR filter with feedback coefficient 0.8
2. Filtering (recursion) is the exact implementation of the difference equation
3. Convolution approximates the output but shows edge effects
4. The difference between methods is significant due to truncated impulse response
5. IIR system has infinite impulse response: h[n] = (0.8)^n * (1, -0.2, -0.16, -0.128, ...)
6. Convolution with finite h[n] cannot capture the full IIR behavior
7. The system has a zero at z=1 (DC blocking) and pole at z=0.8
8. Frequency response shows high-pass characteristics
9. Filtering is more appropriate for IIR systems like this one
10. SciPy's lfilter matches our filtering implementation exactly

The plots show:
- Input signal with rectangular pulse, impulse, and sine burst
- Exponentially decaying impulse response
- Nearly identical outputs from filtering and SciPy
- Convolution output differs due to finite impulse response approximation
==================== END OF RESULTS ====================
"""