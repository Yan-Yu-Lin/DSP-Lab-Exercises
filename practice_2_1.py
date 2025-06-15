import numpy as np
import matplotlib.pyplot as plt

"""
******************************************************************
[Practice 2-1] Implement the convolution by equation (2.1) without using function conv() or
convolve().
******************************************************************

Complete Problem Statement:
Implement the discrete convolution operation without using built-in functions conv() or convolve().
The convolution must be computed using the fundamental convolution sum equation.

Mathematical Expression (LaTeX):
The discrete convolution of two sequences x[n] and h[n] is defined as:
    y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]

For finite sequences, the sum limits are adjusted based on the signal lengths.
Output length: len(y) = len(x) + len(h) - 1
"""

def manual_convolution(x, h):
    """
    Manually compute the convolution of two sequences x and h
    using the convolution sum: y[n] = Σ x[k] * h[n-k]
    """
    # Get lengths
    len_x = len(x)
    len_h = len(h)
    
    # Output length is len_x + len_h - 1
    len_y = len_x + len_h - 1
    
    # Initialize output
    y = np.zeros(len_y)
    
    # Compute convolution sum for each output sample
    for n in range(len_y):
        # For each n, compute y[n] = Σ x[k] * h[n-k]
        for k in range(len_x):
            # Check if h[n-k] exists (i.e., 0 <= n-k < len_h)
            if 0 <= n - k < len_h:
                y[n] += x[k] * h[n - k]
    
    return y

# Example signals to test
# Let's create some simple test signals
x = np.array([1, 2, 3])  # Input signal
h = np.array([1, 0.5, 0.25])  # Impulse response

# Compute convolution manually
y_manual = manual_convolution(x, h)

# Verify with numpy's convolve (just to check our implementation)
y_numpy = np.convolve(x, h)

# Plot the results
fig, axes = plt.subplots(4, 1, figsize=(10, 12))

# Plot input signal x[n]
axes[0].stem(range(len(x)), x, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0].set_xlabel('n')
axes[0].set_ylabel('x[n]')
axes[0].set_title('Input Signal x[n]')
axes[0].grid(True, alpha=0.3)

# Plot impulse response h[n]
axes[1].stem(range(len(h)), h, linefmt='g-', markerfmt='go', basefmt='k-')
axes[1].set_xlabel('n')
axes[1].set_ylabel('h[n]')
axes[1].set_title('Impulse Response h[n]')
axes[1].grid(True, alpha=0.3)

# Plot manual convolution result
axes[2].stem(range(len(y_manual)), y_manual, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[2].set_xlabel('n')
axes[2].set_ylabel('y[n]')
axes[2].set_title('Manual Convolution Result y[n] = x[n] * h[n]')
axes[2].grid(True, alpha=0.3)

# Plot comparison
axes[3].stem(range(len(y_manual)), y_manual, linefmt='r-', markerfmt='ro', 
             basefmt='k-', label='Manual')
axes[3].stem(range(len(y_numpy)), y_numpy + 0.05, linefmt='b--', markerfmt='b^', 
             basefmt='k--', label='NumPy (offset for visibility)')
axes[3].set_xlabel('n')
axes[3].set_ylabel('y[n]')
axes[3].set_title('Comparison: Manual vs NumPy Convolution')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print results
print("Manual Convolution Implementation")
print("=" * 50)
print(f"x[n] = {x}")
print(f"h[n] = {h}")
print(f"y[n] (manual) = {y_manual}")
print(f"y[n] (numpy)  = {y_numpy}")
print(f"\nDifference: {np.max(np.abs(y_manual - y_numpy)):.2e}")

# Show step-by-step calculation for educational purposes
print("\n\nStep-by-step calculation:")
print("=" * 50)
for n in range(len(y_manual)):
    print(f"\ny[{n}] = ", end="")
    terms = []
    value = 0
    for k in range(len(x)):
        if 0 <= n - k < len(h):
            terms.append(f"x[{k}]×h[{n-k}]")
            value += x[k] * h[n-k]
    print(" + ".join(terms))
    print(f"     = ", end="")
    
    terms_values = []
    for k in range(len(x)):
        if 0 <= n - k < len(h):
            terms_values.append(f"{x[k]}×{h[n-k]}")
    print(" + ".join(terms_values))
    print(f"     = {value}")

# Another example with different signals
print("\n\nExample 2: Different signals")
print("=" * 50)

# Rectangle and triangle signals
x2 = np.array([1, 1, 1, 1])  # Rectangle
h2 = np.array([1, 2, 1])      # Triangle

y2_manual = manual_convolution(x2, h2)

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.stem(range(len(x2)), x2, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.title('Rectangle Signal x[n]')
plt.ylabel('x[n]')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.stem(range(len(h2)), h2, linefmt='g-', markerfmt='go', basefmt='k-')
plt.title('Triangle Signal h[n]')
plt.ylabel('h[n]')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.stem(range(len(y2_manual)), y2_manual, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title('Convolution Result y[n] = x[n] * h[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Rectangle x[n] = {x2}")
print(f"Triangle h[n] = {h2}")
print(f"Convolution y[n] = {y2_manual}")

"""
==================== RUN RESULTS ====================
Manual Convolution Implementation
==================================================
x[n] = [1 2 3]
h[n] = [1.   0.5  0.25]
y[n] (manual) = [1.   2.5  4.25 2.   0.75]
y[n] (numpy)  = [1.   2.5  4.25 2.   0.75]

Difference: 0.00e+00


Step-by-step calculation:
==================================================

y[0] = x[0]×h[0]
     = 1×1.0
     = 1.0

y[1] = x[0]×h[1] + x[1]×h[0]
     = 1×0.5 + 2×1.0
     = 2.5

y[2] = x[0]×h[2] + x[1]×h[1] + x[2]×h[0]
     = 1×0.25 + 2×0.5 + 3×1.0
     = 4.25

y[3] = x[1]×h[2] + x[2]×h[1]
     = 2×0.25 + 3×0.5
     = 2.0

y[4] = x[2]×h[2]
     = 3×0.25
     = 0.75


Example 2: Different signals
==================================================
Rectangle x[n] = [1 1 1 1]
Triangle h[n] = [1 2 1]
Convolution y[n] = [1. 3. 4. 4. 3. 1.]

KEY INSIGHTS:
1. Manual implementation matches NumPy's convolve() exactly (0 difference)
2. Output length = len(x) + len(h) - 1 as expected
3. The convolution sum y[n] = Σ x[k] * h[n-k] is computed for each n
4. Rectangle * Triangle produces a trapezoidal shape
5. Step-by-step calculation shows how terms contribute to each output sample
6. Implementation handles edge cases correctly (beginning and end of sequences)

The algorithm complexity is O(N*M) where N=len(x) and M=len(h)
==================== END OF RESULTS ====================
"""