import numpy as np
import matplotlib.pyplot as plt

"""
******************************************************************
[Practice 3-6] Find the 6-point DFT of the 6-point circular convolution of the two sequences
in Fig. 2-1-1(a) and (b).
******************************************************************

Complete Problem Statement:
Find the 6-point DFT of the 6-point circular convolution of two sequences.
Since Fig. 2-1-1(a) and (b) are not provided, we'll use example sequences.

Mathematical Expression (LaTeX):
Circular convolution property of DFT:
If y[n] = x[n] ⊛ h[n] (circular convolution)
Then Y[k] = X[k] · H[k] (element-wise multiplication in frequency domain)

Where ⊛ denotes N-point circular convolution.
"""

# Since the figure is not provided, let's create two example sequences
# that would be typical for this type of problem

# Example sequences (6-point)
x = np.array([1, 2, 3, 2, 1, 0])  # Sequence (a)
h = np.array([1, 1, 0, 0, 0, 1])  # Sequence (b)

print("Circular Convolution and DFT Analysis")
print("=" * 50)
print(f"Sequence x[n]: {x}")
print(f"Sequence h[n]: {h}")

# Method 1: Direct circular convolution
def circular_convolution(x, h):
    """Compute N-point circular convolution"""
    N = len(x)
    y = np.zeros(N)
    
    for n in range(N):
        for k in range(N):
            # Circular indexing
            y[n] += x[k] * h[(n - k) % N]
    
    return y

# Compute circular convolution directly
y_direct = circular_convolution(x, h)

# Method 2: Using DFT property
# Step 1: Compute DFTs of both sequences
X = np.fft.fft(x)
H = np.fft.fft(h)

# Step 2: Multiply in frequency domain
Y = X * H

# Step 3: Compute inverse DFT
y_dft = np.fft.ifft(Y).real

# The 6-point DFT of the circular convolution
Y_final = np.fft.fft(y_direct)

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Plot original sequences
axes[0, 0].stem(range(6), x, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[0, 0].set_xlabel('n')
axes[0, 0].set_ylabel('x[n]')
axes[0, 0].set_title('Sequence x[n]')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(range(6))

axes[0, 1].stem(range(6), h, linefmt='g-', markerfmt='go', basefmt='k-')
axes[0, 1].set_xlabel('n')
axes[0, 1].set_ylabel('h[n]')
axes[0, 1].set_title('Sequence h[n]')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(range(6))

# Plot DFTs of original sequences
axes[1, 0].stem(range(6), np.abs(X), linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1, 0].set_xlabel('k')
axes[1, 0].set_ylabel('|X[k]|')
axes[1, 0].set_title('6-point DFT of x[n]: Magnitude')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(range(6))

axes[1, 1].stem(range(6), np.abs(H), linefmt='g-', markerfmt='go', basefmt='k-')
axes[1, 1].set_xlabel('k')
axes[1, 1].set_ylabel('|H[k]|')
axes[1, 1].set_title('6-point DFT of h[n]: Magnitude')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(range(6))

# Plot circular convolution result and its DFT
axes[2, 0].stem(range(6), y_direct, linefmt='r-', markerfmt='ro', basefmt='k-')
axes[2, 0].set_xlabel('n')
axes[2, 0].set_ylabel('y[n]')
axes[2, 0].set_title('6-point Circular Convolution: y[n] = x[n] ⊛ h[n]')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_xticks(range(6))

axes[2, 1].stem(range(6), np.abs(Y_final), linefmt='m-', markerfmt='mo', basefmt='k-')
axes[2, 1].set_xlabel('k')
axes[2, 1].set_ylabel('|Y[k]|')
axes[2, 1].set_title('6-point DFT of Circular Convolution')
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].set_xticks(range(6))

plt.tight_layout()
plt.show()

# Verify the convolution property
print("\n\nVerification of DFT Convolution Property:")
print("-" * 50)
print("Method 1: Direct circular convolution then DFT")
print(f"y[n] = x[n] ⊛ h[n] = {y_direct}")
print(f"Y[k] = DFT{{y[n]}} magnitude: {np.abs(Y_final).round(2)}")

print("\nMethod 2: DFT, multiply, then IDFT")
print(f"y[n] via DFT method = {y_dft.round(6)}")
print(f"Maximum difference = {np.max(np.abs(y_direct - y_dft)):.2e}")

# Show element-wise multiplication property
print("\n\nDFT Values:")
print("-" * 50)
print("k | X[k] | H[k] | Y[k] = X[k]·H[k]")
print("-" * 50)
for k in range(6):
    print(f"{k} | {X[k]:.2f} | {H[k]:.2f} | {Y[k]:.2f}")

# Phase information
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.stem(range(6), np.abs(Y_final), linefmt='m-', markerfmt='mo', basefmt='k-')
plt.xlabel('k')
plt.ylabel('|Y[k]|')
plt.title('Magnitude Spectrum of Y[k]')
plt.grid(True, alpha=0.3)
plt.xticks(range(6))

plt.subplot(2, 2, 2)
plt.stem(range(6), np.angle(Y_final), linefmt='c-', markerfmt='co', basefmt='k-')
plt.xlabel('k')
plt.ylabel('∠Y[k] (rad)')
plt.title('Phase Spectrum of Y[k]')
plt.grid(True, alpha=0.3)
plt.xticks(range(6))
plt.ylim([-np.pi, np.pi])

# Show the relationship Y[k] = X[k] * H[k]
plt.subplot(2, 2, 3)
plt.plot(range(6), np.abs(X), 'b-o', label='|X[k]|', linewidth=2)
plt.plot(range(6), np.abs(H), 'g-s', label='|H[k]|', linewidth=2)
plt.plot(range(6), np.abs(Y), 'm-^', label='|Y[k]| = |X[k]||H[k]|', linewidth=2)
plt.xlabel('k')
plt.ylabel('Magnitude')
plt.title('DFT Multiplication Property')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(6))

# Linear vs Circular convolution comparison
# Linear convolution would be longer
y_linear = np.convolve(x, h, mode='full')
plt.subplot(2, 2, 4)
plt.stem(range(len(y_linear)), y_linear, linefmt='k-', markerfmt='ko', 
         basefmt='k-', label='Linear convolution')
plt.stem(range(6), y_direct, linefmt='r--', markerfmt='rs', 
         basefmt='k--', label='Circular convolution')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Linear vs Circular Convolution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary
print("\n\nSummary:")
print("=" * 50)
print(f"6-point circular convolution result: {y_direct}")
print(f"\n6-point DFT of circular convolution:")
print(f"Magnitude: {np.abs(Y_final).round(2)}")
print(f"Phase (degrees): {np.degrees(np.angle(Y_final)).round(1)}")
print(f"\nKey property verified: Y[k] = X[k] · H[k]")

"""
==================== RUN RESULTS ====================
Circular Convolution and DFT Analysis
==================================================
Sequence x[n]: [1 2 3 2 1 0]
Sequence h[n]: [1 1 0 0 0 1]


Verification of DFT Convolution Property:
--------------------------------------------------
Method 1: Direct circular convolution then DFT
y[n] = x[n] ⊛ h[n] = [3. 6. 7. 6. 3. 2.]
Y[k] = DFT{y[n]} magnitude: [27.  8.  0.  1.  0.  8.]

Method 2: DFT, multiply, then IDFT
y[n] via DFT method = [3. 6. 7. 6. 3. 2.]
Maximum difference = 0.00e+00


DFT Values:
--------------------------------------------------
k | X[k] | H[k] | Y[k] = X[k]·H[k]
--------------------------------------------------
0 | 9.00+0.00j | 3.00+0.00j | 27.00+0.00j
1 | -2.00-3.46j | 2.00+0.00j | -4.00-6.93j
2 | 0.00+0.00j | 0.00+0.00j | 0.00+0.00j
3 | 1.00+0.00j | -1.00+0.00j | -1.00-0.00j
4 | 0.00+0.00j | 0.00+0.00j | 0.00+0.00j
5 | -2.00+3.46j | 2.00+0.00j | -4.00+6.93j


Summary:
==================================================
6-point circular convolution result: [3. 6. 7. 6. 3. 2.]

6-point DFT of circular convolution:
Magnitude: [27.  8.  0.  1.  0.  8.]
Phase (degrees): [   0. -120.    0.  180.    0.  120.]

Key property verified: Y[k] = X[k] · H[k]

KEY INSIGHTS:
1. Circular convolution in time domain = multiplication in frequency domain
2. Y[k] = X[k] · H[k] for all k (element-wise multiplication)
3. The 6-point circular convolution wraps around due to periodicity
4. Linear convolution would produce 11 samples (6+6-1)
5. Circular convolution produces exactly 6 samples
6. DFT method is computationally efficient for large N (using FFT)
7. Both methods produce identical results (error < 1e-15)
8. Zero DFT values (k=2,4) occur where either X[k] or H[k] is zero
9. Phase of Y[k] = Phase of X[k] + Phase of H[k]
10. This property is fundamental to frequency domain filtering

The plots demonstrate:
- Original sequences and their DFTs
- Circular convolution result
- DFT of the convolution showing Y[k] = X[k]·H[k]
- Comparison with linear convolution showing wrap-around effect
==================== END OF RESULTS ====================
"""