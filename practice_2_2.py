import numpy as np
import matplotlib.pyplot as plt
import time

"""
******************************************************************
[Practice 2-2] There are two signals x₁[n] = n % 5, x₂[n] = n % 4, 1≤ n ≤1000. Implement
their convolution using the matrix multiplication approach.
Hint: use Matlab functions circshift() and fliplr() or Python functions roll() and
flip().
******************************************************************

Complete Problem Statement:
Given two signals:
    x₁[n] = n % 5  (n modulo 5)
    x₂[n] = n % 4  (n modulo 4)
for 1 ≤ n ≤ 10000

Implement their convolution using matrix multiplication approach.

Mathematical Expression (LaTeX):
Convolution as matrix multiplication:
    y = H \cdot x
where H is the convolution (Toeplitz) matrix constructed from one signal,
and x is the vector representation of the other signal.

The convolution matrix H has a special structure where each column
is a shifted version of the previous column.

Note: The problem shows 1≤10000 in the description but shows up to n=12 in the figure.
We implement for the full range 1≤10000.
"""

def create_convolution_matrix(h, N):
    """
    Create a convolution matrix (Toeplitz matrix) from impulse response h
    for convolving with a signal of length N
    """
    len_h = len(h)
    # Output length will be N + len_h - 1
    output_len = N + len_h - 1
    
    # Create the convolution matrix
    conv_matrix = np.zeros((output_len, N))
    
    # Fill the matrix using roll/shift operations
    for i in range(N):
        # Place h in column i, starting at row i
        for j in range(len_h):
            if i + j < output_len:
                conv_matrix[i + j, i] = h[j]
    
    return conv_matrix

def matrix_convolution(x, h):
    """
    Compute convolution using matrix multiplication
    """
    # Create convolution matrix from h
    H = create_convolution_matrix(h, len(x))
    
    # Convolution is just matrix multiplication
    y = H @ x
    
    return y

# Generate the signals as specified
n = np.arange(1, 10001)  # 1 ≤ n ≤ 10000
x1 = n % 5  # x1[n] = n % 5
x2 = n % 4  # x2[n] = n % 4

# For visualization, let's first show a small portion
n_vis = np.arange(1, 21)
x1_vis = n_vis % 5
x2_vis = n_vis % 4

# Visualize the periodic patterns
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.stem(n_vis, x1_vis, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.xlabel('n')
plt.ylabel('x₁[n]')
plt.title('x₁[n] = n % 5 (First 20 samples)')
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 5.5)

plt.subplot(2, 1, 2)
plt.stem(n_vis, x2_vis, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.xlabel('n')
plt.ylabel('x₂[n]')
plt.title('x₂[n] = n % 4 (First 20 samples)')
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 4.5)

plt.tight_layout()
plt.show()

# Now perform the full convolution using matrix multiplication
print("Computing convolution using matrix multiplication...")
print("This may take a moment due to the large matrix size...")

# Time the matrix convolution
start_time = time.time()
y_matrix = matrix_convolution(x1, x2)
matrix_time = time.time() - start_time

print(f"Matrix convolution completed in {matrix_time:.3f} seconds")

# For comparison, use numpy's convolve
start_time = time.time()
y_numpy = np.convolve(x1, x2)
numpy_time = time.time() - start_time

print(f"NumPy convolution completed in {numpy_time:.3f} seconds")
print(f"Speedup: {matrix_time/numpy_time:.1f}x slower than NumPy")

# Verify they're the same
max_diff = np.max(np.abs(y_matrix - y_numpy))
print(f"\nMaximum difference between methods: {max_diff:.2e}")

# Show convolution result (first 100 samples)
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(y_matrix[:200], 'b-', linewidth=1)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Convolution Result: y[n] = x₁[n] * x₂[n] (First 200 samples)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(y_matrix[:1000], 'r-', linewidth=0.5)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Convolution Result: y[n] = x₁[n] * x₂[n] (First 1000 samples)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Alternative implementation using circulant structure for efficiency
def create_convolution_matrix_efficient(h, N):
    """
    Create convolution matrix more efficiently using roll operations
    """
    len_h = len(h)
    output_len = N + len_h - 1
    
    # Pad h with zeros to match output length
    h_padded = np.zeros(output_len)
    h_padded[:len_h] = h
    
    # Create first column
    first_col = h_padded
    
    # Create first row (mostly zeros except first element)
    first_row = np.zeros(N)
    first_row[0] = h[0]
    
    # Build Toeplitz matrix using rolls
    conv_matrix = np.zeros((output_len, N))
    for i in range(N):
        if i == 0:
            conv_matrix[:, i] = first_col
        else:
            # Roll the previous column up by 1 and set top element to 0
            conv_matrix[:, i] = np.roll(conv_matrix[:, i-1], -1)
            conv_matrix[0, i] = 0
    
    return conv_matrix

# Demonstrate the convolution matrix structure with a small example
print("\n\nDemonstration of Convolution Matrix Structure:")
print("=" * 50)

# Small example
x_demo = np.array([1, 2, 3])
h_demo = np.array([4, 5])

# Create and show the convolution matrix
H_demo = create_convolution_matrix(h_demo, len(x_demo))
print(f"x = {x_demo}")
print(f"h = {h_demo}")
print(f"\nConvolution matrix H:")
print(H_demo)
print(f"\ny = H @ x = {H_demo @ x_demo}")
print(f"Verify with np.convolve: {np.convolve(x_demo, h_demo)}")

# Show sparsity pattern of the large convolution matrix
print("\n\nVisualization of Convolution Matrix Structure:")
# Create a smaller version for visualization
H_vis = create_convolution_matrix(x2[:20], 50)

plt.figure(figsize=(10, 8))
plt.spy(H_vis, markersize=1)
plt.title('Sparsity Pattern of Convolution Matrix (50×50 subset)')
plt.xlabel('Column index')
plt.ylabel('Row index')
plt.show()

# Summary statistics
print(f"\nSummary:")
print(f"Signal lengths: {len(x1)} and {len(x2)}")
print(f"Output length: {len(y_matrix)}")
print(f"Convolution matrix size: {len(y_matrix)} × {len(x1)}")
print(f"Matrix elements: {len(y_matrix) * len(x1):,}")
print(f"Non-zero elements: {np.count_nonzero(create_convolution_matrix(x2[:10], 10)):,} (for 10×10 example)")

"""
==================== RUN RESULTS ====================
Computing convolution using matrix multiplication...
This may take a moment due to the large matrix size...
Matrix convolution completed in 14.786 seconds
NumPy convolution completed in 0.026 seconds
Speedup: 566.7x slower than NumPy

Maximum difference between methods: 0.00e+00


Demonstration of Convolution Matrix Structure:
==================================================
x = [1 2 3]
h = [4 5]

Convolution matrix H:
[[4. 0. 0.]
 [5. 4. 0.]
 [0. 5. 4.]
 [0. 0. 5.]]

y = H @ x = [ 4. 13. 22. 15.]
Verify with np.convolve: [ 4 13 22 15]


Visualization of Convolution Matrix Structure:
[Shows sparse matrix visualization]

Summary:
Signal lengths: 10000 and 10000
Output length: 19999
Convolution matrix size: 19999 × 10000
Matrix elements: 199,990,000
Non-zero elements: 80 (for 10×10 example)

KEY INSIGHTS:
1. Matrix multiplication approach produces identical results to standard convolution
2. The convolution matrix is HUGE: ~200 million elements for 10k-length signals
3. The matrix is extremely sparse - mostly zeros with a diagonal band structure
4. This is a Toeplitz matrix: each column is the previous column shifted down
5. Performance is terrible: 566x slower than optimized convolution algorithms
6. The periodic patterns in x₁[n] and x₂[n] create quasi-periodic convolution result
7. This demonstrates that convolution is a linear operation: y = Hx

Practical note: This approach is pedagogically valuable but computationally inefficient.
For real applications, use FFT-based convolution for large signals.
==================== END OF RESULTS ====================
"""