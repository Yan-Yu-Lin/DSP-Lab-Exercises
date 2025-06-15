import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter

"""
******************************************************************
[Practice 7-1] Replace the median filter in Example 7-1 by a moving average filter and show the enhanced image.
******************************************************************

Complete Problem Statement:
Replace the median filter in Example 7-1 with a moving average filter for image enhancement.
Since Example 7-1 is not provided, we'll create a noisy image and compare the performance
of median filtering versus moving average filtering for noise reduction.

Mathematical Expression (LaTeX):
Moving Average Filter (2D):
    y[m,n] = \\frac{1}{M \\times N} \\sum_{i=0}^{M-1} \\sum_{j=0}^{N-1} x[m-i, n-j]

For a 3×3 kernel:
    H = \\frac{1}{9} \\begin{bmatrix} 1 & 1 & 1 \\\\ 1 & 1 & 1 \\\\ 1 & 1 & 1 \\end{bmatrix}

Median Filter:
    y[m,n] = \\text{median}\\{x[i,j] : (i,j) \\in \\text{neighborhood of } (m,n)\\}
"""

# Create a synthetic test image with different features
image_size = 256
x = np.linspace(-5, 5, image_size)
y = np.linspace(-5, 5, image_size)
X, Y = np.meshgrid(x, y)

# Create base image with various features
# 1. Smooth gradient background
base_image = 0.3 * (X + Y) / 10 + 0.5

# 2. Add some geometric shapes
# Circle
circle_mask = (X**2 + Y**2) < 4
base_image[circle_mask] = 0.8

# Rectangle
rect_mask = (np.abs(X - 2) < 1) & (np.abs(Y + 2) < 1.5)
base_image[rect_mask] = 0.2

# 3. Add some edges/lines
line_mask = np.abs(X - Y) < 0.1
base_image[line_mask] = 0.9

# Normalize to [0, 1]
base_image = np.clip(base_image, 0, 1)

# Add different types of noise
# 1. Salt and pepper noise (impulse noise)
salt_pepper_noise = np.random.random(base_image.shape)
salt_mask = salt_pepper_noise < 0.05  # 5% salt
pepper_mask = salt_pepper_noise > 0.95  # 5% pepper

noisy_image_sp = base_image.copy()
noisy_image_sp[salt_mask] = 1.0
noisy_image_sp[pepper_mask] = 0.0

# 2. Gaussian noise
gaussian_noise = np.random.normal(0, 0.1, base_image.shape)
noisy_image_gauss = np.clip(base_image + gaussian_noise, 0, 1)

# Apply filters
# Moving average filters with different kernel sizes
kernel_sizes = [3, 5, 7]
filtered_images_ma = {}
filtered_images_med = {}

for ksize in kernel_sizes:
    # Moving average filter (uniform kernel)
    kernel = np.ones((ksize, ksize)) / (ksize * ksize)
    
    # Apply to salt & pepper noise
    filtered_ma_sp = signal.convolve2d(noisy_image_sp, kernel, mode='same', boundary='symm')
    
    # Apply to Gaussian noise
    filtered_ma_gauss = signal.convolve2d(noisy_image_gauss, kernel, mode='same', boundary='symm')
    
    filtered_images_ma[ksize] = {
        'salt_pepper': np.clip(filtered_ma_sp, 0, 1),
        'gaussian': np.clip(filtered_ma_gauss, 0, 1)
    }
    
    # Median filter for comparison
    filtered_med_sp = median_filter(noisy_image_sp, size=ksize)
    filtered_med_gauss = median_filter(noisy_image_gauss, size=ksize)
    
    filtered_images_med[ksize] = {
        'salt_pepper': filtered_med_sp,
        'gaussian': filtered_med_gauss
    }

# Visualization
fig = plt.figure(figsize=(16, 12))

# Original and noisy images
plt.subplot(4, 5, 1)
plt.imshow(base_image, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.axis('off')

plt.subplot(4, 5, 2)
plt.imshow(noisy_image_sp, cmap='gray', vmin=0, vmax=1)
plt.title('Salt & Pepper Noise')
plt.axis('off')

plt.subplot(4, 5, 3)
plt.imshow(noisy_image_gauss, cmap='gray', vmin=0, vmax=1)
plt.title('Gaussian Noise')
plt.axis('off')

# Moving average results
row_idx = 1
for ksize in kernel_sizes:
    # Salt & pepper - Moving average
    plt.subplot(4, 5, row_idx + 5)
    plt.imshow(filtered_images_ma[ksize]['salt_pepper'], cmap='gray', vmin=0, vmax=1)
    plt.title(f'MA {ksize}×{ksize} (S&P)')
    plt.axis('off')
    
    # Salt & pepper - Median
    plt.subplot(4, 5, row_idx + 10)
    plt.imshow(filtered_images_med[ksize]['salt_pepper'], cmap='gray', vmin=0, vmax=1)
    plt.title(f'Median {ksize}×{ksize} (S&P)')
    plt.axis('off')
    
    # Gaussian - Moving average
    plt.subplot(4, 5, row_idx + 15)
    plt.imshow(filtered_images_ma[ksize]['gaussian'], cmap='gray', vmin=0, vmax=1)
    plt.title(f'MA {ksize}×{ksize} (Gauss)')
    plt.axis('off')
    
    row_idx += 1

plt.tight_layout()
plt.show()

# Detailed comparison for 5×5 kernel
ksize = 5
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Salt & pepper noise comparison
axes[0, 0].imshow(noisy_image_sp, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('Salt & Pepper Noise')
axes[0, 0].axis('off')

axes[0, 1].imshow(filtered_images_ma[ksize]['salt_pepper'], cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title(f'Moving Average {ksize}×{ksize}')
axes[0, 1].axis('off')

axes[0, 2].imshow(filtered_images_med[ksize]['salt_pepper'], cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title(f'Median Filter {ksize}×{ksize}')
axes[0, 2].axis('off')

# Show difference images
diff_ma_sp = np.abs(base_image - filtered_images_ma[ksize]['salt_pepper'])
diff_med_sp = np.abs(base_image - filtered_images_med[ksize]['salt_pepper'])

axes[1, 0].imshow(base_image, cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title('Original (Reference)')
axes[1, 0].axis('off')

axes[1, 1].imshow(diff_ma_sp, cmap='hot', vmin=0, vmax=0.5)
axes[1, 1].set_title('MA Error')
axes[1, 1].axis('off')

axes[1, 2].imshow(diff_med_sp, cmap='hot', vmin=0, vmax=0.5)
axes[1, 2].set_title('Median Error')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Edge preservation analysis
# Extract a line profile across an edge
profile_row = image_size // 2
profile_range = slice(image_size//2 - 30, image_size//2 + 30)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(base_image[profile_row, profile_range], 'k-', linewidth=2, label='Original')
plt.plot(noisy_image_sp[profile_row, profile_range], 'gray', alpha=0.5, label='Noisy')
plt.plot(filtered_images_ma[5]['salt_pepper'][profile_row, profile_range], 'b-', label='MA 5×5')
plt.plot(filtered_images_med[5]['salt_pepper'][profile_row, profile_range], 'r-', label='Median 5×5')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')
plt.title('Edge Profile - Salt & Pepper Noise')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(base_image[profile_row, profile_range], 'k-', linewidth=2, label='Original')
plt.plot(noisy_image_gauss[profile_row, profile_range], 'gray', alpha=0.5, label='Noisy')
plt.plot(filtered_images_ma[5]['gaussian'][profile_row, profile_range], 'b-', label='MA 5×5')
plt.plot(filtered_images_med[5]['gaussian'][profile_row, profile_range], 'r-', label='Median 5×5')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')
plt.title('Edge Profile - Gaussian Noise')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Quantitative analysis
print("Image Filtering Analysis: Moving Average vs Median Filter")
print("=" * 60)

# Calculate metrics
def calculate_metrics(original, noisy, filtered):
    # Mean Squared Error
    mse = np.mean((original - filtered) ** 2)
    
    # Peak Signal-to-Noise Ratio
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # Structural Similarity (simplified version)
    mean_orig = np.mean(original)
    mean_filt = np.mean(filtered)
    std_orig = np.std(original)
    std_filt = np.std(filtered)
    cov = np.mean((original - mean_orig) * (filtered - mean_filt))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mean_orig * mean_filt + c1) * (2 * cov + c2)) / \
           ((mean_orig ** 2 + mean_filt ** 2 + c1) * (std_orig ** 2 + std_filt ** 2 + c2))
    
    return mse, psnr, ssim

print("\nPerformance Metrics (5×5 kernel):")
print("-" * 60)
print("Salt & Pepper Noise:")

mse_ma, psnr_ma, ssim_ma = calculate_metrics(base_image, noisy_image_sp, 
                                              filtered_images_ma[5]['salt_pepper'])
mse_med, psnr_med, ssim_med = calculate_metrics(base_image, noisy_image_sp, 
                                                filtered_images_med[5]['salt_pepper'])

print(f"  Moving Average - MSE: {mse_ma:.4f}, PSNR: {psnr_ma:.2f} dB, SSIM: {ssim_ma:.3f}")
print(f"  Median Filter  - MSE: {mse_med:.4f}, PSNR: {psnr_med:.2f} dB, SSIM: {ssim_med:.3f}")

print("\nGaussian Noise:")
mse_ma_g, psnr_ma_g, ssim_ma_g = calculate_metrics(base_image, noisy_image_gauss, 
                                                    filtered_images_ma[5]['gaussian'])
mse_med_g, psnr_med_g, ssim_med_g = calculate_metrics(base_image, noisy_image_gauss, 
                                                      filtered_images_med[5]['gaussian'])

print(f"  Moving Average - MSE: {mse_ma_g:.4f}, PSNR: {psnr_ma_g:.2f} dB, SSIM: {ssim_ma_g:.3f}")
print(f"  Median Filter  - MSE: {mse_med_g:.4f}, PSNR: {psnr_med_g:.2f} dB, SSIM: {ssim_med_g:.3f}")

# Edge preservation analysis
print("\nEdge Preservation Analysis:")
print("-" * 60)

# Compute gradient magnitude
grad_orig = np.sqrt(np.gradient(base_image, axis=0)**2 + np.gradient(base_image, axis=1)**2)
grad_ma = np.sqrt(np.gradient(filtered_images_ma[5]['salt_pepper'], axis=0)**2 + 
                  np.gradient(filtered_images_ma[5]['salt_pepper'], axis=1)**2)
grad_med = np.sqrt(np.gradient(filtered_images_med[5]['salt_pepper'], axis=0)**2 + 
                   np.gradient(filtered_images_med[5]['salt_pepper'], axis=1)**2)

edge_preserve_ma = np.corrcoef(grad_orig.ravel(), grad_ma.ravel())[0, 1]
edge_preserve_med = np.corrcoef(grad_orig.ravel(), grad_med.ravel())[0, 1]

print(f"Edge correlation (Salt & Pepper):")
print(f"  Moving Average: {edge_preserve_ma:.3f}")
print(f"  Median Filter:  {edge_preserve_med:.3f}")

"""
==================== RUN RESULTS ====================
Image Filtering Analysis: Moving Average vs Median Filter
============================================================

Performance Metrics (5×5 kernel):
------------------------------------------------------------
Salt & Pepper Noise:
  Moving Average - MSE: 0.0063, PSNR: 22.02 dB, SSIM: 0.850
  Median Filter  - MSE: 0.0005, PSNR: 33.23 dB, SSIM: 0.986

Gaussian Noise:
  Moving Average - MSE: 0.0019, PSNR: 27.28 dB, SSIM: 0.949
  Median Filter  - MSE: 0.0026, PSNR: 25.79 dB, SSIM: 0.935

Edge Preservation Analysis:
------------------------------------------------------------
Edge correlation (Salt & Pepper):
  Moving Average: 0.621
  Median Filter:  0.912

KEY INSIGHTS:
1. Median filter excels at removing salt & pepper (impulse) noise
2. Moving average performs better for Gaussian (continuous) noise
3. Median filter achieves 33.23 dB PSNR vs 22.02 dB for MA on impulse noise
4. Edge preservation is significantly better with median filter (0.912 vs 0.621)
5. Moving average blurs edges due to linear averaging
6. Median filter preserves edges as it selects existing pixel values
7. Larger kernel sizes increase smoothing but reduce detail
8. Salt & pepper noise creates bright/dark spots that MA spreads out
9. Median filter completely removes isolated noise pixels
10. For Gaussian noise, both filters perform similarly (27.28 vs 25.79 dB)
11. Moving average is computationally simpler (linear convolution)
12. Median filter requires sorting, making it computationally intensive
13. MA filter causes "ghosting" around impulse noise
14. The choice of filter depends on noise type and edge importance

The plots demonstrate:
- Visual comparison of filter effectiveness on different noise types
- Edge profile showing MA blurring vs median edge preservation
- Error heat maps revealing where each filter fails
==================== END OF RESULTS ====================
"""