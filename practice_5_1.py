import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile as wavfile

"""
******************************************************************
[Practice 5-1] Perform ↑4 and then ↓3 of an audio signal without using the Matlab functions.
Plot the spectrogram of the resulting signal.
******************************************************************

Complete Problem Statement:
Perform upsampling by factor 4 (↑4) followed by downsampling by factor 3 (↓3) on an audio signal.
Do not use Matlab functions (implement from scratch).
Plot the spectrogram of the resulting signal.

Mathematical Expression (LaTeX):
Upsampling by L: y[n] = \\begin{cases} x[n/L] & \\text{if } n \\text{ is multiple of } L \\\\ 0 & \\text{otherwise} \\end{cases}
Downsampling by M: y[n] = x[nM]
Combined rate change: \\frac{4}{3} = 1.333...

Note: Since no audio file is provided, we'll generate a test audio signal.
"""

# Generate a test audio signal
fs = 8000  # Original sampling frequency
duration = 1.0  # 1 second
t = np.arange(0, duration, 1/fs)

# Create a signal with multiple frequency components
# Simulating speech/music with formants
audio = (0.5 * np.sin(2*np.pi*440*t) +      # A4 note
         0.3 * np.sin(2*np.pi*880*t) +      # A5 note  
         0.2 * np.sin(2*np.pi*1320*t) +     # E6 note
         0.1 * np.sin(2*np.pi*200*t))       # Low frequency

# Add some amplitude modulation to make it more interesting
audio *= (1 + 0.3*np.sin(2*np.pi*5*t))

print("Multirate Signal Processing: ↑4 then ↓3")
print("=" * 50)
print(f"Original signal: {len(audio)} samples at {fs} Hz")

# Step 1: Upsample by 4 (without using scipy.signal.resample)
def upsample_by_L(x, L):
    """Insert L-1 zeros between each sample"""
    N = len(x)
    y = np.zeros(N * L)
    y[::L] = x
    return y

# Step 2: Design and apply anti-aliasing filter for upsampling
def design_lowpass_filter(cutoff_normalized, num_taps=101):
    """Design FIR lowpass filter using window method"""
    n = np.arange(num_taps)
    h = cutoff_normalized * np.sinc(cutoff_normalized * (n - (num_taps-1)/2))
    h *= np.hamming(num_taps)
    h /= np.sum(h)
    return h

# Upsample by 4
upsampled = upsample_by_L(audio, 4)
fs_up = fs * 4
print(f"After ↑4: {len(upsampled)} samples at {fs_up} Hz")

# Apply lowpass filter (cutoff at π/4 for upsampling by 4)
h_up = design_lowpass_filter(1/4, num_taps=121)
upsampled_filtered = signal.convolve(upsampled, h_up, mode='same')
upsampled_filtered *= 4  # Compensate for amplitude loss

# Step 3: Downsample by 3
def downsample_by_M(x, M):
    """Keep every M-th sample"""
    return x[::M]

# Apply anti-aliasing filter before downsampling (cutoff at π/3)
h_down = design_lowpass_filter(1/3, num_taps=121)
pre_downsampled = signal.convolve(upsampled_filtered, h_down, mode='same')

# Downsample by 3
downsampled = downsample_by_M(pre_downsampled, 3)
fs_final = fs_up // 3  # Note: 32000/3 ≈ 10666.67 Hz
print(f"After ↓3: {len(downsampled)} samples at {fs_final:.2f} Hz")
print(f"Net rate change: {fs_final/fs:.3f} (should be 4/3 = 1.333)")

# Compute spectrograms
def compute_spectrogram(signal, fs, window_size=256, overlap=128):
    """Compute spectrogram using STFT"""
    hop = window_size - overlap
    n_frames = (len(signal) - window_size) // hop + 1
    
    # Initialize spectrogram
    spectrogram = np.zeros((window_size//2 + 1, n_frames))
    
    # Compute STFT
    window = np.hamming(window_size)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start+window_size] * window
        spectrum = np.fft.rfft(frame)
        spectrogram[:, i] = np.abs(spectrum)
    
    # Time and frequency axes
    times = np.arange(n_frames) * hop / fs
    freqs = np.fft.rfftfreq(window_size, 1/fs)
    
    return spectrogram, freqs, times

# Plotting
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Original signal
axes[0, 0].plot(t[:500], audio[:500], 'b-', linewidth=1)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title(f'Original Signal (first 500 samples, fs={fs} Hz)')
axes[0, 0].grid(True, alpha=0.3)

# Final signal
t_final = np.arange(len(downsampled)) / fs_final
axes[0, 1].plot(t_final[:666], downsampled[:666], 'r-', linewidth=1)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].set_title(f'Final Signal (first 666 samples, fs={fs_final:.0f} Hz)')
axes[0, 1].grid(True, alpha=0.3)

# Frequency response of filters
w, h_up_response = signal.freqz(h_up)
w, h_down_response = signal.freqz(h_down)

axes[1, 0].plot(w/np.pi, np.abs(h_up_response), 'b-', label='Upsampling filter', linewidth=2)
axes[1, 0].plot(w/np.pi, np.abs(h_down_response), 'r-', label='Downsampling filter', linewidth=2)
axes[1, 0].set_xlabel('Normalized Frequency (×π rad/sample)')
axes[1, 0].set_ylabel('Magnitude')
axes[1, 0].set_title('Anti-aliasing Filter Responses')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Spectrograms
spec_orig, f_orig, t_orig = compute_spectrogram(audio, fs)
spec_final, f_final, t_final_spec = compute_spectrogram(downsampled, fs_final)

# Original spectrogram
im1 = axes[2, 0].imshow(20*np.log10(spec_orig + 1e-10), aspect='auto', 
                        origin='lower', cmap='viridis',
                        extent=[t_orig[0], t_orig[-1], f_orig[0], f_orig[-1]])
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Frequency (Hz)')
axes[2, 0].set_title('Original Signal Spectrogram')
axes[2, 0].set_ylim([0, fs/2])

# Final spectrogram
im2 = axes[2, 1].imshow(20*np.log10(spec_final + 1e-10), aspect='auto', 
                        origin='lower', cmap='viridis',
                        extent=[t_final_spec[0], t_final_spec[-1], f_final[0], f_final[-1]])
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Frequency (Hz)')
axes[2, 1].set_title('Final Signal Spectrogram (after ↑4↓3)')
axes[2, 1].set_ylim([0, fs_final/2])

# Add colorbars
plt.colorbar(im1, ax=axes[2, 0], label='dB')
plt.colorbar(im2, ax=axes[2, 1], label='dB')

# Frequency domain comparison
axes[1, 1].magnitude_spectrum(audio, Fs=fs, color='b', alpha=0.7, label='Original')
axes[1, 1].magnitude_spectrum(downsampled, Fs=fs_final, color='r', alpha=0.7, label='After ↑4↓3')
axes[1, 1].set_xlim([0, 2000])
axes[1, 1].legend()
axes[1, 1].set_title('Magnitude Spectrum Comparison')

plt.tight_layout()
plt.show()

# Analysis
print("\n\nAnalysis:")
print("-" * 50)
print(f"Effective rate conversion: {len(downsampled)/len(audio):.4f}")
print(f"Theoretical rate conversion: 4/3 = {4/3:.4f}")
print(f"Error: {abs(len(downsampled)/len(audio) - 4/3):.6f}")

# Check frequency content preservation
print("\nFrequency Content:")
print("Original signal contains: 200Hz, 440Hz, 880Hz, 1320Hz")
print("After ↑4↓3, Nyquist frequency is:", fs_final/2, "Hz")
print("All frequencies are preserved (below new Nyquist)")

# Save the result
output_filename = 'practice_5_1_output.wav'
# Normalize to prevent clipping
downsampled_normalized = downsampled / np.max(np.abs(downsampled))
wavfile.write(output_filename, int(fs_final), (downsampled_normalized * 32767).astype(np.int16))
print(f"\nOutput saved as: {output_filename}")