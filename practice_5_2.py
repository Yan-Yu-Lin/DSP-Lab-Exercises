import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

"""
******************************************************************
[Practice 5-2] Generate a music with melody: So Mi Mi Fa Re Re Do Re Mi Fa So So So ; So Mi Mi Fa Re Re Do Mi So So Do, via Matlab function sound.m at sampling frequency of 8000Hz.
******************************************************************

Complete Problem Statement:
Generate a musical melody with the specified note sequence using digital signal processing.
The melody consists of two phrases separated by a semicolon.
In MATLAB, this would use the sound() function. In Python, we'll use scipy.io.wavfile.write().

Musical Note Sequence:
First phrase: Sol Mi Mi Fa Re Re Do Re Mi Fa Sol Sol Sol
Second phrase: Sol Mi Mi Fa Re Re Do Mi Sol Sol Do

Mathematical Expression (LaTeX):
Musical note frequencies (C4 scale):
    Do (C4) = 261.63 Hz
    Re (D4) = 293.66 Hz
    Mi (E4) = 329.63 Hz
    Fa (F4) = 349.23 Hz
    Sol (G4) = 392.00 Hz

Signal generation: x(t) = A \\sin(2\\pi f_i t) where f_i is the frequency of note i
Sampling: x[n] = x(nT_s) where f_s = 8000 Hz, T_s = 1/f_s
"""

# Sampling parameters
fs = 8000  # Sampling frequency in Hz
duration_per_note = 0.4  # Duration of each note in seconds
pause_duration = 0.1  # Small pause between notes
phrase_pause = 0.3  # Pause between phrases

# Define note frequencies (C4 scale)
note_freq = {
    'Do': 261.63,  # C4
    'Re': 293.66,  # D4
    'Mi': 329.63,  # E4
    'Fa': 349.23,  # F4
    'Sol': 392.00  # G4 (So)
}

# Define the melody
melody = [
    # First phrase
    'Sol', 'Mi', 'Mi', 'Fa', 'Re', 'Re', 'Do', 'Re', 'Mi', 'Fa', 'Sol', 'Sol', 'Sol',
    'pause',  # Semicolon represents a pause
    # Second phrase
    'Sol', 'Mi', 'Mi', 'Fa', 'Re', 'Re', 'Do', 'Mi', 'Sol', 'Sol', 'Do'
]

def generate_note(frequency, duration, fs, amplitude=0.5):
    """Generate a single note with envelope to reduce clicking"""
    t = np.linspace(0, duration, int(duration * fs), False)
    # Basic sine wave
    note = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Apply envelope (attack-decay-sustain-release)
    attack_time = 0.05  # 50ms attack
    release_time = 0.05  # 50ms release
    
    attack_samples = int(attack_time * fs)
    release_samples = int(release_time * fs)
    
    # Create envelope
    envelope = np.ones_like(note)
    # Attack (fade in)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Release (fade out)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    
    return note * envelope

# Generate the complete melody
audio_signal = []
note_times = []  # Store timing information for visualization
current_time = 0

for note in melody:
    if note == 'pause':
        # Add pause between phrases
        pause_samples = int(phrase_pause * fs)
        audio_signal.extend(np.zeros(pause_samples))
        current_time += phrase_pause
    else:
        # Generate note
        frequency = note_freq[note]
        note_signal = generate_note(frequency, duration_per_note, fs)
        audio_signal.extend(note_signal)
        
        # Store note timing info
        note_times.append({
            'note': note,
            'frequency': frequency,
            'start_time': current_time,
            'duration': duration_per_note
        })
        current_time += duration_per_note
        
        # Add small pause between notes
        pause_samples = int(pause_duration * fs)
        audio_signal.extend(np.zeros(pause_samples))
        current_time += pause_duration

# Convert to numpy array
audio_signal = np.array(audio_signal)

# Normalize to prevent clipping
audio_signal = audio_signal / np.max(np.abs(audio_signal)) * 0.8

# Save as WAV file
output_filename = 'melody_output.wav'
wavfile.write(output_filename, fs, (audio_signal * 32767).astype(np.int16))

# Create time axis
t = np.arange(len(audio_signal)) / fs

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 1. Waveform
axes[0].plot(t, audio_signal, 'b-', linewidth=0.5)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Generated Melody Waveform')
axes[0].grid(True, alpha=0.3)

# Add note labels
for note_info in note_times:
    axes[0].axvline(x=note_info['start_time'], color='r', alpha=0.3, linestyle='--')
    axes[0].text(note_info['start_time'] + 0.02, 0.8, note_info['note'], 
                rotation=90, fontsize=8, alpha=0.7)

# 2. Spectrogram
f_spec, t_spec, Sxx = signal.spectrogram(audio_signal, fs, nperseg=1024, noverlap=512)
# Avoid log of zero
Sxx_db = 10 * np.log10(Sxx + 1e-10)
axes[1].pcolormesh(t_spec, f_spec, Sxx_db, shading='gouraud', cmap='viridis')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Spectrogram of Melody')
axes[1].set_ylim([0, 600])  # Focus on fundamental frequencies

# Add frequency lines for each note
for freq_name, freq_value in note_freq.items():
    axes[1].axhline(y=freq_value, color='white', alpha=0.3, linestyle=':', linewidth=0.5)
    axes[1].text(0.1, freq_value + 5, freq_name, color='white', fontsize=8, alpha=0.7)

# 3. Musical notation representation
axes[2].set_xlim(0, current_time)
axes[2].set_ylim(0, 6)
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Note')
axes[2].set_title('Musical Score Representation')

# Create note mapping for y-axis
note_positions = {'Do': 1, 'Re': 2, 'Mi': 3, 'Fa': 4, 'Sol': 5}
y_labels = ['', 'Do\n(C4)', 'Re\n(D4)', 'Mi\n(E4)', 'Fa\n(F4)', 'Sol\n(G4)']
axes[2].set_yticks(range(6))
axes[2].set_yticklabels(y_labels)

# Plot notes as rectangles
for note_info in note_times:
    note_name = note_info['note']
    y_pos = note_positions[note_name]
    rect = plt.Rectangle((note_info['start_time'], y_pos - 0.3), 
                        note_info['duration'], 0.6,
                        facecolor='blue', edgecolor='black', alpha=0.7)
    axes[2].add_patch(rect)

# Add phrase separator
phrase_sep_time = note_times[12]['start_time'] + duration_per_note + pause_duration
axes[2].axvline(x=phrase_sep_time, color='red', linestyle='--', linewidth=2, alpha=0.5)
axes[2].text(phrase_sep_time + 0.05, 5.5, 'Phrase Break', rotation=0, fontsize=10, color='red')

axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Analyze the melody
print("Melody Generation Analysis")
print("=" * 50)
print(f"Sampling frequency: {fs} Hz")
print(f"Total duration: {current_time:.2f} seconds")
print(f"Number of notes: {len([n for n in melody if n != 'pause'])}")
print(f"Output file: {output_filename}")

# Frequency analysis of each note
print("\nNote Frequency Analysis:")
print("-" * 30)
for note_name, freq in note_freq.items():
    period = 1/freq
    samples_per_period = fs * period
    print(f"{note_name}: {freq:.2f} Hz, Period: {period*1000:.2f} ms, Samples/period: {samples_per_period:.1f}")

# Count note occurrences
note_counts = {}
for note in melody:
    if note != 'pause':
        note_counts[note] = note_counts.get(note, 0) + 1

print("\nNote Distribution:")
print("-" * 30)
for note, count in sorted(note_counts.items()):
    print(f"{note}: {count} occurrences ({count/len([n for n in melody if n != 'pause'])*100:.1f}%)")

# Musical intervals
print("\nMusical Intervals in Melody:")
print("-" * 30)
prev_note = None
intervals = []
for note in melody:
    if note != 'pause' and prev_note and prev_note != 'pause':
        freq_ratio = note_freq[note] / note_freq[prev_note]
        cent_diff = 1200 * np.log2(freq_ratio)
        intervals.append((prev_note, note, cent_diff))
        if len(intervals) <= 5:  # Show first few intervals
            print(f"{prev_note} → {note}: {cent_diff:+.0f} cents")
    prev_note = note

"""
==================== RUN RESULTS ====================
Melody Generation Analysis
==================================================
Sampling frequency: 8000 Hz
Total duration: 12.30 seconds
Number of notes: 24
Output file: melody_output.wav

Note Frequency Analysis:
------------------------------
Do: 261.63 Hz, Period: 3.82 ms, Samples/period: 30.6
Re: 293.66 Hz, Period: 3.41 ms, Samples/period: 27.2
Mi: 329.63 Hz, Period: 3.03 ms, Samples/period: 24.3
Fa: 349.23 Hz, Period: 2.86 ms, Samples/period: 22.9
Sol: 392.00 Hz, Period: 2.55 ms, Samples/period: 20.4

Note Distribution:
------------------------------
Do: 3 occurrences (12.5%)
Fa: 3 occurrences (12.5%)
Mi: 6 occurrences (25.0%)
Re: 5 occurrences (20.8%)
Sol: 7 occurrences (29.2%)

Musical Intervals in Melody:
------------------------------
Sol → Mi: -300 cents
Mi → Mi: +0 cents
Mi → Fa: +100 cents
Fa → Re: -300 cents
Re → Re: +0 cents

KEY INSIGHTS:
1. The melody uses 5 distinct notes from the C4 major scale (Do, Re, Mi, Fa, Sol)
2. Sol (G4) is the most frequent note, appearing 7 times (29.2%)
3. Sampling at 8000 Hz provides ~20-30 samples per period for these frequencies
4. The melody consists of two phrases separated by a pause (semicolon in original)
5. ADSR envelope with 50ms attack/release prevents clicking artifacts
6. Spectrogram clearly shows discrete frequency bands for each note
7. The melody spans a perfect fifth interval (Do to Sol = 702 cents)
8. Note durations are uniform (0.4s) creating a steady rhythm
9. Nyquist frequency is 4000 Hz, well above all note frequencies (no aliasing)
10. The pause between phrases (0.3s) creates clear musical structure
11. Total audio data is ~98,400 samples for 12.3 seconds
12. WAV file uses 16-bit PCM encoding, resulting in ~197KB file size
13. Musical intervals include major thirds (Do→Mi), perfect fourths (Re→Sol)
14. The melody exhibits repetitive patterns typical of children's songs

The plots demonstrate:
- Waveform: Individual note envelopes and phrase structure visible
- Spectrogram: Clear horizontal frequency bands showing note transitions
- Musical score: Visual representation mapping pitch (y-axis) to time (x-axis)
==================== END OF RESULTS ====================
"""