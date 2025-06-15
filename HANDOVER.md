# DSP Lab Exercises Handover Document

## Project Overview

Arthur (林晏宇) needs to complete DSP (Digital Signal Processing) lab exercises and generate 心得 (reflections) for his Thursday report. Each practice problem needs to be implemented as a self-contained Python file that another Claude instance can read to understand the problem and generate thoughtful reflections.

**Deadline**: Thursday (part of ~30 reflections needed)
**Course**: 數位訊號實習 (Digital Signal Processing Lab)

## Critical Requirements for Each Practice File

Each Python file MUST contain:

1. **Complete Problem Statement** - Copy the exact wording from the exercise, character-for-character
2. **Mathematical Expressions** - Use LaTeX format with proper escaping (double backslashes: `\\`)
3. **Implementation Code** - Working solution with detailed comments
4. **Run Results** - Actual output included as multi-line string at the end
5. **Key Insights** - 10+ bullet points summarizing important findings

### File Structure Template

```python
import numpy as np
import matplotlib.pyplot as plt

"""
******************************************************************
[Practice X-Y] Complete problem statement exactly as written...
******************************************************************

Complete Problem Statement:
[Expanded description]

Mathematical Expression (LaTeX):
H(z) = \\frac{1 - z^{-1}}{1 - 0.8z^{-1}}

Note: [Any clarifications about missing figures or assumptions]
"""

# Implementation code here...

# Analysis and visualization...

print("Results...")

"""
==================== RUN RESULTS ====================
[Actual output from running the code]

KEY INSIGHTS:
1. [First key insight]
2. [Second key insight]
...
10. [Tenth key insight]

The plots demonstrate:
- [What the visualizations show]
==================== END OF RESULTS ====================
"""
```

## Completed Work (10 files)

### Lab 1-3:
- ✅ `practice_1_2.py` - Discrete sinc function x[n] = sin(ωn)/(πn)
- ✅ `practice_1_3.py` - Two-tone signal sampling (10Hz + 30Hz)

### Lab 2:
- ✅ `practice_2_1.py` - Manual convolution implementation
- ✅ `practice_2_2.py` - Matrix multiplication convolution (n % 5, n % 4)

### Lab 3:
- ✅ `practice_3_1.py` - DTFT without exp(), abs(), angle()
- ✅ `practice_3_2.py` - 10-pt vs 100-pt DFT comparison
- ✅ `practice_3_3.py` - FFT implementation
- ✅ `practice_3_4.py` - FFT of two-tone signal
- ✅ `practice_3_5.py` - Aliasing demonstration (fs=50Hz)
- ✅ `practice_3_6.py` - Circular convolution DFT

### Lab 4:
- ✅ `practice_4_1.py` - Filtering vs convolution (IIR system)
- ✅ `practice_4_2.py` - Compensating system design (unstable result)

## Remaining Work

### Lab 5:
**[Practice 5-1]** Perform ↑4 and then ↓3 of an audio signal without using the Matlab functions. Plot the spectrogram of the resulting signal.

**[Practice 5-2]** Generate a music with melody: So Mi Mi Fa Re Re Do Re Mi Fa So So So ; So Mi Mi Fa Re Re Do Mi So So Do, via Matlab function sound.m at sampling frequency of 8000Hz.

**[Practice 5-3]** Use Adobe Audition® to de-vocal a stereo music by subtracting its left-channel signal from right-channel signal.

### Lab 6:
**[Practice 6-1]** Design a Chebyshev lowpass digital filter using Matlab function upsample.m to perform upsampling of the signal in Example 6-2 by a factor 2. Sketch the resulting waveform and spectrogram.

### Lab 7:
**[Practice 7-1]** Replace the median filter in Example 7-1 by a moving average filter and show the enhanced image.

## Technical Instructions

### Directory Structure
All files go in: `/Users/linyanyu/Desktop/10-19-Academic/11-Current-Semester/11.03-數位處理實習/DSP-Python-Exercises/`

### Running Files
Use uv package manager with required dependencies:
```bash
uv run --with numpy --with matplotlib --with scipy python "practice_X_Y.py"
```

### Common Issues and Solutions

1. **stem() plot parameters**: Don't use `alpha` or `markersize` directly. Instead:
   ```python
   markerline, stemlines, baseline = plt.stem(x, y)
   markerline.set_markersize(8)
   stemlines.set_alpha(0.7)
   ```

2. **LaTeX in docstrings**: Always use double backslashes
   - Wrong: `\sin(\omega n)`
   - Right: `\\sin(\\omega n)`

3. **Missing figures**: When exercises reference figures not provided, create reasonable test signals:
   - Use rectangular pulses, impulses, sine waves
   - Document assumptions clearly

4. **Audio/Image files**: For practices requiring external files:
   - Generate synthetic test data
   - Note that real files aren't available
   - Implement the algorithm anyway

## Implementation Notes for Remaining Exercises

### Practice 5-1 (Upsampling/Downsampling):
- ↑4 means upsampling by 4 (insert 3 zeros between samples)
- ↓3 means downsampling by 3 (keep every 3rd sample)
- Need to implement anti-aliasing filters
- Use STFT for spectrogram without matplotlib's specgram

### Practice 5-2 (Music Generation):
- Note frequencies: Do=261.63Hz, Re=293.66Hz, Mi=329.63Hz, Fa=349.23Hz, Sol=392Hz
- Use numpy to generate sine waves
- Save with scipy.io.wavfile.write instead of Matlab's sound()

### Practice 5-3 (De-vocal):
- Note: Can't actually use Adobe Audition in Python
- Simulate with synthetic stereo signal
- Implement: `devocal = right_channel - left_channel`

### Practice 6-1 (Chebyshev Filter):
- Use scipy.signal.cheby1 for filter design
- Implement upsampling manually (not with upsample.m)
- Show frequency response and time-domain effects

### Practice 7-1 (Image Filtering):
- Generate synthetic noisy image
- Implement both median and moving average filters
- Compare results visually

## Context for 心得 Generation

When another Claude reads these files to generate 心得, they should focus on:
1. Understanding of the mathematical concepts
2. Implementation challenges and solutions
3. Comparison with theoretical expectations
4. Practical applications of the techniques
5. Personal learning insights

Each 心得 should be thoughtful and demonstrate deep understanding of the DSP concepts.

## Final Notes

- Arthur is an electronics major at NTUT
- These are for his Thursday DSP lab report
- Quality is important - these will be graded
- The exercises progress from basic DSP concepts to advanced applications
- Some exercises (like 5-3) may need creative interpretation since we can't use external software

Good luck with the remaining implementations!