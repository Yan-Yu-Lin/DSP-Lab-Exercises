# 數位訊號處理實習 DSP Lab Exercises

## 專案簡介 Project Introduction

這個專案包含了數位訊號處理實習課程的所有練習題目與實作解答。每個練習都使用 Python 實現，並包含詳細的註解說明和視覺化結果。

This project contains all practice exercises and implementation solutions for the Digital Signal Processing Lab course. Each exercise is implemented in Python with detailed comments and visualization results.

**學生 Student**: 林晏宇 Arthur Lin  
**課程 Course**: 數位訊號處理實習 Digital Signal Processing Lab  
**學期 Semester**: 113-1

---

## 目錄 Table of Contents

- [Lab 1: 基礎訊號處理 Basic Signal Processing](#lab-1-基礎訊號處理-basic-signal-processing)
  - [Practice 1-2: 離散 Sinc 函數 Discrete Sinc Function](#practice-1-2-離散-sinc-函數-discrete-sinc-function)
  - [Practice 1-3: 雙頻訊號取樣 Two-tone Signal Sampling](#practice-1-3-雙頻訊號取樣-two-tone-signal-sampling)
- [Lab 2: 卷積運算 Convolution](#lab-2-卷積運算-convolution)
  - [Practice 2-1: 手動實作卷積 Manual Convolution](#practice-2-1-手動實作卷積-manual-convolution)
  - [Practice 2-2: 矩陣乘法卷積 Matrix Multiplication Convolution](#practice-2-2-矩陣乘法卷積-matrix-multiplication-convolution)
- [Lab 3: 傅立葉轉換 Fourier Transform](#lab-3-傅立葉轉換-fourier-transform)
  - [Practice 3-1: DTFT 實作 DTFT Implementation](#practice-3-1-dtft-實作-dtft-implementation)
  - [Practice 3-2: DFT 點數比較 DFT Points Comparison](#practice-3-2-dft-點數比較-dft-points-comparison)
  - [Practice 3-3: FFT 實作 FFT Implementation](#practice-3-3-fft-實作-fft-implementation)
  - [Practice 3-4: 雙頻訊號 FFT Two-tone Signal FFT](#practice-3-4-雙頻訊號-fft-two-tone-signal-fft)
  - [Practice 3-5: 混疊效應 Aliasing Effect](#practice-3-5-混疊效應-aliasing-effect)
  - [Practice 3-6: 圓形卷積 Circular Convolution](#practice-3-6-圓形卷積-circular-convolution)
- [Lab 4: 濾波器設計 Filter Design](#lab-4-濾波器設計-filter-design)
  - [Practice 4-1: 濾波與卷積比較 Filtering vs Convolution](#practice-4-1-濾波與卷積比較-filtering-vs-convolution)
  - [Practice 4-2: 補償系統設計 Compensating System Design](#practice-4-2-補償系統設計-compensating-system-design)
- [Lab 5: 音訊處理 Audio Processing](#lab-5-音訊處理-audio-processing)
  - [Practice 5-1: 升降取樣 Up/Down Sampling](#practice-5-1-升降取樣-updown-sampling)
  - [Practice 5-2: 音樂生成 Music Generation](#practice-5-2-音樂生成-music-generation)
  - [Practice 5-3: 消除人聲 De-vocalization](#practice-5-3-消除人聲-de-vocalization)
- [Lab 6: 進階濾波器 Advanced Filters](#lab-6-進階濾波器-advanced-filters)
  - [Practice 6-1: Chebyshev 濾波器 Chebyshev Filter](#practice-6-1-chebyshev-濾波器-chebyshev-filter)
- [Lab 7: 影像處理 Image Processing](#lab-7-影像處理-image-processing)
  - [Practice 7-1: 移動平均濾波器 Moving Average Filter](#practice-7-1-移動平均濾波器-moving-average-filter)

---

## Lab 1: 基礎訊號處理 Basic Signal Processing

### Practice 1-2: 離散 Sinc 函數 Discrete Sinc Function

**題目說明**  
繪製離散時間訊號 x[n] = sin(ωc·n)/(π·n)，其中 ωc = 0.2π，-30 ≤ n ≤ 30。

**實作重點**
1. **處理零點奇異性**：當 n=0 時，直接計算會產生 0/0 的不定型，需要使用羅必達法則（L'Hôpital's rule）求極限值
2. **Sinc 函數特性**：這是數位訊號處理中的基本函數，代表理想低通濾波器的脈衝響應
3. **視覺化方式**：使用 stem plot 展示離散特性，同時用 line plot 顯示函數包絡線

**程式實作技巧**
```python
# 避免除以零的處理方式
non_zero_idx = n != 0
x[non_zero_idx] = np.sin(w_c * n[non_zero_idx]) / (np.pi * n[non_zero_idx])
x[n == 0] = w_c / np.pi  # 使用極限值
```

**學習心得**

這個練習讓我深入理解了 sinc 函數在數位訊號處理中的重要性。Sinc 函數是理想低通濾波器的時域表現形式，其幾個關鍵特性包括：

1. **主瓣特性**：函數在 n=0 處達到最大值 ωc/π = 0.2，這代表了濾波器的直流增益
2. **零點位置**：零點出現在 n = ±5, ±10, ±15... 的位置，恰好是 π/ωc 的整數倍
3. **衰減特性**：振幅隨著 |n| 增加而逐漸衰減，但衰減速度較慢（1/n）

透過實作，我體會到數學理論與程式實現之間的差異。在理論上，sinc 函數是連續的，但在數位實現時必須考慮離散取樣和數值計算的限制。特別是在 n=0 的處理上，展現了數值分析的重要性。

![Practice 1-2 Result](image/pr-1-2.png)

**結果與討論**

從輸出圖形可以觀察到：
- Stem plot 清楚展示了離散訊號的特性，每個取樣點都是獨立的
- Line plot 則幫助我們理解函數的整體趨勢和包絡線
- 最大值確實出現在 n=0，數值為 0.2000
- 函數呈現典型的 sinc 形狀，具有主瓣和逐漸衰減的旁瓣

這個基礎練習為後續的濾波器設計和頻域分析奠定了重要基礎。理解 sinc 函數的特性，對於掌握取樣定理、濾波器設計和頻譜分析都至關重要。

---

### Practice 1-3: 雙頻訊號取樣 Two-tone Signal Sampling

**題目說明**  
給定連續時間訊號 x(t) = 2sin(2π×10t) + sin(2π×30t)：
- (a) 繪製 x(t) 在 0 ≤ t ≤ 0.5 秒的圖形
- (b) 以 fs = 100 Hz 取樣，繪製 x[n] 在 0 ≤ n ≤ 50 的圖形

**實作重點**
1. **雙頻訊號合成**：結合 10 Hz 和 30 Hz 兩個正弦波，振幅分別為 2 和 1
2. **取樣定理驗證**：確認取樣頻率 fs = 100 Hz 是否滿足奈奎斯特準則
3. **頻譜分析**：使用 FFT 驗證訊號的頻率成分
4. **視覺化比較**：同時展示連續訊號與離散取樣的差異

**程式實作技巧**
```python
# 定義連續訊號函數，方便重複使用
def x_continuous(t):
    return 2 * np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 30 * t)

# 取樣時間點的計算
n = np.arange(0, 51)
t_sampled = n * Ts  # Ts = 1/fs
```

**學習心得**

這個練習讓我實際體驗了取樣定理的重要性。透過實作雙頻訊號的取樣過程，我深刻理解了以下概念：

1. **奈奎斯特定理的實際應用**：訊號最高頻率為 30 Hz，因此最小取樣頻率需要 > 60 Hz。我們使用 100 Hz 的取樣率，提供了足夠的餘裕，確保不會發生混疊現象。

2. **拍頻現象的觀察**：當兩個不同頻率的正弦波疊加時，會產生拍頻（beat）效應。從時域圖形可以清楚看到振幅的週期性變化，這是因為 10 Hz 和 30 Hz 的最小公倍數效應。

3. **離散與連續的關係**：透過將連續訊號和取樣點重疊顯示，我可以直觀地看到離散取樣如何「捕捉」連續訊號的特徵。每個取樣點都準確地落在連續曲線上。

4. **頻域驗證的重要性**：單從時域圖形可能難以確認所有頻率成分，但透過 FFT 分析，可以清楚地看到 10 Hz 和 30 Hz 的頻譜峰值，且振幅比例正確（2:1）。

![Practice 1-3 Time Domain](image/pr-1-3-1.png)

![Practice 1-3 Frequency Domain](image/pr-1-3-2.png)

**結果與討論**

實驗結果完美驗證了理論預期：

1. **時域觀察**：
   - 連續訊號呈現平滑的複合波形
   - 取樣點準確捕捉了訊號的變化
   - 可以觀察到約 0.1 秒的拍頻週期（10 Hz 的倒數）

2. **頻域分析**：
   - FFT 明確顯示兩個頻率峰值：10 Hz 和 30 Hz
   - 峰值高度比例約為 2:1，符合原始訊號的振幅設定
   - 沒有其他頻率成分，證明取樣正確無混疊

3. **實務考量**：
   - 實際應用中，100 Hz 的取樣率對於 30 Hz 訊號提供了 1.67 倍的過取樣
   - 這種過取樣有助於後續的數位濾波處理
   - 如果取樣率降低到 60 Hz 以下，將會產生嚴重的混疊失真

這個練習強化了我對取樣定理的理解，並展示了時域和頻域分析相互驗證的重要性。

---

## Lab 2: 卷積運算 Convolution

### Practice 2-1: 手動實作卷積 Manual Convolution

**題目說明**  
不使用內建函數 conv() 或 convolve()，根據卷積方程式（2.1）手動實作離散卷積運算。

**實作重點**
1. **卷積定義**：離散卷積定義為 y[n] = Σ x[k] × h[n-k]，需要逐一計算每個輸出樣本
2. **雙層迴圈結構**：外層迴圈遍歷輸出索引 n，內層迴圈遍歷輸入索引 k
3. **邊界條件檢查**：確保 h[n-k] 的索引在有效範圍內（0 ≤ n-k < len_h）
4. **輸出長度計算**：輸出序列長度為 len(x) + len(h) - 1

**程式實作技巧**
```python
# 卷積核心運算邏輯
for n in range(len_y):
    for k in range(len_x):
        if 0 <= n - k < len_h:
            y[n] += x[k] * h[n - k]
```

**學習心得**

這個練習讓我深入理解了卷積運算的本質和實作細節。卷積是數位訊號處理中最基本且重要的運算之一，透過手動實作，我能夠：

首先，我體會到卷積的物理意義。卷積不僅僅是數學運算，它代表了一個系統對輸入訊號的響應過程。當輸入訊號 x[n] 通過一個具有脈衝響應 h[n] 的線性時不變系統時，輸出 y[n] 就是這兩者的卷積。這個過程可以理解為：系統「記憶」了過去所有的輸入，並根據其脈衝響應給予不同的權重。

其次，我發現了卷積運算的時間複雜度問題。手動實作的時間複雜度為 O(N×M)，這對於長序列來說是相當耗時的。這也讓我理解為什麼在實際應用中，我們會使用 FFT 來加速卷積運算，將時間複雜度降低到 O(N log N)。

第三，透過逐步計算的展示，我更清楚地看到了每個輸出樣本是如何由多個輸入樣本的加權組合而來。例如，y[2] = x[0]×h[2] + x[1]×h[1] + x[2]×h[0]，這種組合方式完美地體現了線性系統的疊加特性。

最後，我注意到不同信號形狀的卷積會產生不同的結果。矩形訊號與三角形訊號的卷積產生梯形，這種形狀變化反映了系統的平滑化效應。這讓我聯想到實際應用中的濾波器設計，其中不同的脈衝響應會對訊號產生不同的處理效果。

![Practice 2-1 Basic Convolution](image/pr-2-1-1.png)

![Practice 2-1 Rectangle Triangle Convolution](image/pr-2-1-2.png)

**結果與討論**

實驗結果證實了手動實作的正確性：

1. **數值準確性**：手動實作與 NumPy 內建函數的結果差異為 0.00e+00，證明演算法實作完全正確

2. **訊號形狀分析**：
   - 第一個例子中，序列 [1,2,3] 與 [1,0.5,0.25] 的卷積產生 [1,2.5,4.25,2,0.75]
   - 矩形訊號與三角形訊號的卷積產生對稱的梯形，體現了卷積的平滑化特性

3. **計算過程透明度**：逐步計算展示了每個輸出樣本的構成，有助於理解卷積運算的機制

4. **實際應用意義**：這種手動實作雖然效率較低，但對於理解系統響應、濾波器設計和訊號分析的原理具有重要價值

通過這個練習，我不僅掌握了卷積的程式實作技巧，更重要的是建立了對線性系統理論的直觀理解。

---

### Practice 2-2: 矩陣乘法卷積 Matrix Multiplication Convolution

**題目說明**  
給定兩個訊號 x₁[n] = n % 5 和 x₂[n] = n % 4，範圍 1 ≤ n ≤ 10000，使用矩陣乘法方法實作它們的卷積。

**實作重點**
1. **Toeplitz 矩陣構造**：將一個訊號轉換為卷積矩陣，其中每列都是前一列的移位版本
2. **矩陣乘法等效性**：卷積運算 y = x₁ * x₂ 等效於矩陣乘法 y = H × x
3. **週期性訊號特性**：模運算產生週期性的訊號模式，period(x₁) = 5, period(x₂) = 4
4. **記憶體與計算效率**：大矩陣運算的記憶體需求和計算複雜度分析

**程式實作技巧**
```python
# 建立卷積矩陣的核心邏輯
for i in range(N):
    for j in range(len_h):
        if i + j < output_len:
            conv_matrix[i + j, i] = h[j]
```

**學習心得**

這個練習讓我從線性代數的角度深入理解卷積運算，並體會到不同實作方法的優缺點。透過矩陣乘法實作卷積，我獲得了以下重要認知：

首先，我真正理解了卷積的線性性質。當我將卷積運算轉換為矩陣乘法 y = Hx 時，這個線性變換的本質變得非常清楚。卷積矩陣 H 是一個 Toeplitz 矩陣，它的結構揭示了卷積運算的時移不變性質。每一列都是前一列向下移位的結果，這完美地對應了卷積定義中的 h[n-k] 項。

其次，我體會到了演算法複雜度的實際影響。對於長度為 10,000 的訊號，卷積矩陣包含約 2 億個元素，佔用大量記憶體。更重要的是，矩陣乘法的時間複雜度為 O(N²M)，比傳統卷積的 O(NM) 還要高。實驗結果顯示，矩陣方法比 NumPy 的優化實作慢了 566 倍，這讓我深刻理解了演算法優化的重要性。

第三，我觀察到週期性訊號的有趣性質。x₁[n] = n % 5 和 x₂[n] = n % 4 分別具有週期 5 和 4，它們的卷積結果呈現準週期性，這是因為兩個週期的最小公倍數為 20。這種週期性分析在通訊系統和訊號分析中非常重要。

最後，我理解了稀疏矩陣的概念。雖然卷積矩陣很大，但它是高度稀疏的，大部分元素為零。這種結構可以通過特殊的稀疏矩陣演算法來優化，但對於一般的矩陣乘法來說，仍然需要處理所有元素。

![Practice 2-2 Signal Patterns](image/pr-2-2-1.png)

![Practice 2-2 Convolution Result](image/pr-2-2-2.png)

![Practice 2-2 Matrix Structure](image/pr-2-2-3.png)

**結果與討論**

實驗結果證實了矩陣乘法方法的正確性與局限性：

1. **數值正確性**：矩陣乘法與標準卷積的結果完全一致，最大誤差為 0.00e+00

2. **性能比較**：
   - 矩陣乘法：14.786 秒
   - NumPy 卷積：0.026 秒
   - 性能差距：566 倍

3. **記憶體需求**：
   - 卷積矩陣大小：19,999 × 10,000
   - 總元素數：約 2 億個
   - 大部分為零（稀疏矩陣）

4. **實際應用意義**：
   - 教學價值：清楚展示卷積的線性代數本質
   - 理論重要性：為理解濾波器矩陣表示法奠定基礎
   - 實務考量：大型訊號處理中不實用，應使用 FFT 方法

這個練習讓我深刻理解了理論與實踐之間的差異。雖然矩陣乘法方法在計算效率上不如傳統方法，但它提供了卷積運算的另一種數學觀點，對於理解線性系統理論和後續的濾波器設計具有重要意義。

---

## Lab 3: 傅立葉轉換 Fourier Transform

### Practice 3-1: DTFT 實作 DTFT Implementation

**題目說明**  
不使用 exp()、abs() 和 angle() 函數，根據基本數學公式實作離散時間傅立葉轉換（DTFT）。

**實作重點**
1. **歐拉公式展開**：將 e^(-jωn) 分解為 cos(ωn) - j×sin(ωn)
2. **實部計算**：X_R(e^jω) = Σ x[n]cos(ωn)
3. **虛部計算**：X_I(e^jω) = -Σ x[n]sin(ωn)
4. **幅度與相位**：使用 atan2 正確處理所有象限的相位角

**程式實作技巧**
```python
# DTFT 核心計算
for i, freq in enumerate(w):
    X_R[i] = np.sum(x * np.cos(freq * n))
    X_I[i] = -np.sum(x * np.sin(freq * n))
magnitude = np.sqrt(X_R**2 + X_I**2)
phase = np.arctan2(X_I, X_R)
```

**學習心得**

這個練習讓我從最基本的數學原理理解傅立葉轉換的本質，而不僅僅是使用現成的函數。透過手動實作 DTFT，我獲得了深層的理解：

首先，我真正理解了歐拉公式的威力和意義。當我將複數指數 e^(-jωn) 分解為 cos(ωn) - j×sin(ωn) 時，傅立葉轉換的物理意義變得極為清晰：它實際上是將訊號與不同頻率的正弦和餘弦波相關聯的過程。實部反映了訊號與餘弦波的相關性，虛部則反映了與正弦波的相關性（帶負號）。

其次，我體會到了頻域分析的直觀性。矩形脈衝的頻譜呈現 sinc 函數形狀，其零點位置恰好在 2π/N 的整數倍處（N=5）。這種規律性不是偶然的，而是反映了矩形窗函數的固有特性。指數衰減序列則產生平滑的頻譜響應，這與其在時域的平滑特性相對應。

第三，我學會了正確處理複數的幅度和相位計算。特別是 atan2 函數的使用，它比簡單的 atan 函數更強大，能夠正確處理所有四個象限的情況。這在實際的訊號處理中非常重要，因為相位資訊的正確性直接影響到後續的分析和處理。

最後，我發現了理論與實務的微妙差異。手動 DTFT 與 FFT 的結果存在小幅差異（約 10^-3 到 10^-4 量級），這主要是因為 DTFT 是連續頻率的概念，而 FFT 是在離散頻率點上的採樣。這種差異讓我更深刻地理解了數位訊號處理中離散化帶來的影響。

透過逐步計算的過程，我看到了每個頻率點是如何通過與所有時域樣本的加權組合而得出的。這種透明的計算過程讓我對傅立葉轉換不再感到神秘，而是成為了一個可以理解和掌握的數學工具。

![Practice 3-1 Rectangular Pulse DTFT](image/pr-3-1-1.png)

![Practice 3-1 Exponential Decay DTFT](image/pr-3-1-2.png)

**結果與討論**

實驗結果驗證了手動實作的正確性和理論的一致性：

1. **數值驗證**：
   - 在 ω = π/2 處，手動 DTFT 幅度為 1.0063，FFT 為 1.0000
   - 在 ω = π 處，差異僅為 1.19×10^-4
   - 小幅差異源於連續與離散頻率採樣的差異

2. **頻譜特性分析**：
   - 矩形脈衝產生 sinc 形狀的頻譜，零點在 2π/5 的整數倍處
   - 指數衰減序列呈現平滑的低通特性
   - 相位譜正確反映了訊號的時域特性

3. **數學原理驗證**：
   - 歐拉公式分解完全正確：e^(-jωn) = cos(ωn) - j×sin(ωn)
   - 實部和虛部的計算符合理論預期
   - atan2 函數正確處理了所有象限的相位

4. **計算透明度**：
   - 逐步計算展示了 DTFT 的每個細節
   - 手動計算結果與理論公式完全一致
   - 無需使用複數指數函數即可實現完整的頻域分析

這個練習不僅讓我掌握了 DTFT 的實作技巧，更重要的是建立了對頻域分析基本原理的深刻理解，為後續的 FFT 學習和頻譜分析打下了堅實基礎。

---