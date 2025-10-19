# elec5305-project-510019277  
**Comparative Study of Music Synthesis Techniques**

---

## 🎧 Project Overview

This project investigates **music synthesis** as a concrete application of digital audio signal processing, focusing on a comparative analysis between **Additive Synthesis** and **Frequency Modulation (FM) Synthesis**.  
Additive synthesis constructs tones by summing harmonically related sinusoids, offering explicit control over harmonic structure but requiring high computational cost for complex timbres.  
FM synthesis, in contrast, achieves rich, evolving timbres efficiently by modulating a carrier with a secondary oscillator, producing complex sidebands from a small parameter set.  

The goal of this project is to analyze and implement both techniques in **MATLAB**, visualize their spectral structures, and apply **machine learning** to automatically predict FM synthesis parameters from spectral features — bridging traditional DSP with data-driven modeling.

---

## 📚 Background and Motivation

Early sound synthesis relied on additive methods, representing tones as sums of harmonics based on Fourier principles.  
FM synthesis (John Chowning, 1973) revolutionized digital sound design by enabling complex spectra generation through frequency modulation, leading to efficient real-time synthesis in instruments such as the **Yamaha DX7**.  

Additive synthesis provides precise harmonic control, suitable for instrument modeling, whereas FM synthesis provides expressive timbre control with fewer parameters.  
This study builds both systems, compares their **spectral behavior and efficiency**, and introduces an ML-based regression model to predict FM parameters from spectral data, following the philosophy of **Differentiable DSP (DDSP)**.

---

## 🧰 Implementation Overview

**Tools & Platform:** MATLAB (Signal Processing Toolbox)  
**Core techniques:**  
- Additive and FM Synthesis  
- Spectral Analysis (FFT, STFT, spectral centroid, bandwidth, rolloff, MFCC)  
- Ridge Regression Machine Learning (predicting FM parameters)  
- Evaluation via log-STFT spectral loss and perceptual comparison  

**Data Source:**  
All audio is synthesized directly within MATLAB using the implemented algorithms.  
No external datasets are required.  
Each tone (A4 = 440 Hz, 0.8 s duration) is generated with different FM parameters `(r, I)`.

---

## 🧪 Current Progress

Implemented baseline **Additive** and **FM** synthesizers in MATLAB, generated demo audio files and visualized their spectrograms.

- **Additive synthesis:** distinct harmonic lines spaced at multiples of the fundamental frequency.  
- **FM synthesis:** rich sideband structures depending on frequency ratio `r` and modulation index `I`.

### 🔊 Audio Demos
- [Additive demo](results/audio/additive_A4.wav)  
- [FM demo (r=2, I=4)](results/audio/fm_A4_r2_I4.wav)

### 🖼️ Spectrograms
- [Additive spectrogram](results/plots/additive_A4_spec.png)  
- [FM spectrogram](results/plots/fm_A4_r2_I4_spec.png)

---

## 📊 Spectral Observations

A parameter sweep was performed for `r ∈ {0.5, 1, 2}` and `I ∈ {0, 2, 4, 6}`.  
Key findings from frequency-domain analysis:

- Increasing **modulation index (I)** broadens the bandwidth and raises the **spectral centroid**, producing brighter timbres.  
- Changing **ratio (r)** shifts sideband spacing, altering harmonic alignment and perceived color.  
- Additive synthesis shows discrete harmonics, while FM synthesis generates continuous spectral clusters.

---

## 🤖 Machine Learning Results – FM Parameter Prediction

We synthesized an FM dataset on a grid of ratios `r ∈ {0.5, 1, 2}` and indices `I ∈ {0, 2, 4, 6}` (12 samples).  
Spectral features (centroid, bandwidth, rolloff, ZCR, MFCC mean/std) were extracted and used to train **ridge regression models** predicting `(r, I)`.

**Test metrics** (see [`results/ml/metrics.csv`](results/ml/metrics.csv)):  
| Metric | r | I |
|:--|:--:|:--:|
| MAE | 0.554 | 1.776 |
| R² | 0.169 | 0.268 |
| Mean log-STFT L₁ |    –    |    3.530    |

**Plots:**  
- [True vs Predicted Scatter](results/plots/ml_scatter_true_vs_pred.png)  
- [Spectrogram Comparison (True vs Predicted)](results/plots/ml_spectrogram_true_vs_pred.png)

**Observations:**  
- Predictions show moderate alignment between true and estimated parameters.  
- `I` influences timbral brightness strongly, while `r` is subtler to regress due to overlapping spectral patterns.  
- The ML module successfully demonstrates a **data-driven synthesis control** pipeline—an interpretable counterpart to DDSP’s learned parameter estimation.

---

## 📈 Future Work

1. **Expand dataset:** increase `(r, I)` grid density and tone duration for improved ML regression accuracy.  
2. **Add ADSR envelopes:** simulate realistic instrument dynamics.  
3. **Integrate additive synthesis** into the ML stage for feature comparison.  
4. **Evaluate computational efficiency** (runtime vs model accuracy).  
5. **Extend toward DDSP-style models**, using neural networks to learn continuous synthesis control.

---

## 📦 Repository Structure

elec5305-project-510019277/
│
├─ src/
│ └─ synthesis/
│ ├─ additive_synth.m
│ ├─ fm_synth.m
│ └─ adsr.m
│
├─ results/
│ ├─ audio/ ← synthesized WAV files
│ ├─ plots/ ← spectrograms & ML scatter plots
│ └─ ml/ ← metrics.csv and ML outputs
│
├─ main_synthesis_demo.mlx
├─ main_ml_pipeline.mlx
└─ Project Proposal.pdf


---

## 🧾 References

- Reid, G. (2000). *An Introduction to Additive Synthesis.* Sound on Sound.  
- Reid, G. (2000). *An Introduction to Frequency Modulation.* Sound on Sound.  
- Smith, J. O. (n.d.). *Spectral Audio Signal Processing.* DSPRelated.  
- Synclavier Digital (2019). *What is Additive Synthesis Anyway?*  
- Wikipedia (2025). *Additive Synthesis.*  
- Wikipedia (2024). *Frequency Modulation Synthesis.*  
- Chowning, J. M. (1973). *The Synthesis of Complex Audio Spectra by Means of Frequency Modulation.* JAES.

---

## 👤 Author
**Yi Fan (SID 510019277)**  
University of Sydney – ELEC5305 Acoustics, Speech and Signal Processing  
Semester 2 – 2025



[Download Project Proposal (PDF)](Project%20Proposal.pdf)
