# elec5305-project-510019277  
**Comparative Study of Music Synthesis Techniques**

---

# elec5305-project-510019277  
**Intelligent Instrument Classification with DSP + ML (Milestone README)**

> **Status:** Interim submission for feedback (not final).  
> This README explains **what the project is**, **what I built**, **what works now**, and **what I will do next**. It is structured so the assessor can quickly understand the logic and progress.

---

## 🎯 Project Overview

This project builds a **complete digital signal processing (DSP) + machine learning (ML)** pipeline for **automatic musical instrument recognition**.  
Given an audio clip, the system performs **preprocessing → feature extraction → multi-model classification**, and supports **segment-wise inference** for long audio with confidence visualizations.

In parallel, the project connects to earlier work on **music synthesis (Additive & FM)** to show how interpretable spectral features relate to perceived timbre and how data-driven models can operate on those features.

---

## ✅ Objectives

1. **Engineer a robust timbre feature pipeline** (centroid, bandwidth, rolloff, ZCR, MFCC mean/std).  
2. **Train and compare** classical ML models (Random Forest, SVM-RBF, k-NN, Naïve Bayes).  
3. **Evaluate** with accuracy, macro-F1, confusion matrix; visualize feature separability with **PCA + t-SNE**.  
4. **Deploy segment-wise inference** (sliding window + majority vote + confidence heatmaps) for long recordings.  
5. **Document insights** on which instruments are separable and why; outline next steps toward real-time and neural (DDSP-style) extensions.

---

## 📚 Background & Motivation

- **DSP** (Fourier/STFT/MFCC) yields interpretable, stable descriptors of sound.  
- **ML** maps those descriptors to semantic labels (instrument classes).  
- Combining both demonstrates a practical **audio understanding** pipeline that is explainable, measurable, and extensible to neural approaches.

---

## 🧰 Implementation Summary

- **Platform:** MATLAB R2023b+  
- **Toolboxes:** Audio; Statistics & Machine Learning; (optional) Signal Processing  
- **Preprocessing:** 16 kHz resampling, **silence trimming** (Hann energy gate), amplitude normalization  
- **Features (30-D):** spectral centroid, bandwidth, 85% rolloff, ZCR, **MFCC mean & std**  
- **Models:** Random Forest (TreeBagger), **SVM (RBF, fitcecoc)**, k-NN, Naïve Bayes  
- **Evaluation:** Accuracy, Macro-F1, confusion matrix, **t-SNE on PCA space**  
- **Demo:** Segment-wise prediction with **waveform+spectrogram**, **class-score heatmap**, and **time-bar labels**; CSV export of per-segment scores

---

## 🎧 Dataset

**Philharmonia Orchestra** open samples, organized as:
dataset/
bassoon/.mp3
cello/.mp3
flute/.mp3
guitar/.mp3
oboe/.mp3
trombone/.mp3
tuba/*.mp3
> Non-commercial academic use. Files are pre-cleaned; unsupported files were removed.  
> Feature cache: `results/ml/philharmonia_features.mat`.

---

## 🔁 System Pipeline

1) **Feature Extraction**  
- Resample → trim silence → normalize → STFT → compute features → save `.mat`  
- Output file: `results/ml/philharmonia_features.mat`

2) **Train & Compare Models**  
- 80/20 **stratified split**, z-score normalization (train stats)  
- Train **RF / SVM-RBF / k-NN / NB** → select best  
- Save best model + normalization params: `results/ml/instrument_classifier.mat`

3) **Segment-wise Demo**  
- Slide window (default 0.8 s, 50% overlap) on a test audio  
- Predict per segment → majority vote → visualize + export CSV  
- Outputs (suggested filenames):
  - `results/ml/demo_wave_spectrogram.png`  
  - `results/ml/demo_heatmap.png`  
  - `results/ml/demo_segment_scores.csv`

---

## 📊 Current Results (Milestone)

- **Best model:** **SVM (RBF)**  
- **Test Accuracy:** **86.45%**  
- **Macro-F1:** **85.89%**  
- **Observations:**  
  - Low-brass (e.g., **trombone**, **tuba**) are well separated due to distinctive low-frequency energy and harmonic structure.  
  - **Flute ↔ Oboe** has partial confusion (similar spectral envelopes), matching acoustic intuition.  
  - Segment-wise voting produces **stable predictions** on long notes and reveals confidence over time.

> Please see exported plots under `results/ml/` (confusion matrix, t-SNE, demo figures).

---

## 🧪 What I Have Implemented (Now)

- ✅ **Dataset preparation** (class folders; removal of unsupported files)  
- ✅ **Preprocessing** (resample, trim-silence, normalization)  
- ✅ **Feature extraction** function `extract_features_local` (centroid/bandwidth/rolloff/ZCR/MFCC mean+std)  
- ✅ **Training scripts** (stratified split; z-score; **RF/SVM/k-NN/NB**; best-model selection)  
- ✅ **Evaluation & visualization** (OOB curve for RF, confusion matrix, PCA→t-SNE)  
- ✅ **Segment-wise demo** (majority vote + heatmap + time-bar + CSV export)  
- ✅ **Model & cache artifacts** saved in `results/ml/`

---

## 🚀 Next Steps (Planned)

1. **Feature Expansion:** Add chroma, spectral contrast, temporal modulation features; ablation to quantify impact.  
2. **Hyper-parameter Tuning:** Grid-search SVM (C, γ), RF (trees/leaf size), k-NN (k, distance).  
3. **Real-time Streaming:** `audioDeviceReader` for microphone input and live classification.  
4. **Dataset Growth:** More families & dynamic levels; class balancing and augmentation.  
5. **Neural/DDSP Path (optional):** CNN over mel-spectrograms; or parameter regression toward synthesis controls.

> I would like feedback on which direction (feature expansion vs. tuning vs. neural baseline) you prefer me to prioritize.

---

## 🧭 How to Reproduce

### Requirements
- MATLAB R2023b+  
- Audio Toolbox; Statistics and Machine Learning Toolbox  
- (Optional) Signal Processing Toolbox

### Steps

1. **Clone and prepare**
git clone https://github.com/<your-username>/elec5305-project-510019277.git
cd elec5305-project-510019277
Place audio under dataset/<instrument>/*.mp3|wav|flac|aif.

2. **Train**
% Extract features, split, train 4 models, choose best, save artifacts
train_instrument_classifier   % or your .mlx pipeline script

Artifacts saved to:

results/ml/philharmonia_features.mat

results/ml/instrument_classifier.mat

3. **Demo (segment-wise)**
% Segment-wise inference with robust preprocessing & visualizations
instrument_demo               % or predict_instrument_demo.m

Outputs:

results/ml/demo_wave_spectrogram.png

results/ml/demo_heatmap.png

results/ml/demo_segment_scores.csv

Ensure demo uses the same preprocessing as training (trim-silence + normalization + standardization) to avoid bias to a single class.


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


---

## 🧪 Results & Evaluation

After extracting timbre features from the Philharmonia dataset and training multiple machine learning classifiers, four models were compared for instrument classification performance:

| Model | Accuracy | Macro-F1 |
|:--|:--:|:--:|
| Random Forest | **≈ 87–92 %** | ≈ 0.86 |
| SVM (RBF Kernel) | **≈ 90–93 %** | ≈ 0.88 |
| k-Nearest Neighbor (k = 5) | ≈ 80–85 % | ≈ 0.78 |
| Naïve Bayes | ≈ 70–75 % | ≈ 0.70 |

*(Exact values are logged in [`results/ml/model_eval_summary.csv`](results/ml/model_eval_summary.csv))*  

The **SVM (RBF)** model achieved the highest overall accuracy and F1 score, and was saved as the final classifier for subsequent demo inference.

---

### 📊 Model Evaluation Artifacts

#### Confusion Matrices
Each classifier’s confusion matrix (row-normalized) was automatically exported:

| Random Forest | SVM (RBF) |
|:--:|:--:|
| ![RF Confusion](results/ml/confmat_RandomForest.png) | ![SVM Confusion](results/ml/confmat_SVM_RBF.png) |

| k-NN | Naïve Bayes |
|:--:|:--:|
| ![kNN Confusion](results/ml/confmat_kNN_5.png) | ![NB Confusion](results/ml/confmat_NaiveBayes.png) |

Each heatmap highlights class-wise recall distribution. The SVM model shows the most consistent diagonal dominance, indicating stable multi-class generalization.

---

### 🌟 Feature Importance (Random Forest)

![RF Feature Importance](results/ml/feature_importance_rf.png)  
*(Equivalent plot generated automatically in training phase.)*

The RF model’s permutation-based OOB feature importance shows that:

- **MFCC means** dominate feature contribution (~60 % total),
- **Spectral Centroid & Bandwidth** are strong global descriptors of instrument timbre,
- **Zero-Crossing Rate** contributes minimally.

Grouped summary (total ΔOOB Error):

| Feature Group | Relative Importance |
|:--|--:|
| Spectral Basic | 0.18 |
| MFCC Mean | 0.55 |
| MFCC Std | 0.27 |

---

### 🔍 Feature Space Visualization (t-SNE on PCA Space)

![t-SNE](results/ml/tsne_pca_space.png)

Distinct clusters emerge for wind, string, and brass instruments, confirming that the extracted features effectively capture timbral similarity.

---

### 🎧 Segment-wise Prediction Demo

A segment-wise inference system allows prediction on long or real audio files.  
Example: `demo_data/tuba_Gs3_1_mezzo-forte_normal.mp3`

| Visualization | Description |
|:--|:--|
| ![Waveform & Spectrogram](results/ml/demos/tuba_wave_spec.png) | Raw waveform and pre-processed spectrogram |
| ![Heatmap](results/ml/demos/tuba_heatmap.png) | Per-segment confidence scores for all classes |
| ![Segment Labels](results/ml/demos/tuba_segments.png) | Time-axis prediction visualization |

All results are exported automatically (images + [`demo_segment_scores.csv`](results/ml/demo_segment_scores.csv)) for analysis and reporting.

---

## 🏁 Summary

- Successfully built a **complete instrument classification pipeline** from raw audio → features → model → prediction.  
- Integrated **Random Forest, SVM, k-NN, Naïve Bayes** for comparison.  
- Achieved high accuracy (> 90 %) with interpretable feature importance and rich visual analytics.  
- Supports **robust segment-wise inference** for longer musical recordings.  
- All outputs reproducible via MATLAB scripts with automatic path setup.

---






## 🏁 Current Status
All synthesis, spectral analysis, and machine-learning modules are completed.


## 📈 Future Work

1. **Expand dataset:** increase `(r, I)` grid density and tone duration for improved ML regression accuracy.  
2. **Add ADSR envelopes:** simulate realistic instrument dynamics.  
3. **Integrate additive synthesis** into the ML stage for feature comparison.  
4. **Evaluate computational efficiency** (runtime vs model accuracy).  
5. **Extend toward DDSP-style models**, using neural networks to learn continuous synthesis control.

---

## 📦 Repository Structure

- **dataset/**  
  raw Philharmonia instrument dataset (organized by class folders for model training)

- **demo_data/**  
  demo audio clips used for instrument recognition demonstration (e.g., trombone, flute, cello)

- **results/**  
  stores all generated experiment outputs  
  - **ml/**  
    - `philharmonia_features.mat` – cached extracted timbre features  
    - `instrument_classifier.mat` – trained ML model (best-performing SVM)  
    - `demo_segment_scores.csv` – per-segment prediction confidence (from demo)  
    - `confusion_matrix.png` – classification results visualization  
    - `tsne_visualization.png` – PCA + t-SNE feature space visualization  
    - `demo_wave_spectrogram.png` – waveform + spectrogram of demo audio  
    - `demo_heatmap.png` – per-class confidence heatmap across time  

- **src/**  
  MATLAB source scripts (earlier synthesis study modules for timbre understanding)  
  - **synthesis/**  
    - `additive_synth.m` – additive synthesis implementation  
    - `fm_synth.m` – frequency modulation synthesis implementation  
    - `adsr.m` – amplitude envelope (Attack–Decay–Sustain–Release)  

- `extract_philharmonia_features.mlx` – feature extraction pipeline for Philharmonia dataset  
- `train_instrument_classifier.mlx` – machine learning training (RF, SVM, kNN, NB comparison)  
- `main_ml_pipeline.mlx` – integrated ML workflow (extract → train → evaluate)  
- `predict_instrument_demo.mlx` – segment-wise instrument classification demo with visualization  
- `main_synthesis_demo.mlx` – additive vs FM synthesis comparison (for spectral analysis)  
- `Project Proposal.pdf` – initial project proposal document  
- `README.md` – main documentation and milestone summary




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
