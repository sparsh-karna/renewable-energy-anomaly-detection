# Uncertainty-Aware Mental Workload Classification for Wearables

**Project Goal:** Predict cognitive workload levels (N-back task difficulty) from physiological wearable signals while quantifying prediction uncertainty using conformal prediction, enabling trustworthy deployment in HCI and safety-critical contexts.

## Project Novelty

This semester project combines three key innovations:

1. **Conformal Prediction for Uncertainty Quantification** — Non-parametric distribution-free approach to output prediction sets (e.g., {low, medium}) instead of single labels when uncertain
2. **Sensor Domain Shift Analysis** — Explicit study of deploying models across different sensor modalities: clinical ECG → fingertip PPG → wrist wearable PPG
3. **Multi-Modal Physiological Fusion** — Combine ECG, PPG (two sources), GSR, with signal-specific preprocessing and frequency-band features

## Dataset: MAUS

- **22 participants** (IDs 002-025)
- **6 Trials per participant** (3 levels × 2 reps): 0-back (low), 2-back (medium), 3-back (high) workload
- **4 Physiological Modalities**:
  - ECG (infrared, 256 Hz)
  - Fingertip PPG (infrared, 256 Hz)
  - Wrist PPG (PixArt optical, 100 Hz → resampled to 256 Hz)
  - GSR/EDA (infrared, 256 Hz)
- **Subjective Ratings**: NASA-TLX workload scores, PSQI sleep quality

## Project Structure

```
├── preprocessing.ipynb              # Data loading, filtering, windowing
├── feature_extraction.ipynb         # Two feature extraction algorithms
├── MAUS/
│   ├── Data/
│   │   ├── Raw_data/               # (ignored, too large)
│   │   └── IBI_sequence/           # (ignored, too large)
│   ├── Subjective_rating/          # (ignored)
│   └── preprocessed/
│       ├── windowed_data.npz       # Main dataset: windows × channels × time
│       ├── splits.npz              # Train/Cal/Test indices
│       ├── features_stat_spectral.csv    # Algorithm 1 features
│       ├── features_dwt.csv              # Algorithm 2 features
│       ├── features_extracted.npz        # Both combined
│       └── *.png                   # Visualizations
├── requirements.txt
├── .gitignore
└── README.md                        # This file
```

## Pipeline Overview

### 1. Preprocessing (`preprocessing.ipynb`)

**Input:** Raw 256Hz (and 100Hz for wrist PPG) physiological signals
**Output:** 10-second windows, labeled with workload class

Steps:
- Bandpass filtering per signal type (ECG: 0.5-40 Hz, PPG: 0.5-8 Hz, GSR: 0.05-5 Hz)
- 50 Hz powerline notch filtering
- Artifact detection via z-score threshold (|z| > 5) and interpolation
- Resampling wrist PPG from 100 Hz → 256 Hz
- Windowing: 10s windows with 50% overlap
- Z-score normalization per window
- **Clean window filtering**: reject windows with >30% artifacts

**Outputs:**
- 4 single-channel arrays: `X_ecg`, `X_ppg_inf`, `X_ppg_pix`, `X_gsr` (shape: N × 1 × 2560)
- 2 fusion arrays: `X_fused` (4 channels), `X_ppg_gsr` (3 channels)
- Labels `y` (0, 1, 2 = low/medium/high)
- Subject-wise train/calibration/test split (14/4/4 participants, 64%/18%/18% windows)

### 2. Feature Extraction (`feature_extraction.ipynb`)

Two complementary algorithms operating on preprocessed windows:

#### Algorithm 1: Statistical + Spectral Features (52 features total)

**Time-domain per channel (7):**
- Mean, Std, RMS, Skewness, Kurtosis, Peak-to-Peak, Zero-Crossing Rate

**Frequency-domain per channel (6):**
- Power in 3 physiological bands (VLF/LF/HF for cardiac; Tonic/Phasic/Fast for GSR)
- Band power ratio (LF/HF)
- Dominant frequency
- Spectral entropy

Rationale: Captures bursts of activity and frequency-specific autonomic nervous system responses.

#### Algorithm 2: Discrete Wavelet Transform Features (120 features total)

**Per channel:**
- 5-level Daubechies-4 decomposition (D1-D5 + approximation A5)
- Per level: Energy, Energy Ratio, Shannon Entropy, Mean, Std

Decomposition frequency bands (@ 256 Hz):
- D1: 64–128 Hz
- D2: 32–64 Hz
- D3: 16–32 Hz
- D4: 8–16 Hz
- D5: 4–8 Hz
- A5: 0–4 Hz

Rationale: Captures multi-resolution transient patterns; better for detecting brief autonomic events.

**Outputs:**
- `features_stat_spectral.csv` — Algorithm 1 table
- `features_dwt.csv` — Algorithm 2 table
- `features_extracted.npz` — Both as NumPy arrays for model training

## Planned Model Architectures

(To be implemented in subsequent notebooks)

### Baseline: Shallow Classifiers
- Logistic Regression, Random Forest, SVM on extracted features

### Deep Learning: 1D CNN for Raw Signals
```python
Input: (Batch, 1 channel, 2560 samples @ 256 Hz = 10 seconds)
  ↓
Conv1D (64 filters, kernel=16) → ReLU → BatchNorm
  ↓
MaxPool1D (4) → Dropout(0.3)
  ↓
Conv1D (128 filters, kernel=8) → ReLU → BatchNorm
  ↓
GlobalAvgPool → Dense(64) → ReLU → Dense(3 classes)
```

### Uncertainty Quantification: Conformal Prediction
1. Train base model (above CNN)
2. Compute non-conformity scores on calibration set
3. At test time: output prediction set {classes} with coverage guarantee

Example: predict {low, medium} when uncertain, reject {high}.

## Deployment Configurations to Test

| Config | Train Modality | Test Modality | Type | Use Case |
|--------|---|---|---|---|
| In-domain | Wrist PPG | Wrist PPG | Baseline | Deploy on same wearable |
| Cross-modal | ECG | Wrist PPG | Robustness | Transfer from clinic → consumer wearable |
| Sensor fusion | PPG + GSR | PPG + GSR | Robustness | Combine multiple sensors for stability |

## Key Results Expected

### Preprocessing
- ✓ Loaded 1,059 windows from 22 participants
- ✓ 18% artifact rate → filtered to ~900 clean windows
- ✓ Balanced 3-way classification (±10% per class)

### Feature Extraction
- Algorithm 1 PCA: ~60% variance in 2 components → modest class separation
- Algorithm 2 (DWT): ~55% variance in 2 components → captures different patterns
- Key separators: ECG power ratios (LF/HF), wrist PPG entropy, GSR phasic energy

### (Planned) Model Performance
- In-domain CNN Acc: ~75%
- Cross-modal CNN Acc: ~60% (domain shift penalty)
- Conformal prediction: ~85% coverage at 95% confidence

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
```bash
jupyter notebook preprocessing.ipynb
# Outputs: MAUS/preprocessed/windowed_data.npz + splits.npz
```

### 3. Run Feature Extraction
```bash
jupyter notebook feature_extraction.ipynb
# Outputs: features_stat_spectral.csv, features_dwt.csv, visualizations
```

### 4. Load Data for Your Own Models
```python
import numpy as np

# Load preprocessed windows
data = np.load('MAUS/preprocessed/windowed_data.npz')
splits = np.load('MAUS/preprocessed/splits.npz', allow_pickle=True)

# Access by modality
X_wrist_ppg = data['X_ppg_pix']      # (N, 1, 2560) - single channel
X_fused = data['X_fused']             # (N, 4, 2560) - all channels

# Get split indices
train_idx = splits['train_mask']
cal_idx = splits['cal_mask']
test_idx = splits['test_mask']

# Load extracted features
features_data = np.load('MAUS/preprocessed/features_extracted.npz')
X_feat_algo1 = features_data['algo1_features']  # (N, 52)
X_feat_algo2 = features_data['algo2_features']  # (N, 120)
```

## Experiment Checklist

- [ ] Feature Extraction Results (completed)
  - [x] Visualize Algorithm 1 (statistical + spectral)
  - [x] Visualize Algorithm 2 (DWT)
  - [x] PCA comparison

- [ ] Baseline Models
  - [ ] Logistic Regression on features
  - [ ] Random Forest
  - [ ] SVM

- [ ] Deep Learning Models
  - [ ] 1D CNN on wrist PPG
  - [ ] 1D GRU for temporal sequence
  - [ ] Multi-channel fusion CNN

- [ ] Conformal Prediction
  - [ ] Calibrate on cal set
  - [ ] Evaluate coverage-accuracy trade-off
  - [ ] Visualize prediction sets

- [ ] Cross-Modality Transfer
  - [ ] Train on ECG → test on wrist PPG
  - [ ] Analyze domain shift with uncertainty
  - [ ] Adversarial adaptation (optional)

- [ ] Report
  - [ ] Write methods section
  - [ ] Summarize results with tables/figures
  - [ ] Discuss limitations & future work

## References

- **Conformal Prediction**: Vovk et al. (2005), Balasubramanian et al. (2014)
- **DWT in Physiology**: Peng & Zheng (2012), Al-Qahtani (2016)
- **Cognitive Workload**: Hart & Staveland (1988) — NASA-TLX
- **HRV/Heart Rate Variability**: Task Force of ESC (1996)

## Authors

- Your Name (if applicable)

## License

MIT (or specify your preferred license)

---

**Note**: This README describes the completed preprocessing and feature extraction phases. Model training, evaluation, and uncertainty quantification notebooks will be added as the project progresses.
