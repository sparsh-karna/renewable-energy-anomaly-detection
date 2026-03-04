# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-03-04

### Added

- **Feature Extraction Notebook** (`feature_extraction.ipynb`)
  - Algorithm 1: Statistical + Spectral Features (52 features)
    - Time-domain: mean, std, RMS, skewness, kurtosis, peak-to-peak, zero-crossing rate
    - Frequency-domain: power bands (VLF/LF/HF), band ratios, dominant frequency, spectral entropy
  - Algorithm 2: Discrete Wavelet Transform Features (120 features)
    - 5-level Daubechies-4 decomposition
    - Per-level energy, entropy, mean, std
  - Visualizations: box plots, correlation heatmaps, energy distributions, PCA projections
  - Output: CSV and NPZ files for model training

- **Preprocessing Notebook** (`preprocessing.ipynb`)
  - Complete signal pipeline: load → filter → artifact detection → resample → window → normalize
  - Multi-modal filtering per signal type
  - Artifact interpolation and window quality assessment
  - Train/Calibration/Test split (14/4/4 participants)
  - Multiple output formats (NPZ arrays + metadata)

- **Project Documentation**
  - Comprehensive README with project novelty and structure
  - Requirements.txt with all dependencies
  - .gitignore for data and generated files
  - Changelog (this file)

### Technical Details

- **Preprocessing**: 76,800 samples/trial → 2,560 samples/window (10s @ 256 Hz)
- **Artifact Filtering**: z-score threshold | z | > 5, reject windows >30% artifacts
- **Resampling**: Wrist PPG 100 Hz → 256 Hz to align with infrared sensors
- **Feature Output**: 1,059 total windows → ~900 clean windows across 22 participants

### Testing & Validation

- ✓ Signal filtering with bandpass + notch applied
- ✓ Artifact detection with interpolation
- ✓ Window creation with controlled overlap
- ✓ Z-normalization per window
- ✓ Feature extraction on ~900 clean samples

## [0.1.0] - 2026-02-28

### Initial Setup

- Project structure created
- 22-participant MAUS dataset integrated
- 4 modalities loaded: ECG, fingertip PPG, wrist PPG, GSR
