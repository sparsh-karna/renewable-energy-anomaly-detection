# Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 MAUS Raw Dataset (22 subjects)              │
│            ECG, PPG, GSR @ 256Hz | PPG_wrist @ 100Hz       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │    Preprocessing Pipeline        │
        │  (preprocessing.ipynb)          │
        └──────────────────┬───────────────┘
                           │
        ┌──────────────────┴───────────────┐
        │                                  │
        ▼                                  ▼
   ┌─────────────────┐           ┌─────────────────┐
   │   Filtering     │           │   Artifact      │
   │                 │           │   Detection     │
   │ • Bandpass      │           │                 │
   │ • Notch 50Hz    │           │ • Z-score       │
   │ • Per signal    │           │ • Interpolate   │
   └────────┬────────┘           └────────┬────────┘
            │                             │
            └──────────────┬──────────────┘
                           │
                    ┌──────▼──────┐
                    │  Resampling │
                    │  100→256Hz  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │   Windowing     │
                    │  10s, 50% ov.   │
                    │  Z-normalize    │
                    └──────┬──────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
    ┌────────────────────┐        ┌────────────────────┐
    │  Preprocessed      │        │   Train/Cal/Test   │
    │  Windows           │        │   Split Indices    │
    │  (NPZ)             │        │   (NPZ)            │
    │                    │        │                    │
    │ • X_ecg (N,1,2560) │        │ • train_mask      │
    │ • X_ppg_inf        │        │ • cal_mask        │
    │ • X_ppg_pix        │        │ • test_mask       │
    │ • X_gsr            │        │                    │
    │ • X_fused (N,4)    │        │ 64% / 18% / 18%   │
    │ • y (N,)           │        └────────────────────┘
    └────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────┐
  │  Feature Extraction Pipeline         │
  │  (feature_extraction.ipynb)         │
  └────────────────────┬─────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
  ┌────────────────────┐      ┌─────────────────┐
  │  Algorithm 1       │      │  Algorithm 2    │
  │  Stat+Spectral     │      │  DWT            │
  │                    │      │                 │
  │ • Time-domain (7)  │      │ • 5-level db4   │
  │ • Freq-domain (6)  │      │ • Energy (3)    │
  │ • Per channel      │      │ • Entropy (3)   │
  │ • Total: 52 feats  │      │ • Mean/Std (2)  │
  │                    │      │ • Total: 120    │
  └────────┬───────────┘      └────────┬────────┘
           │                          │
           └──────────────┬───────────┘
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
        ▼                                    ▼
   ┌─────────────────┐            ┌──────────────────┐
   │  Feature CSV    │            │  Feature NPZ     │
   │ (Exploratory)   │            │ (Model input)    │
   │                 │            │                  │
   │ • Full table    │            │ • Algo1 (N, 52) │
   │ • Metadata      │            │ • Algo2 (N, 120)│
   │ • Per-window    │            │ • Labels         │
   └─────────────────┘            │ • Participant    │
                                  └──────────────────┘
        │
        └─────────────────┬──────────────────┘
                          │
                    ┌─────▼──────┐
                    │  Visuals   │
                    │ Box plots  │
                    │ Heat maps  │
                    │ PCA        │
                    └────────────┘
```

## Data Flow

### Phase 1: Preprocessing

| Step | Input | Output | Key Parameters |
|------|-------|--------|-----------------|
| Load | Raw CSV (76,800 samples) | Signal arrays | 256 Hz (inf) / 100 Hz (pix) |
| Filter | Raw signal | Filtered signal | FCutoff: 0.5-40 Hz (type-specific) |
| Artifact Remove | Filtered | Clean | z_thresh=5, max_ratio=0.3 |
| Resample | PPG @ 100 Hz | @ 256 Hz | Scipy.signal.resample |
| Normalize | Each channel | Z-score | μ=0, σ=1 per window |
| Window | Full trial | 10s windows | window_sec=10, overlap=0.5 |
| Split | 1,059 windows | Indexed subsets | 14/4/4 subjects |

### Phase 2: Feature Extraction

#### Algorithm 1: Statistical + Spectral
```python
For each window w in X:
  For each channel c in w:

    # Time-domain
    f['mean'] = mean(c)
    f['std'] = std(c)
    f['rms'] = sqrt(mean(c²))
    f['skew'] = skewness(c)
    f['kurt'] = kurtosis(c)
    f['ptp'] = max(c) - min(c)
    f['zcr'] = zero_crossings(c) / len(c)

    # Frequency-domain
    freqs, psd = welch(c, fs=256)
    for band in [VLF, LF, HF]:
      f[f'power_{band}'] = integrate(psd[freqs in band])
    f['lf_hf_ratio'] = power_LF / power_HF
    f['dom_freq'] = freqs[argmax(psd)]
    f['spec_entropy'] = shannon_entropy(psd)
```

**Output**: (N_windows, 4_channels × 13_features) = (N, 52)

#### Algorithm 2: Discrete Wavelet Transform
```python
For each window w in X:
  For each channel c in w:

    # 5-level decomposition
    coeffs = wavedec(c, 'db4', level=5)
    # coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]

    total_energy = sum(coeff² for all coeffs)

    for i, coeff_set in enumerate(coeffs):
      name = f'A5' if i==0 else f'D{5-i+1}'
      features[f'{name}_energy'] = sum(coeff²)
      features[f'{name}_energy_ratio'] = energy / total_energy
      features[f'{name}_entropy'] = shannon_entropy(coeff²)
      features[f'{name}_mean'] = mean(coeff)
      features[f'{name}_std'] = std(coeff)
```

**Output**: (N_windows, 4_channels × 30_features) = (N, 120)

## Frequency Bands

### Cardiac Signals (ECG, PPG)
| Band | Range | Function |
|------|-------|----------|
| VLF | 0.003–0.04 Hz | Very slow autonomic |
| LF | 0.04–0.15 Hz | Sympathetic + parasympathetic |
| HF | 0.15–0.4 Hz | Parasympathetic / respiratory |

### GSR/EDA
| Band | Range | Function |
|------|-------|----------|
| Tonic | 0.05–0.2 Hz | Slow skin conductance level |
| Phasic | 0.2–1.0 Hz | Skin conductance response |
| Fast | 1.0–5.0 Hz | Noise / motion |

### DWT (at 256 Hz)
| Level | Freq Range | Purpose |
|-------|-----------|---------|
| D1 | 64–128 Hz | Muscle artifacts, high-freq noise |
| D2 | 32–64 Hz | Respiratory sinus arrhythmia |
| D3 | 16–32 Hz | Breathing (12–30 breaths/min) |
| D4 | 8–16 Hz | Low-freq sympathetic |
| D5 | 4–8 Hz | Very low-freq trends |
| A5 | 0–4 Hz | Tonic baseline |

## Key Configuration

```python
# Preprocessing
FS_INF = 256        # Infrared sensor sampling rate
FS_PIX = 100        # PixArt (wrist) sampling rate
FS_TARGET = 256     # Resampling target

WINDOW_SEC = 10     # Window duration
OVERLAP_RATIO = 0.5 # 50% overlap

ARTIFACT_ZSCORE_THRESH = 5.0
ARTIFACT_MAX_RATIO = 0.3  # Reject if >30% artifacts

# Labels
TRIAL_CONDITION = {0: 0, 1: 2, 2: 3, 3: 2, 4: 3, 5: 0}
LABEL_MAP = {0: 0, 2: 1, 3: 2}  # n-back level → class index

# Feature Extraction
DWT_WAVELET = 'db4'
DWT_LEVEL = 5
```

## Scaling & Memory

- **Total windows**: 1,059
- **Clean windows**: ~900 (85% pass artifact filter)
- **Per window storage**:
  - Raw signal (4 channels × 2560 samples × float32): ~40 KB
  - Features (52 + 120 values × float32): <1 KB

- **Full dataset**:
  - Windowed arrays (NPZ): ~36 MB
  - Features (CSV): <5 MB
  - Visualizations (PNG): ~2 MB

## Processing Time

- Preprocessing: ~5 min for 22 subjects (single core)
- Feature extraction: ~2 min for ~900 windows

---

**Last Updated**: 2026-03-04
