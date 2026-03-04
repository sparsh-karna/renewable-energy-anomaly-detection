# Preprocessing Guide

Complete walkthrough of the preprocessing pipeline and how to customize it.

## Overview

The preprocessing pipeline transforms raw physiological signals into clean, normalized windows ready for machine learning.

**Input**: Raw CSV files (22 subjects × 4 modalities × 6 trials)
**Output**: Windowed data ready for feature extraction or neural networks

## Quick Start

Open `MAUS/preprocessing.ipynb` and run all cells.

```bash
cd MAUS
jupyter notebook preprocessing.ipynb
```

## Pipeline Stages

### 1. Configuration

```python
FS_INF = 256        # Infrared sensors
FS_PIX = 100        # PixArt wrist PPG
FS_TARGET = 256     # Resampling target

WINDOW_SEC = 10     # 10-second windows
OVERLAP_RATIO = 0.5 # 50% overlap

ARTIFACT_ZSCORE_THRESH = 5.0
ARTIFACT_MAX_RATIO = 0.3  # Reject if >30% artifacts
```

**Why these values?**
- 10s windows capture ~8–10 heartbeats (HR ~60–100 bpm), sufficient for HRV estimation
- 50% overlap provides redundancy and more samples for training
- Z-score > 5 is ~99.9999% outlier in Gaussian distribution
- 30% artifact tolerance allows some noise while preserving data

### 2. Loading Raw Signals

```python
def load_raw_signals(pid):
    """Load all raw signals for a participant."""
    pdir = os.path.join(RAW_DIR, pid)

    # Each CSV has 6 columns (one per trial)
    ecg_df = pd.read_csv(os.path.join(pdir, 'inf_ecg.csv'))
    ppg_inf_df = pd.read_csv(os.path.join(pdir, 'inf_ppg.csv'))
    ppg_pix_df = pd.read_csv(os.path.join(pdir, 'pixart.csv'))
    gsr_df = pd.read_csv(os.path.join(pdir, 'inf_gsr.csv'))

    # Extract per trial, handle NaNs
    # Return organized dict
```

**Output**: Dictionary with 6 trials, each containing 4 signal arrays

### 3. Filtering

#### Bandpass Filtering
```python
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Zero-phase Butterworth filter."""
    # Uses scipy.signal.filtfilt for zero phase distortion
```

**Cutoff frequencies per signal:**
| Signal | Low | High | Reason |
|--------|-----|------|--------|
| ECG | 0.5 Hz | 40 Hz | Remove DC drift, suppress muscle noise |
| PPG (both) | 0.5 Hz | 8 Hz | Heartbeat only (0.6–2.5 Hz), no respiration |
| GSR | 0.05 Hz | 5 Hz | Tonic + phasic components |

#### Notch Filter
```python
# Remove 50 Hz powerline interference
iirnotch(freq=50, Q=30, fs=256)
```

### 4. Artifact Detection & Cleaning

```python
def detect_and_interpolate_artifacts(signal, z_thresh=5.0):
    """
    Find outliers via z-score and replace with linear interpolation.
    Returns: cleaned signal, artifact ratio
    """
    z = np.abs(zscore(signal))
    artifact_mask = z > z_thresh  # Find outliers

    # Interpolate
    clean = signal.copy()
    bad_idx = np.where(artifact_mask)[0]
    good_idx = np.where(~artifact_mask)[0]
    clean[bad_idx] = np.interp(bad_idx, good_idx, clean[good_idx])

    return clean, artifact_mask.sum() / len(signal)
```

**Example:**
```
Original:    [0.5, 0.4, 15.2, 0.3, 0.2]     ← 15.2 is artifact (z≈10)
Cleaned:     [0.5, 0.4, 0.35, 0.3, 0.2]    ← linear interpolation
```

### 5. Resampling

PixArt samples at 100 Hz, infrared at 256 Hz. Upsample to match:

```python
def resample_signal(signal, fs_original, fs_target):
    """Resample to target rate using scipy.signal.resample."""
    num_samples_target = int(len(signal) * fs_target / fs_original)
    return resample(signal, num_samples_target)
```

**Example:**
```
Old: 5000 samples @ 100 Hz = 50 seconds
New: 12,800 samples @ 256 Hz = 50 seconds
```

### 6. Normalization

```python
ecg_norm = zscore(ecg_clean)  # μ=0, σ=1
```

**Why?** Neural networks train better with normalized inputs. Physiological signals have different units (V, μS, a.u.), so z-normalization enables fair comparison.

### 7. Windowing

```python
def create_windows(trial_data, window_sec=10, overlap=0.5, fs=256):
    """
    Segment trial into overlapping windows.
    Returns list of window dicts.
    """
    window_len = int(window_sec * fs)   # 2560 samples
    step = int(window_len * (1 - overlap))  # 1280 samples

    for start in range(0, min_len - window_len, step):
        end = start + window_len
        # Create window dict with signals + metadata
```

**Visualization:**
```
Trial (~50s):  |-------|-------|-------|
Windows (@10s): |-------| |-------| |-------| (50% overlap)
              Window 1  Window 2  Window 3
```

### 8. Quality Filtering

```python
if max_artifact_ratio < ARTIFACT_MAX_RATIO:
    # Keep window
else:
    # Skip window
```

**Impact on dataset:**
```
Total windows: 1,059
Clean windows: ~900 (85%)
Rejected: ~159 (15%)
```

## Advanced Customization

### Reduce Artifact Threshold
```python
ARTIFACT_ZSCORE_THRESH = 4.0  # More aggressive (fewer artifacts)
# or
ARTIFACT_MAX_RATIO = 0.2      # Stricter (more windows rejected)
```

### Larger Windows
```python
WINDOW_SEC = 20     # 20s windows = 5120 samples
# Better for deep learning (more temporal context)
# Fewer windows per trial
```

### No Overlap
```python
OVERLAP_RATIO = 0.0  # Consecutive, non-overlapping windows
# Reduces correlation between adjacent windows
# Fewer total samples (~400 instead of 900)
```

### Process Subset of Subjects
```python
PARTICIPANT_IDS = ['002', '003', '004', '005']  # Only first 4
```

## Output Files

### `windowed_data.npz` structure
```python
np.load('windowed_data.npz')
# Keys:
# - X_ecg:       (900, 1, 2560)     Single channel
# - X_ppg_inf:   (900, 1, 2560)
# - X_ppg_pix:   (900, 1, 2560)
# - X_gsr:       (900, 1, 2560)
# - X_fused:     (900, 4, 2560)     All 4 channels stacked
# - X_ppg_gsr:   (900, 3, 2560)     PPG + GSR fusion
# - y:           (900,)              Labels: 0, 1, 2
# - participants:(900,)              Participant IDs
# - is_clean:    (900,)              Boolean: passes quality check
# - artifact_ratios: (900,)          Per-window artifact ratio
```

### `splits.npz` structure
```python
np.load('splits.npz')
# Keys:
# - train_mask:  (900,)              Boolean index
# - cal_mask:    (900,)              For conformal prediction
# - test_mask:   (900,)
# - train_pids:  (14,)               Subject IDs in train set
# - cal_pids:    (4,)
# - test_pids:   (4,)
```

### `subjective_ratings.csv`
```
participant,trial,condition,label,nasa_tlx_adjusted,mental_demand,...,psqi
002,0,0,0,42.5,55,20,45,60,70,40,10
002,1,2,1,55.3,60,25,50,55,75,38,10
...
```

## Loading & Using Preprocessed Data

```python
import numpy as np

# Load
data = np.load('MAUS/preprocessed/windowed_data.npz')
splits = np.load('MAUS/preprocessed/splits.npz', allow_pickle=True)

# Select modality
X = data['X_ppg_pix']  # Wrist PPG
y = data['y']

# Apply split
train_mask = splits['train_mask']
X_train = X[train_mask]
y_train = y[train_mask]

# Now feed to your model
# X_train shape: (720, 1, 2560) for CNN
# y_train shape: (720,) with values in {0, 1, 2}
```

## Visualization

Check `MAUS/preprocessed/sample_windows.png` for example windows from each modality and condition.

## Troubleshooting

**Q: Why are some windows rejected?**
A: Z-score outliers > 5σ in more than 30% of the window samples. Usually due to motion artifacts or sensor glitches.

**Q: Can I change the sampling rate?**
A: Yes, modify `FS_TARGET`, but feature extraction frequencies will need adjustment.

**Q: How do I process only specific participants?**
A: Modify `PARTICIPANT_IDS` list in the config cell.

---

**Next**: See `feature_extraction.ipynb` to extract features from preprocessed windows.
