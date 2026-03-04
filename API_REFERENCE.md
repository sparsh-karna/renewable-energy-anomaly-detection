# API Reference

Complete reference for module functions and classes used in preprocessing and feature extraction.

## Preprocessing

### Module: `preprocessing` (from `preprocessing.ipynb`)

#### `load_raw_signals(pid)`

Load and process all raw signals for a participant.

**Parameters**
- `pid` (str): Participant ID, e.g., '002'

**Returns**
- `trials` (dict): Keys are trial indices (0–5), values are dicts with:
  - `'ecg'` (array): Filtered, cleaned, normalized ECG signal
  - `'ppg_inf'` (array): Fingertip PPG
  - `'ppg_pix'` (array): Wrist PPG (resampled to 256 Hz)
  - `'gsr'` (array): GSR/EDA signal
  - `'condition'` (int): Workload level (0, 2, or 3)

**Example**
```python
trials = load_raw_signals('002')
for trial_idx, data in trials.items():
    print(f"Trial {trial_idx}: {len(data['ecg'])} samples")
```

---

#### `bandpass_filter(signal, lowcut, highcut, fs, order=4)`

Apply zero-phase Butterworth bandpass filter.

**Parameters**
- `signal` (1D array): Input time-series
- `lowcut` (float): Low cutoff frequency (Hz)
- `highcut` (float): High cutoff frequency (Hz)
- `fs` (int): Sampling rate (Hz)
- `order` (int, default=4): Filter order

**Returns**
- `filtered_signal` (1D array): Filtered output

**Example**
```python
ecg_filtered = bandpass_filter(ecg_raw, lowcut=0.5, highcut=40.0, fs=256)
```

---

#### `detect_and_interpolate_artifacts(signal, z_thresh=5.0)`

Detect outliers via z-score and replace with linear interpolation.

**Parameters**
- `signal` (1D array): Input signal
- `z_thresh` (float, default=5.0): Z-score threshold for outlier detection

**Returns**
- `clean_signal` (1D array): Signal with artifacts interpolated
- `artifact_ratio` (float): Fraction of samples that were artifacts

**Example**
```python
clean_ecg, art_ratio = detect_and_interpolate_artifacts(ecg_filt)
print(f"Artifact ratio: {art_ratio:.2%}")
```

---

#### `create_windows(trial_data, window_sec=10, overlap=0.5, fs=256)`

Segment trial into overlapping windows.

**Parameters**
- `trial_data` (dict): Output from `load_raw_signals()`
- `window_sec` (float, default=10): Window duration (seconds)
- `overlap` (float, default=0.5): Overlap ratio (0–1)
- `fs` (int, default=256): Sampling rate (Hz)

**Returns**
- `windows` (list): Each element is a dict with:
  - Signal arrays (4 channels × `window_sec * fs` samples)
  - `'label'` (int): 0, 1, or 2
  - `'condition'` (int): 0, 2, or 3
  - `'artifact_ratio'` (float)
  - `'is_clean'` (bool): Passes quality threshold

**Example**
```python
windows = create_windows(trials[0], window_sec=10, overlap=0.5)
print(f"{len(windows)} windows extracted")
clean_windows = [w for w in windows if w['is_clean']]
print(f"{len(clean_windows)} pass quality check")
```

---

## Feature Extraction

### Module: `feature_extraction` (from `feature_extraction.ipynb`)

#### `extract_stat_spectral_features(signal, fs, signal_type)`

Extract statistical and spectral features.

**Parameters**
- `signal` (1D array): Signal window
- `fs` (int): Sampling rate (Hz)
- `signal_type` (str): One of {'ecg', 'ppg', 'gsr'}

**Returns**
- `features` (dict): 13 named features
  - Time-domain: mean, std, rms, skewness, kurtosis, peak_to_peak, zcr
  - Frequency-domain: power_VLF, power_LF, power_HF, lf_hf_ratio, dominant_freq, spectral_entropy

**Example**
```python
feats = extract_stat_spectral_features(X_ecg[0, 0, :], fs=256, signal_type='ecg')
print(f"Power LF/HF ratio: {feats['lf_hf_ratio']:.2f}")
print(f"Spectral entropy: {feats['spectral_entropy']:.3f}")
```

---

#### `compute_band_power(signal, fs, band)`

Compute Power Spectral Density (PSD) in a frequency band.

**Parameters**
- `signal` (1D array): Input signal
- `fs` (int): Sampling rate (Hz)
- `band` (tuple): (lowcut, highcut) in Hz

**Returns**
- `power` (float): Integrated power in band

**Example**
```python
lf_power = compute_band_power(ecg_signal, fs=256, band=(0.04, 0.15))
hf_power = compute_band_power(ecg_signal, fs=256, band=(0.15, 0.4))
ratio = lf_power / hf_power
```

---

#### `spectral_entropy(signal, fs)`

Compute normalized Shannon entropy of PSD.

**Parameters**
- `signal` (1D array): Input signal
- `fs` (int): Sampling rate (Hz)

**Returns**
- `entropy` (float): Normalized to [0, 1]

**Interpretation**
- High entropy: broad-spectrum (noisy)
- Low entropy: narrow-spectrum (periodic)

**Example**
```python
entropy_ecg = spectral_entropy(X_ecg[i, 0, :], fs=256)
entropy_ppg = spectral_entropy(X_ppg_pix[i, 0, :], fs=256)
```

---

#### `zero_crossing_rate(signal)`

Compute fraction of samples that cross zero.

**Parameters**
- `signal` (1D array): Input signal

**Returns**
- `zcr` (float): Zero-crossing rate (0–1)

**Example**
```python
zcr = zero_crossing_rate(signal)
print(f"Zero-crossing rate: {zcr:.3f}")
```

---

#### `extract_dwt_features(signal, wavelet='db4', level=5)`

Extract discrete wavelet transform features.

**Parameters**
- `signal` (1D array): Signal window
- `wavelet` (str, default='db4'): Wavelet name (e.g., 'db3', 'db5', 'sym5')
- `level` (int, default=5): Decomposition level

**Returns**
- `features` (dict): 30 named features
  - Per level (A5, D5, D4, D3, D2, D1): energy, energy_ratio, entropy, mean, std

**Example**
```python
dwt_feats = extract_dwt_features(X_ecg[0, 0, :], wavelet='db4', level=5)
print(f"D3 energy ratio: {dwt_feats['D3_energy_ratio']:.3f}")
print(f"A5 entropy: {dwt_feats['A5_entropy']:.3f}")
```

---

#### `coeff_entropy(coeffs)`

Shannon entropy of squared wavelet coefficients.

**Parameters**
- `coeffs` (1D array): Wavelet coefficients at one level

**Returns**
- `entropy` (float): Shannon entropy

**Example**
```python
from pywt import wavedec
coeffs = wavedec(signal, 'db4', level=5)
for i, c in enumerate(coeffs):
    ent = coeff_entropy(c)
    print(f"Level {i} entropy: {ent:.3f}")
```

---

## Data Loading

### `numpy.load()` (Standard NumPy)

Load preprocessed NPZ files.

**Example**
```python
# Load preprocessed windows
data = np.load('MAUS/preprocessed/windowed_data.npz')
X_ecg = data['X_ecg']        # Shape: (900, 1, 2560)
X_fused = data['X_fused']    # Shape: (900, 4, 2560)
y = data['y']                # Shape: (900,)

# Load split indices
splits = np.load('MAUS/preprocessed/splits.npz', allow_pickle=True)
train_mask = splits['train_mask']  # Boolean array
```

**Keys in `windowed_data.npz`**
| Key | Shape | Type | Description |
|-----|-------|------|---|
| X_ecg | (N, 1, 2560) | float32 | ECG windows |
| X_ppg_inf | (N, 1, 2560) | float32 | Fingertip PPG |
| X_ppg_pix | (N, 1, 2560) | float32 | Wrist PPG |
| X_gsr | (N, 1, 2560) | float32 | GSR windows |
| X_fused | (N, 4, 2560) | float32 | All 4 channels stacked |
| X_ppg_gsr | (N, 3, 2560) | float32 | PPG (both) + GSR |
| y | (N,) | int64 | Labels (0, 1, 2) |
| participants | (N,) | object | Participant IDs |
| is_clean | (N,) | bool | Quality mask |
| artifact_ratios | (N,) | float32 | Artifact fraction per window |

---

### `pandas.read_csv()` (Standard Pandas)

Load CSV feature tables or subjective ratings.

**Example**
```python
# Feature tables
df_algo1 = pd.read_csv('MAUS/preprocessed/features_stat_spectral.csv')
df_algo2 = pd.read_csv('MAUS/preprocessed/features_dwt.csv')

# Subjective ratings
df_subj = pd.read_csv('MAUS/preprocessed/subjective_ratings.csv')

# Signal quality report
df_quality = pd.read_csv('MAUS/preprocessed/signal_quality.csv')
```

---

## Constants

### Configuration
```python
# Sampling rates and targets
FS_INF = 256      # Infrared sensors (ECG, PPG, GSR)
FS_PIX = 100      # PixArt wrist PPG
FS_TARGET = 256   # Resampling target

# Windowing
WINDOW_SEC = 10   # Window duration
OVERLAP_RATIO = 0.5
WINDOW_LEN = 2560  # 10 * 256

# Artifact detection
ARTIFACT_ZSCORE_THRESH = 5.0
ARTIFACT_MAX_RATIO = 0.3

# Trial-to-condition mapping
TRIAL_CONDITION = {0: 0, 1: 2, 2: 3, 3: 2, 4: 3, 5: 0}
LABEL_MAP = {0: 0, 2: 1, 3: 2}  # Workload → class index

# Feature extraction
DWT_WAVELET = 'db4'
DWT_LEVEL = 5

# Frequency bands
FREQ_BANDS = {
    'ecg': {'VLF': (0.003, 0.04), 'LF': (0.04, 0.15), 'HF': (0.15, 0.4)},
    'ppg': {'VLF': (0.003, 0.04), 'LF': (0.04, 0.15), 'HF': (0.15, 0.4)},
    'gsr': {'Tonic': (0.05, 0.2), 'Phasic': (0.2, 1.0), 'Fast': (1.0, 5.0)},
}
```

---

## Dependencies

```
numpy>=1.21.0          # Array operations
scipy>=1.7.0           # Signal processing (filter, resample, welch)
pandas>=1.3.0          # Data tables
scikit-learn>=1.0.0    # ML utilities (PCA, scalers)
PyWavelets>=1.2.0      # Wavelet transforms
matplotlib>=3.4.0      # Visualization
jupyter>=1.0.0         # Notebooks
```

---

**Last Updated**: 2026-03-04
