# Feature Extraction Guide

Complete documentation for both feature extraction algorithms and how to customize or extend them.

## Overview

After preprocessing, two complementary feature extraction methods are applied to generate fixed-length feature vectors from variable-length time-series windows.

**Input**: Preprocessed windows (900 × 1 × 2560) per modality
**Output**: Feature tables (900 × 52) for Algorithm 1, (900 × 120) for Algorithm 2

## Algorithm 1: Statistical + Spectral Features

### Rationale

- **Time-domain**: Captures signal shape (skew, kurtosis), energy (RMS), and variability (std)
- **Frequency-domain**: Autonomic nervous system markers (LF/HF ratio), workload-related power shifts

### Time-Domain Features (7 per channel)

```python
mean          = np.mean(signal)
std           = np.std(signal)
rms           = np.sqrt(np.mean(signal ** 2))
skewness      = scipy.stats.skew(signal)
kurtosis      = scipy.stats.kurtosis(signal)
peak_to_peak  = np.ptp(signal)  # max - min
zcr           = zero_crossing_rate(signal)
```

**Interpretation:**
| Feature | Meaning | Workload Effect |
|---------|---------|---|
| std | Signal variability | ↑ in high workload |
| rms | Signal power | ↑ in higher effort |
| skewness | Asymmetry | May differ by modality |
| kurtosis | Tail weight | ↑ in noisy conditions |
| zcr | Oscillation rate | ↑ if higher-freq content |

### Frequency-Domain Features (6 per channel)

```python
# Welch's method for power spectral density
freqs, psd = scipy.signal.welch(signal, fs=256, nperseg=512)

# Band power: integrate PSD within frequency range
power_VLF = np.trapezoid(psd[freqs in (0.003, 0.04)], freqs_in_band)
power_LF  = np.trapezoid(psd[freqs in (0.04, 0.15)], freqs_in_band)
power_HF  = np.trapezoid(psd[freqs in (0.15, 0.4)], freqs_in_band)

# Ratios
lf_hf_ratio = power_LF / power_HF

# Spectral properties
dominant_freq = freqs[np.argmax(psd)]
spectral_entropy = -sum(psd_norm * log2(psd_norm))  # Normalized to [0, 1]
```

**Band Interpretation (Cardiac Signals):**
| Band | Frequency | Origin | Workload Effect |
|------|-----------|--------|---|
| VLF | 0.003–0.04 Hz | Thermoregulation | ↑ stress |
| LF | 0.04–0.15 Hz | Sympathetic + Parasympathetic | ↑ cognitive load |
| HF | 0.15–0.4 Hz | Parasympathetic (respiratory) | ↓ in high workload |
| **LF/HF** | Ratio | Sympathetic dominance | ↑ in high workload |

**For GSR:**
| Band | Frequency | Origin |
|------|-----------|--------|
| Tonic | 0.05–0.2 Hz | Baseline conductance |
| Phasic | 0.2–1.0 Hz | Skin conductance response (SCR) |
| Fast | 1.0–5.0 Hz | Noise / motion |

### Output Dimension

- Per channel: 7 (time) + 6 (freq) = **13 features**
- Total channels: 4 (ECG, PPG_inf, PPG_pix, GSR)
- **Total features: 52**

## Algorithm 2: Discrete Wavelet Transform (DWT)

### Rationale

- Captures multi-resolution time-frequency information
- Better for transient events (autonomic responses, sudden changes)
- Orthogonal decomposition avoids feature correlation

### Decomposition

```python
import pywt

# 5-level Daubechies-4 decomposition
coeffs = pywt.wavedec(signal, 'db4', level=5)
# Returns: [cA5, cD5, cD4, cD3, cD2, cD1]
#           approx + 5 detail levels (high-freq to low-freq)
```

**Frequency Decomposition (@ 256 Hz):**
```
D1:  64–128 Hz    (high-freq noise, muscle)
D2:  32–64 Hz     (respiratory sinus arrhythmia)
D3:  16–32 Hz     (breathing rate: ~12–30 bpm)
D4:  8–16 Hz      (low-freq autonomic)
D5:  4–8 Hz       (very low-freq trends)
A5:  0–4 Hz       (DC baseline, tonic)
```

### Per-Level Features (5 features × 6 levels = 30 per channel)

```python
def extract_dwt_features(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    total_energy = sum(c**2 for c in coeffs)

    for i, coeff_set in enumerate(coeffs):
        name = 'A5' if i == 0 else f'D{5-i+1}'

        # Energy
        energy = np.sum(coeff_set ** 2)
        features[f'{name}_energy'] = energy

        # Normalized energy (ratio to total)
        features[f'{name}_energy_ratio'] = energy / total_energy

        # Entropy (how "spread out" the coefficients are)
        features[f'{name}_entropy'] = shannon_entropy(coeff_set ** 2)

        # Mean and std
        features[f'{name}_mean'] = np.mean(coeff_set)
        features[f'{name}_std'] = np.std(coeff_set)
```

### Energy Interpretation

High energy in a band indicates activity in that frequency range:
- High D1 energy → high-frequency noise or muscle artifacts
- High D3/D4 energy → strong breathing/autonomic response
- High A5 energy → slow tonic baseline shifts

### Output Dimension

- Per level: 5 features × 6 levels = **30 features**
- Total channels: 4
- **Total features: 120**

## Feature Extraction Pipeline

```python
# For each window
for i, window_idx in enumerate(clean_indices):
    row = {'label': labels[window_idx], 'participant': ...}

    # For each modality
    for modality_name, signal_array in modalities.items():
        signal = signal_array[window_idx, 0, :]  # Extract 1D signal

        # Algorithm 1
        algo1_feats = extract_stat_spectral_features(signal, FS=256, type=...)
        for fname, fval in algo1_feats.items():
            row[f'{modality_name}_{fname}'] = fval

        # Algorithm 2
        algo2_feats = extract_dwt_features(signal)
        for fname, fval in algo2_feats.items():
            row[f'{modality_name}_{fname}'] = fval

    features_table.append(row)

# Output: DataFrames with 4 × (52 + 120) = 688 computed features
```

## Output Files

### CSV Format (Exploratory)
```
features_stat_spectral.csv:    (900, 56 columns)
  window_idx, label, participant, ecg_mean, ecg_std, ..., gsr_entropy

features_dwt.csv:              (900, 365 columns)
  window_idx, label, participant, ecg_D1_energy, ecg_D1_energy_ratio, ...
```

### NPZ Format (for Models)
```python
data = np.load('features_extracted.npz')
X_algo1 = data['algo1_features']       # (900, 52) — standardized
X_algo2 = data['algo2_features']       # (900, 120)
y = data['labels']                     # (900,)
feat_names_1 = data['algo1_feature_names']  # Names for interpretation
feat_names_2 = data['algo2_feature_names']
```

## Using Features for Models

### Baseline (Shallow Classifier)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load features
X = data['algo1_features']  # or algo2
y = data['labels']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_scaled, y)

# Feature importance
importances = clf.feature_importances_
top_features = np.argsort(importances)[-10:]  # Top 10
```

### Deep Learning (Dimensionality Reduction)
```python
from sklearn.decomposition import PCA

# PCA to reduce dimensionality
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Feed to small NN
# X_pca shape: (900, 10)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
```

## Customization

###Add New Signal Type
```python
# In feature_extraction.ipynb, add to FREQ_BANDS:
FREQ_BANDS['new_signal'] = {
    'Band1': (freq_low, freq_high),
    'Band2': (freq_low, freq_high),
}

# Modify modalities dict
modalities['new_signal'] = (X_new_signal, 'new_signal')
```

### Change DWT Wavelet
```python
# Other options: 'db3', 'db5', 'sym5', 'coif2'
DWT_WAVELET = 'sym5'  # Symlet-5 (more vanishing moments)
```

### Add Custom Features
```python
def extract_custom_features(signal):
    """Energy in specific frequency bands."""
    freqs, psd = welch(signal, fs=256)
    features = {}

    # Example: 0.1 Hz band (very slow)
    slow_band = (freqs >= 0.05) & (freqs <= 0.15)
    features['power_0p1hz_band'] = np.trapezoid(psd[slow_band], freqs[slow_band])

    return features

# Add to feature loop
custom = extract_custom_features(signal)
for fname, fval in custom.items():
    row[f'{modality}_custom_{fname}'] = fval
```

## Performance Notes

- **Runtime**: Feature extraction on 900 windows @ ~2 minutes (single core)
- **Memory**: ~50 MB for feature tables (both algorithms)
- **Correlation**: Many features are correlated (e.g., mean ↔ RMS)
  - Use PCA or L1 regularization to reduce redundancy

## Quality Visualization

Run the feature extraction notebook to see:
- **Box plots**: Feature distributions across workload levels
- **Heatmaps**: Feature correlations (identify redundancy)
- **Energy stacks**: DWT energy distribution per modality
- **PCA**: Projection of feature space (class separation)

## Next Steps

→ Use extracted features to train baseline models (logistic regression, random forest)
→ Or feed raw windows to CNN for end-to-end learning

---

**Last Updated**: 2026-03-04
