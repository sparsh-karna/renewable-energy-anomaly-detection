# Testing & Validation Guide

Guidelines for testing and validating the preprocessing and feature extraction pipelines.

## Manual Testing Checklist

### Preprocessing Validation

Run `preprocessing.ipynb` and verify:

- [ ] All 22 participants loaded (status: "Processing [PID]...")
- [ ] Each participant produces ~9–10 windows per trial
- [ ] Total windows ≥ 1,000 (before quality filtering)
- [ ] After quality filtering: ~900 clean windows (85%+)
- [ ] Output files created:
  - [ ] `windowed_data.npz` (size ≥ 30 MB)
  - [ ] `splits.npz` (size ≥ 10 KB)
  - [ ] `subjective_ratings.csv`
  - [ ] `signal_quality.csv`
- [ ] Sample windows visualization created (`sample_windows.png`)

**Expected Output**:
```
Loaded 1059 windows
Clean windows: 900 (85%)
Label distribution: {0: 300, 1: 300, 2: 300}
Train/Cal/Test: 640 / 130 / 130
```

### Feature Extraction Validation

Run `feature_extraction.ipynb` and verify:

- [ ] Algorithm 1 produces 52 features per window → 900 × 52 table
- [ ] Algorithm 2 produces 120 features per window → 900 × 120 table
- [ ] Both CSVs created:
  - [ ] `features_stat_spectral.csv` (size ≥ 2 MB)
  - [ ] `features_dwt.csv` (size ≥ 4 MB)
- [ ] Visualizations created:
  - [ ] `algo1_boxplots.png`
  - [ ] `algo1_correlation.png`
  - [ ] `algo2_energy_bars.png`
  - [ ] `algo2_entropy.png`
  - [ ] `feature_pca_comparison.png`
- [ ] Feature statistics output:
  - [ ] Mean/std per class printed
  - [ ] PCA variance > 50% for both algorithms
  - [ ] No NaN or Inf values

**Expected Output**:
```
Algorithm 1 complete: 900 windows x 52 features
Algorithm 2 complete: 900 windows x 120 features
Algorithm 1 PCA variance (2 PCs): 62%
Algorithm 2 PCA variance (2 PCs): 55%
```

---

## Automated Testing (Future)

### Unit Tests (once implemented)

```bash
pytest tests/test_preprocessing.py -v
pytest tests/test_features.py -v
```

**Example test structure**:
```python
# tests/test_preprocessing.py

import numpy as np
from preprocessing import bandpass_filter, create_windows

def test_bandpass_filter():
    """Test bandpass filter on synthetic signal."""
    sig = np.sin(2 * np.pi * 1 * np.linspace(0, 10, 2560))
    filtered = bandpass_filter(sig, 0.5, 8, fs=256)

    assert len(filtered) == len(sig)
    assert not np.isnan(filtered).any()

def test_create_windows():
    """Test window creation."""
    # Mock trial data
    trial_data = {...}
    windows = create_windows(trial_data, window_sec=10)

    assert len(windows) >= 4  # At least 4 non-overlapping 10s windows
    assert windows[0]['is_clean'] in [True, False]
```

---

## Integration Testing

### Pipeline End-to-End

1. **Input**: Raw CSV files in `MAUS/Data/Raw_data/`
2. **Process**: Run both notebooks sequentially
3. **Output**: Check all expected files exist

```bash
# Verify pipeline execution
ls -lh MAUS/preprocessed/windowed_data.npz
ls -lh MAUS/preprocessed/features_*.csv
```

### Data Integrity Checks

```python
# Load preprocessed data
data = np.load('MAUS/preprocessed/windowed_data.npz')
splits = np.load('MAUS/preprocessed/splits.npz', allow_pickle=True)

# Validation checks
assert data['X_ecg'].shape == (900, 1, 2560), "ECG shape mismatch"
assert data['y'].shape == (900,), "Labels shape mismatch"
assert data['y'].min() == 0 and data['y'].max() == 2, "Label range invalid"
assert splits['train_mask'].sum() > 600, "Too few train samples"
assert (splits['train_mask'] + splits['cal_mask'] + splits['test_mask']).sum() == 900, "Splits don't sum to total"

print("✓ All data integrity checks passed")
```

---

## Performance Benchmarking

### Processing Time

Run and record:

```python
import time

# Preprocessing
start = time.time()
# ... run preprocessing.ipynb ...
prep_time = time.time() - start
print(f"Preprocessing: {prep_time:.1f} seconds")

# Feature extraction
start = time.time()
# ... run feature_extraction.ipynb ...
feat_time = time.time() - start
print(f"Feature extraction: {feat_time:.1f} seconds")
```

**Expected** (modern laptop):
- Preprocessing: 300–600 seconds (5–10 min)
- Feature extraction: 120–180 seconds (2–3 min)

### Memory Usage

```bash
# Monitor processing
python -m memory_profiler preprocessing.ipynb
```

**Expected peak**: ~2–3 GB RAM

---

## Quality Assurance

### Visual Inspection

1. **Sample windows** (`sample_windows.png`):
   - [ ] ECG shows clear QRS complexes
   - [ ] PPG shows pulsatile waves
   - [ ] GSR appears smooth with occasional steps
   - [ ] Pattern consistent across 3 classes

2. **Feature distributions** (`algo1_boxplots.png`):
   - [ ] Box plots show separation between classes
   - [ ] Few extreme outliers
   - [ ] Medians trend monotonically with workload (where expected)

3. **PCA projections** (`feature_pca_comparison.png`):
   - [ ] Classes form distinct clusters
   - [ ] No isolated outliers
   -  [ ] Algo 1 shows slightly better separation than Algo 2

### Statistical Checks

```python
# Load features
df = pd.read_csv('MAUS/preprocessed/features_stat_spectral.csv')

# Check for missing/invalid values
assert df.isna().sum().sum() == 0, "Found NaN values"
assert np.isinf(df.select_dtypes(np.float64)).any().any() == False, "Found Inf values"

# Check class balance
class_counts = df['label'].value_counts()
assert (class_counts > 250).all() and (class_counts < 350).all(), "Imbalanced classes"

# Check feature ranges (post-standardization)
feat_cols = [c for c in df.columns if c.startswith(('ecg_', 'ppg_', 'gsr_'))]
for col in feat_cols:
    mean = df[col].mean()
    std = df[col].std()
    assert -0.5 < mean < 0.5, f"{col} mean not centered"
    assert 0.9 < std < 1.1, f"{col} std not unit"

print("✓ All statistical checks passed")
```

---

## Regression Testing

When modifying code, verify:

- [ ] Same number of windows extracted (within 5%)
- [ ] Feature statistics remain similar (within 10% per metric)
- [ ] Train/test split is deterministic (same participants, same fold)

---

## Known Issues & Expectations

### Artifacts
- ~15% of windows rejected (normal for wearables)
- Wrist PPG has higher artifact rate (~18%) — expected due to motion
- Some subjects > 90% clean, others ~75% — natural variation

### Feature Correlations
- Some features highly correlated (e.g., std vs RMS, r > 0.9) — expected
- PCA or L1 regularization recommended for downstream models

### Class Separation
- Classes overlap slightly in PCA space — expected for 3-way classification
- Not all modalities separate equally — ECG > PPG_inf > PPG_pix

---

## Sign-Off Checklist

- [ ] Preprocessing notebook runs top-to-bottom without errors
- [ ] Feature extraction notebook runs top-to-bottom without errors
- [ ] All output files exist and have correct sizes
- [ ] Visual inspections pass (see above)
- [ ] Statistical checks pass (see above)
- [ ] Performance benchmarks within expected ranges
- [ ] Git status clean (no untracked large files)

**Ready for model training**: ✓ Yes / ✗ No

---

**Last Updated**: 2026-03-04
