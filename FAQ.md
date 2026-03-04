# FAQ & Troubleshooting

## Frequently Asked Questions

### Data & Preprocessing

**Q: I don't have the MAUS dataset. Where can I download it?**

A: The MAUS dataset is not publicly available on GitHub due to size (>5GB). Contact the dataset authors or check:
- Original publication venue
- Research institute repositories
- Request from authors directly

For testing the pipeline, you can use synthetic data:
```python
# Generate synthetic signal
import numpy as np
N_SUBJ = 2
N_TRIALS = 6
SAMPLES = 76800

fake_data = {}
for modality in ['ecg', 'ppg', 'gsr']:
    # Gaussian noise + low-freq trend
    trend = np.linspace(0, 1, SAMPLES)
    noise = np.random.randn(SAMPLES, N_TRIALS)
    fake_data[modality] = (trend[:, None] + noise) / 10

# Save as CSV
import pandas as pd
df = pd.DataFrame(fake_data)
df.to_csv('MAUS/Data/Raw_data/999/inf_ecg.csv', index=False)
```

**Q: Why does preprocessing take so long?**

A: Filtering is computationally intensive. To speed up:
- Use a multi-core machine
- Process subjects in parallel (Python `multiprocessing`)
- Reduce `nperseg` in Welch's method (trade off spectral resolution)

**Q: What if some windows are rejected as artifacts?**

A: Normal! Motion artifacts are common in wearable signals. To debug:
```python
# Check artifact rates per subject
df_quality = pd.read_csv('MAUS/preprocessed/signal_quality.csv')
df_quality[df_quality['clean_ratio'] < 0.7]  # Problematic subjects
```

### Feature Extraction

**Q: Which algorithm should I use for my model?**

A: Use both! They capture different information:
- **Algorithm 1 (Stat+Spectral)**: Interpretable physiological bands, fewer features (52)
- **Algorithm 2 (DWT)**: Multi-resolution, captures transients, more features (120)

Start with Algorithm 1 for fast baseline, then try Algorithm 2 for robustness.

**Q: Can I extract features at different window sizes?**

A: Yes, but feature meanings change:
```python
# Shorter windows (5s)
WINDOW_SEC = 5
# → Lower frequency resolution (~5 Hz main lobe)
# → More windows, but noisier estimates

# Longer windows (20s)
WINDOW_SEC = 20
# → Better frequency resolution
# → Fewer windows, but better statistics
```

### Models & Training

**Q: Which modality should I train on?**

A: **In-domain recommendation**: Wrist PPG (PixArt)
- Most practical (consumer wearable)
- Worst quality (high noise)
- Tests real-world robustness

**For comparison:**
- ECG (clinical gold standard) — best accuracy
- Fingertip PPG — intermediate accuracy & quality
- GSR (slower dynamics) — complementary information

**Q: Should I use raw windows or extracted features?**

A: **For baseline (weeks 1–2)**: Extracted features
- Fast training
- Interpretable
- Few hyperparameters

**For final model**: Raw windows with CNN
- Better performance (end-to-end learning)
- Learns optimal filters
- But harder to debug

### Uncertainty & Conformal Prediction

**Q: What's the advantage of conformal prediction over softmax confidence?**

A: Conformal prediction has **theoretical guarantees**:
```
Softmax:         "96% confident on high class"  (no guarantee)
Conformal:       "High + Medium class (90% coverage)"  (guaranteed)
```

At test time, you can decide: Is 1-element set good enough, or do you need 2–3 elements?

**Q: How large should the calibration set be?**

A: Rule of thumb: 10–20% of data
- Too small: noisy threshold estimates
- Too large: less data for training

For 900 windows: 90–180 for calibration is reasonable.

**Q: Can I use conformal prediction with deep learning?**

A: Yes! Just apply it to the neural network outputs:
```python
# Train CNN normally
model.train()

# On calibration set, get predictions
probs = model.predict(X_cal)  # (n_cal, 3) — logits

# Compute non-conformity scores
nc_scores = 1 - np.max(probs, axis=1)  # Inverse confidence

# Find quantile for 90% coverage
quantile_90 = np.percentile(nc_scores, 90)

# At test time
for x_test in X_test:
    probs = model.predict(x_test)
    nc_score = 1 - np.max(probs)
    if nc_score < quantile_90:
        prediction_set = {argmax(probs)}  # High confidence
    else:
        prediction_set = {all classes with prob > threshold}  # Lower confidence
```

---

## Troubleshooting

### Installation Issues

**Problem:** `pip install -r requirements.txt` fails on `torch`

**Diagnosis:**
```bash
pip install torch==2.0.0  # Try specific version
# or use conda
conda install pytorch::pytorch -c pytorch
```

**Problem:** `jupyter: command not found`

**Solution:**
```bash
python -m jupyter notebook  # Use python module syntax
# or
which jupyter
# Check PATH, reinstall if needed
pip install --force-reinstall jupyter
```

### Preprocessing Issues

**Problem:** `FileNotFoundError: MAUS/Data/Raw_data/002/inf_ecg.csv`

**Solution:** Verify dataset location:
```bash
ls MAUS/Data/Raw_data/  # Check if directory exists and has data
# If empty, download dataset first
```

**Problem:** `MemoryError` during preprocessing

**Solution:** Process in batches:
```python
# In preprocessing.ipynb, modify:
PARTICIPANT_IDS = ['002', '003', '004', '005']  # 4 instead of 22
# Run preprocessing, increment to next 4, repeat
```

Or reduce overlap:
```python
OVERLAP_RATIO = 0.0  # No overlap (fewer windows)
```

**Problem:** Too many windows rejected as artifacts

**Diagnosis:**
```python
is_clean.mean()  # Should be > 0.8
```

**Solution 1:** Relax artifact threshold
```python
ARTIFACT_ZSCORE_THRESH = 4.5  # Was 5.0
ARTIFACT_MAX_RATIO = 0.4      # Was 0.3
```

**Solution 2:** Check signal quality
```python
# Are certain subjects/modalities worse?
df_quality = pd.read_csv('MAUS/preprocessed/signal_quality.csv')
df_quality.sort_values('clean_ratio')
```

### Feature Extraction Issues

**Problem:** `RuntimeWarning: invalid value encountered in divide` (NaN/Inf)

**Solution:** This is usually harmless (e.g., 0/0 in band ratio). Handle in feature loading:
```python
X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
```

**Problem:** Feature values are all zeros

**Diagnosis:**
```python
np.isnan(X_features).sum()  # Check for NaNs
X_features.std()            # Should be > 0 after standardization
```

**Solution:** Ensure clean windows are used:
```python
X_features = X_features[is_clean]  # Filter before feature extraction
```

### Model Training Issues

**Problem:** Loss is NaN after first epoch

**Causes:**
- Learning rate too high
- Exploding gradients
- Invalid input (NaNs, Infs)

**Solution:**
```python
# Check data
assert not np.isnan(X_train).any()
assert not np.isinf(X_train).any()

# Reduce learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Was 1e-3
```

**Problem:** Model always predicts the same class

**Diagnosis:**
```python
y.unique()          # Labels are 0, 1, 2
(y == 0).mean()     # Class distribution (should be ~1/3 each)
```

**Cause:** Class imbalance or insufficient training

**Solution:**
```python
# Use class weights
class_weights = torch.tensor([1/0.33, 1/0.33, 1/0.33])
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Or increase epochs
epochs = 100  # Was 50
```

**Problem:** Validation accuracy doesn't improve

**Diagnosis:**
```python
# Plot training curves
plt.plot(train_loss)
plt.plot(val_loss)
```

**Possible causes & solutions:**
- **Model too simple**: Add more layers
- **Underfitting**: Train longer, higher capacity
- **Overfitting**: Add dropout, L2 regularization, reduce model size
- **Bad learning rate**: Try learning rate schedule

### Conformal Prediction Issues

**Problem:** Coverage is much lower than expected (e.g., 70% instead of 90%)

**Diagnosis:**
```python
actual_coverage = (y_test.values[:, None] == pred_sets).sum(axis=1).mean()
print(f"Actual coverage: {actual_coverage:.1%}")
```

**Cause:** Calibration set too small or unrepresentative

**Solution:**
- Increase calibration set size (larger percentile)
- Ensure balanced class distribution in cal set

**Problem:** Prediction sets are always size 3 (all classes)

**Diagnosis:**
```python
pred_set_sizes = [len(s) for s in pred_sets]
np.mean(pred_set_sizes)  # Should be < 2
```

**Cause:** Quantile threshold too high (too conservative)

**Solution:**
```python
# Use higher percentile for lower coverage guarantee
quantile = np.percentile(nc_scores, 95)  # Was 90
# This gives ~95% coverage (fewer predictions needed in sets)
```

### Visualization Issues

**Problem:** `Plots not showing in Jupyter`

**Solution:**
```python
%matplotlib inline  # At top of notebook
import matplotlib.pyplot as plt
```

**Problem:** Out of memory when plotting large heatmaps

**Solution:**
```python
# Reduce to subset of features
plt.figure(figsize=(12, 10))  # Bigger figure
corr = df[[feat1, feat2, feat3, ...]].corr()  # Subset
sns.heatmap(corr, cmap='RdBu_r')
```

---

## Performance Issues

### Slow Preprocessing?

```
Typical times (22 subjects):
- Loading & filtering: 2–3 min
- Artifact detection: 1 min
- Windowing: 1 min
- Total: ~5 min on single core

To speed up:
- Use parallel processing: joblib.Parallel()
- Reduce signal length if possible (truncate early trials)
```

### Slow Feature Extraction?

```
Typical times (900 windows):
- Algorithm 1 (stat+spectral): 1–2 min
- Algorithm 2 (DWT): 0.5–1 min (faster!)
- Total: ~2–3 min
```

### Model Training Too Slow?

- Use GPU: `cuda=True` in torch
- Batch processing: increase batch size (less overhead)
- Reduce model size: fewer layers/units

---

## Getting Help

1. **Check this FAQ** — maybe your issue is documented
2. **Search GitHub Issues** — same problem reported?
3. **Read notebook comments** — inline documentation
4. **Ask on Discussions** — open-ended questions
5. **Email maintainers** — for critical blockers

---

**Last Updated**: 2026-03-04
