# Quick Start Guide

Get up and running with MAUS in 10 minutes.

## 5-Minute Setup

```bash
# 1. Clone repository
git clone https://github.com/[your-org]/ai.git
cd ai

# 2. Create environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies (2 min)
pip install -r requirements.txt

# 4. Download dataset (manual, not shown here)
# Download MAUS dataset and extract to MAUS/Data/

# 5. Run preprocessing (5 min)
cd MAUS
jupyter notebook preprocessing.ipynb
# Run all cells (Ctrl+Shift+Enter in Jupyter)

# 6. Run feature extraction (2 min)
jupyter notebook feature_extraction.ipynb
# Run all cells

# Done! Output in MAUS/preprocessed/
```

**Total**: ~15 minutes (including downloads)

---

## Load & Explore Data

```python
import numpy as np
import pandas as pd

# Load preprocessed windows
data = np.load('MAUS/preprocessed/windowed_data.npz')
splits = np.load('MAUS/preprocessed/splits.npz', allow_pickle=True)

# Get arrays
X_wrist_ppg = data['X_ppg_pix']        # Shape: (900, 1, 2560)
y = data['y']                           # Shape: (900,)
is_clean = data['is_clean']

# Get split indices
train_mask = splits['train_mask']
test_mask = splits['test_mask']

# Prepare data
X_train = X_wrist_ppg[train_mask]
y_train = y[train_mask]
X_test = X_wrist_ppg[test_mask]
y_test = y[test_mask]

print(f"Train: {X_train.shape[0]} windows")
print(f"Test: {X_test.shape[0]} windows")
```

---

## Train Your First Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load features (faster than raw signals)
feat_data = np.load('MAUS/preprocessed/features_extracted.npz')
X = feat_data['algo1_features']  # (900, 52)
y = feat_data['labels']

# Split
train_idx = splits['train_mask']
test_idx = splits['test_mask']

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
acc = model.score(X_test, y_test)
print(f"Accuracy: {acc:.1%}")  # Expected: ~70–75%

# Get feature importance
importance = model.feature_importances_
top_10 = np.argsort(importance)[-10:]
print(f"Top features: {top_10}")
```

---

## Explore Results

```python
# Visualize preprocessing outputs
from PIL import Image
import matplotlib.pyplot as plt

# Sample windows
img = Image.open('MAUS/preprocessed/sample_windows.png')
plt.figure(figsize=(16, 10))
plt.imshow(img)
plt.axis('off')
plt.show()

# Feature distributions
img = Image.open('MAUS/preprocessed/algo1_boxplots.png')
plt.figure(figsize=(16, 10))
plt.imshow(img)
plt.axis('off')
plt.show()

# PCA comparison
img = Image.open('MAUS/preprocessed/feature_pca_comparison.png')
plt.figure(figsize=(12, 5))
plt.imshow(img)
plt.axis('off')
plt.show()
```

---

## Next Steps

### For Baseline Modeling
1. Read `PREPROCESSING_GUIDE.md` and `FEATURE_EXTRACTION_GUIDE.md`
2. Try different classifiers (SVM, XGBoost)
3. Implement cross-validation
4. Report accuracy, F1, confusion matrix

### For Deep Learning
1. Read `ROADMAP.md` Phase 5
2. Implement 1D CNN on raw signals
3. Experiment with architectures (ResNet, GRU)
4. Use GPU for faster training

### For Conformal Prediction
1. Read `API_REFERENCE.md`
2. Implement split conformal on your model
3. Evaluate coverage-accuracy trade-off
4. Document results

---

## Common Tasks

### Change Window Size
```python
# In preprocessing.ipynb, modify config:
WINDOW_SEC = 5  # Instead of 10

# Then re-run preprocessing
```

### Use Different Modality
```python
# In feature extraction or model:
X = data['X_ecg']       # ECG instead of wrist PPG
X = data['X_fused']     # All 4 channels
X = data['X_ppg_gsr']   # PPG + GSR only
```

### Process Subset of Subjects
```python
# In preprocessing.ipynb:
PARTICIPANT_IDS = ['002', '003', '004', '005']  # Only 4

# Then re-run preprocessing (much faster)
```

### Get Feature Names
```python
feat_data = np.load('MAUS/preprocessed/features_extracted.npz')
feat_names = feat_data['algo1_feature_names']
print(f"Features: {list(feat_names)[:10]}")
```

---

## Troubleshooting

**Q: Import error for pywt?**
```bash
pip install PyWavelets
```

**Q: Data files not found?**
```bash
# Ensure MAUS/Data/Raw_data/ exists with CSV files
ls MAUS/Data/Raw_data/002/
# Should show: inf_ecg.csv, inf_ppg.csv, pixart.csv, etc.
```

**Q: Jupyter notebook won't start?**
```bash
python -m jupyter notebook MAUS/preprocessing.ipynb
```

**Q: Too slow?**
```python
# Use already-preprocessed features instead of running notebooks
# Features are saved in CSV/NPZ ready to load
```

---

## Important Files

| File | Purpose |
|------|---------|
| `preprocessing.ipynb` | Load → Filter → Window → Save |
| `feature_extraction.ipynb` | Extract features, visualize |
| `MAUS/preprocessed/*.npz` | Data for models |
| `config.py` (coming) | Centralized configuration |
| `README.md` | Overview |
| `ARCHITECTURE.md` | Pipeline details |
| `ROADMAP.md` | Future phases |

---

## Further Reading

- **[INSTALLATION.md](INSTALLATION.md)** — Detailed setup
- **[PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)** — Preprocessing details
- **[FEATURE_EXTRACTION_GUIDE.md](FEATURE_EXTRACTION_GUIDE.md)** — Feature how-to
- **[FAQ.md](FAQ.md)** — Common issues & solutions
- **[ROADMAP.md](ROADMAP.md)** — What comes next
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — How to contribute

---

**Estimated time to first model**: 30 minutes (with data already downloaded)

Good luck! 🚀

---

**Last Updated**: 2026-03-04
