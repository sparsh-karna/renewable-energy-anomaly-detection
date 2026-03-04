# Installation & Setup Guide

## Prerequisites

- Python 3.8+
- pip or conda
- ~2 GB disk space (for preprocessed data)
- ~4 GB RAM (for processing)
- GPU optional (NVIDIA CUDA for deep learning, later stages)

## Step 1: Clone Repository

```bash
cd /Users/sparshkarna/dev/sem6/ai
git clone https://github.com/[your-username]/ai.git
cd ai
```

## Step 2: Create Virtual Environment

### Using venv
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Using conda
```bash
conda create -n maus-workload python=3.10
conda activate maus-workload
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **Scientific computing**: numpy, scipy, pandas, scikit-learn
- **Signal processing**: PyWavelets
- **Visualization**: matplotlib, seaborn
- **Jupyter**: jupyter, ipykernel, ipywidgets
- **Deep learning** (optional): torch, tensorflow
- **Development**: pytest, black, flake8

## Step 4: Verify Installation

```bash
python -c "import numpy, scipy, pandas, pywt; print('✓ All imports successful')"
```

## Step 5: Set Up Jupyter

```bash
jupyter notebook
# Opens browser at http://localhost:8888
```

## Project Structure

After installation:
```
ai/
├── .gitignore
├── CHANGELOG.md
├── README.md
├── ARCHITECTURE.md
├── MAUS_DATASET.md
├── requirements.txt
│
├── MAUS/
│   ├── preprocessing.ipynb         ← START HERE
│   ├── feature_extraction.ipynb     ← Then here
│   ├── Data/
│   │   ├── Raw_data/               (download separately)
│   │   ├── IBI_sequence/           (download separately)
│   │   └── Subjective_rating/      (download separately)
│   └── preprocessed/               (generated after running preprocessing)
│       ├── windowed_data.npz
│       ├── splits.npz
│       └── features_*.csv/.npz
│
└── [Future notebooks for models]
```

## Step 6: Download Dataset

The MAUS dataset is too large for Git (>5 GB).

**Option A: From Official Source**
- Contact MAUS dataset maintainers (link in MAUS_DATASET.md)
- Extract to `MAUS/Data/`

**Option B: Use Preprocessed Data Only**
(If you only want to train models, not preprocess)
- Request preprocessed NPZ files from project maintainers
- Extract to `MAUS/preprocessed/`

## Step 7: Run Preprocessing

```bash
cd MAUS
jupyter notebook preprocessing.ipynb
```

**Expected runtime**: ~5 minutes
**Output files** (in `MAUS/preprocessed/`):
- `windowed_data.npz` (36 MB)
- `splits.npz`
- `subjective_ratings.csv`
- `signal_quality.csv`

## Step 8: Run Feature Extraction

```bash
jupyter notebook feature_extraction.ipynb
```

**Expected runtime**: ~2 minutes
**Output files** (in `MAUS/preprocessed/`):
- `features_stat_spectral.csv`
- `features_dwt.csv`
- `features_extracted.npz`
- Visualizations (PNG)

## Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'pywt'`
**Solution:**
```bash
pip install PyWavelets
```

### Problem: `numpy.trapz` not found (NumPy 2.0+)
**Solution:** Already fixed in notebooks (uses `np.trapezoid` instead)

### Problem: Jupyter kernel not found
**Solution:**
```bash
python -m ipykernel install --user --name maus-workload
# Then in Jupyter: Kernel → Change kernel → maus-workload
```

### Problem: Memory error during preprocessing
**Solution:** Reduce window overlap or skip certain participants:
```python
# In preprocessing.ipynb, modify:
PARTICIPANT_IDS = ['002', '003', '004', '005']  # Subset
```

### Problem: `FileNotFoundError: MAUS/Data/Raw_data/...`
**Solution:** Download datasets first to correct location (Step 6)

## Performance Tips

- **Faster processing**: Use a multi-core machine (preprocessing uses single core, but feature extraction can be parallelized)
- **Lower memory**: Process participants in batches instead of all at once
- **GPU for models**: Install CUDA-enabled PyTorch for model training (later notebooks)

## Environment Variables (Optional)

```bash
export MAUS_DATA_DIR=/path/to/MAUS/Data
export MAUS_PREPROCESSED_DIR=/path/to/MAUS/preprocessed
```

Then notebooks can auto-detect paths.

## Next Steps

1. ✓ Install dependencies
2. ✓ Run preprocessing
3. ✓ Run feature extraction
4. → Build baseline models (coming soon)
5. → Implement conformal prediction (coming soon)
6. → Cross-modal transfer experiments (coming soon)

## Getting Help

- Check TROUBLESHOOTING section above
- Review notebook inline comments
- Check GitHub Issues: [link to be added]

---

**Last Updated**: 2026-03-04
