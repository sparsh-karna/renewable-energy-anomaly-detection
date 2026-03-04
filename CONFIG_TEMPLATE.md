# Configuration Template

Use this template to customize preprocessing and feature extraction parameters for different experiments.

## Config File: `config.py` (Recommended Future Addition)

```python
# config.py — Centralized configuration for all pipelines

# ============================================================================
# PREPROCESSING PARAMETERS
# ============================================================================

# Sampling rates
FS_INF = 256        # Infrared sensors (ECG, PPG, GSR) in Hz
FS_PIX = 100        # PixArt wrist PPG in Hz
FS_TARGET = 256     # Target resampling rate in Hz

# Windowing
WINDOW_SEC = 10     # Window duration in seconds
OVERLAP_RATIO = 0.5 # Overlap between consecutive windows (0–1)

# Filtering
FILTER_ORDER = 4    # Butterworth filter order

FILTER_PARAMS = {
    'ecg':      {'lowcut': 0.5,  'highcut': 40.0},    # Hz
    'ppg_inf':  {'lowcut': 0.5,  'highcut': 8.0},
    'ppg_pix':  {'lowcut': 0.5,  'highcut': 8.0},
    'gsr':      {'lowcut': 0.05, 'highcut': 5.0},
}

# Artifact detection
ARTIFACT_ZSCORE_THRESH = 5.0    # Z-score threshold for outlier detection
ARTIFACT_MAX_RATIO = 0.3        # Reject window if > 30% artifacts

# Trial-to-condition mapping (from dataset column headers)
TRIAL_CONDITION = {
    0: 0,   # Trial 1 → 0-back (low workload)
   1: 2,   # Trial 2 → 2-back (medium)
    2: 3,   # Trial 3 → 3-back (high)
    3: 2,   # Trial 4 → 2-back
    4: 3,   # Trial 5 → 3-back
    5: 0,   # Trial 6 → 0-back
}

# Label mapping
LABEL_MAP = {0: 0, 2: 1, 3: 2}  # n-back level → class index (0, 1, 2)

# Participants to process (empty list = all)
PARTICIPANT_IDS = []  # If empty, loads dynamically from Data/Raw_data/

# ============================================================================
# FEATURE EXTRACTION PARAMETERS
# ============================================================================

# Frequency bands for spectral features
FREQ_BANDS = {
    'ecg': {
        'VLF': (0.003, 0.04),   # Very low frequency
        'LF':  (0.04, 0.15),    # Low frequency
        'HF':  (0.15, 0.4),     # High frequency
    },
    'ppg': {
        'VLF': (0.003, 0.04),
        'LF':  (0.04, 0.15),
        'HF':  (0.15, 0.4),
    },
    'gsr': {
        'Tonic':  (0.05, 0.2),   # Slow tonic baseline
        'Phasic': (0.2, 1.0),    # Faster phasic response
        'Fast':   (1.0, 5.0),    # High-freq noise
    },
}

# Welch's method parameters (for PSD estimation)
WELCH_NPERSEG = 512  # Window length in samples

# DWT parameters
DWT_WAVELET = 'db4'   # Daubechies-4 wavelet
DWT_LEVEL = 5         # Number of decomposition levels

# ============================================================================
# TRAIN/CALIBRATION/TEST SPLIT
# ============================================================================

TRAIN_SUBJECTS = 14
CAL_SUBJECTS = 4
TEST_SUBJECTS = 4
# (total should be <= 22 available)

RANDOM_SEED = 42  # For reproducibility

# ============================================================================
# FEATURE STANDARDIZATION
# ============================================================================

STANDARDIZE_FEATURES = True  # Z-score normalization before saving

# ============================================================================
# OUTPUT PATHS
# ============================================================================

DATA_DIR = 'MAUS/Data'
RAW_DATA_DIR = 'MAUS/Data/Raw_data'
IBI_DIR = 'MAUS/Data/IBI_sequence'
SUBJ_DIR = 'MAUS/Subjective_rating'
OUTPUT_DIR = 'MAUS/preprocessed'

# ============================================================================
# MODEL TRAINING PARAMETERS (for future use)
# ============================================================================

# Deep learning hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
DROPOUT = 0.3
OPTIMIZER = 'adam'  # or 'sgd'
LOSS = 'categorical_crossentropy'  # for 3-way classification

# Conformal prediction
CONFORMAL_COVERAGE = 0.9  # 90% coverage guarantee

# GPU
USE_GPU = True
GPU_ID = 0

# ============================================================================
# LOGGING & VERBOSITY
# ============================================================================

VERBOSE = True
LOG_INTERVAL = 10  # Print status every N windows

```

## How to Use Config

### In Notebooks

```python
# At top of notebook
exec(open('config.py').read())  # Load configuration

# Then reference constants
FS = FS_TARGET
WINDOW_LEN = WINDOW_SEC * FS

print(f"Processing {len(PARTICIPANT_IDS) or 22} participants")
```

### Override Specific Parameters

For experiments, override in notebook:

```python
# Load default config
exec(open('config.py').read())

# Override for this run
WINDOW_SEC = 20          # Longer windows
ARTIFACT_MAX_RATIO = 0.5 # More lenient
DWT_WAVELET = 'sym5'     # Different wavelet

print(f"Config: {WINDOW_SEC}s windows, {DWT_WAVELET} wavelet")
```

### Create Experiment-Specific Configs

```python
# config_experiment_cross_modality.py

# Load base config
exec(open('config.py').read())

# Override for this experiment
PARTICIPANT_IDS = ['002', '003', '004', '005']  # Smaller subset
WINDOW_SEC = 10
LABEL_MAP = {0: 0, 2: 1, 3: 2}

# Save preprocessed data separately
OUTPUT_DIR = 'MAUS/preprocessed_cross_modal'
```

Then in notebook:

```python
exec(open('config_experiment_cross_modality.py').read())
# Run preprocessing
```

---

## Configuration Profiles

### Profile 1: Quick Test (Fast Prototyping)

```python
# Minimal dataset
PARTICIPANT_IDS = ['002', '003', '004']  # Only 3 subjects
WINDOW_SEC = 5  # Shorter windows
OVERLAP_RATIO = 0.0  # No overlap
ARTIFACT_MAX_RATIO = 0.1  # Stricter filtering
```

**Use when:** Rapidly iterating, debugging pipeline

**Expected**: ~50–100 windows, ~30s processing

---

### Profile 2: Full Quality (Production)

```python
# All participants, strict quality
PARTICIPANT_IDS = []  # All 22
WINDOW_SEC = 10
OVERLAP_RATIO = 0.5
ARTIFACT_ZSCORE_THRESH = 5.0
ARTIFACT_MAX_RATIO = 0.3
DWT_LEVEL = 5
```

**Use when:** Final experiments, reproducibility

**Expected**: ~900 windows, ~5 min processing

---

### Profile 3: High Resolution (Spectral Analysis)

```python
# Better frequency resolution
WINDOW_SEC = 30  # Longer windows → higher freq resolution
OVERLAP_RATIO = 0.75  # More overlap
WELCH_NPERSEG = 1024  # Larger Welch window
```

**Use when:** Studying frequency-specific phenomena

**Trade-off**: Fewer windows, better spectral detail

---

### Profile 4: Minimalist (Fastest Features)

```python
# Only essential features
WINDOW_SEC = 5
OVERLAP_RATIO = 0.0
DWT_LEVEL = 3  # Only 3 decomposition levels
# (Reduces feature dimension from 120 to 72)
```

**Use when:** Real-time or resource-constrained deployment

---

## Parameter Sensitivity

### Window Size
- **5s**: Many windows (~1,800), noisy statistics
- **10s** (default): Good balance (~900 windows)
- **20s**: Fewer windows (~450), better statistics
- **30s**: Highest quality, fewest samples (~300)

### Overlap
- **0** (no overlap): Independent windows (less data)
- **0.5** (default): Balanced (standard practice)
- **0.75**: High correlation, more samples

### Artifact Threshold
- **z > 3**: Aggressive (80–90% windows pass)
- **z > 5** (default): Conservative (85–90% pass)
- **z > 7**: Very lenient (95%+ pass, noisy)

### Frequency Bands
For different populations or task types, adjust bands:

```python
# For high-arousal tasks (exercise)
FREQ_BANDS['ecg'] = {
    'VLF': (0.001, 0.04),  # Extended VLF
    'LF':  (0.04, 0.25),   # Wider LF
    'HF':  (0.25, 0.5),    # Shifted HF
}
```

---

## Reproducibility Checklist

- [ ] Config file committed to Git
- [ ] Random seed set (`RANDOM_SEED = 42`)
- [ ] Participant list fixed
- [ ] Window size documented
- [ ] Artifact thresholds recorded
- [ ] Preprocessing script version noted
- [ ] Feature list enumerated (which features used?)
- [ ] Results table saved with timestamp

**Example notebook header:**

```python
# =================================================================
# Experiment: Baseline Models with Algorithm 1 Features
# Date: 2026-03-04
# Config: config.py + manual overrides below
# =================================================================

exec(open('config.py').read())

# Overrides
PARTICIPANT_IDS = []  # All 22
WINDOW_SEC = 10
DWT_LEVEL = 5  # Not used, only Algo 1

print(f"Config: Algo1 only, {WINDOW_SEC}s windows, {len(PARTICIPANT_IDS or ©)} subjects")
```

---

**Last Updated**: 2026-03-04
