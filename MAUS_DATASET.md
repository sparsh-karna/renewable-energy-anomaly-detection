# MAUS Dataset Documentation

## Overview

**MAUS** (Mental workload, physiological signals, pAy attention, and Uncertainty in Self-reports)

- **22 Subjects** (IDs: 002–025, with 001, 007, 009 missing)
- **6 Trials per subject** (N-back cognitive task)
- **3 Workload Levels**: 0-back (low), 2-back (medium), 3-back (high)
- **4 Physiological Modalities**: ECG, Fingertip PPG, Wrist PPG, GSR/EDA
- **Sampling Duration**: ~50 seconds per trial

## Directory Structure

```
MAUS/
├── MAUS_Documentation.pdf     # Official dataset paper
├── Data/
│   ├── Raw_data/              # Raw time-series signals
│   │   ├── {002..025}/
│   │   │   ├── inf_ecg.csv         # Infrared ECG (6 columns = 6 trials)
│   │   │   ├── inf_ppg.csv         # Infrared fingertip PPG
│   │   │   ├── inf_gsr.csv         # Infrared GSR/EDA
│   │   │   ├── inf_resting.csv     # Resting baseline (ECG/PPG/GSR together)
│   │   │   ├── pixart.csv          # PixArt wrist PPG
│   │   │   ├── pixart_resting.csv  # Wrist PPG resting
│   │   │   └── record.xlsx         # Metadata/log per trial
│   │
│   └── IBI_sequence/          # Derived inter-beat intervals (peak detection)
│       ├── {002..025}/
│       │   ├── rest_1.csv, rest_2.csv, rest_3.csv
│       │   └── trial_{1..6}_{1..3}.csv + trial_*_*_peak.csv
│
└── Subjective_rating/         # Questionnaire responses
    ├── {002..025}/
    │   ├── NASA_TLX.csv       # Workload ratings per trial
    │   └── PSQI.csv           # Sleep quality score
```

## Raw Signal Files (per subject)

### Format
Each raw CSV file contains 6 columns (one per trial), with optional 3 sub-segments per trial.

**Row 1 (Header):**
```
Trial 1:0back,Trial 2:2back,Trial 3:3back,Trial 4:2back,Trial 5:3back,Trial 6:0back
```

**Rows 2+:** Numeric samples

### Files

| File | Modality | Fs | Channels | Typical Rows | Notes |
|------|----------|----|---------|----|-------|
| `inf_ecg.csv` | Infrared ECG | 256 Hz | 6 | 76,801 | Gold standard for HR |
| `inf_ppg.csv` | Infrared fingertip PPG | 256 Hz | 6 | 76,801 | Clinical-grade wrist PPG |
| `inf_gsr.csv` | Infrared GSR/EDA | 256 Hz | 6 | 76,801 | Skin conductance |
| `inf_resting.csv` | Infrared (rest) | 256 Hz | 3 | 74,971 | Baseline (ECG/PPG/GSR stacked) |
| `pixart.csv` | PixArt wrist PPG | 100 Hz | 6 | 30,001 | Consumer wearable |
| `pixart_resting.csv` | PixArt (rest) | 100 Hz | 1 | 29,449 | Wrist baseline |
| `record.xlsx` | Metadata | – | – | – | Trial start/end times |

### Example Structure (inf_ecg.csv)
```
Trial 1:0back,Trial 2:2back,Trial 3:3back,Trial 4:2back,Trial 5:3back,Trial 6:0back
0.357,-0.278,-0.035,0.313,-0.09,0.097
0.328,-0.291,-0.127,0.309,-0.085,0.135
...
[76,800 rows total, ~50 seconds × 256 Hz]
```

## IBI Sequence Files (Derived)

Pre-processed inter-beat intervals via peak detection.

### Format
CSV with 4 columns: `RRI_inf`, `PPI_inf`, `PPI_pix`, `label`

- **RRI_inf**: R-R interval from infrared ECG (milliseconds)
- **PPI_inf**: Pulse-pulse interval from fingertip PPG (ms)
- **PPI_pix**: Pulse-peak interval from wrist PPG (ms)
- **label**: Trial condition encoded (0 = 0-back, etc.)

### Example (trial_1_1.csv)
```
RRI_inf,PPI_inf,PPI_pix,label
628.91,625.0,624.39,0.0
621.09,628.91,624.39,0.0
...
[varies by trial, ~200-300 intervals]
```

**Note:** IBI files are useful for hand-crafted HRV features (SDNN, RMSSD, LF/HF ratio), but the preprocessing pipeline uses raw waveforms for deep learning.

## Subjective Ratings (NASA-TLX & PSQI)

### NASA-TLX (NASA Task Load Index)

6-point weighted scale per trial:
1. **Mental Demand** — cognitive effort
2. **Physical Demand** — motor effort
3. **Temporal Demand** — time pressure
4. **Performance** — success perception
5. **Effort** — overall exertion
6. **Frustration** — irritation

**Output**: Adjusted weighted workload score (0–100) per trial.

**Example (NASA_TLX.csv):**
```
Mental Demand,Physical Demand,Temporal Demand,Performance,Effort,Frustration
50,15,45,60,70,40     # Trial 1 subscales
55,12,50,55,75,38     # Trial 2
...
Row 7: 45.5,47.3,50.2,52.1,48.9,46.7  # Adjusted weighted ratings (6 values = 6 trials)
```

### PSQI (Pittsburgh Sleep Quality Index)

Single global score (0–21) per subject. Higher = worse sleep quality.

**Example (PSQI.csv):**
```
10
```

## Sampling Details

### Infrared Sensors (fs=256 Hz)
- **Technology**: Photoplethysmography with infrared LEDs
- **Location**: ECG → chest (electrode), PPG → fingertip
- **Duration**: ~50s per trial
- **Data type**: Float (normalized amplitude, ±1.0 range typical)

### PixArt Wrist Sensor (fs=100 Hz)
- **Technology**: Optical reflectance PPG (smart watch-like)
- **Location**: Wrist
- **Duration**: ~50s per trial
- **Data type**: Float (arbitrary units, not normalized by device)

### GSR Sensor (fs=256 Hz)
- **Technology**: Two electrodes on fingers
- **Location**: Index + middle finger, same hand as wrist PPG
- **Range**: 0–100 microSiemens typical
- **Data type**: Float (μS conductance)

## Task Protocol

### N-back Cognitive Task
Subjects view a sequence of letters on screen and respond (button/key) when current letter matches the letter from N positions back.

- **0-back**: Press when any specific target letter appears (baseline, minimal WL)
- **2-back**: Press when current letter matches letter from 2 steps prior (medium WL)
- **3-back**: Press when current letter matches letter from 3 steps prior (high WL)

**Trial Structure:**
```
Each trial: ~50s duration
├─ Instruction (2s)
├─ Stimulus sequence (45s, ~10 letters/sec)
└─ Response window

6 trials: [0-back, 2-back, 3-back, 2-back, 3-back, 0-back]
(Balanced within-subjects design, counterbalanced across subjects)
```

## Resting Baseline

Before and/or after the task block:
- 2–3 minutes of eyes-open rest
- Instructions: "Sit quietly and relax"
- No cognitive task

**Used for**: Baseline heart rate, respiratory rate, tonic GSR.

## Trial Labeling

Each column in raw files is labeled by position (1–6), and the header indicates the condition:

```python
TRIAL_CONDITION = {
    0: 0,   # Trial 1 → 0-back (low)
    1: 2,   # Trial 2 → 2-back (medium)
    2: 3,   # Trial 3 → 3-back (high)
    3: 2,   # Trial 4 → 2-back (medium)
    4: 3,   # Trial 5 → 3-back (high)
    5: 0,   # Trial 6 → 0-back (low)
}
```

## Data Characteristics

### Signal Quality
- **Sample Rate Mismatch**: Infrared @ 256 Hz, PixArt @ 100 Hz → resampling required
- **Noise Sources**: Motion artifacts (especially wrist), powerline 50/60 Hz interference
- **Artifact Rate**: ~15–30% of windows contain significant artifacts (z-score > 5)

### Physiological Ranges
| Signal | Typical Range | Units | Notes |
|--------|---|---|---|
| ECG | ±1.0 | V (normalized) | QRS complex ~0.1–0.5 V clinical |
| PPG | ±1.0 | a.u. | DC + AC components; AC = pulsation |
| GSR | 0–10 | μS | Baseline 2–5 μS, phasic response up to 20 μS |

### Class Distribution (22 subjects × 6 trials/subject)
- 0-back (low): 44 trials (22 × 2)
- 2-back (medium): 44 trials
- 3-back (high): 44 trials
- **Total**: 132 trials ≈ 1,100 minutes of recording

## How Preprocessing Uses This Data

1. **Load**: Read 6 columns from each raw CSV → split into 6 trial signals
2. **Label**: Map trial index → workload condition via `TRIAL_CONDITION`
3. **Filter**: Apply bandpass per signal type (ECG 0.5–40 Hz, PPG 0.5–8 Hz, GSR 0.05–5 Hz)
4. **Artifact Remove**: Detect z-score outliers, interpolate
5. **Resample**: PixArt 100 Hz → 256 Hz to match infrared
6. **Window**: Segment each trial into 10s windows with 50% overlap → ~9–10 windows/trial
7. **Normalize**: Z-score per window
8. **Filter Clean**: Remove windows with > 30% artifacts

**Output**: ~900 clean windows with labels, split across train/cal/test.

## References

Official MAUS paper (if published):
- [Link to be added when available]

Related datasets:
- **WESAD**: Wearable stress and affect dataset (multimodal wearables)
- **DEAP**: Database of emotion recognition using physiological signals
- **HTF**: Heart-Rate Variability Datasets

---

**Last Updated**: 2026-03-04
