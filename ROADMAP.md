# Project Roadmap & Future Work

## Vision

Implement and validate uncertainty-aware deep learning for physiological signal classification, with explicit quantification of confidence and robustness across sensor modalities.

## Completed Phases

### Phase 1: Data Acquisition & Exploration ✓
- [x] MAUS dataset loaded and documented
- [x] 22 subjects, 4 physiological modalities analyzed
- [x] NASA-TLX and PSQI ratings integrated

### Phase 2: Preprocessing ✓
- [x] Multi-modal filtering (ECG, PPG, GSR with appropriate cutoffs)
- [x] Artifact detection and interpolation
- [x] Resampling (wrist PPG 100→256 Hz)
- [x] Windowing (10s, 50% overlap)
- [x] Quality filtering (reject windows >30% artifacts)
- [x] Train/Calibration/Test split (14/4/4 subjects)

### Phase 3: Feature Extraction ✓
- [x] Algorithm 1: Statistical + Spectral Features (52 total)
- [x] Algorithm 2: Discrete Wavelet Transform Features (120 total)
- [x] Feature visualization (box plots, correlations, PCA)
- [x] Output in CSV and NPZ formats

## Upcoming Phases

### Phase 4: Baseline Models (Next)
**Goal**: Establish performance ceiling using standard ML

**Tasks:**
- [ ] Logistic Regression on Algorithm 1 features
- [ ] Random Forest on both feature sets
- [ ] SVM (RBF kernel) comparison
- [ ] Nested cross-validation for hyperparameter tuning
- [ ] Report: accuracy, F1, confusion matrices

**Expected Performance:**
- Accuracy: ~65–75% (3-way classification is harder than 2-way)
- Best modality: ECG or fingertip PPG (clinical-grade)
- Worst: wrist PPG (consumer-grade noise)

**Deliverables:**
- `baseline_models.ipynb`
- Performance table (accuracy/F1 per model × modality)
- Feature importance plots

### Phase 5: Deep Learning (Core Novelty)
**Goal**: End-to-end learning on raw signals for robustness

**Architecture Options:**

#### Option A: Simple 1D CNN
```python
Input: (B, 1 channel, 2560 samples) @ 256 Hz = 10s
  ↓ Conv1D (64 filters, kernel=16, ReLU, BatchNorm)
  ↓ MaxPool (4)
  ↓ Conv1D (128 filters, kernel=8, ReLU, BatchNorm)
  ↓ GlobalAvgPool
  ↓ Dense(64, ReLU, Dropout 0.3)
Output: Dense(3, softmax) → probabilities for [low, med, high]
```

#### Option B: Residual CNN (ResNet-18 adapted)
- Better gradient flow for deeper networks
- Skip connections for learning identity mappings
- Tested on physiological signals (WESAD baseline)

#### Option C: Temporal CNN + GRU
```
Conv feature extraction (1-3 layers)
  ↓
GRU for temporal dependencies (64 units)
  ↓
Dense for classification
```

**Tasks:**
- [ ] Implement 1D CNN on wrist PPG (baseline)
- [ ] Implement ResNet-18 variant (comparison)
- [ ] Implement CNN+GRU (if temporal info helps)
- [ ] Hyperparameter search (batch size, LR, dropout)
- [ ] Evaluate on test set with uncertainty quantification

**Deliverables:**
- `deep_learning.ipynb`
- Model checkpoints (best ckpt per architecture)
- Training curves (loss, accuracy)

### Phase 6: Conformal Prediction (Key Novelty)
**Goal**: Output prediction sets instead of single labels when uncertain

**Algorithm: Split Conformal Prediction**

```
1. Split data: Train (60%), Calibration (20%), Test (20%)
2. Train classifier on Train set
3. On Calibration set:
   - Compute non-conformity scores: |y_true - confidence|
   - Find threshold for desired coverage (e.g., 95%)
4. At test time:
   - For each sample, output prediction set (e.g., {low, med})
   - Guaranteed: ≥95% of true labels in prediction set
```

**Tasks:**
- [ ] Implement split conformal prediction wrapper
- [ ] Calibrate on cal set (find quantile)
- [ ] Evaluate coverage vs. accuracy trade-off
- [ ] Visualize prediction sets (as error bars or sets)
- [ ] Compare to softmax confidence (baseline uncertainty)

**Expected Results:**
- Coverage guarantee at 90%: ~90%±5% of test samples
- Average set size: 1.2–1.5 (mostly single predictions, some ambiguous)
- Smaller sets for high-confidence examples

**Deliverables:**
- `conformal_prediction.ipynb`
- Coverage-accuracy plot (x=coverage, y=avg set size)
- Example prediction sets per class

### Phase 7: Cross-Modality Transfer (Robustness Study)
**Goal**: Test domain shift when deploying across sensors

**Experiments:**

#### Experiment A: ECG → Wrist PPG
```
Train on: ECG (clinical gold standard)
Test on:  Wrist PPG (consumer wearable)
Metric:   Accuracy drop (expect 15–25% drop)
```

#### Experiment B: Fingertip PPG → Wrist PPG
```
Train:    Fingertip PPG (infrared, better quality)
Test:     Wrist PPG (worse quality, motion artifacts)
Metric:   Robustness to domain shift
```

#### Experiment C: Multi-Source Training
```
Train:    ECG + Fingertip PPG + GSR (all 3)
Test:     Wrist PPG (single sensor)
Metric:   Benefit of diversity vs. source mismatch
```

**Uncertainty Angle:**
- Does conformal prediction help in domain shift?
- Are confidence estimates reliable under distribution shift?

**Tasks:**
- [ ] Train 3 models (ECG, fingertip PPG, GSR single modality)
- [ ] Evaluate on test (same modality) vs. test (wrist PPG)
- [ ] Apply conformal prediction to each
- [ ] Plot coverage-accuracy separately per transfer scenario

**Deliverables:**
- `transfer_learning.ipynb`
- Domain-shift penalty table (ACCin-domain vs. ACCcross-modal)
- Uncertainty estimates under shift

### Phase 8: Sensor Fusion
**Goal**: Combine multiple modalities for stability

**Strategies:**

#### Early Fusion
```
X = [ecg(t), ppg_inf(t), ppg_pix(t), gsr(t)]  concatenate
      ↓ CNN
      class prediction
```

#### Late Fusion
```
class_ecg = CNN_ecg(ecg)
class_ppg = CNN_ppg(ppg)
class_gsr = CNN_gsr(gsr)
        ↓ Ensemble (voting or weighted average)
        final class
```

#### Attention Fusion
```
Per modality: extract features
        ↓ Multi-head attention (learn modality weights)
        weighted aggregate
        ↓ Classification
```

**Tasks:**
- [ ] Implement early fusion CNN
- [ ] Implement late fusion (voting)
- [ ] Implement attention fusion
- [ ] Compare to single-modality baselines
- [ ] Analyze learned modality weights (which modalities matter for workload?)

**Deliverables:**
- `fusion.ipynb`
- Modality importance heatmap
- Fusion accuracy vs. single-modality

### Phase 9: Interpretability & Explainability
**Goal**: Understand what the model learns

**Techniques:**
1. **Saliency maps**: Which time regions matter?
   - Backprop gradient wrt input
   - Visualize importance of each timestamp

2. **Frequency analysis**: Which frequency bands drive decisions?
   - Extract CNN filters, interpret frequency response
   - Spectrogram-guided attention

3. **Feature attribution**: SHAP or LIME on extracted features
   - Rank importance of each feature
   - Per-sample explanations

**Tasks:**
- [ ] Compute saliency maps (gradient-based)
- [ ] Visualize temporal importance per class
- [ ] Compute spectral importance (FFT of top saliencies)
- [ ] SHAP analysis on extracted features
- [ ] Create interpretation report

**Deliverables:**
- `interpretability.ipynb`
- Saliency visualizations
- Feature importance rankings
- Frequency-band importance heatmap

### Phase 10: Evaluation & Final Report
**Goal**: Comprehensive results summary

**Report Contents:**
1. Methods
   - Preprocessing pipeline
   - Feature extraction (both algorithms)
   - Model architectures (baseline + deep)
   - Uncertainty quantification (conformal prediction)

2. Results
   - Performance tables (accuracy, F1, coverage)
   - Graphs: ROC curves, confusion matrices, trade-off plots
   - Cross-modality results
   - Fusion analysis

3. Discussion
   - Strengths (conformal prediction guarantees, multimodal)
   - Limitations (domain shift, feature importance)
   - Practical implications (wearable deployment)

4. Future Work
   - Real-time deployment
   - Personalization per subject
   - Handling concept drift (over-time adaptation)

**Deliverables:**
- `FINAL_REPORT.pdf` (or Markdown)
- `results_summary.ipynb` (reproducible figures)

---

## Timeline (Estimated)

| Phase | Task | Target Weeks |
|-------|------|---|
| 4 | Baseline ML | 1–2 |
| 5 | Deep CNNs | 2–3 |
| 6 | Conformal Prediction | 1 |
| 7 | Cross-modality | 1–2 |
| 8 | Fusion | 1–2 |
| 9 | Interpretability | 1–2 |
| 10 | Final Report | 1 |
| | **Total** | **~10–15 weeks** |

## Open Questions

1. **Domain shift under uncertainty**: Does conformal prediction remain well-calibrated when test domain ≠ train domain?
2. **Modality complementarity**: Which modalities complement each other best for workload?
3. **Real-time inference**: Can we subsample windows for faster predictions (e.g., 2.5s instead of 10s)?
4. **Personalization**: Should we train per-subject models with conformal prediction?
5. **Continuous workload**: Instead of 3 discrete classes, can we predict continuous workload (0–100)?

## Success Criteria

- [ ] Baseline accuracy > 70%
- [ ] Deep learning accuracy > 75%
- [ ] Conformal coverage ≈ nominal (e.g., 90% ± 5%)
- [ ] Cross-modal accuracy drop < 20 percentage points
- [ ] Interpretability report explains ≥50% of decisions

## Resources Needed

- GPU (for faster training, optional)
- Jupyter environment
- No additional labeled data (already have subjective ratings)

---

**Last Updated**: 2026-03-04
**Status**: Greenlit ✓ Ready for Phase 4
