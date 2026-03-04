# Versioning & Release Notes

## Version History

### v0.2.0 — Feature Extraction Phase (2026-03-04)

**Status**: Released ✓

**Features Added:**
- Algorithm 1: Statistical + Spectral Features (52 features)
- Algorithm 2: Discrete Wavelet Transform (120 features)
- Comprehensive feature visualizations (box plots, PCA, correlations)
- Feature quality reports and artifact analysis

**Technical Improvements:**
- Fixed NumPy 2.0 compatibility (`np.trapz` → `np.trapezoid`)
- Added feature standardization and NaN handling
- Per-channel modality processing (ECG, PPG×2, GSR)

**Documentation:**
- API reference for all feature functions
- Configuration template with profiles
- Troubleshooting guide
- Contributing guidelines

**Testing:**
- ✓ 900 clean windows across 22 participants
- ✓ Feature extraction: 2–3 min per phase
- ✓ Zero NaN/Inf values in output
- ✓ Class-balanced dataset (33% each)

**Known Issues:**
- Wrist PPG artifact rate high (18%) — motion artifacts from consumer-grade hardware
- Feature redundancy (some correlations > 0.9) — recommend PCA for downstream models
- Subjective ratings correlation moderate (r~0.3–0.4) with physiological signals

**Migration Notes:** N/A (new release)

---

### v0.1.0 — Preprocessing Phase (2026-02-28)

**Status**: Released ✓

**Features:**
- Signal loading and multi-modal filtering
- Artifact detection (z-score based)
- Resampling (PixArt 100 Hz → 256 Hz)
- 10s windowing with 50% overlap
- Train/Calibration/Test split (14/4/4 subjects)

**Quality:**
- ✓ 85% window retention rate
- ✓ All participants loaded (22/22)
- ✓ All modalities processed (ECG, PPG_inf, PPG_pix, GSR)

---

## Version Naming Convention

Format: `MAJOR.MINOR.PATCH-STATUS`

- **MAJOR**: Large feature additions or architecture changes (0 → 1 = end of modeling)
- **MINOR**: Phase completions (preprocessing, features, models, etc.)
- **PATCH**: Bug fixes, documentation improvements
- **STATUS**: `-dev`, `-alpha`, `-beta`, `-rc` (pre-release), or none (stable)

### Current Roadmap
- **v0.3.0**: Baseline models (Phase 4)
- **v0.4.0**: Deep learning (Phase 5)
- **v0.5.0**: Conformal prediction (Phase 6)
- **v0.6.0**: Cross-modality transfer (Phase 7)
- **v1.0.0**: Final evaluation & report (Phase 10)

---

## Release Checklist

When preparing a release:

- [ ] Complete feature implementation
- [ ] All tests passing (unit + integration)
- [ ] Documentation updated (API, guides, README)
- [ ] CHANGELOG.md updated with version & date
- [ ] VERSION or setup.py updated
- [ ] Git tag created: `git tag v0.X.Y`
- [ ] Push tag: `git push origin v0.X.Y`
- [ ] GitHub release notes drafted
- [ ] All notebooks validated (cells re-run, output cleared)

Example:
```bash
git tag v0.2.0
git push origin v0.2.0
# Then create release on GitHub with summary
```

---

## Backward Compatibility

**Policy**: Minor versions maintain backward compatibility. Breaking changes only in major versions.

**Data Format Stability**:
- NPZ array layout stable (v0.2+)
- CSV column order stable
- Function signatures stable

**If Breaking Changes Needed**:
- Deprecated functions for 1 minor version before removal
- Migration guide in release notes
- Clear API documentation

---

## Security & Dependency Updates

### Dependency Policy
- Pin major versions in `requirements.txt` for reproducibility
- Regular security updates (patch versions automatically)
- Major version bumps require testing on new version

### Reporting Issues
- Security vulnerability: Email maintainers (do not open GitHub issue)
- Bug report: Open GitHub issue with reproducible example
- Feature request: Open GitHub discussion

---

## Performance Metrics by Version

| Version | Preprocessing (22 subj) | Features (900 wins) | Model Accuracy |
|---------|---|---|---|
| v0.1.0  | 5 min | N/A | N/A |
| v0.2.0  | 5 min | 2.5 min | N/A |
| v0.3.0  | 5 min | 2.5 min | Est. 70% |
| v1.0.0  | 5 min | 2.5 min | Est. 78% |

---

## Archived Versions

- **v0.1.0** (Feb 28, 2026): Initial preprocessing
- **v0.2.0** (Mar 04, 2026): Feature extraction

---

**Last Updated**: 2026-03-04
