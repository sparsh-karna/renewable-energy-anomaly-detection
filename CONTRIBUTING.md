# Contributing Guidelines

Thank you for your interest in contributing to the MAUS Uncertainty-Aware Workload Classification project!

## How to Contribute

### 1. Report Issues

Found a bug or have a feature request? Please open a GitHub issue with:
- **Title**: Clear one-liner (e.g., "Feature: Add EEG support")
- **Description**: Detailed explanation of the issue/feature
- **Steps to reproduce** (if bug): Exact code to trigger the issue
- **Expected vs. actual behavior**
- **Environment**: Python version, OS, key library versions

### 2. Submit Code Changes

#### Fork & Clone
```bash
git clone https://github.com/[your-username]/ai.git
cd ai
git checkout -b feature/your-feature-name  # Use meaningful branch names
```

#### Make Changes

**Code Style:**
- Follow PEP 8
- Use 4 spaces for indentation (not tabs)
- Max line length: 100 characters
- Use descriptive variable names

**For Jupyter Notebooks:**
- Keep cells focused (one logical unit per cell)
- Add markdown with clear explanations
- Include comments for complex logic
- Remove unnecessary output before committing

#### Run Checks
```bash
# Format code
black .

# Check for style issues
flake8 .

# Run tests (if available)
pytest tests/
```

#### Commit & Push
```bash
# Commit with clear message
git commit -m "feat: Add new feature X

Detailed explanation of changes, why they matter, any trade-offs.

Co-Authored-By: Your Name <your.email@example.com>"

# Push to your fork
git push origin feature/your-feature-name
```

#### Create Pull Request
- Open PR against `main` branch
- Title: Same as commit message (short, imperative)
- Description: Link to related issues, explain approach
- Wait for review and address feedback

### 3. Improve Documentation

Documentation is as important as code! Contributions welcome:
- Fix typos or clarification in README, guides, or docstrings
- Add examples to clarify complex concepts
- Create tutorials or blogs about your experiments

**Documentation Style:**
- Markdown for guides (`.md` files)
- Code blocks with syntax highlighting
- Include expected outputs/results
- Link to related sections

### 4. Run Experiments & Report Results

**Investigate open questions from the roadmap:**
1. Run a phase (e.g., Phase 4: Baseline models)
2. Document your approach and hyperparameters
3. Report results with confidence intervals
4. Create a PR with findings

**Expected in PR.**
- Reproducible code
- Hyperparameter justification
- Performance metrics
- Visualization of results
- Discussion of findings

## Code Guidelines

### General Principles
- **Simplicity over cleverness** — readable code is better
- **No premature generalization** — solve the current problem first
- **Meaningful names** — variable names should be self-documenting
- **DRY (Don't Repeat Yourself)** — but avoid over-abstraction

### Functions
```python
def extract_stat_spectral_features(signal, fs, signal_type):
    """
    Extract statistical + spectral features from a signal window.

    Parameters
    ----------
    signal : array-like, shape (n_samples,)
        1D signal sampled at fs Hz.
    fs : int
        Sampling rate in Hz.
    signal_type : str
        One of {'ecg', 'ppg', 'gsr'} to select appropriate filters.

    Returns
    -------
    features : dict
        Named features: 'mean', 'std', 'power_LF', etc.

    Examples
    --------
    >>> import numpy as np
    >>> sig = np.random.randn(1000)
    >>> feats = extract_stat_spectral_features(sig, fs=256, signal_type='ecg')
    >>> feats['mean']
    # ~0.0 (for random zero-mean signal)
    """
```

### Classes & Modules
- Use type hints if Python 3.8+
- Write docstrings (Google or NumPy style)
- Keep modules focused on a single responsibility

### Testing
If adding new functions, include unit tests:
```python
# tests/test_features.py
import numpy as np
from feature_extraction import extract_stat_spectral_features

def test_extract_stat_spectral_features():
    """Test feature extraction on synthetic signal."""
    sig = np.sin(2 * np.pi * 1 * np.linspace(0, 10, 2560))  # 1 Hz sine
    feats = extract_stat_spectral_features(sig, fs=256, signal_type='ecg')

    assert isinstance(feats, dict)
    assert 'mean' in feats
    assert abs(feats['rms'] - np.sqrt(0.5)) < 0.1  # Sine RMS ≈ √0.5
```

## Commit Message Format

```
<type>: <subject>

<body>

Co-Authored-By: Name <email@example.com>
```

**Type:** One of
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation change
- `refactor`: Code restructuring (no behavior change)
- `perf`: Performance improvement
- `test`: Test additions/changes

**Subject:** Imperative, present tense, no period. Max 70 chars.

**Body:** Explain what and why (not how, that's the code). Wrap at 80 chars.

## Naming Conventions

| Type | Convention | Example |
|------|---|---|
| Variables | snake_case | `window_size`, `artifact_ratio` |
| Constants | UPPER_SNAKE_CASE | `FS_INF = 256` |
| Functions | snake_case | `extract_dwt_features()` |
| Classes | PascalCase | `ConformalPredictor` |
| Notebooks | lowercase-with-hyphens | `deep_learning.ipynb` |
| Markdown files | Title-Case-with-hyphens | `PREPROCESSING_GUIDE.md` |

## Review Process

1. **Automated checks** (GitHub Actions)
   - Tests pass
   - Code style (black, flake8)
   - No conflicts with main

2. **Code review** (maintainers)
   - Correctness & clarity
   - Alignment with project goals
   - Documentation completeness

3. **Approval & merge**
   - At least 1 approval required
   - All discussions resolved
   - Branch deleted after merge

## Project Structure Principles

```
/ Root
├── *.md              # Top-level documentation (README, ROADMAP, etc.)
├── requirements.txt  # Python dependencies
├── MAUS/             # Main project code
│   ├── *.ipynb       # Pipelines (preprocessing, models, evaluation)
│   └── Data/         # Datasets (gitignored)
└── [tests/]          # (Future) Unit tests
```

## Areas for Contribution

### High Priority
- [ ] Implement Phase 4 (baseline models)
- [ ] Implement Phase 5 (deep learning)
- [ ] Implement Phase 6 (conformal prediction)
- [ ] Create reproducible experiment notebooks

### Medium Priority
- [ ] Add unit tests
- [ ] Performance optimizations
- [ ] Multi-GPU support

### Low Priority (Nice to Have)
- [ ] Docker containerization
- [ ] Web dashboard for results
- [ ] Paper/thesis writing and submission

## Questions?

- **GitHub Discussions**: For general questions or ideas
- **GitHub Issues**: For bugs or feature requests
- **Email**: [Contact information, to be added]

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors, irrespective of age, body size, disability, ethnicity, gender identity, experience level, nationality, personal appearance, political belief, race, religion, sexual identity, or sexual orientation.

### Expected Behavior
- Use welcoming and inclusive language
- Be respectful of differing opinions and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community

### Enforcement
Violations may result in removal from the project. Report incidents to [maintainers, to be added].

---

**Thank you for contributing!** 🎉

Your work helps advance uncertainty-aware AI for physiological signal analysis.

---

**Last Updated**: 2026-03-04
