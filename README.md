# Anomaly Detection in Hybrid Renewable Energy Generation

A data-driven anomaly detection system for identifying combined wind-solar generation droughts in India's renewable energy infrastructure using open meteorological and capacity datasets.

## Overview

India has rapidly expanded its renewable energy capacity with large-scale deployment of both wind and solar power across multiple states. However, multi-day periods of simultaneously low wind speeds and solar irradiance create "renewable energy droughts" that threaten grid stability. This project addresses the critical gap in detecting and explaining **combined** anomalies in hybrid wind-solar generation.

## Problem Statement

Existing research largely focuses on forecasting either wind or solar output in isolation, or on detecting anomalies in electricity consumption rather than generation. There is currently no publicly available system that detects and explains combined anomalies in hybrid wind-solar generation using open meteorological and capacity datasets.

## Key Features

- **Hybrid Analysis**: Combined anomaly detection for wind and solar generation (not isolated)
- **Real-World Data**: Uses publicly available India renewable energy capacity data and ERA5 reanalysis meteorological data
- **Multi-Method Approach**: Implements statistical methods, machine learning, and deep learning models
- **Explainable AI**: Integrates SHAP/LIME for interpretable anomaly explanations
- **Grid Reliability Focus**: Specifically targets anomalies relevant to power grid stability

## Datasets

### Primary Dataset
- **India Renewable Energy Capacity Dataset**: State-level installed capacity data for solar and wind power
- Source: Publicly available CSV format
- Coverage: Multiple Indian states over several years

### Supplementary Dataset
- **ERA5 Reanalysis Meteorological Data**: Global atmospheric reanalysis with hourly climate variables
- Variables: Wind speed (10m, 100m), surface solar radiation, temperature, surface pressure
- Spatial Coverage: Regular grid over Indian regions with wind and solar farms

### Optional
- National/regional generation statistics for calibration and validation

## Methodology

### Phase 1: Data Integration and EDA
- Merge capacity and meteorological data by state/region and time
- Compute normalized generation proxies using capacity factors
- Identify weather patterns correlated with low output periods
- Temporal trends, spatial heatmaps, and correlation analysis

### Phase 2: Baseline Anomaly Detection
- Statistical methods: Z-score, IQR, STL decomposition
- Unsupervised ML: Isolation Forest, Local Outlier Factor (LOF), K-means clustering
- Time series models: ARIMA with residual analysis
- Evaluation: Precision, Recall, F1-score

### Phase 3: Advanced Models and Ensembling
- LSTM-based autoencoder for multivariate reconstruction
- Hybrid CNN-LSTM architecture for spatial-temporal patterns
- Sequential ensemble:
  - Stage 1: Change point detection for long-duration droughts
  - Stage 2: Statistical process control for short anomalies
- Explainability: SHAP and LIME integration

### Phase 4: Validation and Documentation
- Cross-validation across regions
- Sensitivity analysis on hyperparameters
- Case studies on historical events (e.g., monsoon-related low-output periods)
- Reproducible notebooks and modular codebase

## Expected Outcomes

1. Working anomaly detection pipeline for hybrid wind-solar drought events at regional level
2. Quantitative evidence (F1-scores, case studies) demonstrating superiority of combined analysis
3. Interpretable explanations linking anomalies to meteorological drivers

## Deliverables

- Written research report with problem statement, methodology, and results
- Jupyter notebooks for preprocessing, analysis, modeling, and evaluation
- Modular Python codebase for reusability
- Visual artifacts: time series plots, heatmaps, anomaly timelines

## Literature Foundation

Comprehensive review of 20+ peer-reviewed papers (2023-2025) from:
- IEEE Transactions (Power Systems, Sustainable Energy, Industrial Electronics)
- ScienceDirect (Energy, Applied Energy, Renewable Energy, Expert Systems)
- MDPI (Sensors, Sustainability)
- Elsevier (Renewable and Sustainable Energy Reviews)

Key trends: Transformer-based models, ensemble methods, emerging LLM applications for time series

## Project Structure

```
├── da1/
│   ├── main.tex              # LaTeX research document
│   ├── main.pdf              # Compiled report
│   ├── references.bib        # Bibliography
│   ├── literature_review.csv # Survey of 20+ papers
│   └── university_logo.png   # Institution branding
└── README.md                 # This file
```

## Authors

- Sparsh Karna (23BDS1172)
- Lavanaya Malhotra (23BDS1169)

## Acknowledgments

This project addresses a critical challenge in renewable energy grid management, leveraging open data and reproducible methods to support India's transition to sustainable energy infrastructure.

## License

This project is developed for academic research purposes.

---

*For questions or collaboration opportunities, please open an issue on this repository.*
