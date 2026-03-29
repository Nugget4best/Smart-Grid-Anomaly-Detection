# Smart Power Grid Anomaly Detection using Multi-Paradigm Machine Learning

A comprehensive study comparing unsupervised, supervised, and deep learning approaches for anomaly detection in electrical power grid systems. Evaluated on both simulated grid stability data and synthetic power plant operational data with realistic fault modes.

## Overview

Power grid reliability is critical infrastructure. This project develops and evaluates anomaly detection systems that can identify grid instability and power plant faults using machine learning:

- **Three detection paradigms:** unsupervised (no labels needed), supervised (labeled data), and deep learning (autoencoders)
- **Nine models compared:** Isolation Forest, LOF, One-Class SVM, Random Forest, XGBoost, Gradient Boosting, Voting Ensemble, Dense Autoencoder, LSTM Autoencoder
- **Domain-specific feature engineering** based on electrical engineering knowledge (power factor deviation, thermal efficiency, voltage sag/swell indicators)
- **Two datasets:** UCI Electrical Grid Stability (10K samples) + Synthetic Power Plant with 5 fault modes (5K samples)

## Methodology

```
Data Acquisition
├── UCI Electrical Grid Stability (10,000 samples, 12 features)
└── Synthetic Power Plant (5,000 samples, 13 features, 5 fault modes)
         |
   Exploratory Data Analysis
   - Statistical hypothesis testing (Mann-Whitney U, KS test)
   - Effect size analysis (Cohen's d)
   - Parallel coordinates fault visualization
         |
   Feature Engineering
   - Power quality indicators (PF deviation, voltage sag, freq excursion)
   - Interaction features (cross-variable products/ratios)
   - Rolling statistics (mean, std, range, skewness)
   - Rate-of-change derivatives (1st and 2nd order)
   - 16 → 90+ features
         |
   Model Training
   ├── Unsupervised: Isolation Forest, LOF, One-Class SVM
   ├── Supervised: RF, XGBoost, Gradient Boosting, Voting Ensemble
   └── Deep Learning: Dense Autoencoder, LSTM Autoencoder
         |
   Evaluation
   - Precision, Recall, F1, MCC, ROC-AUC, Average Precision
   - Confusion matrices, ROC/PR curves, radar comparison
   - Cross-paradigm analysis
```

## Project Structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   └── README.md                      # Dataset download instructions
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Load UCI + generate synthetic data
│   ├── 02_eda.ipynb                   # Statistical analysis, hypothesis testing
│   ├── 03_feature_engineering.ipynb   # Domain-specific feature construction
│   ├── 04_unsupervised.ipynb          # IF, LOF, OCSVM
│   ├── 05_supervised.ipynb            # RF, XGBoost, GB, Voting
│   ├── 06_autoencoder.ipynb           # Dense AE, LSTM AE
│   └── 07_evaluation.ipynb            # Cross-method comparison
├── src/
│   ├── data_loader.py                 # UCI loading + synthetic generation
│   ├── feature_engineering.py         # Rolling stats, power quality, interactions
│   ├── models.py                      # Unsupervised + supervised models
│   ├── autoencoder.py                 # Dense AE + LSTM AE architecture
│   └── evaluation.py                  # Metrics, ROC/PR curves, visualization
└── results/
    ├── figures/                        # Generated plots
    └── models/                         # Saved model artifacts
```

## Installation

```bash
git clone https://github.com/Nugget4best/Smart-Grid-Anomaly-Detection.git
cd Smart-Grid-Anomaly-Detection
pip install -r requirements.txt
```

## Usage

1. Download the UCI dataset (see `data/README.md`) — synthetic data is generated automatically
2. Run notebooks in order:

```bash
jupyter notebook notebooks/
```

## Datasets

| Dataset | Samples | Features | Anomaly Rate | Source |
|---------|---------|----------|-------------|--------|
| UCI Electrical Grid Stability | 10,000 | 12 | ~36% (unstable) | UCI ML Repository |
| Synthetic Power Plant | 5,000 | 16 | 8% (5 fault modes) | Generated (deterministic) |

### Power Plant Fault Modes

| Fault | Mechanism | Key Indicators |
|-------|-----------|---------------|
| Overload | Load exceeds rated capacity | High current, low PF, elevated vibration |
| Voltage Sag | Sudden voltage drop | Low voltage, high current, reduced power |
| Frequency Deviation | Grid frequency > ±0.5 Hz | Abnormal frequency, RPM deviation |
| Thermal Runaway | Uncontrolled temperature rise | High exhaust/coolant temp |
| Sensor Drift | Gradual sensor offset | Drifting voltage, oil pressure |

## Models

| Category | Model | Key Property |
|----------|-------|-------------|
| Unsupervised | Isolation Forest | Tree-based isolation scoring |
| Unsupervised | Local Outlier Factor | Density-based neighborhood deviation |
| Unsupervised | One-Class SVM | Kernel boundary around normal data |
| Supervised | Random Forest | Bagging with class-weight balancing |
| Supervised | XGBoost | Gradient boosting with scale_pos_weight |
| Supervised | Gradient Boosting | Alternative boosting baseline |
| Supervised | Voting Ensemble | Soft voting (RF + XGBoost + GB) |
| Deep Learning | Dense Autoencoder | Reconstruction error detection |
| Deep Learning | LSTM Autoencoder | Temporal sequence reconstruction |

## Key Technologies

- Python 3.11+
- scikit-learn, XGBoost
- TensorFlow / Keras (LSTM Autoencoder)
- pandas, NumPy, SciPy
- matplotlib, seaborn, plotly
- Jupyter Notebook

## References

- Arzamasov, V., Bohm, K., & Jochem, P. (2018). "Towards Concise Models of Grid Stability." IEEE PES ISGT-Europe.
- Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). "Isolation Forest." ICDM.
- Malhotra, P. et al. (2016). "LSTM-based Encoder-Decoder for Multi-Sensor Anomaly Detection." ICML Workshop.
- Breunig, M.M. et al. (2000). "LOF: Identifying Density-Based Local Outliers." SIGMOD.

## Author

**Afis Adedayo Akande**
- B.Tech Electrical & Electronics Engineering, LAUTECH
- Member, IEEE | Graduate Member, NSE

## License

MIT License — see [LICENSE](LICENSE) for details.
