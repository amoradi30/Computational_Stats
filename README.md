# Irregular Multivariate Time Series Missing Value Imputation

A comparative study of imputation methods for irregularly sampled ICU time series data using the PhysioNet 2012 Challenge dataset.

## Project Overview

Missing data is a pervasive challenge in intensive care unit (ICU) settings. This project compares three statistical approaches for imputing missing values in irregular multivariate time series:

1. **Smoothing Splines** - Cubic spline interpolation through observed points
2. **Gaussian Processes** - GP regression with RBF kernel for temporal correlation
3. **Bayesian Imputation (MICE)** - Multiple Imputation by Chained Equations

Each method is evaluated under three masking strategies that simulate different missingness patterns:

| Strategy | Description |
|----------|-------------|
| **MCAR** | Missing Completely At Random - randomly mask n% of observed values |
| **Sequence-End** | Mask the last n% of each variable's observed timeline |
| **Variable-Wise** | Mask entire variables with highest natural missingness |

## Dataset

**PhysioNet/Computing in Cardiology Challenge 2012**
- 8,000 ICU patient records (4,000 training, 4,000 test)
- 48 hours of multivariate physiological measurements
- Up to 37 time-series variables per patient
- Irregular sampling intervals (minutes to hours)

After filtering features with >60% missingness, we retain 6 key variables:
`DiasABP`, `HR`, `MAP`, `SysABP`, `Urine`, `Weight`

## Project Structure

```
├── main.py           # Entry point - configure experiments
├── dataloader.py     # Data loading and train/val/test splits
├── masking.py        # Masking strategies (MCAR, sequence-end, variable-wise)
├── imputation.py     # Imputation methods (splines, GP, MICE)
├── evaluation.py     # Metrics computation (MAE, MSE, RMSE, R²)
├── experiment.py     # Experiment pipeline orchestration
├── data/
│   ├── set-a/        # Training data (4,000 patients)
│   ├── set-b/        # Test data (4,000 patients)
│   └── Outcomes-a.txt
└── ISYE__6416.pdf    # Project proposal
```

## Installation

```bash
# Clone the repository
git clone https://github.com/asalroudbari/Computational_Stats.git
cd Computational_Stats

# Install dependencies
pip install numpy pandas scipy scikit-learn
```

## Usage

### Run a Single Experiment

Edit the configuration in `main.py`:

```python
# Select masking strategy
MASKING_STRATEGY = MaskingStrategy.MCAR
# MASKING_STRATEGY = MaskingStrategy.SEQUENCE_END
# MASKING_STRATEGY = MaskingStrategy.VARIABLE_WISE

# Select imputation method
IMPUTATION_METHOD = ImputationMethod.SMOOTHING_SPLINE
# IMPUTATION_METHOD = ImputationMethod.GAUSSIAN_PROCESS
# IMPUTATION_METHOD = ImputationMethod.MICE

# Set mask ratio
MASK_RATIO = 0.2  # Mask 20% of observed values
```

Then run:

```bash
python main.py
```

### Run Full Comparison

To compare all masking strategies and imputation methods:

```python
# In main.py, uncomment:
run_full_comparison()
```

This runs all 9 combinations (3 masking strategies × 3 imputation methods).

## Methodology

### Data Pipeline

1. **Load Data**: Patient records loaded from PhysioNet files
2. **Feature Filtering**: Drop variables with >60% missing (based on training set)
3. **Split Data**: Patient-wise split to avoid data leakage
   - Train: 3,200 patients (80% of set-a)
   - Validation: 800 patients (20% of set-a)
   - Test: 4,000 patients (set-b)

### Evaluation Pipeline

1. **Masking**: Artificially mask observed values using selected strategy
2. **Imputation**: Apply imputation method to fill masked values
3. **Evaluation**: Compute metrics only on artificially masked positions

### Metrics

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination

## Example Output

```
============================================================
IMPUTATION EVALUATION RESULTS
============================================================

--- Overall Metrics ---
  MAE:  12.3456
  MSE:  234.5678
  RMSE: 15.3155
  R²:   0.8765
  N:    45678

--- Per-Variable Metrics ---
  Variable             MAE       RMSE         R²        N
  -------------------------------------------------------
  DiasABP           8.1234    12.3456     0.9123     7890
  HR                5.6789     8.9012     0.9456     9012
  ...
```

## Authors

- Asal Roudbari (asal@gatech.edu)
- Alireza Moradi (alirezamoradi@gatech.edu)

Georgia Institute of Technology - ISYE 6416 Computational Statistics

## References

1. Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet
2. Wang (2011). Smoothing Splines: Methods and Applications
3. Schulz et al. (2018). A Tutorial on Gaussian Process Regression
4. White et al. (2011). Multiple Imputation Using Chained Equations

## License

This project uses the PhysioNet 2012 Challenge dataset, available under the Open Data Commons Attribution License v1.0.
