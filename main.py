"""
Irregular Multivariate Time Series Missing Value Imputation
PhysioNet 2012 Challenge Dataset - ICU Patient Data

This project compares three imputation methods:
1. Smoothing Splines
2. Gaussian Processes
3. Bayesian Imputation (MICE)
"""

from pathlib import Path
from dataloader import load_physionet_data, print_data_summary, pivot_timeseries


def main():
    DATA_DIR = Path(__file__).parent / "data"

    # Load data with train-val-test splits and feature filtering
    data = load_physionet_data(
        data_dir=DATA_DIR,
        val_ratio=0.2,
        missingness_threshold=0.6,
        seed=42,
        verbose=True
    )

    # Print summary
    print_data_summary(data)

    # Example: access a sample patient from training set
    sample_id = data.train.patient_ids[0]
    sample_ts = pivot_timeseries(data.train.timeseries[sample_id])

    print(f"\n{'=' * 60}")
    print(f"Sample Patient (RecordID: {sample_id})")
    print("=" * 60)
    print(f"Time points: {len(sample_ts)}")
    print(f"Variables: {list(sample_ts.columns)}")
    print(f"\nFirst 10 measurements:\n{sample_ts.head(10)}")


if __name__ == "__main__":
    main()
