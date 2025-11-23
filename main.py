"""
Irregular Multivariate Time Series Missing Value Imputation
PhysioNet 2012 Challenge Dataset - ICU Patient Data

This project compares three imputation methods:
1. Smoothing Splines
2. Gaussian Processes
3. Bayesian Imputation (MICE)

Under three masking strategies:
1. Missing Completely At Random (MCAR)
2. Sequence-end masking
3. Variable-wise masking
"""

from pathlib import Path

from dataloader import load_physionet_data, print_data_summary
from masking import MaskingStrategy
from imputation import ImputationMethod
from experiment import (
    ExperimentConfig,
    run_experiment,
    run_experiment_grid,
    print_results_table,
    summarize_results
)


def main():
    DATA_DIR = Path(__file__).parent / "data"

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    # Data loading parameters
    VAL_RATIO = 0.2                 # 20% of training data for validation
    MISSINGNESS_THRESHOLD = 0.6    # Drop features with >60% missing
    SEED = 42

    # Experiment parameters
    MASK_RATIO = 0.2               # Mask 20% of observed values

    # Select masking strategy (choose one)
    MASKING_STRATEGY = MaskingStrategy.MCAR
    # MASKING_STRATEGY = MaskingStrategy.SEQUENCE_END
    # MASKING_STRATEGY = MaskingStrategy.VARIABLE_WISE

    # Select imputation method (choose one)
    IMPUTATION_METHOD = ImputationMethod.SMOOTHING_SPLINE
    # IMPUTATION_METHOD = ImputationMethod.GAUSSIAN_PROCESS
    # IMPUTATION_METHOD = ImputationMethod.MICE

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    print("=" * 60)
    print("IRREGULAR TIME SERIES IMPUTATION - PhysioNet 2012")
    print("=" * 60)

    data = load_physionet_data(
        data_dir=DATA_DIR,
        val_ratio=VAL_RATIO,
        missingness_threshold=MISSINGNESS_THRESHOLD,
        seed=SEED,
        verbose=True
    )

    print_data_summary(data)

    # =========================================================================
    # RUN SINGLE EXPERIMENT
    # =========================================================================

    config = ExperimentConfig(
        masking_strategy=MASKING_STRATEGY,
        imputation_method=IMPUTATION_METHOD,
        mask_ratio=MASK_RATIO,
        seed=SEED
    )

    # Evaluate on validation set
    result = run_experiment(
        data=data,
        config=config,
        evaluate_on=['val'],
        verbose=True
    )

    # Print detailed results
    print("\n" + summarize_results(result.val_metrics))


def run_full_comparison():
    """
    Run a full comparison across all masking strategies and imputation methods.
    """
    DATA_DIR = Path(__file__).parent / "data"

    print("=" * 60)
    print("FULL COMPARISON - All Strategies & Methods")
    print("=" * 60)

    data = load_physionet_data(
        data_dir=DATA_DIR,
        val_ratio=0.2,
        missingness_threshold=0.6,
        seed=42,
        verbose=True
    )

    print_data_summary(data)

    # Run grid of experiments
    results = run_experiment_grid(
        data=data,
        masking_strategies=[
            MaskingStrategy.MCAR,
            MaskingStrategy.SEQUENCE_END,
            MaskingStrategy.VARIABLE_WISE
        ],
        imputation_methods=[
            ImputationMethod.SMOOTHING_SPLINE,
            ImputationMethod.GAUSSIAN_PROCESS,
            ImputationMethod.MICE
        ],
        mask_ratios=[0.2],
        evaluate_on=['val'],
        seed=42,
        verbose=True
    )

    # Print summary table
    print_results_table(results, split='val')

    return results


if __name__ == "__main__":
    # Run single experiment with configured parameters
    main()

    # Uncomment below to run full comparison:
    # run_full_comparison()
