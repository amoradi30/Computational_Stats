"""
Experiment Pipeline for Imputation Evaluation

Orchestrates the full pipeline: masking -> imputation -> evaluation
across different strategies and methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import time

from dataloader import PhysioNetData, DataSplit, pivot_timeseries
from masking import MaskingStrategy, MaskedData, mask_patient_timeseries
from imputation import ImputationMethod, impute_patient_timeseries
from evaluation import PatientMetrics, evaluate_patient_imputation, summarize_results


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    masking_strategy: MaskingStrategy
    imputation_method: ImputationMethod
    mask_ratio: float = 0.2
    seed: int = 42
    # Imputation-specific parameters
    imputation_params: Dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    train_metrics: PatientMetrics
    val_metrics: Optional[PatientMetrics] = None
    test_metrics: Optional[PatientMetrics] = None
    runtime_seconds: float = 0.0


def run_experiment(data: PhysioNetData,
                   config: ExperimentConfig,
                   evaluate_on: List[str] = ['train', 'val'],
                   verbose: bool = True) -> ExperimentResult:
    """
    Run a single experiment with specified masking and imputation.

    Args:
        data: PhysioNetData containing train/val/test splits
        config: Experiment configuration
        evaluate_on: Which splits to evaluate ('train', 'val', 'test')
        verbose: Print progress

    Returns:
        ExperimentResult with metrics for each evaluated split
    """
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running Experiment")
        print(f"  Masking:    {config.masking_strategy.value}")
        print(f"  Imputation: {config.imputation_method.value}")
        print(f"  Mask Ratio: {config.mask_ratio:.0%}")
        print(f"{'='*60}")

    result = ExperimentResult(config=config, train_metrics=None)

    # Process each split
    splits_to_process = []
    if 'train' in evaluate_on:
        splits_to_process.append(('train', data.train))
    if 'val' in evaluate_on:
        splits_to_process.append(('val', data.val))
    if 'test' in evaluate_on:
        splits_to_process.append(('test', data.test))

    for split_name, split_data in splits_to_process:
        if verbose:
            print(f"\nProcessing {split_name} split ({split_data.n_patients} patients)...")

        # Step 1: Apply masking
        if verbose:
            print(f"  Applying {config.masking_strategy.value} masking...")

        masked_data = mask_patient_timeseries(
            timeseries=split_data.timeseries,
            strategy=config.masking_strategy,
            mask_ratio=config.mask_ratio,
            seed=config.seed,
            pivot_fn=pivot_timeseries
        )

        # Count total masked values
        total_masked = sum(m.n_masked for m in masked_data.values())
        if verbose:
            print(f"    Masked {total_masked} values across all patients")

        # Step 2: Apply imputation
        if verbose:
            print(f"  Applying {config.imputation_method.value} imputation...")

        imputed_data = impute_patient_timeseries(
            masked_data=masked_data,
            method=config.imputation_method,
            **config.imputation_params
        )

        # Step 3: Evaluate
        if verbose:
            print(f"  Evaluating imputation quality...")

        metrics = evaluate_patient_imputation(masked_data, imputed_data)

        if verbose:
            print(f"    Overall MAE:  {metrics.overall.mae:.4f}")
            print(f"    Overall RMSE: {metrics.overall.rmse:.4f}")
            print(f"    Overall R²:   {metrics.overall.r2:.4f}")

        # Store metrics
        if split_name == 'train':
            result.train_metrics = metrics
        elif split_name == 'val':
            result.val_metrics = metrics
        elif split_name == 'test':
            result.test_metrics = metrics

    result.runtime_seconds = time.time() - start_time

    if verbose:
        print(f"\nExperiment completed in {result.runtime_seconds:.1f} seconds")

    return result


def run_experiment_grid(data: PhysioNetData,
                        masking_strategies: List[MaskingStrategy],
                        imputation_methods: List[ImputationMethod],
                        mask_ratios: List[float] = [0.2],
                        evaluate_on: List[str] = ['val'],
                        seed: int = 42,
                        verbose: bool = True) -> List[ExperimentResult]:
    """
    Run experiments over a grid of configurations.

    Args:
        data: PhysioNetData
        masking_strategies: List of masking strategies to try
        imputation_methods: List of imputation methods to try
        mask_ratios: List of mask ratios to try
        evaluate_on: Which splits to evaluate
        seed: Random seed
        verbose: Print progress

    Returns:
        List of ExperimentResult for each configuration
    """
    results = []
    total_experiments = len(masking_strategies) * len(imputation_methods) * len(mask_ratios)
    exp_num = 0

    for mask_ratio in mask_ratios:
        for masking in masking_strategies:
            for imputation in imputation_methods:
                exp_num += 1
                if verbose:
                    print(f"\n[{exp_num}/{total_experiments}]", end="")

                config = ExperimentConfig(
                    masking_strategy=masking,
                    imputation_method=imputation,
                    mask_ratio=mask_ratio,
                    seed=seed
                )

                result = run_experiment(data, config, evaluate_on, verbose)
                results.append(result)

    return results


def results_to_dataframe(results: List[ExperimentResult],
                         split: str = 'val') -> pd.DataFrame:
    """
    Convert experiment results to a summary DataFrame.

    Args:
        results: List of ExperimentResult
        split: Which split metrics to include ('train', 'val', 'test')

    Returns:
        DataFrame with one row per experiment
    """
    rows = []

    for r in results:
        # Get metrics for specified split
        if split == 'train':
            metrics = r.train_metrics
        elif split == 'val':
            metrics = r.val_metrics
        elif split == 'test':
            metrics = r.test_metrics
        else:
            continue

        if metrics is None:
            continue

        rows.append({
            'masking_strategy': r.config.masking_strategy.value,
            'imputation_method': r.config.imputation_method.value,
            'mask_ratio': r.config.mask_ratio,
            'mae': metrics.overall.mae,
            'mse': metrics.overall.mse,
            'rmse': metrics.overall.rmse,
            'r2': metrics.overall.r2,
            'n_evaluated': metrics.overall.n_evaluated,
            'runtime_seconds': r.runtime_seconds
        })

    return pd.DataFrame(rows)


def print_results_table(results: List[ExperimentResult],
                        split: str = 'val') -> None:
    """Print a formatted table of results."""
    df = results_to_dataframe(results, split)

    if len(df) == 0:
        print("No results to display.")
        return

    print(f"\n{'='*80}")
    print(f"EXPERIMENT RESULTS SUMMARY ({split} split)")
    print('='*80)
    print(f"\n{'Masking':<18} {'Imputation':<18} {'Ratio':>6} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 80)

    for _, row in df.iterrows():
        r2_str = f"{row['r2']:.4f}" if not np.isnan(row['r2']) else "N/A"
        print(f"{row['masking_strategy']:<18} {row['imputation_method']:<18} "
              f"{row['mask_ratio']:>6.0%} {row['mae']:>10.4f} {row['rmse']:>10.4f} {r2_str:>10}")


if __name__ == "__main__":
    # Demo with a small subset
    from dataloader import load_physionet_data

    DATA_DIR = Path(__file__).parent / "data"

    print("Loading data (small subset for demo)...")
    # For demo, we'll create a minimal test
    data = load_physionet_data(
        data_dir=DATA_DIR,
        val_ratio=0.2,
        missingness_threshold=0.6,
        seed=42,
        verbose=False
    )

    # Limit to 50 patients for quick demo
    demo_train_ids = data.train.patient_ids[:50]
    demo_val_ids = data.val.patient_ids[:20]

    demo_train = DataSplit(
        patient_ids=demo_train_ids,
        timeseries={pid: data.train.timeseries[pid] for pid in demo_train_ids},
        general_info=data.train.general_info[data.train.general_info['RecordID'].isin(demo_train_ids)]
    )
    demo_val = DataSplit(
        patient_ids=demo_val_ids,
        timeseries={pid: data.val.timeseries[pid] for pid in demo_val_ids},
        general_info=data.val.general_info[data.val.general_info['RecordID'].isin(demo_val_ids)]
    )

    demo_data = PhysioNetData(
        train=demo_train,
        val=demo_val,
        test=data.test,  # Keep full test for reference
        features=data.features,
        dropped_features=data.dropped_features,
        missingness_threshold=data.missingness_threshold
    )

    print(f"\nDemo data: {demo_train.n_patients} train, {demo_val.n_patients} val patients")

    # Run a single experiment
    config = ExperimentConfig(
        masking_strategy=MaskingStrategy.MCAR,
        imputation_method=ImputationMethod.SMOOTHING_SPLINE,
        mask_ratio=0.2
    )

    result = run_experiment(demo_data, config, evaluate_on=['val'], verbose=True)

    # Print detailed results
    print("\n" + summarize_results(result.val_metrics))
