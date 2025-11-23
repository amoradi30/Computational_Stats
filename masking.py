"""
Masking Strategies for Missing Value Imputation Evaluation

Implements three masking strategies to artificially create missing values
from observed data for evaluating imputation methods.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Dict, List
from dataclasses import dataclass


class MaskingStrategy(Enum):
    """Available masking strategies."""
    MCAR = "mcar"                    # Missing Completely At Random
    SEQUENCE_END = "sequence_end"    # Mask end of sequences
    VARIABLE_WISE = "variable_wise"  # Mask high-missingness variables


@dataclass
class MaskedData:
    """Container for masked data with tracking information."""
    data: pd.DataFrame          # Data with artificial missing values (NaN)
    mask: pd.DataFrame          # Boolean mask: True = artificially masked
    original: pd.DataFrame      # Original data before masking

    @property
    def n_masked(self) -> int:
        """Number of artificially masked values."""
        return self.mask.sum().sum()

    @property
    def n_observed_original(self) -> int:
        """Number of observed values in original data."""
        return self.original.notna().sum().sum()

    @property
    def mask_ratio_actual(self) -> float:
        """Actual ratio of masked values to original observed values."""
        if self.n_observed_original == 0:
            return 0.0
        return self.n_masked / self.n_observed_original


def mcar_mask(data: pd.DataFrame, mask_ratio: float = 0.2,
              seed: int = 42) -> MaskedData:
    """
    Missing Completely At Random (MCAR) masking.

    Randomly masks mask_ratio proportion of observed values.

    Args:
        data: DataFrame with time as index, variables as columns
        mask_ratio: Proportion of observed values to mask (0-1)
        seed: Random seed for reproducibility

    Returns:
        MaskedData with masked data, mask indicator, and original
    """
    rng = np.random.default_rng(seed)

    # Identify observed (non-NaN) positions
    observed_mask = data.notna()
    observed_indices = np.argwhere(observed_mask.values)

    if len(observed_indices) == 0:
        # No observed values to mask
        return MaskedData(
            data=data.copy(),
            mask=pd.DataFrame(False, index=data.index, columns=data.columns),
            original=data.copy()
        )

    # Randomly select positions to mask
    n_to_mask = int(len(observed_indices) * mask_ratio)
    mask_idx = rng.choice(len(observed_indices), size=n_to_mask, replace=False)
    positions_to_mask = observed_indices[mask_idx]

    # Create mask DataFrame
    mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    for row, col in positions_to_mask:
        mask.iloc[row, col] = True

    # Apply mask to data
    masked_data = data.copy()
    masked_data[mask] = np.nan

    return MaskedData(data=masked_data, mask=mask, original=data.copy())


def sequence_end_mask(data: pd.DataFrame, mask_ratio: float = 0.2,
                      seed: int = 42) -> MaskedData:
    """
    Sequence-end masking strategy.

    Masks the last mask_ratio proportion of observed time points for each variable.
    This simulates forecasting scenarios where recent observations are missing.

    Args:
        data: DataFrame with time as index, variables as columns
        mask_ratio: Proportion of sequence end to mask (0-1)
        seed: Random seed (unused but kept for API consistency)

    Returns:
        MaskedData with masked data, mask indicator, and original
    """
    mask = pd.DataFrame(False, index=data.index, columns=data.columns)

    for col in data.columns:
        # Get indices where this variable is observed
        observed_idx = data[col].dropna().index

        if len(observed_idx) == 0:
            continue

        # Calculate number of time points to mask from the end
        n_to_mask = max(1, int(len(observed_idx) * mask_ratio))

        # Mask the last n_to_mask observed time points
        indices_to_mask = observed_idx[-n_to_mask:]
        mask.loc[indices_to_mask, col] = True

    # Apply mask to data
    masked_data = data.copy()
    masked_data[mask] = np.nan

    return MaskedData(data=masked_data, mask=mask, original=data.copy())


def variable_wise_mask(data: pd.DataFrame, mask_ratio: float = 0.2,
                       seed: int = 42) -> MaskedData:
    """
    Variable-wise masking strategy.

    Masks entire variables (columns) starting from those with highest natural
    missingness until approximately mask_ratio of total observed values are masked.

    Args:
        data: DataFrame with time as index, variables as columns
        mask_ratio: Target proportion of observed values to mask (0-1)
        seed: Random seed for tie-breaking

    Returns:
        MaskedData with masked data, mask indicator, and original
    """
    rng = np.random.default_rng(seed)

    # Calculate missingness rate for each variable
    n_rows = len(data)
    var_missingness = {}
    var_observed_count = {}

    for col in data.columns:
        n_observed = data[col].notna().sum()
        var_observed_count[col] = n_observed
        var_missingness[col] = 1 - (n_observed / n_rows) if n_rows > 0 else 1.0

    # Sort variables by missingness (highest first), with random tie-breaking
    variables = list(data.columns)
    rng.shuffle(variables)  # Shuffle first for random tie-breaking
    variables_sorted = sorted(variables, key=lambda x: var_missingness[x], reverse=True)

    # Calculate total observed values and target
    total_observed = sum(var_observed_count.values())
    target_masked = int(total_observed * mask_ratio)

    # Select variables to mask
    mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    masked_count = 0

    for var in variables_sorted:
        if masked_count >= target_masked:
            break

        # Mask all observed values in this variable
        observed_positions = data[var].notna()
        mask.loc[observed_positions, var] = True
        masked_count += var_observed_count[var]

    # Apply mask to data
    masked_data = data.copy()
    masked_data[mask] = np.nan

    return MaskedData(data=masked_data, mask=mask, original=data.copy())


def apply_masking(data: pd.DataFrame, strategy: MaskingStrategy,
                  mask_ratio: float = 0.2, seed: int = 42) -> MaskedData:
    """
    Apply a masking strategy to data.

    Args:
        data: DataFrame with time as index, variables as columns
        strategy: MaskingStrategy enum value
        mask_ratio: Proportion to mask (interpretation depends on strategy)
        seed: Random seed

    Returns:
        MaskedData object
    """
    if strategy == MaskingStrategy.MCAR:
        return mcar_mask(data, mask_ratio, seed)
    elif strategy == MaskingStrategy.SEQUENCE_END:
        return sequence_end_mask(data, mask_ratio, seed)
    elif strategy == MaskingStrategy.VARIABLE_WISE:
        return variable_wise_mask(data, mask_ratio, seed)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")


def mask_patient_timeseries(timeseries: Dict[int, pd.DataFrame],
                            strategy: MaskingStrategy,
                            mask_ratio: float = 0.2,
                            seed: int = 42,
                            pivot_fn=None) -> Dict[int, MaskedData]:
    """
    Apply masking to all patients' time series.

    Args:
        timeseries: Dict mapping patient_id to time series DataFrame (long format)
        strategy: Masking strategy to apply
        mask_ratio: Proportion to mask
        seed: Base random seed (incremented per patient for variation)
        pivot_fn: Function to convert long to wide format (optional)

    Returns:
        Dict mapping patient_id to MaskedData
    """
    masked_data = {}

    for i, (pid, ts) in enumerate(timeseries.items()):
        # Convert to wide format if needed
        if pivot_fn is not None:
            wide_ts = pivot_fn(ts)
        else:
            wide_ts = ts

        if len(wide_ts) == 0:
            # Empty time series
            masked_data[pid] = MaskedData(
                data=wide_ts,
                mask=pd.DataFrame(False, index=wide_ts.index, columns=wide_ts.columns),
                original=wide_ts
            )
            continue

        # Apply masking with patient-specific seed
        patient_seed = seed + i
        masked_data[pid] = apply_masking(wide_ts, strategy, mask_ratio, patient_seed)

    return masked_data


if __name__ == "__main__":
    # Demo/test
    np.random.seed(42)

    # Create sample data
    times = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    data = pd.DataFrame({
        'HR': [72, 75, np.nan, 80, 78, 76, 74],
        'Temp': [36.5, np.nan, 37.0, 37.2, np.nan, 36.8, 36.9],
        'BP': [120, 118, 122, np.nan, np.nan, 115, 117]
    }, index=times)
    data.index.name = 'time_hours'

    print("Original Data:")
    print(data)
    print(f"\nObserved values: {data.notna().sum().sum()}")

    for strategy in MaskingStrategy:
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy.value}")
        print('='*50)

        result = apply_masking(data, strategy, mask_ratio=0.3, seed=42)
        print(f"\nMasked Data:")
        print(result.data)
        print(f"\nMask (True = artificially masked):")
        print(result.mask)
        print(f"\nMasked {result.n_masked} values ({result.mask_ratio_actual:.1%} of observed)")
