"""
DataLoader for PhysioNet 2012 Challenge Dataset
Handles loading, preprocessing, train-val-test splitting, and feature filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Time-series variables from PhysioNet 2012
TIME_SERIES_VARIABLES = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
    'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP',
    'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight'
]

GENERAL_DESCRIPTORS = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']


@dataclass
class DataSplit:
    """Container for a data split with patient IDs and time series."""
    patient_ids: List[int]
    timeseries: Dict[int, pd.DataFrame]
    general_info: pd.DataFrame

    @property
    def n_patients(self) -> int:
        return len(self.patient_ids)


@dataclass
class PhysioNetData:
    """Container for train/val/test splits with metadata."""
    train: DataSplit
    val: DataSplit
    test: DataSplit
    features: List[str]
    dropped_features: List[str]
    missingness_threshold: float


def parse_time(time_str: str) -> float:
    """Convert HH:MM to hours as float."""
    h, m = map(int, time_str.split(':'))
    return h + m / 60.0


def load_patient_file(filepath: Path) -> Tuple[Dict, pd.DataFrame]:
    """Load single patient file, return general info and time series."""
    df = pd.read_csv(filepath)
    general_info = {}
    ts_rows = []

    for _, row in df.iterrows():
        t = parse_time(row['Time'])
        param, val = row['Parameter'], row['Value']

        if t == 0 and param in GENERAL_DESCRIPTORS:
            general_info[param] = val
        else:
            ts_rows.append({'time_hours': t, 'variable': param, 'value': val})

    return general_info, pd.DataFrame(ts_rows)


def load_raw_dataset(data_dir: Path, dataset_name: str,
                     verbose: bool = True) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Load all patient files from a dataset directory."""
    dataset_path = data_dir / dataset_name
    files = sorted(dataset_path.glob('*.txt'))

    if verbose:
        print(f"Loading {len(files)} patients from {dataset_name}...")

    general_list = []
    timeseries = {}

    for i, f in enumerate(files):
        info, ts = load_patient_file(f)
        rid = int(info.get('RecordID', f.stem))
        general_list.append(info)
        timeseries[rid] = ts

        if verbose and (i + 1) % 1000 == 0:
            print(f"  Loaded {i + 1} patients...")

    if verbose:
        print(f"  Done. {len(files)} patients loaded.")

    return pd.DataFrame(general_list), timeseries


def pivot_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format time series to wide format, replacing -1 with NaN."""
    if len(ts) == 0:
        return pd.DataFrame()

    pivoted = ts.pivot_table(
        index='time_hours', columns='variable', values='value', aggfunc='mean'
    )
    return pivoted.replace(-1, np.nan)


def compute_feature_missingness(timeseries: Dict[int, pd.DataFrame],
                                 features: List[str]) -> Dict[str, float]:
    """Compute missing rate for each feature across all patients."""
    counts = {f: {'obs': 0, 'total': 0} for f in features}

    for ts in timeseries.values():
        if len(ts) == 0:
            continue
        pivoted = pivot_timeseries(ts)
        n_rows = len(pivoted)

        for f in features:
            if f in pivoted.columns:
                counts[f]['obs'] += pivoted[f].notna().sum()
            counts[f]['total'] += n_rows

    return {
        f: 1 - (c['obs'] / c['total']) if c['total'] > 0 else 1.0
        for f, c in counts.items()
    }


def filter_features_by_missingness(timeseries: Dict[int, pd.DataFrame],
                                    threshold: float = 0.6) -> Tuple[List[str], List[str]]:
    """
    Filter features based on missingness threshold.

    Returns:
        kept_features: Features with missing rate <= threshold
        dropped_features: Features with missing rate > threshold
    """
    missingness = compute_feature_missingness(timeseries, TIME_SERIES_VARIABLES)

    kept = [f for f, rate in missingness.items() if rate <= threshold]
    dropped = [f for f, rate in missingness.items() if rate > threshold]

    return sorted(kept), sorted(dropped)


def filter_patient_timeseries(timeseries: Dict[int, pd.DataFrame],
                               features: List[str]) -> Dict[int, pd.DataFrame]:
    """Filter time series to only include specified features."""
    filtered = {}
    for pid, ts in timeseries.items():
        if len(ts) == 0:
            filtered[pid] = ts
        else:
            filtered[pid] = ts[ts['variable'].isin(features)]
    return filtered


def train_val_split(patient_ids: List[int],
                    val_ratio: float = 0.2,
                    seed: int = 42) -> Tuple[List[int], List[int]]:
    """Split patient IDs into train and validation sets."""
    rng = np.random.default_rng(seed)
    ids = np.array(patient_ids)
    rng.shuffle(ids)

    n_val = int(len(ids) * val_ratio)
    return ids[n_val:].tolist(), ids[:n_val].tolist()


def subset_data(patient_ids: List[int],
                timeseries: Dict[int, pd.DataFrame],
                general_info: pd.DataFrame) -> DataSplit:
    """Create a DataSplit for a subset of patients."""
    ts_subset = {pid: timeseries[pid] for pid in patient_ids}
    info_subset = general_info[general_info['RecordID'].isin(patient_ids)].reset_index(drop=True)
    return DataSplit(patient_ids=patient_ids, timeseries=ts_subset, general_info=info_subset)


def load_physionet_data(data_dir: str = "data",
                        val_ratio: float = 0.2,
                        missingness_threshold: float = 0.6,
                        seed: int = 42,
                        verbose: bool = True) -> PhysioNetData:
    """
    Load PhysioNet 2012 data with train-val-test splits and feature filtering.

    Args:
        data_dir: Path to data directory
        val_ratio: Fraction of training data for validation
        missingness_threshold: Drop features with missing rate > threshold
        seed: Random seed for train-val split
        verbose: Print progress

    Returns:
        PhysioNetData with train, val, test splits and feature metadata
    """
    data_path = Path(data_dir)

    # Load training data (set-a)
    if verbose:
        print("=" * 60)
        print("Loading PhysioNet 2012 Dataset")
        print("=" * 60)

    train_info, train_ts = load_raw_dataset(data_path, 'set-a', verbose)

    # Compute feature missingness on training data and filter
    if verbose:
        print(f"\nFiltering features (missingness threshold: {missingness_threshold:.0%})...")

    kept_features, dropped_features = filter_features_by_missingness(
        train_ts, missingness_threshold
    )

    if verbose:
        print(f"  Kept {len(kept_features)} features, dropped {len(dropped_features)} features")

    # Filter training time series
    train_ts_filtered = filter_patient_timeseries(train_ts, kept_features)

    # Split training into train/val
    all_train_ids = list(train_ts_filtered.keys())
    train_ids, val_ids = train_val_split(all_train_ids, val_ratio, seed)

    train_split = subset_data(train_ids, train_ts_filtered, train_info)
    val_split = subset_data(val_ids, train_ts_filtered, train_info)

    # Load test data (set-b)
    test_info, test_ts = load_raw_dataset(data_path, 'set-b', verbose)
    test_ts_filtered = filter_patient_timeseries(test_ts, kept_features)
    test_ids = list(test_ts_filtered.keys())
    test_split = subset_data(test_ids, test_ts_filtered, test_info)

    return PhysioNetData(
        train=train_split,
        val=val_split,
        test=test_split,
        features=kept_features,
        dropped_features=dropped_features,
        missingness_threshold=missingness_threshold
    )


def print_data_summary(data: PhysioNetData) -> None:
    """Print summary of loaded data."""
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"\n{'Split':<12} {'Patients':>10}")
    print("-" * 24)
    print(f"{'Train':<12} {data.train.n_patients:>10}")
    print(f"{'Validation':<12} {data.val.n_patients:>10}")
    print(f"{'Test':<12} {data.test.n_patients:>10}")
    print("-" * 24)
    print(f"{'Total':<12} {data.train.n_patients + data.val.n_patients + data.test.n_patients:>10}")

    print(f"\n{'Features Summary':}")
    print("-" * 40)
    print(f"Missingness threshold: {data.missingness_threshold:.0%}")
    print(f"Kept features ({len(data.features)}): {data.features}")
    print(f"\nDropped features ({len(data.dropped_features)}): {data.dropped_features}")


if __name__ == "__main__":
    # Load data with default parameters
    DATA_DIR = Path(__file__).parent / "data"

    data = load_physionet_data(
        data_dir=DATA_DIR,
        val_ratio=0.2,
        missingness_threshold=0.6,
        seed=42,
        verbose=True
    )

    print_data_summary(data)
