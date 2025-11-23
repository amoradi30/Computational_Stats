"""
Imputation Methods for Irregular Time Series

Implements three imputation approaches:
1. Smoothing Splines - Cubic spline interpolation
2. Gaussian Processes - GP regression with RBF kernel
3. MICE - Multiple Imputation by Chained Equations (Bayesian)
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional, Dict
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class ImputationMethod(Enum):
    """Available imputation methods."""
    SMOOTHING_SPLINE = "smoothing_spline"
    GAUSSIAN_PROCESS = "gaussian_process"
    MICE = "mice"


def smoothing_spline_impute(data: pd.DataFrame,
                            smoothing_factor: Optional[float] = None) -> pd.DataFrame:
    """
    Impute missing values using smoothing splines.

    Fits a cubic smoothing spline to observed values for each variable,
    then evaluates at all time points to fill missing values.

    Args:
        data: DataFrame with time as index, variables as columns
        smoothing_factor: Spline smoothing parameter (None = automatic via GCV)

    Returns:
        DataFrame with missing values imputed
    """
    imputed = data.copy()
    times = data.index.values.astype(float)

    for col in data.columns:
        series = data[col]
        observed_mask = series.notna()

        # Need at least 4 points for cubic spline
        if observed_mask.sum() < 4:
            # Fall back to linear interpolation or mean
            if observed_mask.sum() >= 2:
                imputed[col] = series.interpolate(method='linear')
            elif observed_mask.sum() == 1:
                imputed[col] = series.fillna(series.dropna().iloc[0])
            # If no observations, leave as NaN
            continue

        t_obs = times[observed_mask]
        y_obs = series[observed_mask].values

        try:
            # Fit smoothing spline
            if smoothing_factor is None:
                # Use automatic smoothing factor selection
                # Start with a reasonable default based on data variance
                spline = UnivariateSpline(t_obs, y_obs, s=len(t_obs))
            else:
                spline = UnivariateSpline(t_obs, y_obs, s=smoothing_factor)

            # Predict at all time points
            y_pred = spline(times)

            # Only fill missing values
            missing_mask = series.isna()
            imputed.loc[missing_mask, col] = y_pred[missing_mask.values]

        except Exception:
            # Fall back to linear interpolation
            imputed[col] = series.interpolate(method='linear')

    return imputed


def gaussian_process_impute(data: pd.DataFrame,
                            length_scale: float = 1.0,
                            noise_level: float = 0.1) -> pd.DataFrame:
    """
    Impute missing values using Gaussian Process regression.

    Fits a GP with RBF kernel to observed values for each variable,
    then predicts at missing time points.

    Args:
        data: DataFrame with time as index, variables as columns
        length_scale: Initial length scale for RBF kernel
        noise_level: Initial noise level

    Returns:
        DataFrame with missing values imputed
    """
    imputed = data.copy()
    times = data.index.values.astype(float).reshape(-1, 1)

    # Define kernel: constant * RBF + white noise
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
        RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2)) +
        WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-5, 1e1))
    )

    for col in data.columns:
        series = data[col]
        observed_mask = series.notna()
        missing_mask = series.isna()

        if observed_mask.sum() < 2 or missing_mask.sum() == 0:
            # Not enough data or nothing to impute
            if observed_mask.sum() == 1 and missing_mask.sum() > 0:
                imputed[col] = series.fillna(series.dropna().iloc[0])
            continue

        t_obs = times[observed_mask.values]
        y_obs = series[observed_mask].values
        t_missing = times[missing_mask.values]

        try:
            # Normalize y for numerical stability
            y_mean = y_obs.mean()
            y_std = y_obs.std() if y_obs.std() > 0 else 1.0
            y_normalized = (y_obs - y_mean) / y_std

            # Fit GP
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=3,
                normalize_y=False,
                random_state=42
            )
            gp.fit(t_obs, y_normalized)

            # Predict at missing points
            y_pred_normalized = gp.predict(t_missing)
            y_pred = y_pred_normalized * y_std + y_mean

            imputed.loc[missing_mask, col] = y_pred

        except Exception:
            # Fall back to linear interpolation
            imputed[col] = series.interpolate(method='linear')

    return imputed


def mice_impute(data: pd.DataFrame,
                max_iter: int = 10,
                n_nearest_features: Optional[int] = None,
                seed: int = 42) -> pd.DataFrame:
    """
    Impute missing values using MICE (Multiple Imputation by Chained Equations).

    Uses iterative imputation where each feature is modeled as a function
    of other features, iterating until convergence.

    Args:
        data: DataFrame with time as index, variables as columns
        max_iter: Maximum number of imputation iterations
        n_nearest_features: Number of features to use for imputation (None = all)
        seed: Random seed

    Returns:
        DataFrame with missing values imputed
    """
    if data.isna().sum().sum() == 0:
        return data.copy()

    # Check if we have enough data
    if len(data) < 2 or data.shape[1] < 1:
        return data.copy()

    # IterativeImputer (MICE implementation in sklearn)
    imputer = IterativeImputer(
        max_iter=max_iter,
        n_nearest_features=n_nearest_features,
        random_state=seed,
        initial_strategy='mean',
        skip_complete=True
    )

    try:
        # Fit and transform
        imputed_values = imputer.fit_transform(data.values)
        imputed = pd.DataFrame(
            imputed_values,
            index=data.index,
            columns=data.columns
        )
    except Exception:
        # Fall back to simple mean imputation
        imputed = data.fillna(data.mean())

    return imputed


def impute(data: pd.DataFrame, method: ImputationMethod, **kwargs) -> pd.DataFrame:
    """
    Apply an imputation method to data.

    Args:
        data: DataFrame with time as index, variables as columns
        method: ImputationMethod enum value
        **kwargs: Method-specific parameters

    Returns:
        DataFrame with missing values imputed
    """
    if method == ImputationMethod.SMOOTHING_SPLINE:
        return smoothing_spline_impute(data, **kwargs)
    elif method == ImputationMethod.GAUSSIAN_PROCESS:
        return gaussian_process_impute(data, **kwargs)
    elif method == ImputationMethod.MICE:
        return mice_impute(data, **kwargs)
    else:
        raise ValueError(f"Unknown imputation method: {method}")


def impute_patient_timeseries(masked_data: Dict,
                               method: ImputationMethod,
                               **kwargs) -> Dict[int, pd.DataFrame]:
    """
    Apply imputation to all patients' masked time series.

    Args:
        masked_data: Dict mapping patient_id to MaskedData objects
        method: Imputation method to use
        **kwargs: Method-specific parameters

    Returns:
        Dict mapping patient_id to imputed DataFrame
    """
    imputed_data = {}

    for pid, mdata in masked_data.items():
        if hasattr(mdata, 'data'):
            # MaskedData object
            imputed_data[pid] = impute(mdata.data, method, **kwargs)
        else:
            # Plain DataFrame
            imputed_data[pid] = impute(mdata, method, **kwargs)

    return imputed_data


if __name__ == "__main__":
    # Demo/test
    np.random.seed(42)

    # Create sample data with missing values
    times = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    data = pd.DataFrame({
        'HR': [72, 75, np.nan, 80, np.nan, 76, 74, 73],
        'Temp': [36.5, np.nan, 37.0, 37.2, np.nan, 36.8, np.nan, 36.9],
        'BP': [120, 118, np.nan, np.nan, 116, 115, 117, 118]
    }, index=times)
    data.index.name = 'time_hours'

    print("Original Data with Missing Values:")
    print(data)
    print(f"\nMissing values: {data.isna().sum().sum()}")

    for method in ImputationMethod:
        print(f"\n{'='*50}")
        print(f"Method: {method.value}")
        print('='*50)

        imputed = impute(data, method)
        print(f"\nImputed Data:")
        print(imputed.round(2))
