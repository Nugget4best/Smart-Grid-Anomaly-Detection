"""
Feature engineering for power grid anomaly detection.

Implements domain-specific feature extraction including rolling statistics,
rate-of-change indicators, power quality metrics, and statistical moment features.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_rolling_statistics(df, columns, windows=[5, 10, 20]):
    """
    Compute rolling window statistics for time-series-like features.

    For each specified column and window size, computes:
        - Rolling mean (trend indicator)
        - Rolling standard deviation (volatility indicator)
        - Rolling min/max (range indicator)
        - Rolling skewness (distribution asymmetry)

    Args:
        df: DataFrame with numerical features
        columns: list of column names to compute rolling stats for
        windows: list of window sizes

    Returns:
        DataFrame with original and new rolling features
    """
    df = df.copy()
    new_features = []

    for col in columns:
        for w in windows:
            roll = df[col].rolling(window=w, min_periods=1)
            df[f'{col}_roll_mean_{w}'] = roll.mean()
            df[f'{col}_roll_std_{w}'] = roll.std().fillna(0)
            df[f'{col}_roll_min_{w}'] = roll.min()
            df[f'{col}_roll_max_{w}'] = roll.max()
            df[f'{col}_roll_range_{w}'] = df[f'{col}_roll_max_{w}'] - df[f'{col}_roll_min_{w}']
            df[f'{col}_roll_skew_{w}'] = roll.skew().fillna(0)

            new_features.extend([
                f'{col}_roll_mean_{w}', f'{col}_roll_std_{w}',
                f'{col}_roll_min_{w}', f'{col}_roll_max_{w}',
                f'{col}_roll_range_{w}', f'{col}_roll_skew_{w}',
            ])

    print(f"Added {len(new_features)} rolling features ({len(columns)} cols x {len(windows)} windows x 6 stats)")
    return df, new_features


def compute_rate_of_change(df, columns):
    """
    Compute first and second order rate-of-change (derivatives).

    First derivative: instantaneous rate of change
    Second derivative: acceleration / rate of change of rate of change

    These are critical for detecting sudden transient events in power systems.
    """
    df = df.copy()
    new_features = []

    for col in columns:
        df[f'{col}_diff1'] = df[col].diff().fillna(0)
        df[f'{col}_diff2'] = df[f'{col}_diff1'].diff().fillna(0)
        df[f'{col}_pct_change'] = df[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
        new_features.extend([f'{col}_diff1', f'{col}_diff2', f'{col}_pct_change'])

    print(f"Added {len(new_features)} rate-of-change features")
    return df, new_features


def compute_power_quality_features(df):
    """
    Compute domain-specific power quality indicators.

    These features encode electrical engineering domain knowledge:
        - Power factor deviation from unity (ideal PF = 1.0)
        - Load-to-capacity ratio (utilization percentage)
        - Frequency deviation from nominal (50 Hz)
        - Voltage deviation from nominal (11 kV)
        - Thermal efficiency proxy (active power / exhaust temp)
        - Power imbalance (apparent - active)
        - Current-voltage product deviation
    """
    df = df.copy()
    features = []

    if 'power_factor' in df.columns:
        df['pf_deviation'] = np.abs(1.0 - df['power_factor'])
        features.append('pf_deviation')

    if 'load_mw' in df.columns:
        rated_capacity = 2.0  # MW (rated capacity of generator)
        df['utilization_pct'] = df['load_mw'] / rated_capacity
        df['overload_indicator'] = (df['load_mw'] > rated_capacity * 0.95).astype(int)
        features.extend(['utilization_pct', 'overload_indicator'])

    if 'frequency_hz' in df.columns:
        df['freq_deviation'] = np.abs(df['frequency_hz'] - 50.0)
        df['freq_critical'] = (df['freq_deviation'] > 0.5).astype(int)
        features.extend(['freq_deviation', 'freq_critical'])

    if 'voltage_kv' in df.columns:
        df['voltage_deviation'] = np.abs(df['voltage_kv'] - 11.0) / 11.0 * 100
        df['voltage_sag'] = (df['voltage_kv'] < 10.0).astype(int)
        df['voltage_swell'] = (df['voltage_kv'] > 12.0).astype(int)
        features.extend(['voltage_deviation', 'voltage_sag', 'voltage_swell'])

    if 'active_power_mw' in df.columns and 'exhaust_temp_c' in df.columns:
        df['thermal_efficiency'] = df['active_power_mw'] / (df['exhaust_temp_c'] + 273.15)
        features.append('thermal_efficiency')

    if 'apparent_power_mva' in df.columns and 'active_power_mw' in df.columns:
        df['power_imbalance'] = df['apparent_power_mva'] - df['active_power_mw']
        features.append('power_imbalance')

    if 'voltage_kv' in df.columns and 'current_a' in df.columns:
        expected_power = df['voltage_kv'] * df['current_a'] / 1000  # rough MVA
        df['vi_power_ratio'] = df['active_power_mw'] / expected_power.replace(0, np.nan)
        df['vi_power_ratio'] = df['vi_power_ratio'].fillna(0)
        features.append('vi_power_ratio')

    print(f"Added {len(features)} power quality features")
    return df, features


def compute_statistical_features(df, columns):
    """
    Compute higher-order statistical features for each sample's feature vector.

    Captures distributional properties across the feature space:
        - Z-score of each feature (standardized deviation from mean)
        - Mahalanobis-inspired distance features
    """
    df = df.copy()
    features = []

    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[f'{col}_zscore'] = (df[col] - mean) / std
            features.append(f'{col}_zscore')

    print(f"Added {len(features)} statistical features")
    return df, features


def compute_interaction_features(df):
    """
    Compute interaction features between key power system variables.

    Cross-variable interactions often reveal failure modes invisible in
    individual features. For example, voltage sag + current spike together
    indicate a distinct fault pattern from either alone.
    """
    df = df.copy()
    features = []

    interaction_pairs = [
        ('voltage_kv', 'current_a'),
        ('load_mw', 'power_factor'),
        ('exhaust_temp_c', 'vibration_mm_s'),
        ('frequency_hz', 'rpm'),
        ('oil_pressure_bar', 'coolant_temp_c'),
        ('active_power_mw', 'reactive_power_mvar'),
    ]

    for col1, col2 in interaction_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            df[f'{col1}_div_{col2}'] = df[col1] / df[col2].replace(0, np.nan)
            df[f'{col1}_div_{col2}'] = df[f'{col1}_div_{col2}'].fillna(0)
            features.extend([f'{col1}_x_{col2}', f'{col1}_div_{col2}'])

    print(f"Added {len(features)} interaction features")
    return df, features


def feature_engineering_pipeline(df, rolling_cols=None, windows=[5, 10, 20]):
    """
    Run the complete feature engineering pipeline.

    Steps:
        1. Power quality domain features
        2. Interaction features
        3. Rolling statistics (if rolling_cols specified)
        4. Rate of change (if rolling_cols specified)
        5. Statistical z-score features

    Returns:
        Tuple of (engineered DataFrame, list of all new feature names)
    """
    all_new_features = []

    # Domain-specific power quality features
    df, pq_features = compute_power_quality_features(df)
    all_new_features.extend(pq_features)

    # Interaction features
    df, interact_features = compute_interaction_features(df)
    all_new_features.extend(interact_features)

    # Rolling statistics (for time-series ordered data)
    if rolling_cols:
        df, roll_features = compute_rolling_statistics(df, rolling_cols, windows)
        all_new_features.extend(roll_features)

        df, roc_features = compute_rate_of_change(df, rolling_cols)
        all_new_features.extend(roc_features)

    print(f"\nTotal engineered features: {len(all_new_features)}")
    return df, all_new_features
