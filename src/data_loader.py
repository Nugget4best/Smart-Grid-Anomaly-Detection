"""
Data loading for UCI Electrical Grid Stability dataset and synthetic power plant data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# UCI Electrical Grid Stability feature descriptions
UCI_FEATURE_DESCRIPTIONS = {
    'tau1': 'Reaction time of participant 1 (producer)',
    'tau2': 'Reaction time of participant 2 (consumer)',
    'tau3': 'Reaction time of participant 3 (consumer)',
    'tau4': 'Reaction time of participant 4 (consumer)',
    'p1': 'Nominal power of participant 1 (producer: positive)',
    'p2': 'Nominal power consumed by participant 2 (negative)',
    'p3': 'Nominal power consumed by participant 3 (negative)',
    'p4': 'Nominal power consumed by participant 4 (negative)',
    'g1': 'Price elasticity coefficient of participant 1',
    'g2': 'Price elasticity coefficient of participant 2',
    'g3': 'Price elasticity coefficient of participant 3',
    'g4': 'Price elasticity coefficient of participant 4',
    'stab': 'Maximum real part of characteristic equation root (continuous target)',
    'stabf': 'Stability label: stable / unstable (categorical target)',
}


def load_uci_grid_stability(filepath):
    """
    Load the UCI Electrical Grid Stability Simulated Dataset.

    The dataset contains 10,000 observations of a 4-node star grid topology.
    Each node represents either a power producer or consumer.

    Features capture reaction times (tau), power production/consumption (p),
    and price elasticity coefficients (g).

    Targets:
        - stab: continuous stability measure (max real part of eigenvalue)
        - stabf: binary stability label ('stable' / 'unstable')

    Reference:
        Arzamasov, V., Bohm, K., & Jochem, P. (2018).
        "Towards Concise Models of Grid Stability."
        IEEE PES Innovative Smart Grid Technologies Conference Europe.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded UCI Grid Stability: {df.shape}")
    print(f"Stability distribution: {df['stabf'].value_counts().to_dict()}")
    return df


def generate_power_plant_data(n_samples=5000, anomaly_ratio=0.08, random_state=42):
    """
    Generate synthetic power plant operational data based on realistic parameters.

    Simulates a gas-turbine power plant with parameters commonly monitored in
    industrial settings: voltage, current, power factor, load, frequency,
    exhaust temperature, and ambient conditions.

    Normal operation follows multivariate distributions calibrated to real-world
    ranges. Anomalies are injected via 5 distinct fault modes:
        1. Overload: excessive load with power factor degradation
        2. Voltage sag: sudden voltage drop with current spike
        3. Frequency deviation: grid frequency excursion beyond ±0.5 Hz
        4. Thermal runaway: elevated exhaust temperature with rising trend
        5. Sensor drift: gradual sensor offset accumulation

    Args:
        n_samples: total number of samples
        anomaly_ratio: fraction of samples that are anomalous
        random_state: random seed for reproducibility

    Returns:
        DataFrame with operational parameters and anomaly labels
    """
    rng = np.random.RandomState(random_state)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies

    # Normal operating conditions (realistic power plant ranges)
    normal_data = {
        'voltage_kv': rng.normal(11.0, 0.3, n_normal),        # 11kV distribution
        'current_a': rng.normal(120, 15, n_normal),            # Amperes
        'power_factor': rng.normal(0.92, 0.03, n_normal),      # Lagging PF
        'load_mw': rng.normal(1.5, 0.2, n_normal),            # Megawatts
        'frequency_hz': rng.normal(50.0, 0.05, n_normal),      # Grid frequency
        'exhaust_temp_c': rng.normal(450, 25, n_normal),       # Exhaust temperature
        'ambient_temp_c': rng.normal(30, 5, n_normal),         # Ambient
        'vibration_mm_s': rng.normal(2.5, 0.5, n_normal),     # Vibration
        'oil_pressure_bar': rng.normal(4.5, 0.3, n_normal),   # Oil pressure
        'coolant_temp_c': rng.normal(85, 5, n_normal),         # Coolant
        'rpm': rng.normal(3000, 30, n_normal),                 # Rotational speed
        'active_power_mw': rng.normal(1.4, 0.2, n_normal),    # Active power
        'reactive_power_mvar': rng.normal(0.5, 0.1, n_normal), # Reactive power
    }
    normal_df = pd.DataFrame(normal_data)
    normal_df['anomaly'] = 0
    normal_df['fault_type'] = 'normal'

    # Generate anomalies (5 fault modes)
    fault_types = ['overload', 'voltage_sag', 'freq_deviation',
                   'thermal_runaway', 'sensor_drift']
    n_per_fault = n_anomalies // len(fault_types)
    anomaly_dfs = []

    for fault in fault_types:
        n_f = n_per_fault if fault != fault_types[-1] else n_anomalies - n_per_fault * 4
        anom = {
            'voltage_kv': rng.normal(11.0, 0.3, n_f),
            'current_a': rng.normal(120, 15, n_f),
            'power_factor': rng.normal(0.92, 0.03, n_f),
            'load_mw': rng.normal(1.5, 0.2, n_f),
            'frequency_hz': rng.normal(50.0, 0.05, n_f),
            'exhaust_temp_c': rng.normal(450, 25, n_f),
            'ambient_temp_c': rng.normal(30, 5, n_f),
            'vibration_mm_s': rng.normal(2.5, 0.5, n_f),
            'oil_pressure_bar': rng.normal(4.5, 0.3, n_f),
            'coolant_temp_c': rng.normal(85, 5, n_f),
            'rpm': rng.normal(3000, 30, n_f),
            'active_power_mw': rng.normal(1.4, 0.2, n_f),
            'reactive_power_mvar': rng.normal(0.5, 0.1, n_f),
        }

        if fault == 'overload':
            anom['load_mw'] = rng.normal(2.3, 0.15, n_f)
            anom['current_a'] = rng.normal(180, 20, n_f)
            anom['power_factor'] = rng.normal(0.78, 0.05, n_f)
            anom['vibration_mm_s'] = rng.normal(5.0, 1.0, n_f)
        elif fault == 'voltage_sag':
            anom['voltage_kv'] = rng.normal(8.5, 0.5, n_f)
            anom['current_a'] = rng.normal(160, 25, n_f)
            anom['active_power_mw'] = rng.normal(0.9, 0.15, n_f)
        elif fault == 'freq_deviation':
            anom['frequency_hz'] = rng.choice(
                [rng.normal(49.2, 0.15, n_f), rng.normal(50.8, 0.15, n_f)]
            )
            anom['rpm'] = rng.normal(2920, 50, n_f)
        elif fault == 'thermal_runaway':
            anom['exhaust_temp_c'] = rng.normal(580, 30, n_f)
            anom['coolant_temp_c'] = rng.normal(105, 8, n_f)
            anom['vibration_mm_s'] = rng.normal(4.5, 0.8, n_f)
        elif fault == 'sensor_drift':
            drift = np.linspace(0, 3.0, n_f) + rng.normal(0, 0.3, n_f)
            anom['voltage_kv'] = rng.normal(11.0, 0.3, n_f) + drift
            anom['oil_pressure_bar'] = rng.normal(4.5, 0.3, n_f) - drift * 0.5

        anom_df = pd.DataFrame(anom)
        anom_df['anomaly'] = 1
        anom_df['fault_type'] = fault
        anomaly_dfs.append(anom_df)

    # Combine and shuffle
    df = pd.concat([normal_df] + anomaly_dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Add derived features
    df['apparent_power_mva'] = np.sqrt(df['active_power_mw']**2 + df['reactive_power_mvar']**2)
    df['load_ratio'] = df['load_mw'] / 2.0  # ratio to rated capacity (2 MW)
    df['temp_differential'] = df['exhaust_temp_c'] - df['ambient_temp_c']

    print(f"Generated synthetic power plant data: {df.shape}")
    print(f"Anomaly distribution: {df['anomaly'].value_counts().to_dict()}")
    print(f"Fault types: {df['fault_type'].value_counts().to_dict()}")
    return df


def create_combined_dataset(uci_df, plant_df):
    """
    Create a unified analysis-ready dataset from both sources.

    Standardizes column naming and adds source labels for comparative analysis.
    """
    uci_df = uci_df.copy()
    uci_df['source'] = 'uci_grid'
    uci_df['anomaly'] = (uci_df['stabf'] == 'unstable').astype(int)

    plant_df = plant_df.copy()
    plant_df['source'] = 'power_plant'

    return uci_df, plant_df


def split_data(df, feature_cols, target_col='anomaly', test_size=0.2, random_state=42):
    """Split into train/test with stratification."""
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train anomaly rate: {y_train.mean():.3f}, Test anomaly rate: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test
