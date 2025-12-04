"""
Forecasting Test Data and Fixtures.

Provides historical traffic data, time series datasets, and synthetic data for testing.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime, timedelta


@pytest.fixture
def historical_traffic_data() -> pd.DataFrame:
    """Generate historical traffic data for forecasting tests."""
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
    
    data = {
        'timestamp': dates,
        'lane_0_volume': np.random.poisson(50, 1000) + np.sin(np.arange(1000) * 2 * np.pi / 24) * 20,
        'lane_1_volume': np.random.poisson(45, 1000) + np.sin(np.arange(1000) * 2 * np.pi / 24 + np.pi/4) * 18,
        'lane_2_volume': np.random.poisson(55, 1000) + np.sin(np.arange(1000) * 2 * np.pi / 24 + np.pi/2) * 22,
        'lane_3_volume': np.random.poisson(48, 1000) + np.sin(np.arange(1000) * 2 * np.pi / 24 + 3*np.pi/4) * 19,
        'queue_length_0': np.random.poisson(5, 1000),
        'queue_length_1': np.random.poisson(4, 1000),
        'queue_length_2': np.random.poisson(6, 1000),
        'queue_length_3': np.random.poisson(5, 1000),
        'wait_time_0': np.random.exponential(10, 1000),
        'wait_time_1': np.random.exponential(9, 1000),
        'wait_time_2': np.random.exponential(11, 1000),
        'wait_time_3': np.random.exponential(10, 1000),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def time_series_dataset() -> np.ndarray:
    """Generate time series dataset for LSTM/GNN forecasting."""
    # Generate 1000 timesteps with 16 features
    timesteps = 1000
    features = 16
    
    # Create time series with trends and seasonality
    t = np.arange(timesteps)
    data = np.zeros((timesteps, features))
    
    for i in range(features):
        # Add trend
        trend = 0.01 * t
        # Add seasonality (daily pattern)
        seasonality = 10 * np.sin(2 * np.pi * t / 24)
        # Add noise
        noise = np.random.normal(0, 2, timesteps)
        
        data[:, i] = trend + seasonality + noise
    
    return data


@pytest.fixture
def synthetic_traffic_patterns() -> Dict[str, np.ndarray]:
    """Generate synthetic traffic patterns for different scenarios."""
    patterns = {}
    
    # Rush hour pattern (morning)
    t = np.arange(24)
    morning_rush = np.zeros(24)
    morning_rush[7:9] = 100  # Peak at 7-9 AM
    morning_rush[6:10] += np.random.normal(0, 10, 4)
    patterns['morning_rush'] = morning_rush
    
    # Rush hour pattern (evening)
    evening_rush = np.zeros(24)
    evening_rush[17:19] = 100  # Peak at 5-7 PM
    evening_rush[16:20] += np.random.normal(0, 10, 4)
    patterns['evening_rush'] = evening_rush
    
    # Off-peak pattern
    off_peak = np.ones(24) * 20 + np.random.normal(0, 5, 24)
    patterns['off_peak'] = off_peak
    
    # Special event pattern
    special_event = np.ones(24) * 30
    special_event[14:18] = 150  # High traffic during event
    patterns['special_event'] = special_event
    
    return patterns


@pytest.fixture
def forecasting_test_sequences() -> Tuple[np.ndarray, np.ndarray]:
    """Generate input-output sequences for forecasting model testing."""
    # Generate sequences: 20 timesteps input, 5 timesteps output
    num_sequences = 100
    input_timesteps = 20
    output_timesteps = 5
    features = 16
    
    X = np.random.randn(num_sequences, input_timesteps, features)
    y = np.random.randn(num_sequences, output_timesteps, features)
    
    return X, y


@pytest.fixture
def edge_case_forecasting_data() -> Dict[str, np.ndarray]:
    """Edge case scenarios for forecasting robustness testing."""
    edge_cases = {}
    
    # Sudden spike
    spike_data = np.ones(100) * 50
    spike_data[50:55] = 200  # Sudden spike
    edge_cases['sudden_spike'] = spike_data
    
    # Gradual increase
    gradual_increase = np.linspace(20, 100, 100) + np.random.normal(0, 5, 100)
    edge_cases['gradual_increase'] = gradual_increase
    
    # Zero traffic
    zero_traffic = np.zeros(100)
    edge_cases['zero_traffic'] = zero_traffic
    
    # Missing data (NaN values)
    missing_data = np.random.randn(100) * 50
    missing_data[20:25] = np.nan
    missing_data[60:65] = np.nan
    edge_cases['missing_data'] = missing_data
    
    # Extreme values
    extreme_values = np.random.randn(100) * 50
    extreme_values[30] = 1000  # Outlier
    extreme_values[70] = -100  # Negative outlier
    edge_cases['extreme_values'] = extreme_values
    
    return edge_cases


@pytest.fixture
def forecasting_ground_truth() -> Dict[str, Any]:
    """Ground truth forecasting results for validation."""
    return {
        "mae_threshold": 5.0,
        "rmse_threshold": 7.0,
        "mape_threshold": 0.15,
        "expected_accuracy": 0.85,
        "min_forecast_horizon": 1,
        "max_forecast_horizon": 10
    }

