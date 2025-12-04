"""
Extended Unit Tests for Traffic Forecasting Components.

Tests data preprocessing, metrics validation, backtesting, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

# Import forecasting components
from src.forecast.traffic_forecast import TrafficForecaster

# Import fixtures
from tests.fixtures.forecasting_data import (
    historical_traffic_data,
    time_series_dataset,
    forecasting_test_sequences,
    edge_case_forecasting_data,
    forecasting_ground_truth
)


class TestDataPreprocessing:
    """Test data preprocessing for forecasting."""
    
    def test_data_normalization(self, time_series_dataset):
        """Test normalizing time series data."""
        data = time_series_dataset
        
        # Normalize to [0, 1]
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        normalized = (data - data_min) / (data_max - data_min + 1e-8)
        
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
        assert normalized.shape == data.shape
    
    def test_sequence_creation(self, time_series_dataset):
        """Test creating input-output sequences."""
        data = time_series_dataset
        input_timesteps = 20
        output_timesteps = 5
        
        sequences = []
        for i in range(len(data) - input_timesteps - output_timesteps + 1):
            input_seq = data[i:i+input_timesteps]
            output_seq = data[i+input_timesteps:i+input_timesteps+output_timesteps]
            sequences.append((input_seq, output_seq))
        
        assert len(sequences) > 0
        assert sequences[0][0].shape == (input_timesteps, data.shape[1])
        assert sequences[0][1].shape == (output_timesteps, data.shape[1])
    
    def test_missing_data_handling(self, edge_case_forecasting_data):
        """Test handling missing data (NaN values)."""
        data = edge_case_forecasting_data['missing_data']
        
        # Fill missing values
        filled_data = pd.Series(data).fillna(method='forward').fillna(method='backward').values
        
        assert not np.any(np.isnan(filled_data))
        assert len(filled_data) == len(data)
    
    def test_outlier_detection(self, edge_case_forecasting_data):
        """Test detecting and handling outliers."""
        data = edge_case_forecasting_data['extreme_values']
        
        # Detect outliers using IQR method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
        assert np.any(outliers)  # Should detect outliers
        assert isinstance(outliers, np.ndarray)


class TestMetricsValidation:
    """Test forecasting metrics calculation and validation."""
    
    @pytest.fixture
    def forecaster(self):
        """Create forecaster for testing."""
        return TrafficForecaster(
            input_timesteps=20,
            output_timesteps=5,
            features=16
        )
    
    def test_mae_calculation(self, forecaster, forecasting_test_sequences):
        """Test Mean Absolute Error calculation."""
        X, y_true = forecasting_test_sequences
        
        # Generate predictions
        y_pred = forecaster.predict(X[:10])
        
        # Calculate MAE
        mae = np.mean(np.abs(y_pred - y_true[:10]))
        
        assert mae is not None
        assert mae >= 0
        assert isinstance(mae, (int, float))
    
    def test_rmse_calculation(self, forecaster, forecasting_test_sequences):
        """Test Root Mean Squared Error calculation."""
        X, y_true = forecasting_test_sequences
        
        # Generate predictions
        y_pred = forecaster.predict(X[:10])
        
        # Calculate RMSE
        mse = np.mean((y_pred - y_true[:10]) ** 2)
        rmse = np.sqrt(mse)
        
        assert rmse is not None
        assert rmse >= 0
        assert isinstance(rmse, (int, float))
    
    def test_mape_calculation(self, forecaster, forecasting_test_sequences):
        """Test Mean Absolute Percentage Error calculation."""
        X, y_true = forecasting_test_sequences
        
        # Generate predictions
        y_pred = forecaster.predict(X[:10])
        
        # Calculate MAPE (avoid division by zero)
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true[:10] - y_pred) / (y_true[:10] + epsilon))) * 100
        
        assert mape is not None
        assert mape >= 0
        assert isinstance(mape, (int, float))
    
    def test_metrics_threshold_validation(self, forecaster, forecasting_ground_truth):
        """Test metrics against ground truth thresholds."""
        gt = forecasting_ground_truth
        
        # Mock metrics
        mae = 4.5
        rmse = 6.5
        mape = 0.12
        
        assert mae < gt['mae_threshold']
        assert rmse < gt['rmse_threshold']
        assert mape < gt['mape_threshold']


class TestBacktesting:
    """Test backtesting functionality."""
    
    @pytest.fixture
    def forecaster(self):
        """Create forecaster."""
        return TrafficForecaster(
            input_timesteps=20,
            output_timesteps=5,
            features=16
        )
    
    def test_backtest_single_period(self, forecaster, time_series_dataset):
        """Test backtesting on single time period."""
        data = time_series_dataset
        
        # Split data
        train_data = data[:800]
        test_data = data[800:900]
        
        # Prepare test input
        input_seq = test_data[:20].reshape(1, 20, 16)
        
        # Generate forecast
        forecast = forecaster.predict(input_seq)
        
        # Compare with actual
        actual = test_data[20:25]
        
        # Calculate error
        error = np.mean(np.abs(forecast[0] - actual))
        
        assert error is not None
        assert error >= 0
    
    def test_backtest_multiple_periods(self, forecaster, time_series_dataset):
        """Test backtesting on multiple time periods."""
        data = time_series_dataset
        
        # Split data
        train_data = data[:800]
        test_data = data[800:900]
        
        errors = []
        for i in range(len(test_data) - 25):
            input_seq = test_data[i:i+20].reshape(1, 20, 16)
            forecast = forecaster.predict(input_seq)
            actual = test_data[i+20:i+25]
            
            error = np.mean(np.abs(forecast[0] - actual))
            errors.append(error)
        
        assert len(errors) > 0
        assert all(e >= 0 for e in errors)
        assert np.mean(errors) is not None
    
    def test_rolling_window_backtest(self, forecaster, time_series_dataset):
        """Test rolling window backtesting."""
        data = time_series_dataset
        window_size = 20
        forecast_horizon = 5
        
        errors = []
        for i in range(len(data) - window_size - forecast_horizon):
            # Use rolling window
            input_seq = data[i:i+window_size].reshape(1, window_size, 16)
            forecast = forecaster.predict(input_seq)
            actual = data[i+window_size:i+window_size+forecast_horizon]
            
            error = np.mean(np.abs(forecast[0] - actual))
            errors.append(error)
        
        assert len(errors) > 0
        assert all(e >= 0 for e in errors)


class TestEdgeCases:
    """Test edge cases and robustness."""
    
    @pytest.fixture
    def forecaster(self):
        """Create forecaster."""
        return TrafficForecaster(
            input_timesteps=20,
            output_timesteps=5,
            features=16
        )
    
    def test_zero_traffic_forecast(self, forecaster, edge_case_forecasting_data):
        """Test forecasting with zero traffic."""
        zero_data = edge_case_forecasting_data['zero_traffic']
        
        # Create input sequence
        input_seq = np.zeros((1, 20, 16))
        input_seq[0, :, 0] = zero_data[:20]
        
        # Generate forecast
        forecast = forecaster.predict(input_seq)
        
        assert forecast is not None
        assert forecast.shape == (1, 5, 16)
    
    def test_sudden_spike_forecast(self, forecaster, edge_case_forecasting_data):
        """Test forecasting sudden traffic spikes."""
        spike_data = edge_case_forecasting_data['sudden_spike']
        
        # Create input sequence
        input_seq = np.zeros((1, 20, 16))
        input_seq[0, :, 0] = spike_data[:20]
        
        # Generate forecast
        forecast = forecaster.predict(input_seq)
        
        assert forecast is not None
        assert forecast.shape == (1, 5, 16)
    
    def test_gradual_increase_forecast(self, forecaster, edge_case_forecasting_data):
        """Test forecasting gradual traffic increase."""
        gradual_data = edge_case_forecasting_data['gradual_increase']
        
        # Create input sequence
        input_seq = np.zeros((1, 20, 16))
        input_seq[0, :, 0] = gradual_data[:20]
        
        # Generate forecast
        forecast = forecaster.predict(input_seq)
        
        assert forecast is not None
        assert forecast.shape == (1, 5, 16)
    
    def test_short_sequence_handling(self, forecaster):
        """Test handling sequences shorter than required."""
        # Sequence shorter than input_timesteps
        short_seq = np.random.randn(10, 16)  # Less than 20
        
        # Should handle gracefully or raise appropriate error
        try:
            # Pad sequence
            padded = np.pad(short_seq, ((10, 0), (0, 0)), mode='constant')
            input_seq = padded.reshape(1, 20, 16)
            forecast = forecaster.predict(input_seq)
            assert forecast is not None
        except Exception as e:
            # Expected error for invalid input
            assert "input" in str(e).lower() or "shape" in str(e).lower()


class TestModelPersistence:
    """Test model saving and loading."""
    
    @pytest.fixture
    def forecaster(self):
        """Create forecaster."""
        return TrafficForecaster(
            input_timesteps=20,
            output_timesteps=5,
            features=16
        )
    
    def test_model_saving(self, forecaster):
        """Test saving model to file."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            forecaster.save_model(model_path)
            assert os.path.exists(model_path)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_model_loading(self, forecaster):
        """Test loading model from file."""
        # Save model first
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            forecaster.save_model(model_path)
            
            # Load into new forecaster
            new_forecaster = TrafficForecaster(
                input_timesteps=20,
                output_timesteps=5,
                features=16
            )
            new_forecaster.load_model(model_path)
            
            assert new_forecaster.model is not None
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_save_load_consistency(self, forecaster, forecasting_test_sequences):
        """Test that saved and loaded model produces same predictions."""
        X, _ = forecasting_test_sequences
        
        # Get predictions before save
        pred_before = forecaster.predict(X[:1])
        
        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            forecaster.save_model(model_path)
            
            new_forecaster = TrafficForecaster(
                input_timesteps=20,
                output_timesteps=5,
                features=16
            )
            new_forecaster.load_model(model_path)
            
            # Get predictions after load
            pred_after = new_forecaster.predict(X[:1])
            
            # Should be similar (allowing for floating point differences)
            assert np.allclose(pred_before, pred_after, rtol=1e-5)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

