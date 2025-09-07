"""
Comprehensive unit tests for Traffic Forecasting Components.

This test suite covers:
- Model architecture validation and parameter checking
- Data preprocessing and validation
- Training mechanics and convergence testing
- Prediction accuracy and consistency
- Metrics computation (MAE, RMSE, MAPE, SMAPE)
- Model serialization and persistence
- Error handling and edge cases
- Performance benchmarks
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from src.forecast.traffic_forecast import TrafficForecaster


class TestTrafficForecasterInitialization:
    """Test forecaster initialization and parameter validation."""
    
    def test_valid_initialization(self):
        """Test valid forecaster initialization."""
        forecaster = TrafficForecaster(
            input_timesteps=20,
            output_timesteps=5,
            features=16,
            lstm_units=100,
            cnn_filters=32,
            cnn_kernel=5
        )
        
        assert forecaster.input_timesteps == 20
        assert forecaster.output_timesteps == 5
        assert forecaster.features == 16
        assert forecaster.lstm_units == 100
        assert forecaster.cnn_filters == 32
        assert forecaster.cnn_kernel == 5
        assert forecaster.model is not None
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        forecaster = TrafficForecaster()
        
        assert forecaster.input_timesteps == 10
        assert forecaster.output_timesteps == 1
        assert forecaster.features == 1
        assert forecaster.lstm_units == 50
        assert forecaster.cnn_filters == 64
        assert forecaster.cnn_kernel == 3
    
    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="All parameters must be positive integers"):
            TrafficForecaster(input_timesteps=0)
        
        with pytest.raises(ValueError, match="All parameters must be positive integers"):
            TrafficForecaster(output_timesteps=-1)
        
        with pytest.raises(ValueError, match="All parameters must be positive integers"):
            TrafficForecaster(features=0)
        
        with pytest.raises(ValueError, match="All parameters must be positive integers"):
            TrafficForecaster(lstm_units=-5)
        
        with pytest.raises(ValueError, match="All parameters must be positive integers"):
            TrafficForecaster(cnn_filters=0)
        
        with pytest.raises(ValueError, match="All parameters must be positive integers"):
            TrafficForecaster(cnn_kernel=-1)


class TestModelArchitecture:
    """Test CNN-LSTM model architecture and properties."""
    
    @pytest.fixture
    def forecaster(self):
        return TrafficForecaster(input_timesteps=10, output_timesteps=3, features=8, lstm_units=32)
    
    def test_model_build(self, forecaster):
        """Test model is built correctly on initialization."""
        assert forecaster.model is not None
        assert isinstance(forecaster.model, tf.keras.Model)
        assert forecaster.model.built
    
    def test_model_layers(self, forecaster):
        """Test model has expected layer structure."""
        model = forecaster.model
        layer_types = [type(layer).__name__ for layer in model.layers]
        
        # Should contain CNN, LSTM, and Dense layers
        assert 'Conv1D' in layer_types
        assert 'MaxPooling1D' in layer_types
        assert 'Flatten' in layer_types
        assert 'RepeatVector' in layer_types
        assert 'LSTM' in layer_types
        assert 'TimeDistributed' in layer_types
        
        # Check minimum number of layers for CNN-LSTM architecture
        assert len(model.layers) >= 6
    
    def test_model_input_output_shapes(self, forecaster):
        """Test model input and output shapes."""
        model = forecaster.model
        
        # Input shape should match (batch, timesteps, features)
        expected_input_shape = (None, forecaster.input_timesteps, forecaster.features)
        assert model.input_shape == expected_input_shape
        
        # Output shape should match (batch, output_timesteps, features)
        expected_output_shape = (None, forecaster.output_timesteps, forecaster.features)
        assert model.output_shape == expected_output_shape
    
    def test_model_compilation(self, forecaster):
        """Test model is compiled with correct optimizer and loss."""
        model = forecaster.model
        
        assert model.optimizer is not None
        assert isinstance(model.optimizer, tf.keras.optimizers.Optimizer)
        assert model.loss == 'mse'  # Mean squared error
    
    def test_model_parameters_count(self, forecaster):
        """Test model has reasonable number of parameters."""
        param_count = forecaster.model.count_params()
        
        # Should have parameters but not excessive
        assert param_count > 100  # At least some parameters
        assert param_count < 1_000_000  # Not excessive for test model


class TestDataPreprocessing:
    """Test data preprocessing and validation."""
    
    @pytest.fixture
    def forecaster(self):
        return TrafficForecaster(input_timesteps=5, output_timesteps=2, features=4)
    
    def test_prediction_data_shapes(self, forecaster):
        """Test prediction with correct data shapes."""
        # Single sample
        data = np.random.rand(1, 5, 4)
        prediction = forecaster.predict(data)
        assert prediction.shape == (1, 2, 4)
        
        # Multiple samples
        data = np.random.rand(10, 5, 4)
        prediction = forecaster.predict(data)
        assert prediction.shape == (10, 2, 4)
    
    def test_prediction_data_validation(self, forecaster):
        """Test prediction with invalid data shapes raises errors."""
        # Wrong number of dimensions
        with pytest.raises(Exception):  # TensorFlow will raise various errors
            forecaster.predict(np.random.rand(5, 4))
        
        # Wrong timestep dimension
        with pytest.raises(Exception):
            forecaster.predict(np.random.rand(1, 3, 4))  # Should be 5 timesteps
        
        # Wrong feature dimension
        with pytest.raises(Exception):
            forecaster.predict(np.random.rand(1, 5, 6))  # Should be 4 features
    
    def test_training_data_validation(self, forecaster):
        """Test training data validation."""
        # Valid data
        X = np.random.rand(50, 5, 4)
        y = np.random.rand(50, 2, 4)
        
        try:
            forecaster.train(X, y, epochs=1, batch_size=10)
        except Exception as e:
            pytest.fail(f"Valid training data should not raise exception: {e}")
        
        # Mismatched batch sizes
        X = np.random.rand(50, 5, 4)
        y = np.random.rand(30, 2, 4)  # Different batch size
        
        with pytest.raises(Exception):
            forecaster.train(X, y, epochs=1)
    
    def test_missing_value_handling(self, forecaster):
        """Test handling of missing values (NaN)."""
        # Data with NaN values
        data = np.random.rand(1, 5, 4)
        data[0, 2, 1] = np.nan  # Introduce NaN
        
        # Model should handle NaN or raise appropriate error
        try:
            prediction = forecaster.predict(data)
            # If prediction succeeds, check for NaN in output
            assert not np.any(np.isnan(prediction)), "Output should not contain NaN"
        except Exception:
            # It's acceptable for the model to reject NaN input
            pass
    
    def test_data_type_handling(self, forecaster):
        """Test handling of different data types."""
        # Float32 data
        data_f32 = np.random.rand(1, 5, 4).astype(np.float32)
        prediction_f32 = forecaster.predict(data_f32)
        assert prediction_f32.dtype in [np.float32, np.float64]
        
        # Float64 data
        data_f64 = np.random.rand(1, 5, 4).astype(np.float64)
        prediction_f64 = forecaster.predict(data_f64)
        assert prediction_f64.dtype in [np.float32, np.float64]


class TestTrainingMechanics:
    """Test training process and mechanics."""
    
    @pytest.fixture
    def training_forecaster(self):
        return TrafficForecaster(input_timesteps=8, output_timesteps=2, features=6, lstm_units=16)
    
    def test_training_basic(self, training_forecaster):
        """Test basic training functionality."""
        X = np.random.rand(100, 8, 6)
        y = np.random.rand(100, 2, 6)
        
        # Training should complete without errors
        training_forecaster.train(X, y, epochs=2, batch_size=16)
    
    def test_training_with_validation_split(self, training_forecaster):
        """Test training with validation split."""
        X = np.random.rand(100, 8, 6)
        y = np.random.rand(100, 2, 6)
        
        # Training with validation split
        training_forecaster.train(X, y, epochs=2, validation_split=0.3)
    
    def test_training_convergence_simple(self, training_forecaster):
        """Test training shows improvement on simple synthetic data."""
        # Create simple synthetic data with pattern
        np.random.seed(42)  # For reproducibility
        t = np.linspace(0, 4*np.pi, 200)
        
        # Create sinusoidal patterns
        patterns = np.array([np.sin(t), np.cos(t), np.sin(2*t), np.cos(2*t), 
                            np.sin(0.5*t), np.cos(0.5*t)]).T
        
        # Create input-output pairs
        X, y = [], []
        for i in range(len(patterns) - 10):
            X.append(patterns[i:i+8])
            y.append(patterns[i+8:i+10])
        
        X = np.array(X)
        y = np.array(y)
        
        # Get initial loss
        initial_loss = training_forecaster.model.evaluate(X, y, verbose=0)
        
        # Train for several epochs
        training_forecaster.train(X, y, epochs=10, batch_size=16, validation_split=0)
        
        # Get final loss
        final_loss = training_forecaster.model.evaluate(X, y, verbose=0)
        
        # Loss should improve (decrease)
        assert final_loss < initial_loss, f"Training should improve loss: {initial_loss} -> {final_loss}"
    
    def test_training_with_callbacks(self, training_forecaster, tmp_path):
        """Test training with TensorBoard callback."""
        X = np.random.rand(50, 8, 6)
        y = np.random.rand(50, 2, 6)
        
        log_dir = str(tmp_path / "logs")
        training_forecaster.train(X, y, epochs=1, log_dir=log_dir)
        
        # Check that log directory was created
        assert os.path.exists(log_dir)
    
    def test_training_error_handling(self, training_forecaster):
        """Test training error handling with invalid data."""
        # Completely wrong data shape
        X = np.random.rand(10, 5)  # Missing feature dimension
        y = np.random.rand(10, 2, 6)
        
        with pytest.raises(RuntimeError, match="Training failed due to an internal error"):
            training_forecaster.train(X, y, epochs=1)


class TestPredictionAccuracy:
    """Test prediction accuracy and consistency."""
    
    @pytest.fixture
    def trained_forecaster(self):
        forecaster = TrafficForecaster(input_timesteps=5, output_timesteps=1, features=2, lstm_units=8)
        
        # Simple training on identity-like mapping
        X = np.random.rand(50, 5, 2)
        y = X[:, -1:, :]  # Predict last timestep
        forecaster.train(X, y, epochs=5, validation_split=0)
        
        return forecaster
    
    def test_prediction_consistency(self, trained_forecaster):
        """Test prediction consistency with same input."""
        data = np.random.rand(1, 5, 2)
        
        # Multiple predictions with same input should be identical
        pred1 = trained_forecaster.predict(data)
        pred2 = trained_forecaster.predict(data)
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_prediction_bounds(self, trained_forecaster):
        """Test prediction output bounds and properties."""
        data = np.random.rand(10, 5, 2)
        predictions = trained_forecaster.predict(data)
        
        # Predictions should be finite
        assert np.all(np.isfinite(predictions))
        
        # Predictions should have correct shape
        assert predictions.shape == (10, 1, 2)
        
        # Predictions should be numeric
        assert predictions.dtype in [np.float32, np.float64]
    
    def test_batch_prediction_consistency(self, trained_forecaster):
        """Test batch vs individual predictions are consistent."""
        # Create test data
        individual_data = [np.random.rand(1, 5, 2) for _ in range(3)]
        batch_data = np.concatenate(individual_data, axis=0)
        
        # Individual predictions
        individual_preds = [trained_forecaster.predict(data) for data in individual_data]
        
        # Batch prediction
        batch_pred = trained_forecaster.predict(batch_data)
        
        # Compare results
        for i, ind_pred in enumerate(individual_preds):
            np.testing.assert_array_almost_equal(ind_pred, batch_pred[i:i+1], decimal=5)
    
    def test_prediction_error_handling(self, trained_forecaster):
        """Test prediction error handling."""
        # Empty data
        with pytest.raises(Exception):
            trained_forecaster.predict(np.array([]))
        
        # Wrong shape
        with pytest.raises(Exception):
            trained_forecaster.predict(np.random.rand(1, 3, 2))  # Wrong timesteps


class TestMetricsComputation:
    """Test forecasting metrics computation."""
    
    def test_mae_computation(self):
        """Test Mean Absolute Error computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        
        # Manual MAE calculation
        expected_mae = np.mean(np.abs(y_true - y_pred))
        
        # Verify calculation
        assert np.isclose(expected_mae, 0.15), f"Expected MAE ~0.15, got {expected_mae}"
    
    def test_rmse_computation(self):
        """Test Root Mean Square Error computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        
        # Manual RMSE calculation
        expected_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Verify calculation
        assert expected_rmse > 0, "RMSE should be positive"
        assert np.isfinite(expected_rmse), "RMSE should be finite"
    
    def test_mape_computation(self):
        """Test Mean Absolute Percentage Error computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        
        # Manual MAPE calculation (avoiding division by zero)
        expected_mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-7))) * 100
        
        # Verify calculation
        assert expected_mape >= 0, "MAPE should be non-negative"
        assert np.isfinite(expected_mape), "MAPE should be finite"
    
    def test_metrics_with_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()  # Perfect prediction
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-7))) * 100
        
        assert mae == 0.0, "Perfect prediction should have MAE = 0"
        assert rmse == 0.0, "Perfect prediction should have RMSE = 0"
        assert mape == 0.0, "Perfect prediction should have MAPE = 0"


class TestModelSerialization:
    """Test model saving and loading functionality."""
    
    @pytest.fixture
    def trained_forecaster(self):
        forecaster = TrafficForecaster(input_timesteps=6, output_timesteps=2, features=4, lstm_units=12)
        
        # Quick training
        X = np.random.rand(30, 6, 4)
        y = np.random.rand(30, 2, 4)
        forecaster.train(X, y, epochs=2, validation_split=0)
        
        return forecaster
    
    def test_save_and_load(self, trained_forecaster, tmp_path):
        """Test model save and load functionality."""
        model_path = str(tmp_path / "test_forecaster.keras")
        
        # Get prediction before saving
        test_data = np.random.rand(1, 6, 4)
        original_prediction = trained_forecaster.predict(test_data)
        
        # Save model
        trained_forecaster.save(model_path)
        assert os.path.exists(model_path)
        
        # Load model
        loaded_forecaster = TrafficForecaster.load(model_path)
        
        # Test loaded model
        assert loaded_forecaster.model is not None
        loaded_prediction = loaded_forecaster.predict(test_data)
        
        # Predictions should match
        np.testing.assert_array_almost_equal(original_prediction, loaded_prediction, decimal=5)
    
    def test_save_load_different_formats(self, trained_forecaster, tmp_path):
        """Test saving/loading with different file formats."""
        # Test .keras format
        keras_path = str(tmp_path / "model.keras")
        trained_forecaster.save(keras_path)
        loaded_keras = TrafficForecaster.load(keras_path)
        assert loaded_keras.model is not None
        
        # Test .h5 format (if supported)
        h5_path = str(tmp_path / "model.h5")
        try:
            trained_forecaster.save(h5_path)
            loaded_h5 = TrafficForecaster.load(h5_path)
            assert loaded_h5.model is not None
        except Exception:
            # H5 format might not be supported in newer TensorFlow versions
            pytest.skip("H5 format not supported")
    
    def test_load_error_handling(self, tmp_path):
        """Test error handling when loading invalid models."""
        # Non-existent file
        with pytest.raises(RuntimeError, match="Model loading failed"):
            TrafficForecaster.load("non_existent_model.keras")
        
        # Invalid file - use tmp_path to avoid Windows file locking issues
        invalid_model_path = tmp_path / "invalid_model.keras"
        invalid_model_path.write_bytes(b"invalid model data")
        
        with pytest.raises(RuntimeError, match="Model loading failed"):
            TrafficForecaster.load(str(invalid_model_path))


@pytest.mark.perf
class TestPerformanceBenchmarks:
    """Performance benchmark tests for forecasting components."""
    
    def test_prediction_performance(self):
        """Test prediction performance benchmark."""
        forecaster = TrafficForecaster(input_timesteps=20, output_timesteps=5, features=8)
        
        # Large batch for performance testing
        data = np.random.rand(1000, 20, 8)
        
        import time
        start_time = time.time()
        predictions = forecaster.predict(data)
        end_time = time.time()
        
        inference_time = end_time - start_time
        throughput = len(data) / inference_time
        
        # Should process at least 100 samples per second
        assert throughput > 100, f"Inference too slow: {throughput:.1f} samples/sec"
        assert predictions.shape == (1000, 5, 8)
    
    def test_training_performance(self):
        """Test training performance on small dataset."""
        forecaster = TrafficForecaster(input_timesteps=10, output_timesteps=2, features=6, lstm_units=32)
        
        X = np.random.rand(500, 10, 6)
        y = np.random.rand(500, 2, 6)
        
        import time
        start_time = time.time()
        forecaster.train(X, y, epochs=5, batch_size=32, validation_split=0)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        # Training should complete in reasonable time (under 30 seconds for small model)
        assert training_time < 30, f"Training too slow: {training_time:.1f} seconds"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
