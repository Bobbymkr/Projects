import pytest
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecast.traffic_forecast import TrafficForecaster

@pytest.fixture
def forecaster():
    return TrafficForecaster(input_timesteps=10, output_timesteps=5, features=8, lstm_units=50, cnn_filters=64, cnn_kernel=3)

def test_initialization(forecaster):
    """Test forecaster initialization."""
    assert forecaster.input_timesteps == 10
    assert forecaster.output_timesteps == 5
    assert forecaster.features == 8
    assert forecaster.model is not None

def test_build_model(forecaster):
    """Test model building."""
    assert forecaster.model is not None
    assert len(forecaster.model.layers) > 0

def test_train(forecaster):
    """Test training functionality."""
    # Generate sample data
    X = np.random.rand(100, 10, 8)  # (samples, timesteps, features)
    y = np.random.rand(100, 5, 8)   # (samples, output_timesteps, features)
    
    # Test training without TensorBoard
    forecaster.train(X, y, epochs=1, batch_size=32)
    assert forecaster.model is not None

def test_predict(forecaster):
    """Test prediction functionality."""
    # Train the model first
    X_train = np.random.rand(50, 10, 8)
    y_train = np.random.rand(50, 5, 8)
    forecaster.train(X_train, y_train, epochs=1, batch_size=16)
    
    # Test prediction
    X_test = np.random.rand(1, 10, 8)
    prediction = forecaster.predict(X_test)
    assert prediction.shape == (1, 5, 8)

def test_save_and_load(forecaster, tmp_path):
    """Test model saving and loading."""
    # Train the model first
    X = np.random.rand(20, 10, 8)
    y = np.random.rand(20, 5, 8)
    forecaster.train(X, y, epochs=1, batch_size=16)
    
    # Save model
    model_path = tmp_path / "test_model.h5"
    forecaster.save(str(model_path))
    assert model_path.exists()
    
    # Load model
    loaded_forecaster = TrafficForecaster.load(str(model_path))
    assert loaded_forecaster.model is not None
    
    # Test prediction consistency
    X_test = np.random.rand(1, 10, 8)
    original_pred = forecaster.predict(X_test)
    loaded_pred = loaded_forecaster.predict(X_test)
    np.testing.assert_array_almost_equal(original_pred, loaded_pred)

def test_validation_params():
    """Test parameter validation."""
    with pytest.raises(ValueError):
        TrafficForecaster(input_timesteps=0, output_timesteps=5, features=8)
    
    with pytest.raises(ValueError):
        TrafficForecaster(input_timesteps=10, output_timesteps=-1, features=8)