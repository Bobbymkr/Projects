import pytest
import numpy as np
from src.forecast.traffic_forecast import TrafficForecaster

@pytest.fixture
def forecaster():
    return TrafficForecaster(input_timesteps=10, output_timesteps=1, features=8, lstm_units=50)

def test_build_model(forecaster):
    # Model is built in __init__, so just check it exists
    assert forecaster.model is not None
    assert len(forecaster.model.layers) >= 4  # CNN + LSTM layers

def test_train(forecaster):
    X = np.random.rand(100, 10, 8)  # Sample data: (batch, timesteps, features)
    y = np.random.rand(100, 1, 8)   # Target data: (batch, output_timesteps, features)
    forecaster.train(X, y, epochs=1, batch_size=32)
    # No assertion, just check if it runs without error

def test_predict(forecaster):
    # Model already built in __init__
    data = np.random.rand(1, 10, 8)  # (batch, timesteps, features)
    prediction = forecaster.predict(data)
    assert prediction.shape == (1, 1, 8)  # (batch, output_timesteps, features)

def test_save_and_load(forecaster, tmp_path):
    # Model already built in __init__
    model_path = tmp_path / "test_model.keras"
    forecaster.save(str(model_path))
    new_forecaster = TrafficForecaster.load(str(model_path))
    assert new_forecaster.model is not None
