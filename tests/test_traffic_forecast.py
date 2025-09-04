import pytest
import numpy as np
from src.forecast.traffic_forecast import TrafficForecaster

@pytest.fixture
def forecaster():
    return TrafficForecaster(input_dim=10, hidden_dim=50, num_layers=2, output_dim=1)

def test_build_model(forecaster):
    forecaster.build_model()
    assert forecaster.model is not None
    assert len(forecaster.model.layers) == 2  # Assuming two LSTM layers

def test_train(forecaster):
    X = np.random.rand(100, 10, 10)  # Sample data
    y = np.random.rand(100, 1)
    forecaster.train(X, y, epochs=1, batch_size=32)
    # No assertion, just check if it runs without error

def test_predict(forecaster):
    forecaster.build_model()
    data = np.random.rand(1, 10, 10)
    prediction = forecaster.predict(data)
    assert prediction.shape == (1, 1)

def test_save_and_load(forecaster, tmp_path):
    forecaster.build_model()
    model_path = tmp_path / "test_model.h5"
    forecaster.save_model(str(model_path))
    new_forecaster = TrafficForecaster(input_dim=10, hidden_dim=50, num_layers=2, output_dim=1)
    new_forecaster.load_model(str(model_path))
    assert new_forecaster.model is not None