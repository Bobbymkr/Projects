import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class TrafficForecaster:
    """LSTM-based model for predicting traffic volumes."""

    def __init__(self, input_timesteps=10, output_timesteps=5, features=8):
        # Function-level comment: Initialize the forecaster with model parameters.
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.features = features
        self.model = self._build_model()

    def _build_model(self):
        # Function-level comment: Build the LSTM model architecture.
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(self.input_timesteps, self.features)))
        model.add(Dense(self.output_timesteps * self.features))
        model.add(tf.keras.layers.Reshape((self.output_timesteps, self.features)))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X, y, epochs=50, batch_size=32):
        # Function-level comment: Train the model on historical data.
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        # Function-level comment: Predict future traffic states.
        return self.model.predict(X)

    def save(self, path):
        # Function-level comment: Save the trained model.
        self.model.save(path)

    @classmethod
    def load(cls, path):
        # Function-level comment: Load a saved model.
        forecaster = cls()
        forecaster.model = tf.keras.models.load_model(path)
        return forecaster