import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import TensorBoard
import os

class TrafficForecaster:
    """Traffic forecasting model using a hybrid CNN-LSTM architecture for time series prediction.

    This class handles model building, training, prediction, saving, and loading.
    """

    def __init__(self, input_timesteps=10, output_timesteps=1, features=1, lstm_units=50, cnn_filters=64, cnn_kernel=3):
        # Detailed comment: Initialize the forecaster with configurable parameters.
        # Parameters:
        # - input_timesteps: Number of past timesteps for input.
        # - output_timesteps: Number of future timesteps to predict.
        # - features: Number of features in the data.
        # - lstm_units: Number of units in LSTM layer.
        # - cnn_filters: Number of filters in CNN layer.
        # - cnn_kernel: Kernel size for CNN layer.
        self._validate_params(input_timesteps, output_timesteps, features, lstm_units, cnn_filters, cnn_kernel)
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.features = features
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.cnn_kernel = cnn_kernel
        self.model = self._build_model()

    def _validate_params(self, input_timesteps, output_timesteps, features, lstm_units, cnn_filters, cnn_kernel):
        # Detailed comment: Validate initialization parameters to ensure they are positive integers.
        if input_timesteps < 1 or output_timesteps < 1 or features < 1 or lstm_units < 1 or cnn_filters < 1 or cnn_kernel < 1:
            raise ValueError("All parameters must be positive integers")

    def _build_model(self):
        # Detailed comment: Build the CNN-LSTM model architecture.
        # Returns: Compiled Keras model.
        model = Sequential()
        model.add(Conv1D(filters=self.cnn_filters, kernel_size=self.cnn_kernel, activation='relu', input_shape=(self.input_timesteps, self.features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(tf.keras.layers.RepeatVector(self.output_timesteps))
        model.add(LSTM(self.lstm_units, activation='relu', return_sequences=True))
        model.add(tf.keras.layers.TimeDistributed(Dense(self.features)))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2, log_dir=None):
        # Detailed comment: Train the model on provided data.
        # Parameters:
        # - X: Input data.
        # - y: Target data.
        # - epochs: Number of training epochs.
        # - batch_size: Batch size for training.
        # - validation_split: Fraction of data for validation.
        # - log_dir: Directory for TensorBoard logs.
        try:
            callbacks = []
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
                callbacks.append(tensorboard_callback)
            
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, 
                          validation_split=validation_split, callbacks=callbacks)
        except Exception as e:
            raise RuntimeError("Training failed due to an internal error.")

    def predict(self, X):
        # Detailed comment: Make predictions using the trained model.
        # Parameters:
        # - X: Input data for prediction.
        # Returns: Predicted values.
        try:
            return self.model.predict(X)
        except Exception as e:
            raise RuntimeError("Prediction failed due to an internal error.")

    def save(self, path):
        # Detailed comment: Save the model to a file.
        # Parameters:
        # - path: Path to save the model.
        self.model.save(path)

    @classmethod
    def load(cls, path):
        # Detailed comment: Load a saved model from a file.
        # Parameters:
        # - path: Path to the saved model.
        try:
            model = tf.keras.models.load_model(path)
            forecaster = cls()  # Use defaults, or extract from model if needed
            forecaster.model = model
            return forecaster
        except Exception as e:
            raise RuntimeError("Model loading failed due to an internal error.")