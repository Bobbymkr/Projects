#!/usr/bin/env python3
"""
LSTM Traffic Forecasting Demonstration

This script demonstrates the LSTM-based traffic forecasting system by:
1. Generating synthetic traffic data with realistic patterns
2. Training a CNN-LSTM hybrid model
3. Making predictions and visualizing results
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime, timedelta

from src.forecast.traffic_forecast import TrafficForecaster

def generate_synthetic_traffic_data(n_samples=1000, n_features=4):
    """Generate synthetic traffic data with realistic patterns."""
    print("Generating synthetic traffic data...")
    
    # Time-based patterns
    time_steps = np.arange(n_samples)
    
    # Base traffic patterns (morning rush, evening rush, night low)
    base_pattern = (
        0.3 +  # Base traffic
        0.4 * np.sin(2 * np.pi * time_steps / 24) +  # Daily cycle
        0.2 * np.sin(2 * np.pi * time_steps / 168) +  # Weekly cycle
        0.1 * np.random.randn(n_samples)  # Random noise
    )
    
    # Generate data for 4 lanes (North, South, East, West)
    traffic_data = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        # Add lane-specific variations
        lane_pattern = base_pattern + 0.1 * np.sin(2 * np.pi * time_steps / 12 + i * np.pi/2)
        # Ensure non-negative values and add some randomness
        traffic_data[:, i] = np.maximum(0, lane_pattern + 0.05 * np.random.randn(n_samples))
    
    return traffic_data

def prepare_sequences(data, input_steps=10, output_steps=5):
    """Prepare sequences for LSTM training."""
    print("Preparing training sequences...")
    
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:(i + input_steps)])
        y.append(data[(i + input_steps):(i + input_steps + output_steps)])
    
    return np.array(X), np.array(y)

def train_forecasting_model(X, y, epochs=20):
    """Train the LSTM forecasting model."""
    print("Training LSTM forecasting model...")
    
    # Initialize forecaster
    forecaster = TrafficForecaster(
        input_timesteps=X.shape[1],
        output_timesteps=y.shape[1],
        features=X.shape[2],
        lstm_units=64,
        cnn_filters=32,
        cnn_kernel=3
    )
    
    # Train the model
    forecaster.train(X, y, epochs=epochs, batch_size=32, validation_split=0.2)
    
    return forecaster

def make_predictions(forecaster, test_data, input_steps=10, output_steps=5):
    """Make predictions using the trained model."""
    print("Making traffic predictions...")
    
    # Prepare test sequence
    test_sequence = test_data[-input_steps:].reshape(1, input_steps, test_data.shape[1])
    
    # Make prediction
    prediction = forecaster.predict(test_sequence)
    
    return prediction[0]  # Remove batch dimension

def visualize_results(actual_data, predicted_data, input_steps=10, output_steps=5):
    """Visualize the forecasting results."""
    print("Creating visualization...")
    
    # Create time axis
    total_steps = len(actual_data)
    time_axis = np.arange(total_steps)
    
    # Plot for each lane
    lane_names = ['North', 'South', 'East', 'West']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LSTM Traffic Forecasting Results', fontsize=16)
    
    for i, (ax, lane_name) in enumerate(zip(axes.flat, lane_names)):
        # Plot historical data
        ax.plot(time_axis, actual_data[:, i], 'b-', label='Historical', linewidth=2)
        
        # Plot prediction
        pred_start = total_steps - input_steps
        pred_end = pred_start + output_steps
        ax.plot(range(pred_start, pred_end), predicted_data[:, i], 'r--', 
                label='Predicted', linewidth=3, markersize=8)
        
        # Highlight input window
        input_start = total_steps - input_steps
        ax.axvspan(input_start, total_steps, alpha=0.2, color='gray', label='Input Window')
        
        ax.set_title(f'{lane_name} Lane Traffic')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Traffic Volume')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'runs/traffic_forecast_results.png'
    os.makedirs('runs', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return fig

def calculate_metrics(actual, predicted):
    """Calculate forecasting accuracy metrics."""
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def main():
    parser = argparse.ArgumentParser(description="LSTM Traffic Forecasting Demo")
    parser.add_argument('--samples', type=int, default=500, help='Number of data samples')
    parser.add_argument('--input-steps', type=int, default=10, help='Input sequence length')
    parser.add_argument('--output-steps', type=int, default=5, help='Output prediction length')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model')
    
    args = parser.parse_args()
    
    print("🚦 LSTM Traffic Forecasting Demonstration")
    print("=" * 50)
    
    # Generate synthetic traffic data
    traffic_data = generate_synthetic_traffic_data(args.samples)
    print(f"Generated {traffic_data.shape[0]} samples with {traffic_data.shape[1]} features")
    
    # Prepare sequences for training
    X, y = prepare_sequences(traffic_data, args.input_steps, args.output_steps)
    print(f"Prepared {X.shape[0]} training sequences")
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    # Train the model
    forecaster = train_forecasting_model(X, y, args.epochs)
    print("Model training completed!")
    
    # Make predictions
    predictions = make_predictions(forecaster, traffic_data, args.input_steps, args.output_steps)
    print(f"Generated predictions for {predictions.shape[0]} future time steps")
    
    # Calculate metrics
    actual_future = traffic_data[-args.output_steps:]
    metrics = calculate_metrics(actual_future, predictions)
    
    print("\n📊 Forecasting Performance Metrics:")
    print(f"  Mean Squared Error (MSE): {metrics['MSE']:.4f}")
    print(f"  Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
    
    # Create visualization
    fig = visualize_results(traffic_data, predictions, args.input_steps, args.output_steps)
    
    # Save model if requested
    if args.save_model:
        model_path = 'runs/traffic_forecaster_model.keras'
        os.makedirs('runs', exist_ok=True)
        forecaster.save(model_path)
        print(f"Model saved to {model_path}")
    
    print("\n✅ LSTM Traffic Forecasting demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("  • Hybrid CNN-LSTM architecture for time series prediction")
    print("  • Multi-lane traffic pattern forecasting")
    print("  • Real-time prediction capabilities")
    print("  • Comprehensive performance metrics")
    print("  • Visual result analysis")

if __name__ == "__main__":
    main()
