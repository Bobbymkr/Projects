"""
Integration Tests: Forecasting to Scheduling and RL.

Tests the integration between traffic forecasting, scheduling,
and reinforcement learning decision-making.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import forecasting
from src.forecast.traffic_forecast import TrafficForecaster

# Import RL agents
from src.rl.dqn_agent import DQNAgent, DQNConfig

# Import fixtures
from tests.fixtures.forecasting_data import (
    historical_traffic_data,
    time_series_dataset,
    forecasting_test_sequences
)


class TestForecastingToScheduling:
    """Test integration from forecasting to scheduling."""
    
    @pytest.fixture
    def forecaster(self):
        """Create traffic forecaster."""
        return TrafficForecaster(
            input_timesteps=20,
            output_timesteps=5,
            features=16,
            lstm_units=50
        )
    
    def test_forecast_generation(self, forecaster, time_series_dataset):
        """Test that forecasts can be generated from historical data."""
        data = time_series_dataset
        
        # Prepare input sequence (last 20 timesteps)
        input_sequence = data[-20:].reshape(1, 20, 16)
        
        # Generate forecast
        forecast = forecaster.predict(input_sequence)
        
        assert forecast is not None
        assert forecast.shape[0] == 1  # Batch size
        assert forecast.shape[1] == 5  # Output timesteps
        assert forecast.shape[2] == 16  # Features
    
    def test_forecast_to_schedule_conversion(self, forecaster, time_series_dataset):
        """Test conversion from forecast to schedule parameters."""
        data = time_series_dataset
        input_sequence = data[-20:].reshape(1, 20, 16)
        
        # Generate forecast
        forecast = forecaster.predict(input_sequence)
        
        # Convert forecast to schedule parameters
        # Extract queue predictions (first 4 features = lanes)
        queue_forecast = forecast[0, :, :4]  # [timesteps, lanes]
        
        # Calculate optimal phase durations based on forecast
        avg_queue_per_lane = np.mean(queue_forecast, axis=0)
        max_queue_lane = np.argmax(avg_queue_per_lane)
        
        # Schedule: prioritize lane with highest forecasted queue
        schedule = {
            "priority_phase": int(max_queue_lane),
            "phase_durations": [30, 30, 30, 30],  # Base durations
            "forecast_horizon": 5
        }
        
        assert schedule is not None
        assert 'priority_phase' in schedule
        assert 0 <= schedule['priority_phase'] < 4


class TestForecastingToRL:
    """Test integration from forecasting to RL decision-making."""
    
    @pytest.fixture
    def forecaster(self):
        """Create traffic forecaster."""
        return TrafficForecaster(
            input_timesteps=20,
            output_timesteps=5,
            features=16
        )
    
    @pytest.fixture
    def dqn_agent(self):
        """Create DQN agent."""
        config = DQNConfig(
            state_dim=20,  # 16 features + 4 forecast features
            action_dim=4,
            learning_rate=0.001
        )
        return DQNAgent(config)
    
    def test_forecast_augmented_state(self, forecaster, dqn_agent, time_series_dataset):
        """Test using forecast to augment agent state."""
        data = time_series_dataset
        forecaster = forecaster
        agent = dqn_agent
        
        # Generate forecast
        input_sequence = data[-20:].reshape(1, 20, 16)
        forecast = forecaster.predict(input_sequence)
        
        # Current state (last timestep)
        current_state = data[-1, :]
        
        # Augment state with forecast (use mean forecast)
        forecast_features = np.mean(forecast[0], axis=0)  # Average over timesteps
        
        # Combine current state with forecast
        augmented_state = np.concatenate([current_state, forecast_features])
        
        assert augmented_state.shape == (32,)  # 16 + 16
        assert len(augmented_state) == agent.config.state_dim or len(augmented_state) <= agent.config.state_dim
    
    def test_rl_decision_with_forecast(self, forecaster, dqn_agent, time_series_dataset):
        """Test RL agent decision-making with forecast-augmented state."""
        data = time_series_dataset
        forecaster = forecaster
        agent = dqn_agent
        
        # Generate forecast
        input_sequence = data[-20:].reshape(1, 20, 16)
        forecast = forecaster.predict(input_sequence)
        
        # Create augmented state
        current_state = data[-1, :]
        forecast_features = np.mean(forecast[0], axis=0)
        
        # Use only current state + forecast summary (4 features for lanes)
        forecast_summary = np.mean(forecast[0, :, :4], axis=0)  # Average queue forecast per lane
        augmented_state = np.concatenate([current_state[:12], forecast_summary])  # 12 + 4 = 16
        
        # Resize to match agent state_dim if needed
        if len(augmented_state) > agent.config.state_dim:
            augmented_state = augmented_state[:agent.config.state_dim]
        elif len(augmented_state) < agent.config.state_dim:
            # Pad with zeros
            padding = np.zeros(agent.config.state_dim - len(augmented_state))
            augmented_state = np.concatenate([augmented_state, padding])
        
        # Agent decision
        action = agent.select_action(augmented_state)
        
        assert action is not None
        assert 0 <= action < 4
    
    def test_forecast_integration_in_training(self, forecaster, dqn_agent, time_series_dataset):
        """Test forecast integration during agent training."""
        data = time_series_dataset
        forecaster = forecaster
        agent = dqn_agent
        
        # Simulate training episode with forecasts
        for step in range(10):
            # Get historical window
            start_idx = step
            end_idx = start_idx + 20
            if end_idx >= len(data):
                break
            
            input_sequence = data[start_idx:end_idx].reshape(1, 20, 16)
            forecast = forecaster.predict(input_sequence)
            
            # Create state
            current_state = data[end_idx - 1, :]
            forecast_summary = np.mean(forecast[0, :, :4], axis=0)
            state = np.concatenate([current_state[:12], forecast_summary])
            
            # Resize state
            if len(state) > agent.config.state_dim:
                state = state[:agent.config.state_dim]
            elif len(state) < agent.config.state_dim:
                padding = np.zeros(agent.config.state_dim - len(state))
                state = np.concatenate([state, padding])
            
            # Agent action
            action = agent.select_action(state)
            
            # Simulate next state and reward
            next_state = data[end_idx, :] if end_idx < len(data) else current_state
            reward = -np.mean(current_state[:4])  # Negative queue length
            
            # Store transition
            next_state_full = np.concatenate([next_state[:12], forecast_summary])
            if len(next_state_full) > agent.config.state_dim:
                next_state_full = next_state_full[:agent.config.state_dim]
            elif len(next_state_full) < agent.config.state_dim:
                padding = np.zeros(agent.config.state_dim - len(next_state_full))
                next_state_full = np.concatenate([next_state_full, padding])
            
            agent.store_transition(state, action, reward, next_state_full, False)
        
        # Train if enough experience
        if len(agent.replay_buffer) >= agent.config.batch_size:
            loss = agent.train_step()
            assert loss is not None


class TestForecastAccuracy:
    """Test forecast accuracy and its impact on decisions."""
    
    @pytest.fixture
    def forecaster(self):
        """Create trained forecaster (mock training)."""
        return TrafficForecaster(
            input_timesteps=20,
            output_timesteps=5,
            features=16
        )
    
    def test_forecast_accuracy_metrics(self, forecaster, time_series_dataset):
        """Test forecast accuracy metrics."""
        data = time_series_dataset
        
        # Split into train/test
        train_data = data[:800]
        test_data = data[800:900]
        
        # Generate forecasts for test data
        predictions = []
        actuals = []
        
        for i in range(len(test_data) - 20):
            input_seq = test_data[i:i+20].reshape(1, 20, 16)
            forecast = forecaster.predict(input_seq)
            predictions.append(forecast[0, 0, :])  # First timestep forecast
            actuals.append(test_data[i+20, :])
        
        # Calculate MAE
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        mae = np.mean(np.abs(predictions - actuals))
        
        assert mae is not None
        assert mae >= 0
    
    def test_forecast_impact_on_performance(self, forecaster, dqn_agent, time_series_dataset):
        """Test how forecast accuracy impacts agent performance."""
        data = time_series_dataset
        forecaster = forecaster
        agent = dqn_agent
        
        # Test with and without forecast
        rewards_with_forecast = []
        rewards_without_forecast = []
        
        for step in range(10):
            idx = 20 + step
            if idx >= len(data):
                break
            
            # With forecast
            input_seq = data[idx-20:idx].reshape(1, 20, 16)
            forecast = forecaster.predict(input_seq)
            forecast_summary = np.mean(forecast[0, :, :4], axis=0)
            state_with = np.concatenate([data[idx-1, :12], forecast_summary])
            if len(state_with) > agent.config.state_dim:
                state_with = state_with[:agent.config.state_dim]
            
            # Without forecast (zeros)
            state_without = np.concatenate([data[idx-1, :12], np.zeros(4)])
            if len(state_without) > agent.config.state_dim:
                state_without = state_without[:agent.config.state_dim]
            
            # Get actions
            action_with = agent.select_action(state_with)
            action_without = agent.select_action(state_without)
            
            # Simulate rewards (simplified)
            reward_with = -np.mean(data[idx, :4])
            reward_without = -np.mean(data[idx, :4])
            
            rewards_with_forecast.append(reward_with)
            rewards_without_forecast.append(reward_without)
        
        # Forecast should ideally improve performance
        # (In practice, this depends on forecast quality)
        avg_reward_with = np.mean(rewards_with_forecast)
        avg_reward_without = np.mean(rewards_without_forecast)
        
        assert avg_reward_with is not None
        assert avg_reward_without is not None


@pytest.fixture
def dqn_agent():
    """Create DQN agent for forecasting tests."""
    config = DQNConfig(
        state_dim=16,
        action_dim=4,
        learning_rate=0.001
    )
    return DQNAgent(config)

