# 📚 API Reference Documentation

This comprehensive API reference provides detailed documentation for all public interfaces in the Adaptive Traffic Control System.

## 📋 **Table of Contents**

- [Core Environment APIs](#core-environment-apis)
- [Reinforcement Learning APIs](#reinforcement-learning-apis)
- [Computer Vision APIs](#computer-vision-apis)
- [Traffic Forecasting APIs](#traffic-forecasting-apis)
- [Control System APIs](#control-system-apis)
- [Utility APIs](#utility-apis)

---

## 🌍 **Core Environment APIs**

### TrafficEnv

The foundational traffic intersection environment for reinforcement learning.

```python
class TrafficEnv(gym.Env):
    """
    Traffic environment simulating a single intersection with multiple lanes and two phases.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing:
            - num_lanes (int): Number of approach lanes (default: 4)
            - phase_lanes (List[List[int]]): Lane groupings per phase
            - min_green (int): Minimum green duration in seconds (default: 5)
            - max_green (int): Maximum green duration in seconds (default: 60)
            - green_step (int): Green duration increment (default: 5)
            - arrival_rates (List[float]): Vehicle arrival rates per lane
            - queue_capacity (int): Maximum queue length per lane (default: 40)
            - reward_weights (Dict): Reward function weights
    
    Returns:
        TrafficEnv: Configured traffic environment instance
    """
```

#### Methods

##### `reset(seed=None, options=None)`
```python
def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]
```
**Description**: Reset the environment to initial state.

**Parameters**:
- `seed` (int, optional): Random seed for reproducibility
- `options` (Dict, optional): Additional reset options

**Returns**:
- `observation` (np.ndarray): Normalized queue lengths [0, 1]
- `info` (Dict): Environment state information

**Example**:
```python
env = TrafficEnv(config)
obs, info = env.reset(seed=42)
print(f"Initial queues: {info['queues']}")
```

##### `step(action)`
```python
def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]
```
**Description**: Execute one simulation step with the given action.

**Parameters**:
- `action` (int): Green light duration index (0 to len(green_values)-1)

**Returns**:
- `observation` (np.ndarray): Next state observation
- `reward` (float): Reward for the action taken
- `terminated` (bool): Whether episode ended naturally
- `truncated` (bool): Whether episode was truncated
- `info` (Dict): Step information and statistics

**Example**:
```python
action = 5  # Select action index
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}, Queues: {info['queues']}")
```

### SumoEnv

SUMO-based realistic traffic simulation environment.

```python
class SumoEnv(gym.Env):
    """
    SUMO-based traffic environment with TraCI communication.
    
    Args:
        config_file (str): Path to SUMO configuration file
        use_gui (bool): Whether to use SUMO GUI (default: False)
        step_length (float): Simulation step length in seconds (default: 1.0)
        
    Features:
        - Realistic vehicle dynamics
        - Complex road networks
        - Traffic light control via TraCI
        - Comprehensive traffic metrics
    """
```

### MarlEnv

Multi-Agent Reinforcement Learning environment for multiple intersections.

```python
class MarlEnv(gym.Env):
    """
    Multi-agent environment for coordinated intersection control.
    
    Args:
        num_agents (int): Number of traffic agents/intersections
        config (Dict): Base configuration for each intersection
        communication_radius (float): Agent communication range
        enable_forecasting (bool): Enable traffic forecasting integration
        
    Features:
        - Multi-agent coordination
        - Inter-agent communication
        - Distributed decision making
        - Traffic forecasting integration
    """
```

---

## 🧠 **Reinforcement Learning APIs**

### DQNAgent

Deep Q-Network agent for traffic signal control.

```python
class DQNAgent:
    """
    Deep Q-Network agent for learning optimal traffic signal policies.
    
    Args:
        state_dim (int): Dimension of state space
        action_dim (int): Number of possible actions
        config (DQNConfig): Agent configuration parameters
        
    Architecture:
        - 3-layer fully connected network
        - Experience replay buffer
        - Target network for stability
        - Epsilon-greedy exploration
    """
```

#### Methods

##### `act(state, epsilon=None)`
```python
def act(self, state: np.ndarray, epsilon: float = None) -> int
```
**Description**: Select action using epsilon-greedy policy.

**Parameters**:
- `state` (np.ndarray): Current environment state
- `epsilon` (float, optional): Exploration rate override

**Returns**:
- `action` (int): Selected action index

**Example**:
```python
agent = DQNAgent(state_dim=4, action_dim=12, config=config)
action = agent.act(state, epsilon=0.1)
```

##### `learn(experiences)`
```python
def learn(self, experiences: List[Tuple]) -> float
```
**Description**: Update Q-network using batch of experiences.

**Parameters**:
- `experiences` (List[Tuple]): Batch of (state, action, reward, next_state, done) tuples

**Returns**:
- `loss` (float): Training loss value

### DQNConfig

Configuration class for DQN agent parameters.

```python
@dataclass
class DQNConfig:
    """
    Configuration for DQN agent hyperparameters.
    
    Attributes:
        lr (float): Learning rate (default: 0.001)
        gamma (float): Discount factor (default: 0.99)
        epsilon_start (float): Initial exploration rate (default: 1.0)
        epsilon_end (float): Final exploration rate (default: 0.01)
        epsilon_decay (float): Exploration decay rate (default: 0.995)
        batch_size (int): Training batch size (default: 32)
        buffer_size (int): Replay buffer capacity (default: 100000)
        target_update_freq (int): Target network update frequency (default: 100)
        tau (float): Soft update parameter (default: 0.005)
    """
```

---

## 👁️ **Computer Vision APIs**

### YOLOQueueEstimator

YOLO-based vehicle detection and queue length estimation.

```python
class YOLOQueueEstimator:
    """
    YOLOv8-based vehicle detection and queue length estimation.
    
    Args:
        model_path (str): Path to YOLO model weights
        confidence_threshold (float): Detection confidence threshold (default: 0.5)
        device (str): Computation device ('cpu' or 'cuda')
        
    Features:
        - Real-time vehicle detection
        - ROI-based queue estimation
        - Multiple vehicle class support
        - Optimized inference pipeline
    """
```

#### Methods

##### `detect_vehicles(frame, rois)`
```python
def detect_vehicles(self, frame: np.ndarray, rois: List[Dict]) -> Dict[str, int]
```
**Description**: Detect vehicles in frame and estimate queue lengths per ROI.

**Parameters**:
- `frame` (np.ndarray): Input video frame
- `rois` (List[Dict]): List of region-of-interest definitions

**Returns**:
- `queue_counts` (Dict[str, int]): Vehicle counts per ROI

**Example**:
```python
estimator = YOLOQueueEstimator("yolov8n.pt")
rois = [{"name": "lane_1", "polygon": [[x1,y1], [x2,y2], ...]}]
counts = estimator.detect_vehicles(frame, rois)
print(f"Lane 1 queue: {counts['lane_1']} vehicles")
```

### VideoPipeline

Complete video processing pipeline for traffic monitoring.

```python
class VideoPipeline:
    """
    End-to-end video processing pipeline for traffic analysis.
    
    Args:
        source (Union[int, str]): Video source (webcam index, file path, URL)
        roi_config (Dict): ROI configuration for lane detection
        output_path (str, optional): Path for output video saving
        
    Features:
        - Multiple video source support
        - Real-time processing
        - ROI management
        - Output recording
        - Performance monitoring
    """
```

---

## 🔮 **Traffic Forecasting APIs**

### TrafficForecaster

CNN-LSTM hybrid model for traffic volume prediction.

```python
class TrafficForecaster:
    """
    CNN-LSTM hybrid model for predicting future traffic conditions.
    
    Args:
        input_timesteps (int): Number of historical timesteps (default: 10)
        output_timesteps (int): Number of prediction timesteps (default: 5)
        features (int): Number of features per timestep (default: 4)
        lstm_units (int): LSTM layer units (default: 64)
        cnn_filters (int): CNN filter count (default: 16)
        cnn_kernel (int): CNN kernel size (default: 3)
        
    Architecture:
        - CNN layer for spatial feature extraction
        - LSTM layer for temporal modeling
        - Dense output layer for predictions
    """
```

#### Methods

##### `train(X_train, y_train, validation_data=None)`
```python
def train(self, X_train: np.ndarray, y_train: np.ndarray, 
          validation_data: Tuple = None, epochs: int = 100) -> Dict
```
**Description**: Train the forecasting model on traffic data.

**Parameters**:
- `X_train` (np.ndarray): Training input data (samples, timesteps, features)
- `y_train` (np.ndarray): Training target data (samples, output_timesteps, features)
- `validation_data` (Tuple, optional): Validation data tuple (X_val, y_val)
- `epochs` (int): Number of training epochs

**Returns**:
- `history` (Dict): Training history with loss and metrics

##### `predict(X)`
```python
def predict(self, X: np.ndarray) -> np.ndarray
```
**Description**: Generate traffic predictions for input sequences.

**Parameters**:
- `X` (np.ndarray): Input data (samples, timesteps, features)

**Returns**:
- `predictions` (np.ndarray): Predicted traffic volumes

**Example**:
```python
forecaster = TrafficForecaster(input_timesteps=20, output_timesteps=5)
forecaster.train(X_train, y_train, epochs=50)
predictions = forecaster.predict(X_test)
```

---

## ⚙️ **Control System APIs**

### FuzzyController

Fuzzy logic-based traffic signal controller.

```python
class FuzzyController:
    """
    Fuzzy logic controller for traffic signal timing.
    
    Args:
        config (Dict): Controller configuration with membership functions
        
    Features:
        - Linguistic rule-based control
        - Interpretable decision making
        - Configurable membership functions
        - Real-time inference
    """
```

#### Methods

##### `control(queue_lengths, wait_times)`
```python
def control(self, queue_lengths: List[float], wait_times: List[float]) -> int
```
**Description**: Determine optimal green duration using fuzzy logic.

**Parameters**:
- `queue_lengths` (List[float]): Current queue lengths per lane
- `wait_times` (List[float]): Average wait times per lane

**Returns**:
- `green_duration` (int): Recommended green light duration in seconds

### WebsterController

Classical Webster's method for signal timing optimization.

```python
class WebsterController:
    """
    Webster's method for calculating optimal signal timing.
    
    Args:
        saturation_flows (List[float]): Saturation flow rates per lane
        lost_time (float): Lost time per cycle (default: 4.0)
        
    Features:
        - Classical traffic engineering approach
        - Optimal cycle length calculation
        - Phase timing optimization
        - Capacity-based timing
    """
```

---

## 🛠️ **Utility APIs**

### ConfigManager

Configuration management and validation.

```python
class ConfigManager:
    """
    Centralized configuration management with validation.
    
    Features:
        - JSON/YAML configuration loading
        - Pydantic-based validation
        - Environment variable support
        - Configuration merging
        - Default value handling
    """
```

#### Methods

##### `load_config(path, validate=True)`
```python
def load_config(self, path: str, validate: bool = True) -> Dict
```
**Description**: Load and validate configuration from file.

**Parameters**:
- `path` (str): Path to configuration file
- `validate` (bool): Whether to validate configuration schema

**Returns**:
- `config` (Dict): Loaded and validated configuration

### MetricsCollector

Performance metrics collection and analysis.

```python
class MetricsCollector:
    """
    Comprehensive metrics collection for system monitoring.
    
    Features:
        - Real-time performance tracking
        - Statistical analysis
        - Export capabilities
        - Visualization support
    """
```

#### Methods

##### `record_episode(episode_data)`
```python
def record_episode(self, episode_data: Dict) -> None
```
**Description**: Record episode performance data.

**Parameters**:
- `episode_data` (Dict): Episode metrics including rewards, queue lengths, wait times

##### `get_statistics(window=None)`
```python
def get_statistics(self, window: int = None) -> Dict
```
**Description**: Calculate performance statistics.

**Parameters**:
- `window` (int, optional): Rolling window size for statistics

**Returns**:
- `stats` (Dict): Calculated statistics and aggregations

### HealthChecker

System health monitoring and diagnostics.

```python
class HealthChecker:
    """
    System health monitoring and diagnostic utilities.
    
    Features:
        - Component health checks
        - Performance monitoring
        - Resource usage tracking
        - Alert generation
    """
```

---

## 📊 **Usage Examples**

### Complete Training Pipeline

```python
from src.env.traffic_env import TrafficEnv
from src.rl.dqn_agent import DQNAgent, DQNConfig
from src.utils.metrics import MetricsCollector

# 1. Setup environment
config = {
    "num_lanes": 4,
    "phase_lanes": [[0, 1], [2, 3]],
    "min_green": 5,
    "max_green": 60,
    "arrival_rates": [0.3, 0.25, 0.35, 0.2]
}
env = TrafficEnv(config)

# 2. Initialize agent
agent_config = DQNConfig(lr=0.001, gamma=0.99)
agent = DQNAgent(state_dim=4, action_dim=12, config=agent_config)

# 3. Setup metrics
metrics = MetricsCollector()

# 4. Training loop
for episode in range(100):
    obs, info = env.reset()
    episode_reward = 0
    
    while True:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        agent.store_experience(obs, action, reward, next_obs, terminated)
        agent.learn()
        
        episode_reward += reward
        obs = next_obs
        
        if terminated or truncated:
            break
    
    metrics.record_episode({
        "episode": episode,
        "reward": episode_reward,
        "queue_lengths": info["queues"]
    })

# 5. Analyze results
stats = metrics.get_statistics()
print(f"Average reward: {stats['avg_reward']:.2f}")
```

### Real-Time Video Processing

```python
from src.vision.video_pipeline import VideoPipeline
from src.vision.yolo_queue import YOLOQueueEstimator

# 1. Setup video pipeline
pipeline = VideoPipeline(source=0)  # Webcam

# 2. Configure ROIs
rois = [
    {"name": "north_lane", "polygon": [[100, 50], [200, 50], [200, 150], [100, 150]]},
    {"name": "south_lane", "polygon": [[100, 250], [200, 250], [200, 350], [100, 350]]},
]

# 3. Process video stream
estimator = YOLOQueueEstimator("yolov8n.pt")

for frame in pipeline.stream():
    queue_counts = estimator.detect_vehicles(frame, rois)
    print(f"Queue counts: {queue_counts}")
    
    # Use queue counts for traffic control
    # ... integrate with RL agent
```

---

## 🔗 **Related Documentation**

- **[Architecture Guide](../architecture/README.md)**: System architecture overview
- **[User Manual](../user-guide/README.md)**: Comprehensive usage guide
- **[Developer Guide](../developer-guide/README.md)**: Development setup and best practices
- **[Configuration Reference](../configuration/README.md)**: Complete configuration options

---

## 📝 **Notes**

- All APIs follow Google-style docstring conventions
- Type hints are provided for all public methods
- Error handling includes custom exception classes
- Performance considerations are documented for each component
- Thread safety information is included where applicable

For additional API details, please refer to the inline documentation in the source code.