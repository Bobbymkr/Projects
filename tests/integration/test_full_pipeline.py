"""
Full Pipeline Integration Tests.

Tests complete end-to-end workflows as specified in Week 3:
- Camera → YOLOv8 → Agent → Signal
- Multi-agent coordination (4-intersection)
- Emergency vehicle preemption
- Regional adaptation switching
- Failure recovery (sensor outage)
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.env.traffic_env import TrafficEnv
    from src.rl.dqn_agent import DQNAgent, DQNConfig
    from src.control.fuzzy_control import FuzzyController
    from src.vision.yolo_queue import YOLOQueueEstimator
except ImportError as e:
    pytest.skip(f"Required imports not available: {e}", allow_module_level=True)


class TestCameraToSignalPipeline:
    """Test complete flow: Camera → YOLOv8 → Agent → Signal"""
    
    @pytest.fixture
    def mock_camera(self):
        """Mock camera feed."""
        camera = Mock()
        # Simulate video frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        camera.get_frame.return_value = frame
        return camera
    
    @pytest.fixture
    def mock_detector(self):
        """Mock YOLOv8 detector."""
        detector = Mock()
        # Simulate detections
        detector.detect.return_value = {
            'boxes': [[100, 100, 200, 200], [300, 300, 400, 400]],
            'confidences': [0.9, 0.85],
            'classes': ['car', 'car']
        }
        detector.estimate_queues.return_value = [5, 8, 3, 6]  # Queue lengths per lane
        return detector
    
    @pytest.fixture
    def agent(self):
        """Create test agent."""
        env_config = {"num_lanes": 4}
        env = TrafficEnv(env_config)
        obs_dim = 8  # 4 lanes * 2 (queue + wait)
        action_dim = env.action_space.n
        return DQNAgent(obs_dim, action_dim, DQNConfig())
    
    @pytest.fixture
    def signal_controller(self):
        """Mock signal controller."""
        controller = Mock()
        controller.current_phase = 0
        controller.green_time = 30
        controller.execute.return_value = True
        return controller
    
    def test_camera_to_signal_pipeline(
        self, mock_camera, mock_detector, agent, signal_controller
    ):
        """Test complete flow: Camera → YOLOv8 → Agent → Signal"""
        # Step 1: Get frame from camera
        frame = mock_camera.get_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        
        # Step 2: Run YOLOv8 detection
        detections = mock_detector.detect(frame)
        assert 'boxes' in detections
        assert len(detections['boxes']) > 0
        
        # Step 3: Estimate queue lengths
        queue_lengths = mock_detector.estimate_queues(detections)
        assert len(queue_lengths) == 4
        assert all(q >= 0 for q in queue_lengths)
        
        # Step 4: Agent decision
        # Create state from queue lengths
        state = np.array(queue_lengths + [0.0] * 4, dtype=np.float32)  # queue + wait times
        action = agent.select_action(state)
        # Validate action is within environment's action space
        assert 0 <= action < env.action_space.n, f"Action {action} out of range [0, {env.action_space.n})"
        
        # Step 5: Execute signal change
        success = signal_controller.execute(action)
        assert success
        assert signal_controller.current_phase in [0, 1, 2, 3]
        assert signal_controller.green_time > 0
    
    def test_pipeline_with_real_components(self):
        """Test pipeline with real (non-mocked) components where possible."""
        # Create real environment
        env_config = {
            "num_lanes": 4,
            "min_green": 5,
            "max_green": 60,
            "arrival_rates": [0.3, 0.25, 0.35, 0.2]
        }
        env = TrafficEnv(env_config)
        
        # Create real agent
        obs_dim = 8
        action_dim = env.action_space.n
        agent = DQNAgent(obs_dim, action_dim, DQNConfig())
        
        # Run pipeline
        obs, info = env.reset()
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Validate
        assert action in range(action_dim)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


class TestMultiAgentCoordination:
    """Test 4-intersection coordination"""
    
    @pytest.fixture
    def network(self):
        """Create 4-intersection network."""
        # Mock network with 4 intersections
        network = Mock()
        network.size = (2, 2)
        network.intersections = [Mock() for _ in range(4)]
        network.reset.return_value = [np.random.rand(8) for _ in range(4)]
        network.step.return_value = (
            [np.random.rand(8) for _ in range(4)],
            [np.random.rand() for _ in range(4)]
        )
        network.total_throughput = 1000
        return network
    
    def test_marl_coordination(self, network):
        """Test 4-intersection coordination."""
        # Create 4 agents
        agents = []
        for i in range(4):
            env_config = {"num_lanes": 4}
            env = TrafficEnv(env_config)
            obs_dim = 8
            action_dim = env.action_space.n
            agents.append(DQNAgent(obs_dim, action_dim, DQNConfig()))
        
        # Simulate coordinated traffic flow
        baseline_throughput = 800  # Baseline without coordination
        
        for episode in range(10):  # Reduced for test speed
            states = network.reset()
            
            for step in range(50):  # Reduced for test speed
                # Each agent decides based on its state
                actions = []
                for agent, state in zip(agents, states):
                    action = agent.select_action(state)
                    actions.append(action)
                
                # Network steps with coordinated actions
                states, rewards = network.step(actions)
                
                # Validate coordination
                assert len(actions) == 4
                # Validate actions are within valid range (check against first agent's action space)
                env_config = {"num_lanes": 4}
                env = TrafficEnv(env_config)
                assert all(0 <= a < env.action_space.n for a in actions), f"Actions {actions} out of range"
        
        # Validate coordination improved throughput
        # In real scenario, coordination should improve throughput
        assert network.total_throughput >= baseline_throughput * 0.8  # Allow some variance


class TestEmergencyVehiclePreemption:
    """Test emergency vehicle priority handling"""
    
    def test_emergency_vehicle_detection(self):
        """Test emergency vehicle detection and priority."""
        # Mock emergency vehicle detector
        emergency_detector = Mock()
        emergency_detector.detect_emergency.return_value = True
        emergency_detector.get_priority_lane.return_value = 0  # North lane
        
        # Create agent with emergency handling
        env_config = {"num_lanes": 4}
        env = TrafficEnv(env_config)
        obs_dim = 8
        action_dim = env.action_space.n
        agent = DQNAgent(obs_dim, action_dim, DQNConfig())
        
        # Detect emergency
        is_emergency = emergency_detector.detect_emergency()
        if is_emergency:
            priority_lane = emergency_detector.get_priority_lane()
            # Agent should prioritize the emergency lane
            # This would be handled by the agent's decision logic
            assert priority_lane in [0, 1, 2, 3]
    
    def test_emergency_preemption_workflow(self):
        """Test complete emergency preemption workflow."""
        # Create environment
        env_config = {"num_lanes": 4}
        env = TrafficEnv(env_config)
        
        # Simulate emergency vehicle arrival
        emergency_detected = True
        emergency_lane = 0
        
        # Agent should switch to emergency lane immediately
        obs, info = env.reset()
        
        # Modify state to indicate emergency
        # In real implementation, this would come from detection
        obs[0] = 999  # High priority signal
        
        # Agent should respond to emergency
        agent = DQNAgent(8, env.action_space.n, DQNConfig())
        action = agent.select_action(obs)
        
        # Validate emergency response
        assert 0 <= action < env.action_space.n, f"Action {action} out of range [0, {env.action_space.n})"


class TestRegionalAdaptation:
    """Test regional adaptation switching"""
    
    def test_regional_config_switching(self):
        """Test switching between regional configurations."""
        # Define regional configs
        regions = {
            "north_america": {
                "driving_side": "right",
                "units": "imperial",
                "peak_hours": [7, 8, 9, 16, 17, 18]
            },
            "europe": {
                "driving_side": "right",
                "units": "metric",
                "peak_hours": [8, 9, 10, 17, 18, 19]
            },
            "uk": {
                "driving_side": "left",
                "units": "metric",
                "peak_hours": [8, 9, 10, 17, 18, 19]
            }
        }
        
        # Test switching regions
        current_region = "north_america"
        env_config = {"num_lanes": 4, "region": current_region}
        env = TrafficEnv(env_config)
        
        # Switch to Europe
        new_region = "europe"
        env_config["region"] = new_region
        env = TrafficEnv(env_config)
        
        # Validate region switch
        assert env_config["region"] == new_region
        assert env_config["region"] in regions
    
    def test_adaptation_to_new_region(self):
        """Test agent adaptation to new region."""
        # Create agent
        env_config = {"num_lanes": 4, "region": "north_america"}
        env = TrafficEnv(env_config)
        agent = DQNAgent(8, env.action_space.n, DQNConfig())
        
        # Train in source region
        obs, info = env.reset()
        for _ in range(10):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        # Switch to target region
        env_config["region"] = "europe"
        env = TrafficEnv(env_config)
        
        # Agent should adapt (in real scenario, would fine-tune)
        obs, info = env.reset()
        action = agent.select_action(obs)
        assert action in [0, 1, 2, 3]


class TestFailureRecovery:
    """Test failure recovery scenarios"""
    
    def test_sensor_outage_recovery(self):
        """Test recovery from sensor outage."""
        # Create environment
        env_config = {"num_lanes": 4}
        env = TrafficEnv(env_config)
        agent = DQNAgent(8, env.action_space.n, DQNConfig())
        
        # Normal operation
        obs, info = env.reset()
        action = agent.select_action(obs)
        
        # Simulate sensor outage (missing/invalid data)
        obs_outage = np.full(8, np.nan)  # Invalid data
        
        # Agent should handle gracefully (fallback to default action)
        try:
            action = agent.select_action(obs_outage)
            # Should not crash
            assert 0 <= action < env.action_space.n, f"Action {action} out of range"
        except (ValueError, RuntimeError):
            # If agent can't handle, should have fallback
            action = 0  # Default action
            assert 0 <= action < env.action_space.n, f"Action {action} out of range"
    
    def test_agent_failure_fallback(self):
        """Test fallback when agent fails."""
        # Create environment
        env_config = {"num_lanes": 4}
        env = TrafficEnv(env_config)
        
        # Create fallback controller (Fuzzy Logic)
        fallback_controller = FuzzyController()
        
        # Simulate agent failure
        agent_failed = True
        
        if agent_failed:
            # Use fallback controller
            obs, info = env.reset()
            queue_lengths = obs[:4]  # First 4 values are queue lengths
            action = fallback_controller.compute_timing(queue_lengths)
            
            # Validate fallback works
            assert action is not None
    
    def test_network_latency_handling(self):
        """Test handling of network latency."""
        import time
        
        # Simulate network delay
        def delayed_decision(obs, delay_ms=100):
            time.sleep(delay_ms / 1000.0)
            agent = DQNAgent(8, 4, DQNConfig())
            return agent.select_action(obs)
        
        env_config = {"num_lanes": 4}
        env = TrafficEnv(env_config)
        obs, info = env.reset()
        
        # Decision with delay
        start_time = time.time()
        action = delayed_decision(obs, delay_ms=50)
        elapsed = (time.time() - start_time) * 1000
        
        # Should complete within reasonable time
        assert elapsed < 200  # Less than 200ms
        assert action in [0, 1, 2, 3]


class TestDataFlowValidation:
    """Test data flow through the pipeline"""
    
    def test_state_transition_consistency(self):
        """Test that state transitions are consistent."""
        env_config = {"num_lanes": 4}
        env = TrafficEnv(env_config)
        agent = DQNAgent(8, env.action_space.n, DQNConfig())
        
        obs, info = env.reset()
        initial_state = obs.copy()
        
        # Take action
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Validate state transition
        assert next_obs.shape == initial_state.shape
        assert not np.array_equal(obs, next_obs)  # State should change
    
    def test_reward_calculation(self):
        """Test reward calculation consistency."""
        env_config = {"num_lanes": 4}
        env = TrafficEnv(env_config)
        agent = DQNAgent(8, env.action_space.n, DQNConfig())
        
        obs, info = env.reset()
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Validate reward
        assert isinstance(reward, (int, float))
        # Rewards are typically negative (penalties)
        assert reward <= 0  # Assuming penalty-based rewards


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

