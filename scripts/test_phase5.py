"""
Test script for Phase 5: Ensemble Methods.

Tests Intelligent Ensemble and Meta-Learning.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.intelligent_ensemble import (
    IntelligentEnsemble, EnsembleConfig,
    PerformanceTracker, ContextAwareSelector,
    ConfidenceBasedWeighting, AdaptiveWeightLearner, MetaLearner
)


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name: str, state_dim: int = 8, action_dim: int = 4):
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bias = np.random.randint(0, action_dim)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action."""
        # Simple deterministic action based on state
        return (np.sum(state) + self.bias) % self.action_dim
    
    def get_confidence(self, state: np.ndarray) -> float:
        """Get confidence."""
        return np.random.uniform(0.5, 1.0)


def test_performance_tracker():
    """Test Phase 5.1: Performance Tracker."""
    print("="*80)
    print("Testing Phase 5.1: Performance Tracker")
    print("="*80)
    
    tracker = PerformanceTracker(window_size=100)
    
    # Register agents
    agents = ["Transformer", "DQN", "Hierarchical RL"]
    for agent in agents:
        tracker.register_agent(agent)
    
    # Update performance
    for i in range(50):
        tracker.update("Transformer", reward=10.0 + i * 0.1, performance=0.8, stability=0.9)
        tracker.update("DQN", reward=8.0 + i * 0.05, performance=0.7, stability=0.8)
        tracker.update("Hierarchical RL", reward=9.0 + i * 0.08, performance=0.75, stability=0.85)
    
    # Get metrics
    print(f"\n📊 Performance Metrics:")
    for agent in agents:
        metrics = tracker.get_performance_metrics(agent)
        print(f"   {agent}:")
        print(f"     Mean Reward: {metrics['mean_reward']:.2f}")
        print(f"     Mean Performance: {metrics['mean_performance']:.2f}")
        print(f"     Stability: {metrics['stability']:.2f}")
        print(f"     Variance: {metrics['variance']:.4f}")
    
    # Calculate weights
    weights = tracker.calculate_weights()
    print(f"\n📊 Calculated Weights:")
    for agent, weight in weights.items():
        print(f"   {agent}: {weight:.4f}")
    
    print(f"\n✅ Performance Tracker Test Complete!")
    return True


def test_context_aware_selector():
    """Test Phase 5.1: Context-Aware Selector."""
    print("\n" + "="*80)
    print("Testing Phase 5.1: Context-Aware Selector")
    print("="*80)
    
    selector = ContextAwareSelector()
    available_agents = ["Transformer", "DQN", "Hierarchical RL", "Model-Based RL", "GNN"]
    
    # Test different contexts
    contexts = [
        {"traffic_density": 0.9, "hour": 8, "emergency": False},  # Rush hour
        {"traffic_density": 0.2, "hour": 2, "emergency": False},  # Night
        {"traffic_density": 0.8, "hour": 12, "emergency": False},  # High traffic
        {"traffic_density": 0.3, "hour": 12, "emergency": False},  # Low traffic
        {"traffic_density": 0.5, "hour": 12, "emergency": True},  # Emergency
        {"traffic_density": 0.5, "hour": 12, "emergency": False},  # Normal
    ]
    
    print(f"\n📊 Context-Aware Selection:")
    for context in contexts:
        selected = selector.select_agents(context, available_agents)
        context_type = selector._determine_context(context)
        print(f"   Context: {context_type}")
        print(f"   Selected agents: {selected}")
    
    print(f"\n✅ Context-Aware Selector Test Complete!")
    return True


def test_confidence_based_weighting():
    """Test Phase 5.1: Confidence-Based Weighting."""
    print("\n" + "="*80)
    print("Testing Phase 5.1: Confidence-Based Weighting")
    print("="*80)
    
    weighting = ConfidenceBasedWeighting()
    
    base_weights = {
        "Transformer": 0.35,
        "DQN": 0.30,
        "Hierarchical RL": 0.20,
        "Model-Based RL": 0.15
    }
    
    predictions = {
        "Transformer": (2, 0.9),  # (action, confidence)
        "DQN": (2, 0.7),
        "Hierarchical RL": (1, 0.8),
        "Model-Based RL": (2, 0.6)
    }
    
    confidence_weights = weighting.calculate_confidence_weights(predictions, base_weights)
    
    print(f"\n📊 Confidence-Based Weights:")
    print(f"   Base weights: {base_weights}")
    print(f"   Confidence weights:")
    for agent, weight in confidence_weights.items():
        action, conf = predictions[agent]
        print(f"     {agent}: {weight:.4f} (action={action}, confidence={conf:.2f})")
    
    print(f"\n✅ Confidence-Based Weighting Test Complete!")
    return True


def test_adaptive_weight_learner():
    """Test Phase 5.1: Adaptive Weight Learner."""
    print("\n" + "="*80)
    print("Testing Phase 5.1: Adaptive Weight Learner")
    print("="*80)
    
    num_agents = 4
    learner = AdaptiveWeightLearner(num_agents, learning_rate=0.01)
    
    print(f"\n📊 Initial Weights: {learner.get_weights()}")
    
    # Simulate updates
    for i in range(20):
        agent_rewards = [10.0, 8.0, 9.0, 7.0]  # Agent 0 performs best
        ensemble_reward = 9.5
        learner.update_weights(agent_rewards, ensemble_reward)
        
        if (i + 1) % 5 == 0:
            weights = learner.get_weights()
            print(f"   Step {i+1}: {weights}")
    
    final_weights = learner.get_weights()
    print(f"\n📊 Final Weights: {final_weights}")
    print(f"   Best agent (index 0) has highest weight: {final_weights[0] > final_weights[1]}")
    
    print(f"\n✅ Adaptive Weight Learner Test Complete!")
    return True


def test_intelligent_ensemble():
    """Test Phase 5.1: Intelligent Ensemble."""
    print("\n" + "="*80)
    print("Testing Phase 5.1: Intelligent Ensemble")
    print("="*80)
    
    # Create mock agents
    agents = {
        "Transformer": MockAgent("Transformer"),
        "DQN": MockAgent("DQN"),
        "Hierarchical RL": MockAgent("Hierarchical RL")
    }
    
    # Test weighted voting
    print(f"\n📊 Testing Weighted Voting...")
    config = EnsembleConfig(method="weighted_voting", performance_weights={
        "Transformer": 0.4,
        "DQN": 0.3,
        "Hierarchical RL": 0.3
    })
    ensemble = IntelligentEnsemble(agents, config)
    
    state = np.random.randn(8)
    action = ensemble.select_action(state)
    weights = ensemble.get_weights()
    print(f"   State: {state[:3]}...")
    print(f"   Selected action: {action}")
    print(f"   Weights: {weights}")
    
    # Test dynamic ensemble
    print(f"\n📊 Testing Dynamic Ensemble...")
    config = EnsembleConfig(
        method="dynamic",
        context_aware=True,
        confidence_based=True,
        adaptive=True
    )
    ensemble = IntelligentEnsemble(agents, config)
    
    context = {"traffic_density": 0.8, "hour": 8, "emergency": False}
    action = ensemble.select_action(state, context)
    print(f"   Context: rush_hour")
    print(f"   Selected action: {action}")
    
    # Update performance
    ensemble.update_performance("Transformer", reward=10.0, performance=0.9, stability=0.95)
    ensemble.update_performance("DQN", reward=8.0, performance=0.7, stability=0.8)
    ensemble.update_performance("Hierarchical RL", reward=9.0, performance=0.8, stability=0.85)
    
    updated_weights = ensemble.get_weights()
    print(f"   Updated weights: {updated_weights}")
    
    # Test confidence-based
    print(f"\n📊 Testing Confidence-Based Ensemble...")
    config = EnsembleConfig(method="confidence_based", confidence_based=True)
    ensemble = IntelligentEnsemble(agents, config)
    action = ensemble.select_action(state)
    print(f"   Selected action: {action}")
    
    print(f"\n✅ Intelligent Ensemble Test Complete!")
    return True


def test_meta_learner():
    """Test Phase 5.2: Meta-Learning (Stacking)."""
    print("\n" + "="*80)
    print("Testing Phase 5.2: Meta-Learning (Stacking)")
    print("="*80)
    
    num_agents = 3
    state_dim = 8
    meta_learner = MetaLearner(num_agents, state_dim, hidden_dim=32)
    
    # Test forward pass
    state = torch.FloatTensor(np.random.randn(2, state_dim))
    agent_predictions = torch.zeros(2, num_agents, 4)
    agent_predictions[0, 0, 2] = 0.9  # Agent 0 predicts action 2
    agent_predictions[0, 1, 1] = 0.8  # Agent 1 predicts action 1
    agent_predictions[0, 2, 2] = 0.7  # Agent 2 predicts action 2
    
    weights = meta_learner(state, agent_predictions)
    print(f"\n📊 Meta-Learner Output:")
    print(f"   Input state shape: {state.shape}")
    print(f"   Agent predictions shape: {agent_predictions.shape}")
    print(f"   Ensemble weights shape: {weights.shape}")
    print(f"   Ensemble weights (batch 0): {weights[0].detach().numpy()}")
    print(f"   Weights sum: {weights[0].sum().item():.4f} (should be ~1.0)")
    
    # Test with ensemble
    print(f"\n📊 Testing Stacking Ensemble...")
    agents = {
        "Transformer": MockAgent("Transformer"),
        "DQN": MockAgent("DQN"),
        "Hierarchical RL": MockAgent("Hierarchical RL")
    }
    
    config = EnsembleConfig(method="stacking", meta_learner_dim=32)
    ensemble = IntelligentEnsemble(agents, config)
    
    state = np.random.randn(8)
    action = ensemble.select_action(state)
    print(f"   Selected action: {action}")
    
    print(f"\n✅ Meta-Learning Test Complete!")
    return True


def test_combined():
    """Test Phase 5: Combined."""
    print("\n" + "="*80)
    print("Testing Phase 5: Combined")
    print("="*80)
    
    # Create ensemble with all features
    agents = {
        "Transformer": MockAgent("Transformer"),
        "DQN": MockAgent("DQN"),
        "Hierarchical RL": MockAgent("Hierarchical RL"),
        "Model-Based RL": MockAgent("Model-Based RL")
    }
    
    config = EnsembleConfig(
        method="dynamic",
        context_aware=True,
        confidence_based=True,
        adaptive=True,
        performance_weights={
            "Transformer": 0.35,
            "DQN": 0.30,
            "Hierarchical RL": 0.20,
            "Model-Based RL": 0.15
        }
    )
    
    ensemble = IntelligentEnsemble(agents, config)
    
    # Test with different contexts
    contexts = [
        {"traffic_density": 0.9, "hour": 8, "emergency": False},  # Rush hour
        {"traffic_density": 0.2, "hour": 2, "emergency": False},  # Night
        {"traffic_density": 0.5, "hour": 12, "emergency": True},  # Emergency
    ]
    
    state = np.random.randn(8)
    
    print(f"\n📊 Testing Combined Ensemble:")
    for context in contexts:
        action = ensemble.select_action(state, context)
        weights = ensemble.get_weights()
        context_type = ensemble.context_selector._determine_context(context)
        print(f"   Context: {context_type}")
        print(f"   Action: {action}")
        print(f"   Weights: {dict(list(weights.items())[:2])}...")
    
    # Update performance
    ensemble.update_performance("Transformer", reward=12.0, performance=0.95, stability=0.98)
    ensemble.update_performance("DQN", reward=9.0, performance=0.75, stability=0.85)
    
    final_weights = ensemble.get_weights()
    print(f"\n   Final weights after updates: {final_weights}")
    
    print(f"\n✅ Combined Test Complete!")
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("Phase 5 Complete Test Suite")
    print("="*80)
    
    # Test Phase 5.1 components
    test_performance_tracker()
    test_context_aware_selector()
    test_confidence_based_weighting()
    test_adaptive_weight_learner()
    test_intelligent_ensemble()
    
    # Test Phase 5.2
    test_meta_learner()
    
    # Test Combined
    test_combined()
    
    print("\n" + "="*80)
    print("All Tests Complete!")
    print("="*80)

if __name__ == "__main__":
    import torch
    main()

