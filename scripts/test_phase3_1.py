"""
Test script for Phase 3.1: Graph Neural Networks (GNN) for Multi-Intersection.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.gnn_agent import GNNAgent, GNNConfig, build_intersection_graph

def test_gnn_agent():
    """Test GNN agent for multi-intersection control."""
    print("="*80)
    print("Testing Phase 3.1: Graph Neural Networks (GNN) for Multi-Intersection")
    print("="*80)
    
    # Configuration
    num_intersections = 4
    state_dim = 8  # Per intersection
    action_dim = 4  # Per intersection
    
    # Build adjacency matrix (2x2 grid)
    adj_matrix = build_intersection_graph(num_intersections, topology="grid")
    print(f"\n📊 Intersection Graph (2x2 Grid):")
    print(f"   Adjacency Matrix:\n{adj_matrix}")
    
    # Create GNN agent
    config = GNNConfig(
        hidden_dim=64,
        num_gcn_layers=2,
        num_gat_layers=1,
        gat_heads=4,
        use_temporal=True
    )
    
    agent = GNNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_intersections=num_intersections,
        adj_matrix=adj_matrix,
        config=config
    )
    
    print(f"\n✅ GNN Agent Created")
    print(f"   Intersections: {num_intersections}")
    print(f"   State Dimension: {state_dim} per intersection")
    print(f"   Action Dimension: {action_dim} per intersection")
    print(f"   Hidden Dimension: {config.hidden_dim}")
    print(f"   GCN Layers: {config.num_gcn_layers}")
    print(f"   GAT Layers: {config.num_gat_layers}")
    print(f"   Temporal Encoder: {'Enabled' if config.use_temporal else 'Disabled'}")
    
    # Test action selection
    print(f"\n📊 Testing Action Selection...")
    states = np.random.randn(num_intersections, state_dim)
    actions = agent.select_action(states, training=True)
    print(f"   States shape: {states.shape}")
    print(f"   Actions: {actions}")
    print(f"   Actions shape: {actions.shape}")
    
    # Test with multiple batches
    batch_states = np.random.randn(3, num_intersections, state_dim)
    for i, state in enumerate(batch_states):
        actions = agent.select_action(state, training=False)
        print(f"   Batch {i+1} actions: {actions}")
    
    # Test training step
    print(f"\n📊 Testing Training Step...")
    # Add some experiences
    for _ in range(100):
        state = np.random.randn(num_intersections, state_dim)
        action = np.random.randint(0, action_dim, size=num_intersections)
        reward = np.random.randn(num_intersections)
        next_state = np.random.randn(num_intersections, state_dim)
        done = False
        agent.push(state, action, reward, next_state, done)
    
    print(f"   Replay buffer size: {len(agent.memory)}")
    
    # Train step
    loss = agent.train_step()
    if loss is not None:
        print(f"   Training loss: {loss:.6f}")
        print(f"   Epsilon: {agent.epsilon:.4f}")
        print(f"   Steps: {agent.steps}")
    else:
        print(f"   Not enough samples for training (need {config.batch_size})")
    
    # Test different topologies
    print(f"\n📊 Testing Different Graph Topologies...")
    topologies = ["grid", "line", "ring", "fully_connected"]
    for topology in topologies:
        adj = build_intersection_graph(num_intersections, topology=topology)
        print(f"   {topology.capitalize()}: {adj.sum() - num_intersections} edges (excluding self-loops)")
    
    print(f"\n✅ GNN Agent Test Complete!")
    return True


def test_gnn_forward_pass():
    """Test GNN forward pass with different configurations."""
    print("\n" + "="*80)
    print("Testing GNN Forward Pass")
    print("="*80)
    
    from src.rl.gnn_agent import TrafficGNN
    
    num_intersections = 4
    state_dim = 8
    action_dim = 4
    batch_size = 2
    
    # Test without temporal encoder
    print(f"\n📊 Testing GNN without Temporal Encoder...")
    config_no_temp = GNNConfig(use_temporal=False, hidden_dim=64)
    model_no_temp = TrafficGNN(state_dim, action_dim, num_intersections, config_no_temp)
    
    states = torch.randn(batch_size, num_intersections, state_dim)
    adj = torch.eye(num_intersections).unsqueeze(0).expand(batch_size, -1, -1)
    
    q_values = model_no_temp(states, adj)
    print(f"   Input shape: {states.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    
    # Test with temporal encoder
    print(f"\n📊 Testing GNN with Temporal Encoder...")
    config_temp = GNNConfig(use_temporal=True, hidden_dim=64, temporal_dim=32)
    model_temp = TrafficGNN(state_dim, action_dim, num_intersections, config_temp)
    
    seq_len = 5
    temporal_seq = torch.randn(batch_size, seq_len, num_intersections, state_dim)
    
    q_values = model_temp(states, adj, temporal_seq)
    print(f"   Input shape: {states.shape}")
    print(f"   Temporal sequence shape: {temporal_seq.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    
    print(f"\n✅ Forward Pass Test Complete!")
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("Phase 3.1 Complete Test Suite")
    print("="*80)
    
    # Test GNN agent
    test_gnn_agent()
    
    # Test forward pass
    test_gnn_forward_pass()
    
    print("\n" + "="*80)
    print("All Tests Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

