"""
Test script for Phase 3.2 and 3.3.

Tests Enhanced Transformer Architecture and Memory-Augmented Networks.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.enhanced_transformer import (
    EnhancedTransformerAgent, EnhancedTransformerConfig,
    PerformerAttention, LongformerAttention, BigBirdAttention, VisionTransformerEncoder
)
from src.rl.memory_augmented import (
    MemoryAugmentedAgent, MemoryConfig,
    NeuralTuringMachine, DifferentiableNeuralComputer, EpisodicMemory
)


def test_enhanced_transformer():
    """Test Phase 3.2: Enhanced Transformer Architecture."""
    print("="*80)
    print("Testing Phase 3.2: Enhanced Transformer Architecture")
    print("="*80)
    
    state_dim = 8
    action_dim = 4
    batch_size = 2
    seq_len = 10
    
    # Test Performer
    print(f"\n📊 Testing Performer Architecture...")
    config_performer = EnhancedTransformerConfig(
        architecture="performer",
        d_model=128,
        nhead=8,
        num_layers=2,
        max_seq_len=1000,
        performer_nb_features=256
    )
    model_performer = EnhancedTransformerAgent(state_dim, action_dim, config_performer)
    
    x = torch.randn(batch_size, seq_len, state_dim)
    q_values = model_performer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    print(f"   ✅ Performer working")
    
    # Test Longformer
    print(f"\n📊 Testing Longformer Architecture...")
    config_longformer = EnhancedTransformerConfig(
        architecture="longformer",
        d_model=128,
        nhead=8,
        num_layers=2,
        max_seq_len=10000,
        attention_window=512
    )
    model_longformer = EnhancedTransformerAgent(state_dim, action_dim, config_longformer)
    
    x_long = torch.randn(batch_size, 100, state_dim)  # Longer sequence
    q_values = model_longformer(x_long)
    print(f"   Input shape: {x_long.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    print(f"   ✅ Longformer working")
    
    # Test BigBird
    print(f"\n📊 Testing BigBird Architecture...")
    config_bigbird = EnhancedTransformerConfig(
        architecture="bigbird",
        d_model=128,
        nhead=8,
        num_layers=2,
        max_seq_len=1000,
        num_random_blocks=3,
        block_size=64
    )
    model_bigbird = EnhancedTransformerAgent(state_dim, action_dim, config_bigbird)
    
    q_values = model_bigbird(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    print(f"   ✅ BigBird working")
    
    # Test Vision Transformer
    print(f"\n📊 Testing Vision Transformer Architecture...")
    config_vision = EnhancedTransformerConfig(
        architecture="vision",
        d_model=128,
        nhead=8,
        num_layers=2,
        image_size=224,
        patch_size=16
    )
    model_vision = EnhancedTransformerAgent(state_dim, action_dim, config_vision)
    
    x_vision = torch.randn(batch_size, 3, 224, 224)  # Image input
    q_values = model_vision(x_vision)
    print(f"   Input shape: {x_vision.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    print(f"   ✅ Vision Transformer working")
    
    print(f"\n✅ Enhanced Transformer Test Complete!")
    return True


def test_memory_augmented():
    """Test Phase 3.3: Memory-Augmented Networks."""
    print("\n" + "="*80)
    print("Testing Phase 3.3: Memory-Augmented Networks")
    print("="*80)
    
    state_dim = 8
    action_dim = 4
    batch_size = 2
    seq_len = 10
    
    # Test Neural Turing Machine
    print(f"\n📊 Testing Neural Turing Machine (NTM)...")
    config_ntm = MemoryConfig(
        memory_type="ntm",
        memory_size=128,
        memory_dim=64,
        num_read_heads=4,
        num_write_heads=1,
        controller_dim=128
    )
    model_ntm = MemoryAugmentedAgent(state_dim, action_dim, config_ntm)
    
    x = torch.randn(batch_size, seq_len, state_dim)
    q_values = model_ntm(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    print(f"   ✅ NTM working")
    
    # Test Differentiable Neural Computer
    print(f"\n📊 Testing Differentiable Neural Computer (DNC)...")
    config_dnc = MemoryConfig(
        memory_type="dnc",
        memory_size=128,
        memory_dim=64,
        num_read_heads=4,
        num_write_heads=1,
        controller_dim=128
    )
    model_dnc = MemoryAugmentedAgent(state_dim, action_dim, config_dnc)
    
    q_values = model_dnc(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    print(f"   ✅ DNC working")
    
    # Test Episodic Memory
    print(f"\n📊 Testing Episodic Memory...")
    episodic_memory = EpisodicMemory(capacity=1000, similarity_threshold=0.7)
    
    # Store some experiences
    for i in range(10):
        state = np.random.randn(state_dim)
        action = np.random.randint(0, action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        is_rare = (i % 3 == 0)  # Some rare events
        episodic_memory.store(state, action, reward, next_state, is_rare)
    
    print(f"   Stored {len(episodic_memory.memory)} experiences")
    print(f"   Stored {len(episodic_memory.rare_event_memory)} rare events")
    
    # Retrieve similar experiences
    query_state = np.random.randn(state_dim)
    retrieved = episodic_memory.retrieve(query_state, k=5)
    print(f"   Retrieved {len(retrieved)} similar experiences")
    
    # Retrieve rare events
    rare_retrieved = episodic_memory.retrieve_rare_events(query_state, k=3)
    print(f"   Retrieved {len(rare_retrieved)} rare events")
    print(f"   ✅ Episodic Memory working")
    
    # Test Episodic Memory Agent
    print(f"\n📊 Testing Episodic Memory Agent...")
    config_episodic = MemoryConfig(
        memory_type="episodic",
        controller_dim=128,
        episode_capacity=1000,
        similarity_threshold=0.7
    )
    model_episodic = MemoryAugmentedAgent(state_dim, action_dim, config_episodic)
    
    x_single = torch.randn(batch_size, state_dim)
    q_values = model_episodic(x_single)
    print(f"   Input shape: {x_single.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
    print(f"   ✅ Episodic Memory Agent working")
    
    print(f"\n✅ Memory-Augmented Networks Test Complete!")
    return True


def test_combined():
    """Test Phase 3.2 and 3.3 together."""
    print("\n" + "="*80)
    print("Testing Phase 3.2 + 3.3: Combined")
    print("="*80)
    
    state_dim = 8
    action_dim = 4
    batch_size = 2
    seq_len = 10
    
    # Create Performer with NTM memory
    print(f"\n📊 Testing Performer + NTM...")
    config_transformer = EnhancedTransformerConfig(architecture="performer", d_model=128)
    config_memory = MemoryConfig(memory_type="ntm", memory_size=64)
    
    transformer_model = EnhancedTransformerAgent(state_dim, action_dim, config_transformer)
    memory_model = MemoryAugmentedAgent(state_dim, action_dim, config_memory)
    
    x = torch.randn(batch_size, seq_len, state_dim)
    
    # Forward through both
    transformer_out = transformer_model(x)
    memory_out = memory_model(x)
    
    print(f"   Transformer output shape: {transformer_out.shape}")
    print(f"   Memory output shape: {memory_out.shape}")
    print(f"   ✅ Combined models working")
    
    print(f"\n✅ Combined Test Complete!")
    return True


def main():
    """Run all tests."""
    print("="*80)
    print("Phase 3.2 & 3.3 Complete Test Suite")
    print("="*80)
    
    # Test Phase 3.2
    test_enhanced_transformer()
    
    # Test Phase 3.3
    test_memory_augmented()
    
    # Test Combined
    test_combined()
    
    print("\n" + "="*80)
    print("All Tests Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

