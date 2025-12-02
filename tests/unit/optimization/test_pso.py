"""
Unit tests for particle swarm optimization.
"""

import pytest
import numpy as np
from src.optimization.pso import ParticleSwarmOptimizer


class TestParticleSwarmOptimizer:
    """Test particle swarm optimizer."""
    
    @pytest.fixture
    def pso(self):
        """Create PSO instance."""
        return ParticleSwarmOptimizer(num_particles=10, iterations=5, num_phases=4)
    
    def test_initialization(self, pso):
        """Test PSO initialization."""
        assert pso is not None
        assert pso.num_particles == 10
        assert pso.iterations == 5
        assert pso.num_phases == 4
    
    def test_initialize_particles(self, pso):
        """Test particle initialization."""
        positions, velocities = pso.initialize_particles()
        assert positions.shape == (pso.num_particles, pso.num_phases)
        assert velocities.shape == (pso.num_particles, pso.num_phases)
        assert np.all(positions >= pso.min_time)
        assert np.all(positions <= pso.max_time)
    
    def test_fitness(self, pso):
        """Test fitness function."""
        position = np.array([20, 30, 25, 35])
        queue_lengths = np.array([5, 8, 3, 6])
        wait_times = np.array([10, 15, 8, 12])
        
        fitness = pso.fitness(position, queue_lengths, wait_times)
        assert isinstance(fitness, (int, float))
        assert fitness >= 0
    
    def test_afsa_swarm(self, pso):
        """Test AFSA swarming behavior."""
        positions = pso.initialize_particles()[0]
        fitnesses = np.random.rand(pso.num_particles)
        
        new_position = pso.afsa_swarm(positions, fitnesses, 0)
        assert new_position.shape == positions[0].shape
    
    def test_afsa_follow(self, pso):
        """Test AFSA following behavior."""
        positions = pso.initialize_particles()[0]
        fitnesses = np.random.rand(pso.num_particles)
        
        new_position = pso.afsa_follow(positions, fitnesses, 0)
        assert new_position.shape == positions[0].shape

