"""
Unit tests for genetic algorithm.
"""

import pytest
import numpy as np
from src.optimization.genetic_algo import GeneticAlgorithm


class TestGeneticAlgorithm:
    """Test genetic algorithm."""
    
    @pytest.fixture
    def ga(self):
        """Create genetic algorithm instance."""
        return GeneticAlgorithm(population_size=20, generations=10, num_phases=4)
    
    def test_initialization(self, ga):
        """Test genetic algorithm initialization."""
        assert ga is not None
        assert ga.population_size == 20
        assert ga.generations == 10
        assert ga.num_phases == 4
    
    def test_initialize_population(self, ga):
        """Test population initialization."""
        population = ga.initialize_population()
        assert population.shape == (ga.population_size, ga.num_phases)
        assert np.all(population >= ga.min_time)
        assert np.all(population <= ga.max_time)
    
    def test_fitness(self, ga):
        """Test fitness function."""
        individual = np.array([20, 30, 25, 35])
        queue_lengths = np.array([5, 8, 3, 6])
        wait_times = np.array([10, 15, 8, 12])
        
        fitness = ga.fitness(individual, queue_lengths, wait_times)
        assert isinstance(fitness, (int, float))
    
    def test_select(self, ga):
        """Test selection."""
        population = ga.initialize_population()
        fitnesses = np.random.rand(ga.population_size)
        
        selected = ga.select(population, fitnesses)
        assert selected.shape == population.shape
    
    def test_crossover(self, ga):
        """Test crossover."""
        parents = ga.initialize_population()
        offspring = ga.crossover(parents)
        assert offspring.shape == parents.shape
    
    def test_mutate(self, ga):
        """Test mutation."""
        population = ga.initialize_population()
        original = population.copy()
        mutated = ga.mutate(population)
        assert mutated.shape == original.shape

