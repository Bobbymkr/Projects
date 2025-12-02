import numpy as np
import random

class GeneticAlgorithm:
    """Genetic Algorithm for optimizing traffic signal timings."""
    def __init__(self, population_size=50, generations=100, mutation_rate=0.01,
                 num_phases=4, min_time=10, max_time=60):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.num_phases = num_phases
        self.min_time = min_time
        self.max_time = max_time

    def initialize_population(self):
        """Initialize population with random timings for each phase."""
        return np.random.randint(self.min_time, self.max_time + 1,
                                 size=(self.population_size, self.num_phases))

    def fitness(self, individual, queue_lengths, wait_times):
        """Fitness function: minimize weighted sum of queues and wait times."""
        # Simulate simple score: lower queues and waits are better (negative for minimization)
        queue_score = np.sum(queue_lengths) / individual.sum()  # Simplified
        wait_score = np.mean(wait_times)
        return - (0.6 * queue_score + 0.4 * wait_score)

    def select(self, population, fitnesses):
        """Tournament selection."""
        selected = []
        for _ in range(self.population_size):
            i1, i2 = random.sample(range(self.population_size), 2)
            selected.append(population[i1] if fitnesses[i1] > fitnesses[i2] else population[i2])
        return np.array(selected)

    def crossover(self, parents):
        """Single-point crossover."""
        offspring = []
        for i in range(0, self.population_size, 2):
            p1, p2 = parents[i], parents[i+1]
            point = random.randint(1, self.num_phases - 1)
            child1 = np.concatenate((p1[:point], p2[point:]))
            child2 = np.concatenate((p2[:point], p1[point:]))
            offspring.extend([child1, child2])
        return np.array(offspring)

    def mutate(self, population):
        """Mutate timings with given probability."""
        for ind in population:
            if random.random() < self.mutation_rate:
                phase = random.randint(0, self.num_phases - 1)
                ind[phase] = random.randint(self.min_time, self.max_time)
        return population

    def optimize(self, queue_lengths, wait_times):
        """Run GA to find optimal timings."""
        population = self.initialize_population()
        for _ in range(self.generations):
            fitnesses = [self.fitness(ind, queue_lengths, wait_times) for ind in population]
            parents = self.select(population, fitnesses)
            offspring = self.crossover(parents)
            population = self.mutate(offspring)
        best_idx = np.argmax(fitnesses)
        return population[best_idx]

# Integration example
def get_ga_timing(state, ga: GeneticAlgorithm):
    """Get optimized timings using GA given state."""
    queue_lengths = state[:4]  # Assuming queues
    wait_times = state[4:8]    # Assuming waits
    return ga.optimize(queue_lengths, wait_times)