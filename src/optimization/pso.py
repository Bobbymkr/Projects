import numpy as np
import random

class ParticleSwarmOptimizer:
    """PSO adapted with Artificial Fish Swarm concepts for signal optimization."""
    def __init__(self, num_particles=30, iterations=100, num_phases=4,
                 min_time=10, max_time=60, c1=2.0, c2=2.0, w=0.7,
                 visual=0.5, crowd_factor=0.6):
        self.num_particles = num_particles
        self.iterations = iterations
        self.num_phases = num_phases
        self.min_time = min_time
        self.max_time = max_time
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.w = w    # Inertia weight
        self.visual = visual  # AFSA visual distance
        self.crowd_factor = crowd_factor  # AFSA crowd factor

    def initialize_particles(self):
        """Initialize positions and velocities."""
        positions = np.random.uniform(self.min_time, self.max_time,
                                      (self.num_particles, self.num_phases))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.num_phases))
        return positions, velocities

    def fitness(self, position, queue_lengths, wait_times):
        """Fitness: minimize weighted queues and waits."""
        queue_score = np.sum(queue_lengths) / np.sum(position)
        wait_score = np.mean(wait_times)
        return 0.6 * queue_score + 0.4 * wait_score  # Minimize

    def afsa_swarm(self, positions, fitnesses, i):
        """AFSA swarming behavior: move towards center if not crowded."""
        center = np.mean(positions, axis=0)
        nf = len(positions)  # Number of fish in visual
        if nf / self.num_particles < self.crowd_factor:
            direction = center - positions[i]
            step = random.random() * direction
            return positions[i] + step
        return positions[i]

    def afsa_follow(self, positions, fitnesses, i):
        """AFSA following: move towards best neighbor if not crowded."""
        best_neighbor = np.argmin(fitnesses)
        if best_neighbor != i:
            direction = positions[best_neighbor] - positions[i]
            step = random.random() * direction
            return positions[i] + step
        return positions[i]

    def optimize(self, queue_lengths, wait_times):
        """Run adapted PSO."""
        positions, velocities = self.initialize_particles()
        pbest_positions = positions.copy()
        pbest_fitness = [self.fitness(p, queue_lengths, wait_times) for p in positions]
        gbest_idx = np.argmin(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx].copy()

        for _ in range(self.iterations):
            for i in range(self.num_particles):
                # Standard PSO update
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                                 self.c2 * r2 * (gbest_position - positions[i]))

                # Adapt with AFSA behaviors
                if random.random() < 0.3:  # Probability to apply swarming
                    positions[i] = self.afsa_swarm(positions, pbest_fitness, i)
                elif random.random() < 0.3:  # Probability to apply following
                    positions[i] = self.afsa_follow(positions, pbest_fitness, i)
                else:
                    positions[i] += velocities[i]

                positions[i] = np.clip(positions[i], self.min_time, self.max_time)

                current_fitness = self.fitness(positions[i], queue_lengths, wait_times)
                if current_fitness < pbest_fitness[i]:
                    pbest_fitness[i] = current_fitness
                    pbest_positions[i] = positions[i].copy()
                    if current_fitness < pbest_fitness[gbest_idx]:
                        gbest_idx = i
                        gbest_position = positions[i].copy()

        return gbest_position

# Integration example
def get_pso_timing(state, pso: ParticleSwarmOptimizer):
    """Get optimized timings using adapted PSO."""
    queue_lengths = state[:4]
    wait_times = state[4:8]
    return pso.optimize(queue_lengths, wait_times)