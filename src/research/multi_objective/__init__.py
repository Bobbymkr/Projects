"""
Phase 8: Multi-Objective Optimization Framework

Provides:
- Multi-objective reward functions
- NSGA-II genetic algorithm
- MO-PPO (Multi-Objective PPO)
- Constraint optimization
- CPO (Constrained Policy Optimization)
"""

from .phase8_multi_objective import (
    MultiObjectiveReward,
    MultiObjectiveWeights,
    NSGA2Solver,
    MOPPOAgent,
    MOPPOConfig,
    ConstraintOptimizer,
    ConstraintConfig,
    CPOAgent,
)

__all__ = [
    'MultiObjectiveReward',
    'MultiObjectiveWeights',
    'NSGA2Solver',
    'MOPPOAgent',
    'MOPPOConfig',
    'ConstraintOptimizer',
    'ConstraintConfig',
    'CPOAgent',
]

