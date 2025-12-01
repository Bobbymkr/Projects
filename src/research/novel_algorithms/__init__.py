"""
Novel Algorithms for Traffic Control.

Phase 5: Cutting-edge algorithms including imitation learning,
model-based RL, and hierarchical RL.
"""

from .imitation_learning import (
    BehavioralCloningAgent,
    InverseReinforcementLearning,
    ImitationLearningTrainer,
    ExpertDemonstration,
)

from .model_based_rl import (
    WorldModel,
    ModelPredictiveControl,
    ModelBasedRLAgent,
    WorldModelState,
)

from .hierarchical_rl import (
    OptionDiscovery,
    HierarchicalPolicy,
    HierarchicalRLAgent,
    Option,
    OptionType,
)

__all__ = [
    # Imitation Learning
    "BehavioralCloningAgent",
    "InverseReinforcementLearning",
    "ImitationLearningTrainer",
    "ExpertDemonstration",
    # Model-Based RL
    "WorldModel",
    "ModelPredictiveControl",
    "ModelBasedRLAgent",
    "WorldModelState",
    # Hierarchical RL
    "OptionDiscovery",
    "HierarchicalPolicy",
    "HierarchicalRLAgent",
    "Option",
    "OptionType",
]

