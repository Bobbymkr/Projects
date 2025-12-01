"""
Federated Learning Framework.

Phase 5: Privacy-preserving distributed learning for traffic control
across multiple intersections without centralizing data.
"""

from .federated_coordinator import (
    FederatedCoordinator,
    FederatedClient,
    AggregationStrategy,
)

from .privacy_mechanisms import (
    DifferentialPrivacy,
    SecureAggregation,
    PrivacyBudget,
)

__all__ = [
    "FederatedCoordinator",
    "FederatedClient",
    "AggregationStrategy",
    "DifferentialPrivacy",
    "SecureAggregation",
    "PrivacyBudget",
]

