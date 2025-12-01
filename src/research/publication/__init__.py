"""
Research Publication Framework.

Phase 5: Tools and templates for research publications,
reproducibility packages, and knowledge sharing.
"""

from .paper_templates import (
    ResearchPaperTemplate,
    ConferencePaperTemplate,
    JournalPaperTemplate,
)

from .reproducibility import (
    ReproducibilityPackage,
    ExperimentSnapshot,
    CodeRelease,
)

__all__ = [
    "ResearchPaperTemplate",
    "ConferencePaperTemplate",
    "JournalPaperTemplate",
    "ReproducibilityPackage",
    "ExperimentSnapshot",
    "CodeRelease",
]

