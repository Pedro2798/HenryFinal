"""Specialized collaborative agents for the contract-comparison pipeline.

The pipeline uses two single-responsibility agents with an explicit handoff:

    ContextualizationAgent  ->  builds a comparative structure map
    ExtractionAgent         ->  consumes the map + texts, emits validated JSON
"""

from .contextualization_agent import ContextualizationAgent
from .extraction_agent import ExtractionAgent

__all__ = ["ContextualizationAgent", "ExtractionAgent"]
