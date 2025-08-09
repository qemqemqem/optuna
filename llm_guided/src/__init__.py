"""
LLM-Guided Optuna: Large Language Model Guided Hyperparameter Optimization

This package extends Optuna with LLM-guided sampling capabilities, allowing
hyperparameter optimization to leverage domain knowledge and pattern recognition
capabilities of Large Language Models.

Key Components:
- LLMGuidedSampler: Main sampler that integrates with Optuna
- Context builders for extracting optimization history
- Structured output models for reliable LLM communication
- Distribution extraction and combination utilities
"""

from .context_builder import ContextBuilder
from .llm_client import LLMClient
from .models import OptimizationContext
from .models import TrialConfiguration
from .sampler import LLMGuidedSampler


__version__ = "0.1.0"
__all__ = [
    "LLMGuidedSampler",
    "TrialConfiguration",
    "OptimizationContext",
    "ContextBuilder",
    "LLMClient",
]
