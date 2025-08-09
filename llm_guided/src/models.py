"""
Pydantic models for structured LLM communication and data validation.

This module defines the data structures used for:
- LLM request/response formats
- Optimization context representation
- Parameter validation and type safety
"""

from enum import Enum
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator


class OptimizationStage(str, Enum):
    """Current stage of the optimization process."""

    EARLY_EXPLORATION = "early_exploration"
    ACTIVE_SEARCH = "active_search"
    FOCUSED_OPTIMIZATION = "focused_optimization"
    REFINEMENT = "refinement"


class ExplorationStrategy(str, Enum):
    """Strategy for parameter exploration."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class TrendType(str, Enum):
    """Type of optimization trend."""

    IMPROVING = "improving"
    PLATEAUING = "plateauing"
    DEGRADING = "degrading"
    INSUFFICIENT_DATA = "insufficient_data"


class TrialConfiguration(BaseModel):
    """
    Structured response format for LLM-generated trial configurations.

    This is the primary model that LLMs must conform to when suggesting
    hyperparameter configurations.
    """

    parameters: Dict[str, Union[float, int, str, bool]] = Field(
        description="Complete hyperparameter configuration mapping parameter names to values",
        example={"learning_rate": 0.001, "batch_size": 32, "dropout": 0.3, "optimizer": "adam"},
    )

    reasoning: str = Field(
        description="Brief explanation (1-3 sentences) of why these parameters were chosen",
        min_length=10,
        max_length=500,
        example="Chose moderate learning rate for stable training, smaller batch size for better gradient estimates, and moderate dropout to prevent overfitting.",
    )

    confidence: Optional[float] = Field(
        default=None,
        description="Confidence level in this configuration (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.75,
    )

    strategy: Optional[ExplorationStrategy] = Field(
        default=ExplorationStrategy.BALANCED,
        description="Whether this configuration is conservative, balanced, or aggressive",
    )

    expected_performance: Optional[str] = Field(
        default=None,
        description="Brief prediction of expected performance",
        max_length=200,
        example="Expected to achieve 85-90% validation accuracy with moderate training time",
    )

    @field_validator("parameters")
    @classmethod
    def parameters_not_empty(cls, v):
        """Ensure parameters dictionary is not empty."""
        if not v:
            raise ValueError("Parameters dictionary cannot be empty")
        return v

    @field_validator("reasoning")
    @classmethod
    def reasoning_meaningful(cls, v):
        """Ensure reasoning is meaningful and not just boilerplate."""
        boilerplate_phrases = [
            "good parameters",
            "should work well",
            "typical values",
            "standard configuration",
            "these parameters",
            "will work",
        ]

        v_lower = v.lower()
        if len([phrase for phrase in boilerplate_phrases if phrase in v_lower]) >= 2:
            raise ValueError(
                "Reasoning appears to be boilerplate - provide specific justification"
            )

        return v

    model_config = ConfigDict(
        extra="forbid",  # Don't allow extra fields
        validate_assignment=True,  # Validate on assignment
    )


class SearchSpaceParameter(BaseModel):
    """Structured representation of a single parameter in the search space."""

    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type", pattern="^(float|int|categorical)$")

    # For numeric parameters
    low: Optional[Union[float, int]] = Field(default=None, description="Lower bound")
    high: Optional[Union[float, int]] = Field(default=None, description="Upper bound")
    log_scale: bool = Field(default=False, description="Whether parameter uses log scale")
    step: Optional[Union[float, int]] = Field(default=None, description="Step size constraint")

    # For categorical parameters
    choices: Optional[List[Union[str, int, float]]] = Field(
        default=None, description="Available choices for categorical parameters"
    )

    # Metadata
    description: Optional[str] = Field(
        default=None, description="Human-readable parameter description"
    )
    typical_range: Optional[str] = Field(default=None, description="Typical successful range")

    @field_validator("choices")
    @classmethod
    def choices_required_for_categorical(cls, v, info):
        """Ensure choices are provided for categorical parameters."""
        if info.data.get("type") == "categorical" and not v:
            raise ValueError("Choices must be provided for categorical parameters")
        return v


class TrialResult(BaseModel):
    """Structured representation of a completed trial result."""

    trial_number: int = Field(ge=0, description="Trial number")
    parameters: Dict[str, Union[float, int, str, bool]] = Field(description="Trial parameters")
    value: float = Field(description="Objective function value")
    duration: Optional[float] = Field(
        default=None, ge=0.0, description="Trial duration in seconds"
    )
    state: str = Field(
        default="COMPLETE",
        pattern="^(COMPLETE|FAIL|PRUNED)$",
        description="Trial completion state",
    )

    # Optional metadata
    intermediate_values: Optional[List[float]] = Field(
        default=None, description="Intermediate values during optimization"
    )
    user_attrs: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional trial metadata"
    )


class ProgressAnalysis(BaseModel):
    """Analysis of optimization progress and trends."""

    stage: OptimizationStage = Field(description="Current optimization stage")
    trend: TrendType = Field(description="Recent performance trend")
    trend_strength: float = Field(
        ge=0.0, le=1.0, description="Strength of the trend (0=weak, 1=strong)"
    )
    trials_since_improvement: int = Field(
        ge=0, description="Number of trials since last improvement"
    )
    convergence_indicator: float = Field(
        ge=0.0, le=1.0, description="Convergence assessment (0=not converged, 1=converged)"
    )
    best_value_trend: Optional[List[float]] = Field(
        default=None, description="Recent best values for trend analysis"
    )
    recommendation: str = Field(
        description="Strategic recommendation based on progress", max_length=200
    )


class OptimizationContext(BaseModel):
    """
    Complete context provided to LLM for generating configurations.

    This model encapsulates all the information needed for the LLM to make
    informed hyperparameter suggestions.
    """

    # Basic information
    objective_name: str = Field(description="Name of the objective function")
    objective_direction: str = Field(
        pattern="^(MINIMIZE|MAXIMIZE)$", description="Optimization direction"
    )
    n_trials_completed: int = Field(ge=0, description="Number of completed trials")

    # Search space
    search_space: List[SearchSpaceParameter] = Field(
        description="Complete search space definition"
    )

    # Historical results
    recent_trials: List[TrialResult] = Field(
        default_factory=list, max_length=20, description="Recent trial results"
    )
    best_trial: Optional[TrialResult] = Field(default=None, description="Best trial found so far")

    # Progress analysis
    progress_analysis: ProgressAnalysis = Field(description="Optimization progress analysis")

    # Problem context
    problem_type: str = Field(
        default="unknown",
        description="Type of optimization problem",
        example="neural_network_training",
    )
    problem_description: str = Field(
        default="No description provided",
        max_length=1000,
        description="Human-readable problem description",
    )

    # Additional context
    constraints: List[str] = Field(
        default_factory=list,
        description="Any constraints or requirements",
        example=["training_time < 2 hours", "memory_usage < 8GB"],
    )

    domain_knowledge: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Domain-specific context and knowledge"
    )

    # Strategy guidance
    strategy_guidance: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Strategic guidance for this trial"
    )


class BatchTrialConfiguration(BaseModel):
    """
    Model for requesting multiple trial configurations simultaneously.
    Useful for generating diverse exploration strategies.
    """

    configurations: List[TrialConfiguration] = Field(
        description="List of diverse trial configurations", min_length=1, max_length=10
    )

    diversity_strategy: str = Field(
        description="Explanation of how diversity was achieved across configurations",
        min_length=20,
        max_length=300,
    )

    @field_validator("configurations")
    @classmethod
    def ensure_consistency(cls, v):
        """Ensure all configurations have consistent parameter structure."""
        if len(v) < 2:
            return v

        # Get parameter names from first configuration
        param_names = set(v[0].parameters.keys())

        # Verify all configs have same parameter structure
        for i, config in enumerate(v[1:], 1):
            if set(config.parameters.keys()) != param_names:
                raise ValueError(
                    f"Configuration {i} has different parameter structure than configuration 0"
                )

        return v


# Utility functions for model creation
def create_search_space_parameter(
    name: str, distribution, description: Optional[str] = None
) -> SearchSpaceParameter:
    """Create SearchSpaceParameter from Optuna distribution."""
    import optuna

    if isinstance(distribution, optuna.distributions.FloatDistribution):
        return SearchSpaceParameter(
            name=name,
            type="float",
            low=distribution.low,
            high=distribution.high,
            log_scale=distribution.log,
            step=distribution.step,
            description=description,
        )
    elif isinstance(distribution, optuna.distributions.IntDistribution):
        return SearchSpaceParameter(
            name=name,
            type="int",
            low=distribution.low,
            high=distribution.high,
            step=distribution.step,
            description=description,
        )
    elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
        return SearchSpaceParameter(
            name=name,
            type="categorical",
            choices=list(distribution.choices),
            description=description,
        )
    else:
        raise ValueError(f"Unsupported distribution type: {type(distribution)}")


def create_trial_result(trial) -> TrialResult:
    """Create TrialResult from Optuna trial."""
    import optuna

    return TrialResult(
        trial_number=trial.number,
        parameters=trial.params,
        value=trial.value,
        duration=trial.duration.total_seconds() if trial.duration else None,
        state=trial.state.name,
        intermediate_values=(
            list(trial.intermediate_values.values()) if trial.intermediate_values else None
        ),
        user_attrs=dict(trial.user_attrs) if trial.user_attrs else {},
    )
