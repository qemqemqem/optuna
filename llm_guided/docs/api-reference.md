# API Reference

## Core Classes

### LLMGuidedSampler

The main sampler class that integrates with Optuna.

```python
class LLMGuidedSampler(BaseSampler):
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.3,
        timeout: int = 30,
        max_retries: int = 3,
        max_context_trials: int = 10
    )
```

**Parameters:**
- `model`: LLM model identifier (any LiteLLM-supported model)
- `temperature`: Sampling temperature (0.0-1.0, lower = more consistent)
- `timeout`: API request timeout in seconds
- `max_retries`: Maximum retry attempts for failed requests
- `max_context_trials`: Maximum trials to include in LLM context

**Methods:**

#### `get_statistics() -> Dict[str, Any]`
Returns comprehensive performance statistics.

**Returns:**
```python
{
    "sampler_stats": {
        "total_trials": int,
        "successful_generations": int,
        "failed_generations": int,
        "success_rate": float,
        "average_llm_time": float,
        "fallback_uses": int,
        "validation_fixes": int
    },
    "validation_stats": {
        "total_validations": int,
        "clamped_parameters": int,
        "fixed_parameters": int,
        "failed_validations": int
    },
    "configuration": {
        "model": str,
        "temperature": float,
        "timeout": int,
        "max_retries": int,
        "max_context_trials": int
    }
}
```

#### `reset_statistics() -> None`
Resets all performance statistics to zero.

---

### ContextBuilder

Extracts optimization context from Optuna studies.

```python
class ContextBuilder:
    def __init__(
        self,
        max_recent_trials: int = 15,
        max_context_trials: int = 10
    )
```

**Parameters:**
- `max_recent_trials`: Maximum trials for trend analysis
- `max_context_trials`: Maximum trials included in LLM context

**Methods:**

#### `build_context(study: optuna.Study) -> OptimizationContext`
Builds comprehensive optimization context.

**Parameters:**
- `study`: Optuna study to extract context from

**Returns:** `OptimizationContext` with trial history, trends, and search space

---

### LLMClient

Handles communication with Large Language Models.

```python
class LLMClient:
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.3,
        timeout: int = 30,
        max_retries: int = 3
    )
```

**Methods:**

#### `generate_trial_configuration(context: OptimizationContext, temperature: Optional[float] = None) -> TrialConfiguration`
Generates trial configuration using LLM.

**Parameters:**
- `context`: Complete optimization context
- `temperature`: Override default temperature

**Returns:** Validated `TrialConfiguration`

**Raises:** `LLMError` if generation fails after retries

---

### ParameterValidator

Validates and clamps LLM-generated parameters.

```python
class ParameterValidator:
    def __init__(self)
```

**Methods:**

#### `validate_and_clamp_configuration(config: TrialConfiguration, search_space: Dict[str, optuna.distributions.BaseDistribution]) -> Dict[str, Any]`
Validates complete configuration against search space.

**Parameters:**
- `config`: LLM-generated trial configuration
- `search_space`: Optuna search space definition

**Returns:** Validated parameter dictionary

**Raises:** `ValidationError` if validation fails completely

#### `get_validation_stats() -> Dict[str, int]`
Returns validation statistics.

---

## Data Models

### TrialConfiguration

LLM response format for trial configurations.

```python
class TrialConfiguration(BaseModel):
    parameters: Dict[str, Union[float, int, str, bool]]
    reasoning: str = Field(min_length=10, max_length=500)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    strategy: Optional[ExplorationStrategy] = Field(default=ExplorationStrategy.BALANCED)
    expected_performance: Optional[str] = Field(default=None, max_length=200)
```

### OptimizationContext

Complete context provided to LLM.

```python
class OptimizationContext(BaseModel):
    objective_name: str
    objective_direction: str  # "MINIMIZE" or "MAXIMIZE"
    n_trials_completed: int
    search_space: List[SearchSpaceParameter]
    recent_trials: List[TrialResult]
    best_trial: Optional[TrialResult]
    progress_analysis: ProgressAnalysis
    problem_type: str = "unknown"
    problem_description: str = "No description provided"
    constraints: List[str] = Field(default_factory=list)
    domain_knowledge: Optional[Dict[str, Any]] = Field(default_factory=dict)
    strategy_guidance: Optional[Dict[str, Any]] = Field(default_factory=dict)
```

### ProgressAnalysis

Analysis of optimization progress and trends.

```python
class ProgressAnalysis(BaseModel):
    stage: OptimizationStage  # EARLY_EXPLORATION, ACTIVE_SEARCH, etc.
    trend: TrendType  # IMPROVING, PLATEAUING, DEGRADING
    trend_strength: float = Field(ge=0.0, le=1.0)
    trials_since_improvement: int
    convergence_indicator: float = Field(ge=0.0, le=1.0)
    best_value_trend: Optional[List[float]] = None
    recommendation: str = Field(max_length=200)
```

### SearchSpaceParameter

Parameter definition in search space.

```python
class SearchSpaceParameter(BaseModel):
    name: str
    type: str  # "float", "int", "categorical"
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    log_scale: bool = False
    step: Optional[Union[float, int]] = None
    choices: Optional[List[Union[str, int, float]]] = None
    description: Optional[str] = None
    typical_range: Optional[str] = None
```

### TrialResult

Completed trial representation.

```python
class TrialResult(BaseModel):
    trial_number: int = Field(ge=0)
    parameters: Dict[str, Union[float, int, str, bool]]
    value: float
    duration: Optional[float] = Field(default=None, ge=0.0)
    state: str = Field(default="COMPLETE", pattern="^(COMPLETE|FAIL|PRUNED)$")
    intermediate_values: Optional[List[float]] = None
    user_attrs: Optional[Dict[str, Any]] = Field(default_factory=dict)
```

## Enums

### OptimizationStage
```python
class OptimizationStage(str, Enum):
    EARLY_EXPLORATION = "early_exploration"
    ACTIVE_SEARCH = "active_search"
    FOCUSED_OPTIMIZATION = "focused_optimization"
    REFINEMENT = "refinement"
```

### TrendType
```python
class TrendType(str, Enum):
    IMPROVING = "improving"
    PLATEAUING = "plateauing"
    DEGRADING = "degrading"
    INSUFFICIENT_DATA = "insufficient_data"
```

### ExplorationStrategy
```python
class ExplorationStrategy(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
```

## Exceptions

### LLMError
Base exception for LLM-related errors.

### LLMTimeoutError
LLM request timed out.

### LLMRequestError
Invalid request to LLM API (authentication, quota, etc.).

### LLMParsingError
Failed to parse LLM response.

### ValidationError
Parameter validation failed.

### LLMSamplingError
LLM sampling failed completely (raised by sampler).

## Utility Functions

### create_search_space_parameter
```python
def create_search_space_parameter(
    name: str, 
    distribution: optuna.distributions.BaseDistribution, 
    description: Optional[str] = None
) -> SearchSpaceParameter
```

Creates `SearchSpaceParameter` from Optuna distribution.

### create_trial_result
```python
def create_trial_result(trial: optuna.Trial) -> TrialResult
```

Creates `TrialResult` from Optuna trial.

## Configuration Examples

### Model Configuration
```python
# OpenAI
sampler = LLMGuidedSampler(model="gpt-4o-2024-08-06")
sampler = LLMGuidedSampler(model="gpt-4o-mini")

# Anthropic
sampler = LLMGuidedSampler(model="claude-3-sonnet-20240229")
sampler = LLMGuidedSampler(model="claude-3-haiku-20240307")

# Local (Ollama)
sampler = LLMGuidedSampler(model="ollama/llama3")
sampler = LLMGuidedSampler(model="ollama/mistral")

# Other providers
sampler = LLMGuidedSampler(model="gemini-pro")
sampler = LLMGuidedSampler(model="cohere/command")
```

### Study Context Configuration
```python
# Minimal
study.set_user_attr("problem_type", "neural_network_training")

# Comprehensive
study.set_user_attr("problem_type", "neural_network_training")
study.set_user_attr("problem_description", "CNN for image classification")
study.set_user_attr("constraints", ["memory < 8GB", "time < 1hr"])
study.set_user_attr("domain_knowledge", {
    "dataset": "CIFAR-10",
    "architecture": "ResNet-18",
    "best_practices": ["batch_norm", "data_augmentation"],
    "typical_ranges": {
        "learning_rate": [1e-4, 1e-2],
        "batch_size": [32, 128]
    }
})
```