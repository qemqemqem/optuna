# Implementation Architecture

## Core Architecture Overview

The LLM-guided Optuna integration is built around a custom sampler that generates complete trial configurations using LLM reasoning about the optimization context.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Code     │───▶│  Optuna Study    │───▶│ LLMGuidedSampler│
│                 │    │                  │    │                 │
│ def objective() │    │ study.optimize() │    │ sample_*()      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                              ┌─────────────────────────────────────┐
                              │         Context Builder             │
                              │                                     │
                              │ • Extract study history            │
                              │ • Format search space              │
                              │ • Analyze optimization progress    │
                              │ • Build LLM prompt                 │
                              └─────────────────────────────────────┘
                                                         │
                                                         ▼
                              ┌─────────────────────────────────────┐
                              │        LLM Client                   │
                              │                                     │
                              │ • LiteLLM integration              │
                              │ • Structured output (Pydantic)    │
                              │ • Error handling & retries        │
                              │ • Response validation              │
                              └─────────────────────────────────────┘
                                                         │
                                                         ▼
                              ┌─────────────────────────────────────┐
                              │    Parameter Validator              │
                              │                                     │
                              │ • Bounds checking                  │
                              │ • Type conversion                  │
                              │ • Constraint enforcement          │
                              │ • Default value handling          │
                              └─────────────────────────────────────┘
```

## Core Components

### 1. LLMGuidedSampler

The main integration point with Optuna's sampler interface.

```python
class LLMGuidedSampler(optuna.samplers.BaseSampler):
    """
    Optuna sampler that uses LLM to suggest complete trial configurations.
    
    This sampler calls the LLM for every trial to generate a complete
    hyperparameter configuration based on the current optimization context.
    """
    
    def __init__(self, 
                 model: str = "gpt-4o-2024-08-06",
                 temperature: float = 0.3,
                 max_context_trials: int = 20,
                 timeout: int = 30):
        """
        Initialize LLM-guided sampler.
        
        Args:
            model: LLM model identifier for LiteLLM
            temperature: Sampling temperature for LLM
            max_context_trials: Maximum trials to include in context
            timeout: API timeout in seconds
        """
        self.model = model
        self.temperature = temperature
        self.max_context_trials = max_context_trials
        self.timeout = timeout
        
        # Initialize components
        self.context_builder = ContextBuilder(max_context_trials)
        self.llm_client = LLMClient(model, temperature, timeout)
        self.validator = ParameterValidator()
        
        # Performance tracking
        self.performance_monitor = PerformanceMonitor()
    
    def sample_independent(self, 
                          study: optuna.Study, 
                          trial: optuna.Trial, 
                          param_name: str, 
                          param_distribution: optuna.distributions.BaseDistribution) -> Any:
        """
        Sample a parameter value for the given trial.
        
        This method is called by Optuna for each parameter in the search space.
        We generate the complete configuration on the first parameter request
        and cache it for subsequent parameter requests in the same trial.
        """
        # Generate complete configuration on first parameter request
        if not hasattr(trial, '_llm_config_cache'):
            trial._llm_config_cache = self._generate_complete_trial(study, trial)
        
        # Return cached parameter value
        if param_name not in trial._llm_config_cache:
            raise ValueError(f"LLM did not provide value for parameter: {param_name}")
        
        return trial._llm_config_cache[param_name]
    
    def _generate_complete_trial(self, study: optuna.Study, trial: optuna.Trial) -> Dict[str, Any]:
        """Generate complete trial configuration using LLM."""
        try:
            # Build context from current study state
            context = self.context_builder.build_context(study)
            
            # Query LLM for configuration
            llm_response = self.llm_client.generate_configuration(context, study.search_space)
            
            # Validate and clamp to search space
            validated_config = self.validator.validate_and_clamp(
                llm_response.parameters, 
                study.search_space
            )
            
            # Track for performance monitoring
            self.performance_monitor.track_suggestion(validated_config, trial.number)
            
            return validated_config
            
        except Exception as e:
            # Log error and re-raise - let Optuna handle it
            logger.error(f"LLM configuration generation failed for trial {trial.number}: {e}")
            raise LLMSamplingError(f"Failed to generate LLM configuration: {e}")
```

### 2. Context Builder

Extracts relevant information from the Optuna study to build LLM context.

```python
class ContextBuilder:
    """Builds comprehensive context for LLM queries from Optuna study state."""
    
    def __init__(self, max_context_trials: int = 20):
        self.max_context_trials = max_context_trials
    
    def build_context(self, study: optuna.Study) -> OptimizationContext:
        """Build complete optimization context from study state."""
        
        # Extract completed trials
        completed_trials = [
            trial for trial in study.trials 
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        
        context = OptimizationContext(
            # Basic study information
            objective_name=study.study_name or "optimization_objective",
            objective_direction=study.direction.name,
            n_trials_completed=len(completed_trials),
            
            # Search space
            search_space=self._format_search_space(study.search_space),
            
            # Trial history  
            recent_trials=self._format_trials(completed_trials[-self.max_context_trials:]),
            
            # Best results
            best_trial=self._format_trial(study.best_trial) if study.best_trial else None,
            
            # Progress analysis
            optimization_progress=self._analyze_progress(completed_trials),
            
            # User-provided context
            problem_description=study.user_attrs.get('problem_description', 'No description provided'),
            problem_type=study.user_attrs.get('problem_type', 'unknown'),
            additional_context=study.user_attrs.get('llm_context', {})
        )
        
        return context
    
    def _format_search_space(self, search_space: Dict) -> Dict[str, Any]:
        """Convert Optuna search space to LLM-friendly format."""
        formatted_space = {}
        
        for param_name, distribution in search_space.items():
            if isinstance(distribution, optuna.distributions.FloatDistribution):
                formatted_space[param_name] = {
                    'type': 'float',
                    'low': distribution.low,
                    'high': distribution.high,
                    'log': distribution.log,
                    'step': distribution.step
                }
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                formatted_space[param_name] = {
                    'type': 'int',
                    'low': distribution.low, 
                    'high': distribution.high,
                    'step': distribution.step
                }
            elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                formatted_space[param_name] = {
                    'type': 'categorical',
                    'choices': list(distribution.choices)
                }
        
        return formatted_space
    
    def _analyze_progress(self, completed_trials: List[optuna.Trial]) -> Dict[str, Any]:
        """Analyze optimization progress and trends."""
        if len(completed_trials) < 3:
            return {
                'stage': 'early_exploration',
                'trend': 'insufficient_data',
                'trials_since_improvement': 0,
                'best_value_trend': None
            }
        
        # Extract values for trend analysis
        values = [trial.value for trial in completed_trials]
        recent_values = values[-10:]  # Last 10 trials
        
        # Determine trend (assuming minimization)
        if len(recent_values) >= 5:
            recent_best = min(recent_values)
            older_best = min(values[:-5][-5:]) if len(values) > 5 else recent_best
            
            if recent_best < older_best * 0.95:  # 5% improvement
                trend = 'improving'
            elif recent_best > older_best * 1.05:  # 5% worse
                trend = 'degrading' 
            else:
                trend = 'plateauing'
        else:
            trend = 'insufficient_data'
        
        # Determine optimization stage
        n_trials = len(completed_trials)
        if n_trials < 10:
            stage = 'early_exploration'
        elif n_trials < 50:
            stage = 'active_search'
        else:
            stage = 'refinement'
        
        # Calculate trials since last improvement
        best_value = min(values)
        trials_since_improvement = 0
        for i in range(len(values) - 1, -1, -1):
            if values[i] == best_value:
                break
            trials_since_improvement += 1
        
        return {
            'stage': stage,
            'trend': trend,
            'trials_since_improvement': trials_since_improvement,
            'best_value_trend': recent_values[-5:] if len(recent_values) >= 5 else None,
            'convergence_indicator': self._assess_convergence(values)
        }
```

### 3. LLM Client

Handles LLM communication with structured output parsing.

```python
class LLMClient:
    """Handles LLM communication with robust error handling and structured output."""
    
    def __init__(self, model: str, temperature: float, timeout: int):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        
        # Enable LiteLLM features
        litellm.enable_json_schema_validation = True
        litellm.set_verbose = False
    
    def generate_configuration(self, 
                             context: OptimizationContext, 
                             search_space: Dict) -> TrialConfiguration:
        """Generate complete trial configuration using LLM."""
        
        prompt = self._build_prompt(context)
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=TrialConfiguration,
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            # Extract structured response
            config = response.choices[0].message.content
            
            # Additional validation
            self._validate_response_completeness(config, search_space)
            
            return config
            
        except litellm.Timeout:
            raise LLMTimeoutError(f"LLM request timed out after {self.timeout}s")
        except litellm.BadRequestError as e:
            raise LLMRequestError(f"Invalid request to LLM: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected LLM error: {e}")
    
    def _build_prompt(self, context: OptimizationContext) -> str:
        """Build comprehensive prompt for LLM configuration generation."""
        
        prompt_parts = [
            "You are an expert hyperparameter optimization assistant.",
            "",
            "OPTIMIZATION CONTEXT:",
            f"- Objective: {context.objective_direction} '{context.objective_name}'",
            f"- Problem Type: {context.problem_type}",
            f"- Description: {context.problem_description}",
            f"- Trials completed: {context.n_trials_completed}",
            f"- Optimization stage: {context.optimization_progress['stage']}",
            f"- Recent trend: {context.optimization_progress['trend']}",
            "",
            "SEARCH SPACE:",
            self._format_search_space_for_prompt(context.search_space),
            ""
        ]
        
        if context.best_trial:
            prompt_parts.extend([
                "BEST RESULT SO FAR:",
                f"- Value: {context.best_trial['value']}",
                f"- Parameters: {json.dumps(context.best_trial['params'], indent=2)}",
                ""
            ])
        
        if context.recent_trials:
            prompt_parts.extend([
                "RECENT TRIAL RESULTS:",
                self._format_recent_trials_for_prompt(context.recent_trials[-5:]),
                ""
            ])
        
        prompt_parts.extend([
            "TASK:",
            "Suggest the next hyperparameter configuration to test. Consider:",
            "1. Parameter relationships and known interactions", 
            "2. Successful patterns from recent trials",
            "3. Unexplored regions of the parameter space",
            "4. Current optimization stage and trends",
            "5. Domain knowledge for this problem type",
            "",
            "Provide a complete configuration with brief reasoning for your choices."
        ])
        
        return "\n".join(prompt_parts)
```

### 4. Pydantic Models

Define structured output format for LLM responses.

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class TrialConfiguration(BaseModel):
    """Structured LLM response for trial configuration."""
    
    parameters: Dict[str, Any] = Field(
        description="Complete hyperparameter configuration for the trial"
    )
    reasoning: str = Field(
        description="Brief explanation of the configuration choices"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence level in this configuration (0-1)",
        ge=0.0,
        le=1.0
    )
    exploration_factor: Optional[str] = Field(
        default=None,
        description="Whether this is exploratory or exploitative",
        pattern="^(exploratory|exploitative|balanced)$"
    )

class OptimizationContext(BaseModel):
    """Complete context for LLM optimization queries."""
    
    objective_name: str
    objective_direction: str  # "MINIMIZE" or "MAXIMIZE"
    n_trials_completed: int
    search_space: Dict[str, Any]
    recent_trials: List[Dict[str, Any]]
    best_trial: Optional[Dict[str, Any]]
    optimization_progress: Dict[str, Any]
    problem_description: str
    problem_type: str
    additional_context: Dict[str, Any] = Field(default_factory=dict)

class FormattedTrial(BaseModel):
    """Formatted trial information for LLM context."""
    
    trial_number: int
    parameters: Dict[str, Any]
    value: float
    duration: Optional[float] = None
    state: str = "COMPLETE"
```

### 5. Parameter Validator

Ensures LLM responses conform to search space constraints.

```python
class ParameterValidator:
    """Validates and clamps LLM-generated parameters to search space bounds."""
    
    def validate_and_clamp(self, 
                          config: Dict[str, Any], 
                          search_space: Dict) -> Dict[str, Any]:
        """Validate complete configuration against search space."""
        
        validated_config = {}
        
        for param_name, distribution in search_space.items():
            if param_name not in config:
                raise ValidationError(f"Missing required parameter: {param_name}")
            
            raw_value = config[param_name]
            validated_value = self._validate_parameter(raw_value, distribution)
            validated_config[param_name] = validated_value
        
        return validated_config
    
    def _validate_parameter(self, 
                           value: Any, 
                           distribution: optuna.distributions.BaseDistribution) -> Any:
        """Validate and clamp single parameter."""
        
        if isinstance(distribution, optuna.distributions.FloatDistribution):
            return self._validate_float_parameter(value, distribution)
        elif isinstance(distribution, optuna.distributions.IntDistribution):
            return self._validate_int_parameter(value, distribution)
        elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
            return self._validate_categorical_parameter(value, distribution)
        else:
            raise ValidationError(f"Unsupported distribution type: {type(distribution)}")
    
    def _validate_float_parameter(self, 
                                 value: Any, 
                                 dist: optuna.distributions.FloatDistribution) -> float:
        """Validate and clamp float parameter."""
        try:
            float_val = float(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value {value}, using midpoint")
            return (dist.low + dist.high) / 2
        
        # Clamp to bounds
        clamped = max(dist.low, min(dist.high, float_val))
        
        # Handle step constraints
        if dist.step is not None:
            steps_from_low = round((clamped - dist.low) / dist.step)
            clamped = dist.low + steps_from_low * dist.step
            clamped = max(dist.low, min(dist.high, clamped))
        
        return clamped
```

## Error Handling Strategy

### Exception Hierarchy
```python
class LLMSamplingError(Exception):
    """Base exception for LLM sampling failures."""
    pass

class LLMTimeoutError(LLMSamplingError):
    """LLM request timed out."""
    pass

class LLMRequestError(LLMSamplingError):
    """Invalid request to LLM API."""
    pass

class ValidationError(LLMSamplingError):
    """Parameter validation failed."""
    pass
```

### Retry Logic
```python
def with_retry(max_attempts=3, backoff_factor=2):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (LLMTimeoutError, LLMRequestError) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = backoff_factor ** attempt
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
                        
            raise last_exception
        return wrapper
    return decorator
```

## Integration with Optuna

### Usage Pattern
```python
# Standard Optuna usage with LLM guidance
import optuna
from llm_guided_optuna import LLMGuidedSampler

# Create LLM-guided sampler
sampler = LLMGuidedSampler(
    model="gpt-4o-2024-08-06",
    temperature=0.3
)

# Create study with LLM sampler
study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    study_name="neural_network_hyperparameter_optimization"
)

# Provide context for better LLM guidance
study.set_user_attr("problem_type", "neural_network_training")
study.set_user_attr("problem_description", "CNN optimization for image classification")
study.set_user_attr("llm_context", {
    "dataset": "CIFAR-10",
    "model_architecture": "ResNet-18",
    "constraints": ["training_time < 1 hour", "memory < 8GB"]
})

# Define objective function
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 256, step=16)  
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    # Train model and return validation loss
    validation_loss = train_model(lr, batch_size, dropout, weight_decay)
    return validation_loss

# Run optimization
study.optimize(objective, n_trials=100)

print("Best trial:", study.best_trial.params)
print("Best value:", study.best_value)
```

This architecture provides a clean, maintainable, and extensible foundation for LLM-guided hyperparameter optimization while maintaining full compatibility with Optuna's existing ecosystem.