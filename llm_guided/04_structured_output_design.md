# Structured Output Design

## Overview

This document details the structured output strategy for extracting reliable hyperparameter configurations from LLM responses using Pydantic models and LiteLLM's structured output capabilities.

## Core Requirements

1. **Type Safety**: Guarantee that LLM responses conform to expected data structures
2. **Validation**: Ensure parameters are within valid ranges and types
3. **Error Handling**: Graceful degradation when LLM responses are malformed
4. **Extensibility**: Support for different types of optimization problems
5. **Debugging**: Clear error messages and response traceability

## Pydantic Model Hierarchy

### Base Configuration Model

```python
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union
from enum import Enum

class OptimizationStage(str, Enum):
    """Current stage of optimization process."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    REFINEMENT = "refinement"

class ExplorationStrategy(str, Enum):
    """Strategy for parameter exploration."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

class TrialConfiguration(BaseModel):
    """
    Primary model for LLM-generated trial configurations.
    
    This is the main structured output format that LLMs must conform to
    when suggesting hyperparameter configurations.
    """
    
    parameters: Dict[str, Union[float, int, str, bool]] = Field(
        description="Complete hyperparameter configuration mapping parameter names to values",
        example={
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout": 0.3,
            "optimizer": "adam"
        }
    )
    
    reasoning: str = Field(
        description="Brief explanation (1-3 sentences) of why these parameters were chosen",
        min_length=10,
        max_length=500,
        example="Chose moderate learning rate for stable training, smaller batch size for better gradient estimates, and moderate dropout to prevent overfitting."
    )
    
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence level in this configuration (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.75
    )
    
    strategy: Optional[ExplorationStrategy] = Field(
        default=ExplorationStrategy.BALANCED,
        description="Whether this configuration is conservative, balanced, or aggressive"
    )
    
    expected_performance: Optional[str] = Field(
        default=None,
        description="Brief prediction of expected performance",
        max_length=200,
        example="Expected to achieve 85-90% validation accuracy with moderate training time"
    )
    
    @validator('parameters')
    def parameters_not_empty(cls, v):
        """Ensure parameters dictionary is not empty."""
        if not v:
            raise ValueError("Parameters dictionary cannot be empty")
        return v
    
    @validator('reasoning')
    def reasoning_meaningful(cls, v):
        """Ensure reasoning is meaningful and not just boilerplate."""
        boilerplate_phrases = [
            "good parameters",
            "should work well", 
            "typical values",
            "standard configuration"
        ]
        
        if any(phrase in v.lower() for phrase in boilerplate_phrases):
            raise ValueError("Reasoning appears to be boilerplate - provide specific justification")
        
        return v

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment
        schema_extra = {
            "example": {
                "parameters": {
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "dropout": 0.2,
                    "weight_decay": 1e-4
                },
                "reasoning": "Selected moderate learning rate for stable convergence, larger batch size for efficiency, low dropout to preserve model capacity, and small weight decay for regularization.",
                "confidence": 0.8,
                "strategy": "balanced",
                "expected_performance": "Should achieve 87-92% validation accuracy within 50 epochs"
            }
        }
```

### Batch Configuration Model

```python
class BatchTrialConfiguration(BaseModel):
    """
    Model for requesting multiple trial configurations at once.
    Useful for generating diverse exploration strategies.
    """
    
    configurations: List[TrialConfiguration] = Field(
        description="List of diverse trial configurations",
        min_items=1,
        max_items=10
    )
    
    diversity_strategy: str = Field(
        description="Explanation of how diversity was achieved across configurations",
        min_length=20,
        max_length=300,
        example="Generated configurations exploring different learning rate regimes: conservative (1e-4), moderate (1e-3), and aggressive (1e-2), while varying batch sizes and regularization accordingly."
    )
    
    @validator('configurations')
    def ensure_diversity(cls, v):
        """Ensure configurations are sufficiently diverse."""
        if len(v) < 2:
            return v
        
        # Check parameter diversity (simplified check)
        param_names = set(v[0].parameters.keys())
        
        # Verify all configs have same parameter structure
        for config in v:
            if set(config.parameters.keys()) != param_names:
                raise ValueError("All configurations must have the same parameter structure")
        
        return v
```

### Context Models

```python
class SearchSpaceDefinition(BaseModel):
    """Structured representation of Optuna search space for LLM context."""
    
    parameter_name: str
    parameter_type: str = Field(pattern="^(float|int|categorical)$")
    
    # For numeric parameters
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    log_scale: bool = False
    step: Optional[Union[float, int]] = None
    
    # For categorical parameters
    choices: Optional[List[Union[str, int, float]]] = None
    
    # Metadata
    description: Optional[str] = None
    typical_range: Optional[str] = None
    
    @validator('choices')
    def choices_required_for_categorical(cls, v, values):
        """Ensure choices are provided for categorical parameters."""
        if values.get('parameter_type') == 'categorical' and not v:
            raise ValueError("Choices must be provided for categorical parameters")
        return v

class TrialResult(BaseModel):
    """Structured representation of completed trial results."""
    
    trial_number: int = Field(ge=0)
    parameters: Dict[str, Union[float, int, str, bool]]
    value: float
    duration: Optional[float] = Field(default=None, ge=0.0, description="Trial duration in seconds")
    state: str = Field(default="COMPLETE", pattern="^(COMPLETE|FAIL|PRUNED)$")
    
    # Optional metadata
    intermediate_values: Optional[List[float]] = Field(default=None, description="Intermediate values during training")
    user_attrs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional trial metadata")

class OptimizationContext(BaseModel):
    """Complete context provided to LLM for generating configurations."""
    
    # Basic information
    objective_name: str
    objective_direction: str = Field(pattern="^(MINIMIZE|MAXIMIZE)$")
    n_trials_completed: int = Field(ge=0)
    
    # Search space
    search_space: List[SearchSpaceDefinition]
    
    # Historical results
    recent_trials: List[TrialResult] = Field(default_factory=list, max_items=20)
    best_trial: Optional[TrialResult] = None
    
    # Progress analysis
    current_stage: OptimizationStage
    recent_trend: str = Field(
        pattern="^(improving|plateauing|degrading|insufficient_data)$"
    )
    trials_since_improvement: int = Field(ge=0)
    
    # Problem context
    problem_type: str = Field(
        default="unknown",
        example="neural_network_training"
    )
    problem_description: str = Field(
        default="No description provided",
        max_length=1000
    )
    
    # Additional context
    constraints: List[str] = Field(
        default_factory=list,
        description="Any constraints or requirements",
        example=["training_time < 2 hours", "memory_usage < 8GB"]
    )
    
    domain_knowledge: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Domain-specific context and knowledge"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "objective_name": "validation_loss",
                "objective_direction": "MINIMIZE",
                "n_trials_completed": 15,
                "current_stage": "exploration",
                "recent_trend": "improving",
                "trials_since_improvement": 2,
                "problem_type": "neural_network_training",
                "problem_description": "CNN training for image classification on CIFAR-10"
            }
        }
```

## LiteLLM Integration

### Configuration and Setup

```python
import litellm
from litellm import completion
from typing import Type, TypeVar

# Enable structured output features
litellm.enable_json_schema_validation = True
litellm.set_verbose = False  # Set to True for debugging

T = TypeVar('T', bound=BaseModel)

class StructuredLLMClient:
    """Client for structured LLM interactions using LiteLLM."""
    
    def __init__(self, 
                 model: str = "gpt-4o-2024-08-06",
                 temperature: float = 0.3,
                 timeout: int = 30,
                 max_retries: int = 3):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        
    def generate_structured_response(self, 
                                   prompt: str, 
                                   response_model: Type[T],
                                   temperature: Optional[float] = None) -> T:
        """Generate structured response using specified Pydantic model."""
        
        effective_temperature = temperature if temperature is not None else self.temperature
        
        for attempt in range(self.max_retries):
            try:
                response = completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=response_model,
                    temperature=effective_temperature,
                    timeout=self.timeout
                )
                
                # Extract and validate structured response
                structured_response = response.choices[0].message.content
                
                # Additional validation if needed
                self._post_process_response(structured_response, response_model)
                
                return structured_response
                
            except litellm.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                    continue
                raise LLMTimeoutError(f"Request timed out after {self.max_retries} attempts")
                
            except litellm.BadRequestError as e:
                # Don't retry bad requests
                raise LLMRequestError(f"Invalid request: {e}")
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Error on attempt {attempt + 1}: {e}")
                    continue
                raise LLMError(f"Failed after {self.max_retries} attempts: {e}")
    
    def _post_process_response(self, response: T, model_class: Type[T]) -> None:
        """Additional validation and post-processing of structured responses."""
        
        # Model-specific validation
        if isinstance(response, TrialConfiguration):
            self._validate_trial_configuration(response)
        elif isinstance(response, BatchTrialConfiguration):
            self._validate_batch_configuration(response)
    
    def _validate_trial_configuration(self, config: TrialConfiguration) -> None:
        """Additional validation for trial configurations."""
        
        # Check for suspicious parameter values
        for param_name, value in config.parameters.items():
            if isinstance(value, (int, float)):
                if not (-1e6 <= value <= 1e6):  # Reasonable bounds check
                    logger.warning(f"Suspicious parameter value: {param_name} = {value}")
        
        # Check reasoning quality
        if len(config.reasoning.split()) < 5:
            logger.warning("Reasoning appears too brief")
```

### Prompt Integration

```python
class PromptBuilder:
    """Builds prompts optimized for structured output generation."""
    
    def build_configuration_prompt(self, context: OptimizationContext) -> str:
        """Build prompt for trial configuration generation."""
        
        prompt_sections = [
            self._build_header(),
            self._build_context_section(context),
            self._build_search_space_section(context.search_space),
            self._build_history_section(context),
            self._build_task_section(context),
            self._build_format_instructions()
        ]
        
        return "\n\n".join(prompt_sections)
    
    def _build_format_instructions(self) -> str:
        """Provide clear format instructions for structured output."""
        
        return """
FORMAT REQUIREMENTS:
Your response must be valid JSON matching this exact structure:
{
  "parameters": {
    "param1": value1,
    "param2": value2,
    ...
  },
  "reasoning": "Brief explanation of parameter choices (1-3 sentences)",
  "confidence": 0.8,
  "strategy": "balanced",
  "expected_performance": "Brief performance prediction"
}

IMPORTANT:
- Include ALL required parameters from the search space
- Use exact parameter names as specified
- Provide specific numerical values within the given ranges
- Keep reasoning concise but informative
- Confidence should reflect your certainty (0.0 to 1.0)
"""
```

## Error Handling and Recovery

### Validation Error Handling

```python
class ValidationErrorHandler:
    """Handles validation errors with recovery strategies."""
    
    def __init__(self, search_space: Dict):
        self.search_space = search_space
    
    def handle_validation_error(self, 
                              raw_response: str, 
                              error: Exception,
                              context: OptimizationContext) -> TrialConfiguration:
        """Attempt to recover from validation errors."""
        
        logger.warning(f"Validation error: {error}")
        
        # Strategy 1: Try to parse partial JSON
        partial_config = self._extract_partial_config(raw_response)
        if partial_config:
            completed_config = self._complete_missing_parameters(partial_config, context)
            if completed_config:
                return completed_config
        
        # Strategy 2: Use regex to extract parameter values
        extracted_params = self._regex_extract_parameters(raw_response)
        if extracted_params:
            return self._build_config_from_extracted_params(extracted_params, context)
        
        # Strategy 3: Generate fallback configuration
        return self._generate_fallback_config(context)
    
    def _extract_partial_config(self, raw_response: str) -> Optional[Dict]:
        """Try to extract partial configuration from malformed JSON."""
        
        # Try to find JSON-like structures
        import re
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, raw_response, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _complete_missing_parameters(self, 
                                   partial_config: Dict, 
                                   context: OptimizationContext) -> Optional[TrialConfiguration]:
        """Fill in missing parameters using context and defaults."""
        
        if 'parameters' not in partial_config:
            return None
        
        parameters = partial_config['parameters']
        
        # Fill missing parameters with reasonable defaults
        for param_name, param_def in self.search_space.items():
            if param_name not in parameters:
                default_value = self._get_default_value(param_def, context)
                parameters[param_name] = default_value
        
        # Create configuration with defaults for missing fields
        try:
            return TrialConfiguration(
                parameters=parameters,
                reasoning=partial_config.get('reasoning', 'Recovered from partial response'),
                confidence=partial_config.get('confidence', 0.5),
                strategy=partial_config.get('strategy', 'balanced')
            )
        except Exception:
            return None
    
    def _generate_fallback_config(self, context: OptimizationContext) -> TrialConfiguration:
        """Generate reasonable fallback configuration when all else fails."""
        
        parameters = {}
        
        for param_name, param_def in self.search_space.items():
            if param_def['type'] == 'float':
                # Use geometric mean for log-scale, arithmetic mean otherwise
                if param_def.get('log_scale', False):
                    value = (param_def['low'] * param_def['high']) ** 0.5
                else:
                    value = (param_def['low'] + param_def['high']) / 2
                parameters[param_name] = value
                
            elif param_def['type'] == 'int':
                value = (param_def['low'] + param_def['high']) // 2
                parameters[param_name] = value
                
            elif param_def['type'] == 'categorical':
                # Choose first option as default
                parameters[param_name] = param_def['choices'][0]
        
        return TrialConfiguration(
            parameters=parameters,
            reasoning="Fallback configuration using parameter midpoints due to LLM response parsing failure",
            confidence=0.3,
            strategy="conservative"
        )
```

## Testing and Validation

### Unit Tests for Models

```python
import pytest
from pydantic import ValidationError

class TestTrialConfiguration:
    """Test cases for TrialConfiguration model."""
    
    def test_valid_configuration(self):
        """Test valid configuration creation."""
        config = TrialConfiguration(
            parameters={"lr": 0.01, "batch_size": 32},
            reasoning="Test configuration for unit tests"
        )
        
        assert config.parameters["lr"] == 0.01
        assert config.strategy == ExplorationStrategy.BALANCED  # Default
    
    def test_empty_parameters_rejected(self):
        """Test that empty parameters are rejected."""
        with pytest.raises(ValidationError):
            TrialConfiguration(
                parameters={},
                reasoning="Empty parameters should fail"
            )
    
    def test_boilerplate_reasoning_rejected(self):
        """Test that boilerplate reasoning is rejected."""
        with pytest.raises(ValidationError):
            TrialConfiguration(
                parameters={"lr": 0.01},
                reasoning="These are good parameters that should work well"
            )
    
    def test_confidence_bounds(self):
        """Test confidence value validation."""
        # Valid confidence
        config = TrialConfiguration(
            parameters={"lr": 0.01},
            reasoning="Specific reasoning about learning rate choice",
            confidence=0.8
        )
        assert config.confidence == 0.8
        
        # Invalid confidence
        with pytest.raises(ValidationError):
            TrialConfiguration(
                parameters={"lr": 0.01},
                reasoning="Valid reasoning",
                confidence=1.5  # > 1.0
            )
```

### Integration Tests

```python
class TestStructuredLLMClient:
    """Integration tests for structured LLM client."""
    
    @pytest.fixture
    def llm_client(self):
        return StructuredLLMClient(model="gpt-4o-2024-08-06")
    
    def test_generate_trial_configuration(self, llm_client):
        """Test generating valid trial configuration."""
        
        prompt = """
        Generate a hyperparameter configuration for neural network training.
        Parameters: learning_rate (0.0001 to 0.1), batch_size (16 to 128), dropout (0.0 to 0.5)
        """
        
        config = llm_client.generate_structured_response(
            prompt=prompt,
            response_model=TrialConfiguration
        )
        
        assert isinstance(config, TrialConfiguration)
        assert "learning_rate" in config.parameters
        assert "batch_size" in config.parameters  
        assert "dropout" in config.parameters
        assert len(config.reasoning) > 10
```

This structured output design ensures reliable, type-safe communication with LLMs while providing robust error handling and recovery mechanisms for production use.