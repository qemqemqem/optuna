# How LLM-Guided Optuna Works

## Overview

LLM-Guided Optuna replaces traditional mathematical sampling strategies (TPE, CMA-ES, etc.) with intelligent parameter suggestions from Large Language Models. The system analyzes optimization context and generates informed hyperparameter configurations.

## Architecture Flow

```
Study State → Context Building → LLM Querying → Response Parsing → Parameter Validation → Trial Execution
```

### 1. Context Building (`context_builder.py`)

For each trial, the system extracts comprehensive context:

**Optimization History**:
- Recent trial results with parameters and values
- Best trial found so far
- Progress trends (improving, plateauing, degrading)
- Number of trials since last improvement

**Search Space Analysis**:
- Parameter types (float, int, categorical) and ranges
- Log scaling information
- Parameter descriptions and domain knowledge

**Progress Analysis**:
- Current optimization stage (early_exploration, active_search, focused_optimization, refinement)
- Trend strength and convergence indicators
- Strategic recommendations

**Problem Context**:
- Problem type (e.g., "neural_network_training")
- Domain-specific knowledge and constraints
- Previous successful configurations

### 2. LLM Communication (`llm_client.py`)

**Prompt Generation**:
The system creates rich prompts containing:
```
- Optimization objective and direction
- Complete search space with parameter descriptions  
- Recent trial history and trends
- Best result so far
- Domain knowledge and constraints
- Strategic guidance based on optimization stage
```

**Structured Output**:
Uses LiteLLM with Pydantic models to ensure reliable parsing:
```python
class TrialConfiguration(BaseModel):
    parameters: Dict[str, Union[float, int, str, bool]]
    reasoning: str  # LLM's explanation
    confidence: float  # LLM's confidence level
    strategy: str  # Exploration approach
```

**Error Handling**:
- Exponential backoff retry logic
- Timeout management
- Graceful degradation for API failures

### 3. Parameter Validation (`parameter_validator.py`)

**Constraint Enforcement**:
- Clamps numeric values to search space bounds
- Validates categorical choices with fuzzy matching
- Handles log-scale parameters correctly
- Respects step size constraints

**Error Recovery**:
- Partial JSON parsing for malformed responses
- Regex extraction of parameter values
- Default value generation for missing parameters

**Validation Statistics**:
Tracks clamping and correction operations for monitoring.

### 4. Integration with Optuna (`sampler.py`)

**Sampler Interface**:
Implements Optuna's `BaseSampler` interface:
- `sample_independent()`: Called for each parameter
- `infer_relative_search_space()`: Returns complete search space
- `after_trial()`: Cleanup and learning opportunities

**Configuration Caching**:
Generates complete configurations on first parameter request and caches for subsequent parameters in the same trial.

**Performance Monitoring**:
Tracks success rates, timing, fallback usage, and validation statistics.

## Key Design Decisions

### Trial-Level vs Parameter-Level Generation

**Chosen**: Trial-Level (Complete Configuration)
- LLMs can reason about parameter relationships
- Avoids inconsistent parameter combinations
- Leverages domain knowledge about successful patterns

**Alternative**: Parameter-Level (Individual Parameters)
- Would lose inter-parameter reasoning
- Risk of suboptimal combinations

### Context Freshness Strategy

**Chosen**: Fresh Context Reconstruction
- Always uses current study state
- Incorporates latest trial results and trends  
- No stale cached context

**Alternative**: Incremental Context Updates
- More efficient but risks staleness
- Complex state management

### Fallback Strategy

**Chosen**: Intelligent Error Recovery + Fallbacks
- Multi-stage recovery (partial parsing, regex, defaults)
- Graceful degradation rather than failure
- Maintains optimization progress

**Alternative**: Traditional Sampler Fallback
- Would break pure LLM-guided approach
- Users requested no traditional fallbacks

## Performance Characteristics

**Latency**: 1-5 seconds per trial (LLM API dependent)
**Success Rate**: ~95% with proper API keys and limits
**Context Size**: 2000-4000 characters (token-efficient)
**Memory Usage**: Stateless - scales with study size only

## Comparison with Traditional Samplers

| Aspect | Traditional (TPE/CMA-ES) | LLM-Guided |
|--------|--------------------------|------------|
| **Domain Knowledge** | None | Extensive ML literature |
| **Parameter Relationships** | Statistical only | Semantic understanding |
| **Exploration Strategy** | Mathematical | Reasoning-based |
| **Interpretability** | Black box | Explainable reasoning |
| **Cold Start** | Poor | Excellent (pre-trained knowledge) |
| **Latency** | <1ms | 1-5s |
| **Cost** | Free | API costs |

## Extensions and Future Work

**Ensemble Methods**: Combine multiple LLM suggestions
**Learning Integration**: Fine-tune models on optimization history
**Multi-Objective**: Extend to Pareto frontier optimization
**Domain Specialization**: Specialized models for specific problem types