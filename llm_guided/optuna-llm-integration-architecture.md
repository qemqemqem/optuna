# LLM-Guided Optuna Integration Architecture

## Core Design Decisions

- **Trial-Level**: LLM suggests complete trial configurations (all parameters together)
- **Every Trial**: LLM called on every single trial
- **Fresh Context**: Rebuild context from study state each time
- **No Fallback**: Pure LLM-guided approach (no fallback to traditional samplers)
- **Single Objective**: Focus on single-objective optimization

## Architecture Overview

```python
class LLMGuidedSampler(optuna.samplers.BaseSampler):
    """
    Sampler that uses LLM to suggest complete trial configurations.
    Calls LLM on every trial with fresh context from study history.
    """
    
    def __init__(self, llm_client, model="gpt-4o", temperature=0.3):
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
    
    def sample_independent(self, study, trial, param_name, param_distribution):
        # This is called by Optuna for each parameter
        # We'll generate the complete trial config and cache it
        
        if not hasattr(trial, '_llm_config'):
            # First parameter request - generate complete config
            trial._llm_config = self._generate_complete_trial(study, trial)
        
        # Return the cached value for this parameter
        return trial._llm_config[param_name]
    
    def _generate_complete_trial(self, study, trial):
        """Generate complete trial configuration using LLM"""
        context = self._build_context(study)
        llm_config = self._query_llm(context, study.search_space)
        return llm_config
```

## Context Building Strategy

```python
def _build_context(self, study):
    """Rebuild complete context from study state"""
    
    context = {
        # Basic study information
        'objective_name': study.study_name,
        'objective_direction': study.direction.name,  # MINIMIZE or MAXIMIZE
        'n_trials_completed': len(study.trials),
        
        # Search space definition
        'search_space': self._format_search_space(study.search_space),
        
        # Trial history (last 20 trials)
        'recent_trials': self._format_recent_trials(study.trials[-20:]),
        
        # Best results so far
        'best_trial': self._format_best_trial(study.best_trial) if study.best_trial else None,
        
        # Performance trends
        'optimization_progress': self._analyze_progress(study.trials),
        
        # Problem context (if provided by user)
        'problem_description': study.user_attrs.get('problem_description', 'No description provided'),
        'problem_type': study.user_attrs.get('problem_type', 'unknown'),
    }
    
    return context

def _format_search_space(self, search_space):
    """Convert Optuna search space to LLM-friendly format"""
    formatted = {}
    for param_name, distribution in search_space.items():
        if isinstance(distribution, optuna.distributions.FloatDistribution):
            formatted[param_name] = {
                'type': 'float',
                'low': distribution.low,
                'high': distribution.high,
                'log': distribution.log
            }
        elif isinstance(distribution, optuna.distributions.IntDistribution):
            formatted[param_name] = {
                'type': 'int', 
                'low': distribution.low,
                'high': distribution.high,
                'step': distribution.step
            }
        elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
            formatted[param_name] = {
                'type': 'categorical',
                'choices': distribution.choices
            }
    return formatted

def _format_recent_trials(self, trials):
    """Format recent trials for LLM context"""
    formatted_trials = []
    for trial in trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            formatted_trials.append({
                'params': trial.params,
                'value': trial.value,
                'trial_number': trial.number
            })
    return formatted_trials

def _analyze_progress(self, trials):
    """Analyze optimization progress for LLM context"""
    completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) < 3:
        return "Early stage - insufficient data for trend analysis"
    
    values = [t.value for t in completed_trials[-10:]]
    
    # Simple trend analysis
    if len(values) >= 5:
        recent_trend = "improving" if values[-1] < values[-5] else "plateauing"  # assuming minimization
    else:
        recent_trend = "insufficient_data"
    
    return {
        'recent_trend': recent_trend,
        'best_value_so_far': min(values) if values else None,
        'trials_since_improvement': self._trials_since_last_improvement(completed_trials)
    }
```

## LLM Query Implementation

```python
from pydantic import BaseModel
from typing import Dict, Any
import litellm

class TrialConfiguration(BaseModel):
    """Pydantic model for LLM-generated trial configuration"""
    parameters: Dict[str, Any]
    reasoning: str

def _query_llm(self, context, search_space):
    """Query LLM for complete trial configuration"""
    
    prompt = self._build_prompt(context)
    
    try:
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=TrialConfiguration,
            temperature=self.temperature
        )
        
        config = response.choices[0].message.content
        
        # Validate against search space
        validated_config = self._validate_and_clamp(config.parameters, search_space)
        
        return validated_config
        
    except Exception as e:
        # If LLM fails, we have a problem since there's no fallback
        # This is by design - we want to fail fast and fix issues
        raise LLMSamplingError(f"LLM query failed: {e}")

def _build_prompt(self, context):
    """Build comprehensive prompt for LLM"""
    
    prompt = f"""
You are an expert hyperparameter optimization assistant. Based on the optimization context below, suggest the next trial configuration to test.

OPTIMIZATION OBJECTIVE:
- Goal: {context['objective_direction']} the objective "{context['objective_name']}"
- Problem Type: {context['problem_type']}
- Description: {context['problem_description']}

SEARCH SPACE:
{self._format_search_space_for_prompt(context['search_space'])}

OPTIMIZATION PROGRESS:
- Trials completed: {context['n_trials_completed']}
- Current trend: {context['optimization_progress']}

BEST RESULT SO FAR:
{context['best_trial'] if context['best_trial'] else "No completed trials yet"}

RECENT TRIAL RESULTS:
{self._format_trials_for_prompt(context['recent_trials'][-5:] if context['recent_trials'] else [])}

TASK:
Suggest the next hyperparameter configuration to test. Consider:
1. The search space constraints
2. Which parameters tend to work well together  
3. The optimization progress and trends
4. Explore regions that haven't been tested much
5. Balance exploration with exploitation based on progress

Provide your suggestion as a complete parameter configuration with brief reasoning.
"""
    return prompt

def _validate_and_clamp(self, config, search_space):
    """Validate LLM config against Optuna search space and clamp to bounds"""
    validated = {}
    
    for param_name, distribution in search_space.items():
        if param_name not in config:
            raise ValueError(f"LLM didn't provide value for required parameter: {param_name}")
        
        value = config[param_name]
        
        if isinstance(distribution, optuna.distributions.FloatDistribution):
            value = max(distribution.low, min(distribution.high, float(value)))
        elif isinstance(distribution, optuna.distributions.IntDistribution):
            value = max(distribution.low, min(distribution.high, int(value)))
        elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
            if value not in distribution.choices:
                # Default to first choice if invalid
                value = distribution.choices[0]
        
        validated[param_name] = value
    
    return validated
```

## Usage Example

```python
import optuna
from llm_guided_sampler import LLMGuidedSampler
import litellm

# Set up LLM client
litellm.set_verbose = True

# Create LLM-guided sampler
sampler = LLMGuidedSampler(
    llm_client=litellm,
    model="gpt-4o-2024-08-06",
    temperature=0.3
)

# Create study with problem context
study = optuna.create_study(
    direction="minimize",
    sampler=sampler,
    study_name="neural_network_optimization"
)

# Add problem context
study.set_user_attr("problem_type", "neural_network_training")
study.set_user_attr("problem_description", "Optimizing CNN for CIFAR-10 classification")

# Define objective function
def objective(trial):
    # Get LLM-suggested parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 256, step=16)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Train model with these parameters
    accuracy = train_model(learning_rate, batch_size, dropout)
    
    return 1.0 - accuracy  # Minimize error (maximize accuracy)

# Run optimization - LLM will be called for every trial
study.optimize(objective, n_trials=50)

print(f"Best trial: {study.best_trial}")
```

## Key Benefits of This Architecture

1. **Simple Integration**: Drop-in replacement for any Optuna sampler
2. **Complete Context**: LLM sees full optimization state every time
3. **Parameter Relationships**: LLM can reason about parameter interactions
4. **No State Management**: Fresh context eliminates state synchronization issues  
5. **Transparent**: Easy to debug - just look at the prompt and response
6. **Flexible**: Can easily modify prompt engineering without changing architecture

## Error Handling Strategy

```python
class LLMSamplingError(Exception):
    """Custom exception for LLM sampling failures"""
    pass

# In the sampler:
def sample_independent(self, study, trial, param_name, param_distribution):
    try:
        if not hasattr(trial, '_llm_config'):
            trial._llm_config = self._generate_complete_trial(study, trial)
        return trial._llm_config[param_name]
    except LLMSamplingError:
        # Log the error and re-raise - let the study handle it
        logger.error(f"LLM sampling failed for trial {trial.number}")
        raise
    except Exception as e:
        # Unexpected error - wrap and re-raise
        raise LLMSamplingError(f"Unexpected error in LLM sampling: {e}")
```

This architecture is clean, simple, and focuses entirely on the LLM-guided approach. Every trial gets the full benefit of LLM reasoning about the complete parameter space and optimization history.