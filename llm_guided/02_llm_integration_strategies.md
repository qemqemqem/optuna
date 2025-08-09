# LLM Integration Strategies and Best Practices

## LLM Interaction Patterns

### Primary Method: Configuration Generation (Recommended)

**Approach**: Ask LLM to generate complete hyperparameter configurations
```
"Based on this optimization context, suggest a promising hyperparameter configuration for the next trial"
```

**Advantages**:
- Captures holistic understanding of parameter relationships
- Cost-effective (single API call per trial)
- Natural for LLM reasoning patterns
- Preserves domain knowledge about parameter interactions

**Implementation Pattern**:
```python
def generate_trial_config(context):
    prompt = build_comprehensive_prompt(context)
    response = llm_client.complete(prompt, response_format=TrialConfiguration)
    return validate_and_clamp(response.parameters, search_space)
```

### Secondary Method: Batch Scoring (For Dense Evaluation)

**Approach**: Generate multiple candidates and have LLM score them
```
"Rate these 10 configurations (0-100) based on likelihood of good performance"
```

**Use Cases**:
- When you need dense coverage of parameter space
- For validation of LLM preferences
- When exploration needs to be more systematic

**Implementation Pattern**:
```python
def score_candidate_configs(candidates, context):
    prompt = build_scoring_prompt(candidates, context)
    scores = llm_client.complete(prompt, response_format=ConfigurationScores)
    return build_density_from_scores(candidates, scores.ratings)
```

## Prompt Engineering Framework

### Context Structure
Every LLM query should include:

1. **Problem Context**
   - Objective function description
   - Problem type (neural network training, AutoML, etc.)
   - Dataset characteristics
   - Model architecture details

2. **Search Space Definition**
   - Parameter names, types, and bounds
   - Any known parameter relationships
   - Scale information (linear vs log scales)

3. **Optimization History**
   - Best configuration found so far
   - Recent trial results (last 5-10 trials)
   - Overall optimization progress and trends
   - Number of trials completed

4. **Strategic Context**
   - Current optimization stage (exploration vs exploitation)
   - Time/budget constraints
   - Performance targets or requirements

### Prompt Template Structure
```python
PROMPT_TEMPLATE = """
You are an expert hyperparameter optimization assistant.

PROBLEM CONTEXT:
- Objective: {objective_direction} {objective_name}
- Problem Type: {problem_type}  
- Description: {problem_description}

SEARCH SPACE:
{format_search_space_with_bounds_and_scales}

OPTIMIZATION PROGRESS:
- Trials completed: {n_trials}
- Best result so far: {best_value} with {best_params}
- Recent trend: {trend_analysis}

RECENT TRIAL RESULTS:
{format_recent_trials_with_performance}

TASK:
Suggest the next hyperparameter configuration to test. Consider:
1. Parameter relationships and interactions
2. Promising regions based on recent results
3. Balance between exploration and exploitation
4. Known best practices for this problem type

Provide a complete configuration with brief reasoning.
"""
```

### Chain-of-Thought Integration
Encourage systematic reasoning:
```python
REASONING_PROMPT = """
Let me analyze this step by step:

1. CURRENT STATE ANALYSIS:
   - What patterns do I see in the successful trials?
   - Which parameters seem most important?
   - Where are the promising regions?

2. PARAMETER RELATIONSHIPS:
   - Which parameters should be adjusted together?
   - What are typical successful combinations?
   - Any parameters that conflict with each other?

3. EXPLORATION STRATEGY:
   - Should I explore new regions or refine current best?
   - What gaps exist in the tested parameter space?
   - How aggressive should this next trial be?

4. CONFIGURATION PROPOSAL:
   Based on this analysis, here's my suggested configuration:
   {provide configuration with reasoning}
"""
```

## Consistency and Reliability Strategies

### Temperature and Sampling Control
```python
# For consistent, focused suggestions
FOCUSED_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.9,
    "frequency_penalty": 0.0
}

# For diverse exploration
DIVERSE_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95, 
    "frequency_penalty": 0.2
}

# Adaptive temperature based on optimization stage
def get_temperature(n_trials, total_trials):
    # Start diverse, become focused
    progress = n_trials / total_trials
    return 0.8 * (1 - progress) + 0.1 * progress
```

### Ensemble Consensus
```python
def ensemble_llm_suggestion(context, n_runs=3):
    suggestions = []
    for i in range(n_runs):
        config = llm_client.generate_config(
            context, 
            temperature=0.2 + 0.1 * i  # Slight temperature variation
        )
        suggestions.append(config)
    
    # Method 1: Parameter-wise median
    consensus_config = {}
    for param in search_space:
        values = [s[param] for s in suggestions]
        consensus_config[param] = np.median(values)
    
    return consensus_config
```

### Validation and Quality Control
```python
def validate_llm_suggestion(config, context):
    quality_checks = {}
    
    # Bounds checking
    quality_checks['in_bounds'] = all(
        is_within_bounds(config[p], search_space[p]) 
        for p in search_space
    )
    
    # Reasonableness check
    quality_checks['reasonable'] = check_against_typical_ranges(
        config, context.problem_type
    )
    
    # Consistency check
    quality_checks['consistent'] = check_parameter_consistency(config)
    
    # Diversity check (not too similar to recent trials)
    quality_checks['diverse'] = check_diversity(
        config, context.recent_trials
    )
    
    return quality_checks

def auto_correct_config(config, quality_checks, search_space):
    if not quality_checks['in_bounds']:
        config = clamp_to_bounds(config, search_space)
    
    if not quality_checks['consistent']:
        config = apply_consistency_rules(config)
    
    return config
```

## Cost Management and Efficiency

### Hierarchical Querying Strategy
```python
class HierarchicalLLMQuery:
    def __init__(self):
        self.cache_duration = 3600  # 1 hour
        self.strategy_cache = {}
    
    def get_next_config(self, context):
        # Level 1: Strategy determination (cheap, cached)
        strategy_key = self.get_strategy_cache_key(context)
        if strategy_key not in self.strategy_cache:
            strategy = self.determine_optimization_strategy(context)
            self.strategy_cache[strategy_key] = strategy
        else:
            strategy = self.strategy_cache[strategy_key]
        
        # Level 2: Configuration generation (expensive, per trial)
        config = self.generate_config_for_strategy(context, strategy)
        
        return config
    
    def determine_optimization_strategy(self, context):
        """Cheap query to determine broad approach"""
        prompt = f"Given {context.n_trials} trials, should we: A) Explore new regions B) Refine best area C) Balance both?"
        return self.llm_client.complete_simple(prompt)
```

### Caching Strategies
```python
class CachedLLMClient:
    def __init__(self, cache_ttl=1800):  # 30 minutes
        self.context_cache = {}
        self.config_cache = {}
        self.cache_ttl = cache_ttl
    
    def generate_config(self, context):
        # Create cache key from context hash
        context_hash = self.hash_context(context)
        
        if self.is_cache_valid(context_hash):
            return self.config_cache[context_hash]
        
        # Generate fresh config
        config = self.llm_client.generate(context)
        
        # Cache result
        self.config_cache[context_hash] = {
            'config': config,
            'timestamp': time.time(),
            'context_hash': context_hash
        }
        
        return config
    
    def hash_context(self, context):
        # Hash relevant parts of context for caching
        hashable = {
            'n_trials': context.n_trials,
            'best_value': context.best_value,
            'recent_trials_hash': hash(str(context.recent_trials[-5:])),
            'problem_type': context.problem_type
        }
        return hashlib.sha256(str(hashable).encode()).hexdigest()[:16]
```

## Error Handling and Robustness

### Graceful Degradation
```python
class RobustLLMClient:
    def __init__(self, max_retries=3, timeout=30):
        self.max_retries = max_retries
        self.timeout = timeout
        self.failure_count = 0
        self.last_successful_config = None
    
    def generate_config(self, context):
        for attempt in range(self.max_retries):
            try:
                config = self.try_llm_generation(context, timeout=self.timeout)
                self.failure_count = 0
                self.last_successful_config = config
                return config
                
            except LLMTimeoutError:
                if attempt < self.max_retries - 1:
                    # Reduce complexity and retry
                    context = self.simplify_context(context)
                    continue
                else:
                    return self.fallback_strategy(context)
                    
            except LLMParsingError as e:
                if attempt < self.max_retries - 1:
                    # Adjust prompt and retry
                    context.prompt_style = 'simple'
                    continue
                else:
                    return self.emergency_fallback(context)
                    
            except Exception as e:
                self.failure_count += 1
                if self.failure_count > 5:
                    raise LLMPermanentFailure("Too many consecutive failures")
                return self.fallback_strategy(context)
    
    def fallback_strategy(self, context):
        if self.last_successful_config:
            # Mutate last successful config
            return self.mutate_config(self.last_successful_config, context.search_space)
        else:
            # Generate reasonable default
            return self.generate_default_config(context.search_space, context.problem_type)
```

### Response Validation Pipeline
```python
def validate_llm_response(raw_response, search_space):
    """Multi-stage validation pipeline"""
    
    # Stage 1: JSON parsing
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        parsed = extract_json_with_regex(raw_response)
        if not parsed:
            raise ValidationError("Could not parse JSON from LLM response")
    
    # Stage 2: Schema validation
    required_params = set(search_space.keys())
    provided_params = set(parsed.get('parameters', {}).keys())
    
    if not required_params.issubset(provided_params):
        missing = required_params - provided_params
        raise ValidationError(f"Missing parameters: {missing}")
    
    # Stage 3: Type and bounds validation
    validated_config = {}
    for param_name, value in parsed['parameters'].items():
        if param_name in search_space:
            validated_config[param_name] = validate_and_clamp_parameter(
                value, search_space[param_name]
            )
    
    return validated_config

def validate_and_clamp_parameter(value, distribution):
    """Validate single parameter against Optuna distribution"""
    if isinstance(distribution, optuna.distributions.FloatDistribution):
        value = float(value)
        return max(distribution.low, min(distribution.high, value))
    
    elif isinstance(distribution, optuna.distributions.IntDistribution):
        value = int(round(float(value)))
        return max(distribution.low, min(distribution.high, value))
    
    elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
        if value in distribution.choices:
            return value
        else:
            # Default to first choice or closest match
            return distribution.choices[0]
    
    else:
        raise ValueError(f"Unsupported distribution type: {type(distribution)}")
```

## Performance Monitoring and Adaptation

### LLM Quality Metrics
```python
class LLMPerformanceMonitor:
    def __init__(self):
        self.suggestion_history = []
        self.performance_history = []
    
    def track_suggestion(self, llm_config, trial_result):
        self.suggestion_history.append({
            'config': llm_config,
            'result': trial_result,
            'timestamp': time.time()
        })
    
    def get_llm_effectiveness_metrics(self):
        if len(self.suggestion_history) < 10:
            return None
        
        recent_suggestions = self.suggestion_history[-20:]
        
        # Success rate (better than random baseline)
        baseline_performance = self.estimate_random_baseline()
        successful_suggestions = sum(
            1 for s in recent_suggestions 
            if s['result'] > baseline_performance
        )
        success_rate = successful_suggestions / len(recent_suggestions)
        
        # Average improvement over random
        improvements = [
            s['result'] - baseline_performance 
            for s in recent_suggestions
        ]
        avg_improvement = np.mean(improvements)
        
        return {
            'success_rate': success_rate,
            'avg_improvement': avg_improvement,
            'total_suggestions': len(self.suggestion_history)
        }
```

This comprehensive framework ensures reliable, cost-effective, and high-quality LLM integration for hyperparameter optimization.