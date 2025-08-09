# LLM-Guided Optuna Usage Guide

## Installation & Setup

### Dependencies
```bash
cd llm_guided
pip install -r requirements.txt
```

### API Key Configuration
```bash
# OpenAI (recommended)
export OPENAI_API_KEY="sk-..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# Or set in Python
import os
os.environ["OPENAI_API_KEY"] = "your-key"
```

## Basic Usage

### Simple Example
```python
import optuna
from src.sampler import LLMGuidedSampler

# Create LLM sampler
sampler = LLMGuidedSampler(model="gpt-4o-2024-08-06")
study = optuna.create_study(direction="minimize", sampler=sampler)

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return x**2 + y**2

study.optimize(objective, n_trials=10)
```

### With Rich Context (Recommended)
```python
import optuna
from src.sampler import LLMGuidedSampler

# Create study with detailed context
sampler = LLMGuidedSampler(
    model="gpt-4o-2024-08-06",
    temperature=0.3,  # Lower = more consistent
    timeout=30,       # API timeout
    max_retries=3     # Retry failed requests
)

study = optuna.create_study(direction="minimize", sampler=sampler)

# Add rich context for better LLM guidance
study.set_user_attr("problem_type", "neural_network_training")
study.set_user_attr(
    "problem_description", 
    "Training ResNet-18 on CIFAR-10 for image classification. "
    "Goal: minimize validation loss while avoiding overfitting."
)
study.set_user_attr("constraints", [
    "training_time < 2 hours per trial",
    "memory_usage < 8GB GPU",
    "target_accuracy > 90%"
])
study.set_user_attr("domain_knowledge", {
    "dataset": "CIFAR-10 (32x32 RGB, 10 classes)",
    "model": "ResNet-18 with batch normalization",
    "best_known": "lr=0.001, bs=128, dropout=0.1",
    "common_issues": "High LR causes instability, small batch sizes are slow"
})

def neural_net_objective(trial):
    # Hyperparameter suggestions will be informed by above context
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 256, step=16)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    # Your training code here
    model = create_model(dropout=dropout)
    opt = create_optimizer(model, lr=lr, weight_decay=weight_decay, name=optimizer)
    
    validation_loss = train_model(model, opt, batch_size=batch_size)
    return validation_loss

# LLM will make intelligent suggestions based on context
study.optimize(neural_net_objective, n_trials=20)
```

## Advanced Configuration

### LLM Model Selection
```python
# Different models for different use cases
samplers = {
    "best_quality": LLMGuidedSampler(model="gpt-4o-2024-08-06"),
    "fast": LLMGuidedSampler(model="gpt-4o-mini"),
    "claude": LLMGuidedSampler(model="claude-3-sonnet-20240229"),
    "local": LLMGuidedSampler(model="ollama/llama3"),
}
```

### Temperature Tuning
```python
# Conservative (more consistent)
sampler = LLMGuidedSampler(model="gpt-4o", temperature=0.1)

# Balanced (recommended)
sampler = LLMGuidedSampler(model="gpt-4o", temperature=0.3)

# Creative (more exploration)
sampler = LLMGuidedSampler(model="gpt-4o", temperature=0.7)
```

### Context Configuration
```python
sampler = LLMGuidedSampler(
    model="gpt-4o",
    max_context_trials=15,  # Include last 15 trials in context
    temperature=0.3,
    timeout=45,             # Longer timeout for complex problems
    max_retries=5          # More retries for reliability
)
```

## Problem-Specific Examples

### ⚡ Lightning Rod Optimization (Geometric)
```python
# Real-world geometric optimization example
OPENAI_API_KEY="your-key" python examples/lightning_rod_optimization.py

# With full LLM response debugging
python examples/lightning_rod_optimization.py --show-full-llm

# Different models and trial counts
python examples/lightning_rod_optimization.py --model gpt-4o-mini --trials 10
```

This example demonstrates LLM-guided optimization on a complex geometric problem: placing lightning rods optimally around a facility to minimize unprotected areas. Features complex constraints, domain knowledge injection, and no analytical solution.

### Computer Vision
```python
study.set_user_attr("problem_type", "computer_vision")
study.set_user_attr("domain_knowledge", {
    "architecture": "ResNet/EfficientNet",
    "data_augmentation": "rotation, flip, crop, color_jitter",
    "typical_lr": "1e-4 to 1e-2 with cosine annealing",
    "batch_size_guidance": "larger is better for batch norm",
    "regularization": "dropout, weight_decay, label_smoothing"
})
```

### NLP
```python
study.set_user_attr("problem_type", "natural_language_processing")
study.set_user_attr("domain_knowledge", {
    "model_type": "transformer",
    "sequence_length": "512 tokens max",
    "typical_lr": "1e-5 to 5e-4 for pre-trained models",
    "batch_size": "constrained by memory, gradient accumulation helps",
    "warmup": "linear warmup over first 10% of training"
})
```

### Tabular Data
```python
study.set_user_attr("problem_type", "tabular_classification")
study.set_user_attr("domain_knowledge", {
    "model_type": "gradient_boosting",
    "typical_lr": "0.01 to 0.3",
    "tree_depth": "3 to 8 for most datasets",
    "regularization": "l1/l2, min_child_samples",
    "feature_importance": "available for interpretation"
})
```

## Monitoring & Debugging

### Performance Statistics
```python
# After optimization
stats = sampler.get_statistics()
print(f"Success rate: {stats['sampler_stats']['success_rate']:.1%}")
print(f"Average LLM time: {stats['sampler_stats']['average_llm_time']:.1f}s")
print(f"Fallback usage: {stats['sampler_stats']['fallback_uses']}")
```

### Logging Configuration
```python
import logging
logging.basicConfig(level=logging.INFO)

# See LLM requests and responses
llm_logger = logging.getLogger('llm_client')
llm_logger.setLevel(logging.DEBUG)

# See parameter validation details
validator_logger = logging.getLogger('parameter_validator')
validator_logger.setLevel(logging.INFO)
```

### Error Handling
```python
try:
    study.optimize(objective, n_trials=20)
except Exception as e:
    print(f"Optimization failed: {e}")
    
    # Check sampler stats for issues
    stats = sampler.get_statistics()
    if stats['sampler_stats']['success_rate'] < 0.5:
        print("Low success rate - check API key and limits")
    
    # Continue with completed trials
    if study.trials:
        print(f"Best result from {len(study.trials)} trials:")
        print(f"Value: {study.best_value}")
        print(f"Params: {study.best_params}")
```

## Testing & Validation

### Quick Functionality Test
```python
# Run demo without API key
python demo.py
```

### Import Verification
```python
# Ensure all components work
python -m pytest tests/test_imports.py -v
```

### Real LLM Test
```python
# Test with actual API (uses 1 API call)
OPENAI_API_KEY="your-key" python quick_real_test.py
```

## Best Practices

### 1. Provide Rich Context
- Always set `problem_type`
- Include domain knowledge and constraints
- Describe the objective clearly

### 2. Choose Appropriate Models
- GPT-4o for best quality
- GPT-4o-mini for speed/cost balance  
- Local models (Ollama) for privacy

### 3. Handle API Limitations
- Set reasonable timeouts (30-60s)
- Use retries (3-5)
- Monitor success rates

### 4. Monitor Performance
- Track LLM success rates
- Watch for excessive fallback usage
- Adjust temperature based on results

### 5. Iterative Improvement
- Start with basic context
- Add domain knowledge as you learn
- Refine constraints based on results

## Troubleshooting

### Common Issues

**"Parameter not found" error**: First trial initialization issue
- Run a few regular trials first, then switch to LLM sampler

**Low success rate**: API or prompt issues
- Check API key validity
- Reduce context complexity
- Increase timeout

**Poor suggestions**: Insufficient context
- Add more domain knowledge
- Include constraints and objectives
- Provide examples of good/bad parameters

**High latency**: Model or API issues
- Use faster model (gpt-4o-mini)
- Reduce context size
- Check network connectivity