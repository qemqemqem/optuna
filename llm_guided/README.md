# LLM-Guided Optuna

Large Language Model guided hyperparameter optimization for Optuna.

## Overview

This project extends Optuna with LLM-guided sampling capabilities, allowing hyperparameter optimization to leverage domain knowledge and pattern recognition capabilities of Large Language Models. Instead of relying purely on mathematical optimization algorithms, we combine Optuna's systematic approach with LLM domain knowledge from vast ML literature and successful configurations.

## Key Features

- **Trial-Level LLM Guidance**: LLMs suggest complete hyperparameter configurations by reasoning about parameter relationships and optimization history
- **Fresh Context Awareness**: Each trial gets updated context from optimization progress
- **Structured Output Parsing**: Reliable parameter extraction using Pydantic models and LiteLLM
- **Robust Error Handling**: Graceful fallbacks and parameter validation
- **Drop-in Compatibility**: Works as a standard Optuna sampler

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
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
    sampler=sampler
)

# Add problem context for better guidance
study.set_user_attr("problem_type", "neural_network_training")
study.set_user_attr("problem_description", "CNN optimization for image classification")

# Define objective function
def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 256, step=16)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Your model training code here
    accuracy = train_model(lr, batch_size, dropout)
    return 1.0 - accuracy  # Minimize error

# Run optimization
study.optimize(objective, n_trials=50)
```

### Configuration Options

```python
sampler = LLMGuidedSampler(
    model="gpt-4o-2024-08-06",    # LLM model to use
    temperature=0.3,               # Sampling temperature (0=deterministic, 1=creative)
    timeout=30,                    # API timeout in seconds
    max_retries=3,                 # Retry attempts for failed requests
    max_context_trials=10          # Max trials to include in LLM context
)
```

## Architecture

The system consists of several key components:

### 1. LLMGuidedSampler
Main integration with Optuna's sampler interface. Generates complete trial configurations using LLM reasoning.

### 2. Context Builder
Extracts and formats optimization context from Optuna study state, including:
- Search space definition
- Trial history and trends  
- Best results so far
- Optimization progress analysis
- Problem-specific context

### 3. LLM Client
Handles structured communication with LLMs using LiteLLM and Pydantic:
- Robust error handling and retries
- Structured output parsing
- Multiple model provider support

### 4. Parameter Validator
Ensures LLM suggestions conform to search space constraints:
- Bounds checking and clamping
- Type conversion and validation
- Fallback value generation

## Advanced Usage

### Adding Domain Knowledge

```python
study.set_user_attr("problem_type", "neural_network_training")
study.set_user_attr("problem_description", "ResNet-18 training on CIFAR-10")
study.set_user_attr("constraints", [
    "training_time < 2 hours",
    "memory_usage < 8GB"
])
study.set_user_attr("domain_knowledge", {
    "dataset": "CIFAR-10",
    "model_architecture": "ResNet-18", 
    "expected_accuracy": "85-92%"
})
```

### Performance Monitoring

```python
# Get comprehensive statistics
stats = sampler.get_statistics()

print(f"Success rate: {stats['sampler_stats']['success_rate']:.2%}")
print(f"Average LLM time: {stats['sampler_stats']['average_llm_time']:.2f}s")
print(f"Fallback uses: {stats['sampler_stats']['fallback_uses']}")
```

## Examples

- `examples/basic_usage.py`: Simple neural network hyperparameter optimization
- `examples/advanced_usage.py`: Advanced features and domain knowledge integration
- `examples/comparison_study.py`: Comparison with traditional Optuna samplers

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Or run basic functionality tests:

```bash
cd tests && python test_basic_functionality.py
```

## Design Documents

Comprehensive design documentation is available in the project:

- `00_project_overview.md`: High-level project vision and architecture
- `01_mathematical_foundations.md`: Mathematical theory for distribution combination
- `02_llm_integration_strategies.md`: LLM interaction patterns and best practices  
- `03_implementation_architecture.md`: Detailed implementation design
- `04_structured_output_design.md`: Pydantic models and parsing strategy
- `05_context_building_strategy.md`: Context extraction and formatting

## Key Benefits

1. **Faster Convergence**: Reduce trials needed by starting with educated guesses
2. **Better Solutions**: Leverage LLM domain knowledge to find superior configurations
3. **Parameter Relationships**: LLMs understand complex parameter interactions
4. **Adaptive Strategy**: Optimization approach adapts based on progress and trends

## Supported Models

Works with any model supported by LiteLLM:
- OpenAI: GPT-4, GPT-4o, GPT-3.5
- Anthropic: Claude-3, Claude-3.5 
- Google: Gemini Pro, Gemini Flash
- Local models via Ollama, LM Studio
- Many others via LiteLLM

## Requirements

- Python 3.8+
- optuna>=3.0.0
- litellm>=1.0.0
- pydantic>=2.0.0
- numpy, scipy, scikit-learn

## Contributing

This project follows XP programming principles with focus on:
- Test-driven development
- Clean, simple code
- Continuous refactoring
- Fast iteration cycles

See `CONTRIBUTING.md` for guidelines.

## License

MIT License - see `LICENSE` file for details.

## Citation

If you use this work in academic research, please cite:

```bibtex
@software{llm_guided_optuna,
  title={LLM-Guided Optuna: Large Language Model Guided Hyperparameter Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/llm-guided-optuna}
}
```