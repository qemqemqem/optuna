# LLM-Guided Optuna

> Large Language Model guided hyperparameter optimization for Optuna

## Overview

LLM-Guided Optuna extends Optuna with intelligent hyperparameter suggestions powered by Large Language Models. Instead of purely mathematical optimization, it combines Optuna's systematic approach with LLM domain knowledge to make smarter parameter choices.

## Key Features

- 🧠 **Intelligent Parameter Suggestions**: LLMs analyze optimization context and suggest complete configurations
- 📊 **Rich Context Awareness**: Incorporates trial history, trends, and domain knowledge  
- 🔒 **Robust Validation**: Ensures all suggestions fit within search space constraints
- 🔄 **Graceful Fallbacks**: Handles LLM failures transparently
- 🎯 **Drop-in Compatibility**: Works as a standard Optuna sampler

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set your LLM API key (example with OpenAI)
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```python
import optuna
from src.sampler import LLMGuidedSampler

# Create LLM-guided sampler
sampler = LLMGuidedSampler(
    model="gpt-4o-2024-08-06",
    temperature=0.3
)

# Create study with rich context for better LLM guidance
study = optuna.create_study(direction="minimize", sampler=sampler)
study.set_user_attr("problem_type", "neural_network_training")
study.set_user_attr("problem_description", "CNN optimization for CIFAR-10")

def objective(trial):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    
    # Your model training code here
    return validation_loss

# Run optimization - each trial queries the LLM!
study.optimize(objective, n_trials=20)
```

## How It Works

1. **Context Building**: Extracts optimization history, progress trends, and search space
2. **LLM Querying**: Sends rich context to LLM requesting intelligent parameter suggestions
3. **Response Parsing**: Validates LLM responses using structured Pydantic models
4. **Parameter Validation**: Clamps suggestions to search space bounds with fallbacks
5. **Trial Execution**: Returns validated parameters to Optuna for objective evaluation

## Testing & Demos

### ⚡ **Lightning Rod Optimization** (Recommended!)
```bash
export OPENAI_API_KEY="your-key"
python examples/lightning_rod_optimization.py
```
A geometric optimization problem that showcases LLM-guided optimization on a real-world scenario: placing lightning rods optimally around a facility to minimize unprotected areas. Non-ML example with complex constraints and no analytical solution.

### 🔍 Complete System Examination
```bash
python examples/examine_this.py  # Detailed walkthrough of every component
```

### Core Functionality Demo
```bash
python demo.py  # Shows all components working (no API key needed)
```

### Import Tests
```bash
python -m pytest tests/test_imports.py -v  # All 6 tests should pass
```

## Project Structure

```
llm_guided/
├── src/                    # Core implementation
│   ├── sampler.py         # Main LLMGuidedSampler
│   ├── llm_client.py      # LLM communication
│   ├── context_builder.py # Optimization context extraction
│   ├── models.py          # Pydantic data models
│   └── parameter_validator.py # Parameter validation
├── tests/                 # Test suite
│   ├── test_imports.py    # Import verification (✅ all pass)
│   └── test_basic_functionality.py # Unit tests
├── examples/              # Usage examples
├── docs/                  # Detailed documentation
└── demo.py               # Interactive demonstration
```

## Architecture

- **Trial-Level Generation**: LLM suggests complete parameter configurations (not individual parameters)
- **Fresh Context**: Rebuilds optimization context from study state each trial
- **No Traditional Fallback**: Pure LLM-guided approach with intelligent error recovery
- **Structured Communication**: Uses LiteLLM + Pydantic for reliable LLM interaction

## Documentation

- [How It Works](docs/how-it-works.md) - Detailed system architecture
- [Usage Guide](docs/usage-guide.md) - Comprehensive usage examples
- [API Reference](docs/api-reference.md) - Complete API documentation

## Current Status

✅ **Working**: All core components, imports, context building, validation  
✅ **Tested**: Import tests passing, demo functional  
⚠️ **Known Issue**: First trial initialization (common Optuna sampler pattern)  

## Contributing

This implementation follows Optuna's coding standards:
- Code formatted with `black --line-length 99`
- Imports sorted with `isort --profile black`
- Tests run with `pytest`

## License

MIT License - See main Optuna repository for details.