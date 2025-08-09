#!/usr/bin/env python3
"""
Demo script showing LLM-Guided Optuna functionality.

This demonstrates the key features without requiring an actual LLM API key.
"""

import sys
import os
sys.path.insert(0, 'src')

import optuna
from sampler import LLMGuidedSampler
from context_builder import ContextBuilder
from models import TrialConfiguration
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("🚀 LLM-Guided Optuna Demo")
print("=" * 50)

# 1. Show basic component instantiation
print("\n1. Creating LLM-Guided Components...")
sampler = LLMGuidedSampler(model="gpt-4o-2024-08-06")
context_builder = ContextBuilder()
print(f"✓ LLM Sampler: {sampler}")
print(f"✓ Context Builder created with max_recent_trials={context_builder.max_recent_trials}")

# 2. Demonstrate context building with empty study
print("\n2. Context Building - Empty Study...")
empty_study = optuna.create_study(direction="minimize")
empty_context = context_builder.build_context(empty_study)
print(f"✓ Empty study context: {empty_context.n_trials_completed} trials")
print(f"  - Stage: {empty_context.progress_analysis.stage.value}")
print(f"  - Search space parameters: {len(empty_context.search_space)}")

# 3. Create study with some trial history
print("\n3. Creating Study with Trial History...")

def mock_objective(trial):
    """Mock neural network training objective."""
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    
    # Mock loss calculation (lower is better)
    import random, math
    base_loss = 0.5
    lr_penalty = abs(math.log10(lr) + 3) * 0.1  # optimal around 1e-3
    batch_penalty = abs(batch_size - 64) / 64 * 0.05  # optimal around 64
    dropout_penalty = abs(dropout - 0.25) * 0.2  # optimal around 0.25
    optimizer_bonus = {"adam": 0.0, "sgd": 0.02, "rmsprop": 0.01}[optimizer]
    noise = random.gauss(0, 0.02)
    
    loss = base_loss + lr_penalty + batch_penalty + dropout_penalty + optimizer_bonus + noise
    return loss

# Run some trials with regular sampler first to build history
regular_study = optuna.create_study(direction="minimize")
regular_study.set_user_attr("problem_type", "neural_network_training")
regular_study.set_user_attr("problem_description", "CNN hyperparameter optimization for image classification")

print("Running 5 trials to build optimization history...")
regular_study.optimize(mock_objective, n_trials=5)

# 4. Show context building with trial history
print(f"\n4. Context Building - Study with {len(regular_study.trials)} Trials...")
rich_context = context_builder.build_context(regular_study)
print(f"✓ Rich context built:")
print(f"  - Trials completed: {rich_context.n_trials_completed}")
print(f"  - Search space parameters: {len(rich_context.search_space)}")
print(f"  - Recent trials: {len(rich_context.recent_trials)}")
print(f"  - Best trial value: {rich_context.best_trial.value:.6f}")
print(f"  - Optimization stage: {rich_context.progress_analysis.stage.value}")
print(f"  - Trend: {rich_context.progress_analysis.trend.value}")
print(f"  - Recommendation: {rich_context.progress_analysis.recommendation}")

print(f"\n  Search Space Details:")
for param in rich_context.search_space:
    if param.type == "float":
        print(f"    - {param.name} ({param.type}): {param.low} to {param.high} {'(log)' if param.log_scale else ''}")
    elif param.type == "int":
        print(f"    - {param.name} ({param.type}): {param.low} to {param.high}")
    elif param.type == "categorical":
        print(f"    - {param.name} ({param.type}): {param.choices}")

print(f"\n  Recent Trial Results:")
for trial in rich_context.recent_trials[-3:]:  # Show last 3
    print(f"    - Trial {trial.trial_number}: {trial.value:.6f} with {trial.parameters}")

# 5. Show what would be sent to LLM
print(f"\n5. LLM Prompt Preview...")
from llm_client import LLMClient
llm_client = LLMClient(model="gpt-4o-test")
prompt = llm_client._build_configuration_prompt(rich_context)
print(f"✓ Generated prompt with {len(prompt)} characters")
print(f"First 500 characters of LLM prompt:")
print("-" * 50)
print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
print("-" * 50)

# 6. Show parameter validation in action
print(f"\n6. Parameter Validation Demo...")
from parameter_validator import ParameterValidator
validator = ParameterValidator()

# Create a test configuration with some invalid values
test_config = TrialConfiguration(
    parameters={
        "learning_rate": 10.0,  # Too high (should be clamped)
        "batch_size": 1000,     # Too high
        "dropout": -0.1,        # Too low
        "optimizer": "invalid_opt"  # Invalid choice
    },
    reasoning="Test configuration with invalid values"
)

print("Testing parameter validation with invalid values:")
print(f"  Original: {test_config.parameters}")

# Get search space from the study
search_space = regular_study.trials[-1].distributions
validated_params = validator.validate_and_clamp_configuration(test_config, search_space)
print(f"  Validated: {validated_params}")

validation_stats = validator.get_validation_stats()
print(f"  Validation stats: {validation_stats}")

print(f"\n🎯 Demo Complete!")
print("=" * 50)
print("Key Features Demonstrated:")
print("✓ Component instantiation and configuration")
print("✓ Context building from empty and populated studies")
print("✓ Progress analysis and optimization stage detection")
print("✓ Search space extraction and parameter descriptions")
print("✓ LLM prompt generation with rich context")
print("✓ Parameter validation and constraint enforcement")
print("✓ Statistical tracking and monitoring")
print("\nThe system is ready for LLM-guided optimization!")
print("Just add an API key and you're ready to go! 🚀")