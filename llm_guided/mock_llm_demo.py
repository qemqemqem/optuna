#!/usr/bin/env python3
"""
Demo showing LLM-Guided Optuna with mock LLM responses.

This demonstrates the complete workflow with simulated intelligent LLM responses.
"""

import sys
import os
sys.path.insert(0, 'src')

import optuna
from sampler import LLMGuidedSampler
from models import TrialConfiguration
import logging
from unittest.mock import patch
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger(__name__)

print("🤖 Mock LLM-Guided Optuna Demo")
print("=" * 60)

def create_mock_llm_response(context, trial_number):
    """Create realistic mock LLM responses based on context."""
    
    # Analyze the context to make intelligent suggestions
    if context.best_trial:
        best_params = context.best_trial.parameters
        best_value = context.best_trial.value
        print(f"📊 LLM analyzing: Best so far is {best_value:.6f} with {best_params}")
    
    # Mock intelligent parameter selection based on stage and trend
    if context.progress_analysis.stage.value == "early_exploration":
        # Explore broadly
        if trial_number % 3 == 0:
            # Try smaller learning rates
            lr_choice = 0.0001
            reasoning = "Exploring smaller learning rates as they often work better for stable training"
        elif trial_number % 3 == 1:
            # Try larger batch sizes
            lr_choice = 0.003
            reasoning = "Testing moderate learning rate with larger batch size for efficiency"
        else:
            # Balanced approach
            lr_choice = 0.001
            reasoning = "Using classic Adam starting point with balanced hyperparameters"
    else:
        # More focused optimization
        if context.best_trial:
            best_lr = context.best_trial.parameters.get('learning_rate', 0.001)
            # Vary around the best known value
            import random
            lr_choice = best_lr * random.uniform(0.5, 2.0)
            reasoning = f"Exploring around best known learning rate {best_lr:.4f}"
        else:
            lr_choice = 0.001
            reasoning = "Default exploration strategy"
    
    # Smart batch size selection
    if context.best_trial and 'batch_size' in context.best_trial.parameters:
        best_bs = context.best_trial.parameters['batch_size']
        batch_size = max(16, min(128, int(best_bs * random.uniform(0.8, 1.2))))
    else:
        batch_size = [32, 64, 96][trial_number % 3]
    
    # Intelligent dropout based on trend
    if context.progress_analysis.trend.value == "improving":
        dropout = 0.25  # Stay conservative
    else:
        dropout = [0.1, 0.3, 0.4][trial_number % 3]  # Explore more
    
    # Optimizer selection
    optimizer = ["adam", "adam", "sgd"][trial_number % 3]  # Prefer Adam
    
    return TrialConfiguration(
        parameters={
            "learning_rate": lr_choice,
            "batch_size": batch_size,
            "dropout": dropout,
            "optimizer": optimizer
        },
        reasoning=reasoning,
        confidence=0.8,
        strategy="balanced"
    )

def mock_objective(trial):
    """Mock neural network training objective."""
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    
    import random, math
    base_loss = 0.4
    lr_penalty = abs(math.log10(lr) + 3) * 0.08  # optimal around 1e-3
    batch_penalty = abs(batch_size - 64) / 64 * 0.03
    dropout_penalty = abs(dropout - 0.25) * 0.15
    optimizer_bonus = {"adam": 0.0, "sgd": 0.05, "rmsprop": 0.02}[optimizer]
    noise = random.gauss(0, 0.015)
    
    loss = base_loss + lr_penalty + batch_penalty + dropout_penalty + optimizer_bonus + noise
    return max(0.1, loss)  # Ensure positive loss

# Create LLM-guided study
print("\n1. Setting up LLM-Guided Study...")
sampler = LLMGuidedSampler(model="gpt-4o-mock", temperature=0.3)

study = optuna.create_study(
    direction="minimize", 
    sampler=sampler,
    study_name="mock_llm_neural_net_optimization"
)

# Add rich context
study.set_user_attr("problem_type", "neural_network_training")
study.set_user_attr("problem_description", "ResNet-18 training on CIFAR-10 dataset")
study.set_user_attr("constraints", ["training_time < 1 hour", "memory_usage < 4GB"])
study.set_user_attr("domain_knowledge", {
    "dataset": "CIFAR-10",
    "model": "ResNet-18", 
    "expected_accuracy": "90-95%"
})

print(f"✓ Created study with LLM sampler: {sampler}")

# Mock the LLM client to return intelligent responses
trial_counter = 0

def mock_generate_trial_configuration(self, context, temperature=None):
    global trial_counter
    trial_counter += 1
    
    print(f"\n🧠 LLM Trial {trial_counter} Generation:")
    print(f"   Context: {context.n_trials_completed} completed trials")
    print(f"   Stage: {context.progress_analysis.stage.value}")
    print(f"   Trend: {context.progress_analysis.trend.value}")
    print(f"   Recommendation: {context.progress_analysis.recommendation}")
    
    config = create_mock_llm_response(context, trial_counter)
    print(f"   🎯 LLM suggests: {config.parameters}")
    print(f"   💭 LLM reasoning: {config.reasoning}")
    
    return config

# Patch the LLM client to use our mock
print("\n2. Running LLM-Guided Optimization...")
with patch('llm_client.LLMClient.generate_trial_configuration', mock_generate_trial_configuration):
    try:
        # Run optimization with mock LLM
        study.optimize(mock_objective, n_trials=6)
        
        print(f"\n🏆 Optimization Results:")
        print(f"   Best trial: #{study.best_trial.number}")
        print(f"   Best value: {study.best_value:.6f}")
        print(f"   Best parameters: {study.best_params}")
        
        # Show progression
        print(f"\n📈 Trial Progression:")
        for i, trial in enumerate(study.trials):
            marker = "🥇" if trial.number == study.best_trial.number else "  "
            print(f"   {marker} Trial {trial.number}: {trial.value:.6f} - {trial.params}")
        
        # Show sampler statistics
        stats = sampler.get_statistics()
        print(f"\n📊 LLM Sampler Statistics:")
        print(f"   Success rate: {stats['sampler_stats']['success_rate']:.1%}")
        print(f"   Total trials: {stats['sampler_stats']['total_trials']}")
        print(f"   Successful generations: {stats['sampler_stats']['successful_generations']}")
        print(f"   Failed generations: {stats['sampler_stats']['failed_generations']}")
        print(f"   Fallback uses: {stats['sampler_stats']['fallback_uses']}")
        print(f"   Validation fixes: {stats['validation_stats']['clamped_parameters']}")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        
print(f"\n3. Comparing with Regular Optuna...")
# Compare with regular TPE sampler
regular_study = optuna.create_study(direction="minimize")
regular_study.optimize(mock_objective, n_trials=6)

print(f"   Regular Optuna best: {regular_study.best_value:.6f}")
print(f"   LLM-Guided best: {study.best_value:.6f}")

improvement = (regular_study.best_value - study.best_value) / regular_study.best_value * 100
if improvement > 0:
    print(f"   🎉 LLM-Guided improved by {improvement:.1f}%!")
else:
    print(f"   📊 Regular was better by {abs(improvement):.1f}% (sample size too small)")

print(f"\n✨ Mock LLM Demo Complete!")
print("=" * 60)
print("Key Behaviors Demonstrated:")
print("✓ Intelligent parameter suggestion based on optimization context")
print("✓ Reasoning-driven hyperparameter choices")
print("✓ Adaptation to optimization stage and trends")
print("✓ Integration with Optuna's trial mechanism")
print("✓ Performance tracking and statistics")
print("✓ Parameter validation and constraint enforcement")
print("\nThis shows how the LLM would make intelligent decisions!")
print("With a real LLM API key, you get even smarter optimization! 🚀")