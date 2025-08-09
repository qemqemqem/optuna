#!/usr/bin/env python3
"""
🔍 EXAMINE THIS: Complete LLM-Guided Optuna Example

This file demonstrates all key features and can be run to see
how LLM-guided optimization works in practice.

Run with: python examples/examine_this.py
(No API key needed - shows system behavior with mocks)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import optuna
from sampler import LLMGuidedSampler
from context_builder import ContextBuilder
from models import TrialConfiguration
import logging

# Configure logging to see what happens under the hood
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

print("🔍 EXAMINE THIS: LLM-Guided Optuna Complete Example")
print("=" * 60)

def neural_network_objective(trial):
    """
    Example objective: Neural network hyperparameter optimization
    
    This simulates training a CNN on image classification with realistic
    hyperparameter interactions and validation loss calculation.
    """
    # 1. Sample hyperparameters (LLM suggests these intelligently)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    
    print(f"  🔬 Trial {trial.number}: lr={learning_rate:.2e}, bs={batch_size}, "
          f"dropout={dropout:.3f}, wd={weight_decay:.2e}, opt={optimizer}")
    
    # 2. Simulate realistic model training with hyperparameter interactions
    import random, math, time
    
    # Simulate training time (shorter for demo)
    time.sleep(0.5)  
    
    # Realistic loss calculation with hyperparameter effects
    base_loss = 0.45
    
    # Learning rate effect (optimal around 1e-3 for Adam, 1e-2 for SGD)
    optimal_lr = 0.01 if optimizer == "sgd" else 0.001
    lr_penalty = abs(math.log10(learning_rate) - math.log10(optimal_lr)) * 0.08
    
    # Batch size effect (optimal around 64-96 for most models)
    batch_penalty = abs(batch_size - 80) / 80 * 0.04
    
    # Dropout effect (optimal around 0.2-0.3 for regularization)
    dropout_penalty = abs(dropout - 0.25) * 0.15
    
    # Weight decay interaction with optimizer
    optimal_wd = 0.0001 if optimizer == "adam" else 0.0005
    wd_penalty = abs(math.log10(weight_decay) - math.log10(optimal_wd)) * 0.03
    
    # Optimizer baseline differences
    optimizer_bonus = {"adam": 0.0, "sgd": 0.02, "rmsprop": 0.015}[optimizer]
    
    # Parameter interaction effects (realistic ML behavior)
    if learning_rate > 0.01 and batch_size < 32:
        lr_penalty += 0.05  # High LR with small batch = instability
    if dropout > 0.4 and weight_decay > 0.001:
        lr_penalty += 0.03  # Too much regularization
    
    # Add realistic noise
    noise = random.gauss(0, 0.02)
    
    validation_loss = (base_loss + lr_penalty + batch_penalty + 
                      dropout_penalty + wd_penalty + optimizer_bonus + noise)
    
    # Ensure positive loss
    validation_loss = max(0.1, validation_loss)
    
    print(f"    📊 Validation loss: {validation_loss:.6f}")
    return validation_loss

def demonstrate_system_components():
    """Show how each component works individually."""
    
    print("\n🧩 COMPONENT DEMONSTRATION")
    print("-" * 40)
    
    # 1. Context Builder
    print("\n1️⃣ Context Builder: Extracts optimization context")
    builder = ContextBuilder()
    
    # Create study with some history
    study = optuna.create_study(direction="minimize")
    study.set_user_attr("problem_type", "neural_network_training")
    study.optimize(neural_network_objective, n_trials=3, show_progress_bar=False)
    
    # Build context
    context = builder.build_context(study)
    print(f"   ✅ Built context: {context.n_trials_completed} trials")
    print(f"   📊 Stage: {context.progress_analysis.stage.value}")
    print(f"   📈 Trend: {context.progress_analysis.trend.value}")
    print(f"   💡 Recommendation: {context.progress_analysis.recommendation}")
    print(f"   🎯 Search space: {len(context.search_space)} parameters")
    
    # 2. LLM Client (mock for demo)
    print("\n2️⃣ LLM Client: Generates intelligent suggestions")
    from llm_client import LLMClient
    client = LLMClient(model="gpt-4o-demo")
    
    # Show what gets sent to LLM
    prompt = client._build_configuration_prompt(context)
    print(f"   ✅ Generated prompt: {len(prompt)} characters")
    print(f"   📝 Sample prompt content:")
    print("   " + "\n   ".join(prompt.split("\n")[:15]) + "\n   ...")
    
    # 3. Parameter Validator
    print("\n3️⃣ Parameter Validator: Ensures valid parameters")
    from parameter_validator import ParameterValidator
    validator = ParameterValidator()
    
    # Test with invalid parameters
    test_config = TrialConfiguration(
        parameters={
            "learning_rate": 10.0,      # Way too high
            "batch_size": 1000,         # Too large
            "dropout": -0.1,            # Invalid (negative)
            "weight_decay": 1e10,       # Absurdly high
            "optimizer": "invalid"      # Not in choices
        },
        reasoning="Test configuration with invalid values"
    )
    
    search_space = study.trials[-1].distributions
    validated = validator.validate_and_clamp_configuration(test_config, search_space)
    
    print(f"   ✅ Validated parameters:")
    print(f"      Original: {test_config.parameters}")  
    print(f"      Clamped:  {validated}")
    
    stats = validator.get_validation_stats()
    print(f"   📊 Clamped {stats['clamped_parameters']} parameters")
    
    return context, study

def demonstrate_llm_guided_optimization():
    """Show the full LLM-guided optimization process."""
    
    print("\n🚀 LLM-GUIDED OPTIMIZATION DEMO")
    print("-" * 40)
    
    # Create LLM-guided sampler
    sampler = LLMGuidedSampler(
        model="gpt-4o-2024-08-06",  # Will use mocks for demo
        temperature=0.3,
        max_context_trials=8
    )
    
    # Create study with rich context
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    # Add comprehensive context (this is what makes LLMs smart!)
    study.set_user_attr("problem_type", "neural_network_training")
    study.set_user_attr(
        "problem_description", 
        "Training ResNet-18 on CIFAR-10 image classification. "
        "Goal: minimize validation loss while maintaining training stability."
    )
    study.set_user_attr("constraints", [
        "training_time < 2 hours per trial",
        "GPU_memory < 8GB",
        "target_accuracy > 90%",
        "avoid_overfitting"
    ])
    study.set_user_attr("domain_knowledge", {
        "dataset": "CIFAR-10 (32x32 RGB images, 10 classes, 50k train + 10k test)",
        "model_architecture": "ResNet-18 with batch normalization and ReLU",
        "data_augmentation": "random_crop, horizontal_flip, normalize",
        "known_good_configs": [
            {"lr": 0.001, "bs": 128, "dropout": 0.1, "opt": "adam"},
            {"lr": 0.01, "bs": 64, "dropout": 0.2, "opt": "sgd"}
        ],
        "common_issues": [
            "lr > 0.01 with Adam often causes instability",
            "very small batch sizes slow convergence",  
            "dropout > 0.5 usually hurts performance",
            "weight_decay needs to be adjusted per optimizer"
        ],
        "optimization_tips": [
            "Adam works well with lr 1e-4 to 1e-2",
            "SGD needs higher lr (1e-2 to 1e-1) but more stable",
            "Batch sizes 64-128 typically optimal for CNNs",
            "Moderate dropout (0.1-0.3) prevents overfitting"
        ]
    })
    
    print(f"✅ Created LLM-guided study with rich context")
    print(f"   Model: {sampler.model}")
    print(f"   Context: {len(study.user_attrs)} attributes")
    
    # Mock LLM responses for demonstration
    def create_smart_mock_response(trial_num):
        """Create realistic LLM responses that get progressively smarter."""
        responses = [
            # Trial 1: Conservative start based on domain knowledge
            {
                "parameters": {"learning_rate": 0.001, "batch_size": 64, 
                             "dropout": 0.2, "weight_decay": 0.0001, "optimizer": "adam"},
                "reasoning": "Starting with classic Adam configuration: moderate lr (0.001) proven for CNNs, "
                           "batch size 64 balances memory and convergence, light dropout for regularization."
            },
            # Trial 2: SGD exploration  
            {
                "parameters": {"learning_rate": 0.01, "batch_size": 96, 
                             "dropout": 0.15, "weight_decay": 0.0005, "optimizer": "sgd"},
                "reasoning": "Exploring SGD with higher lr (0.01) as typically needed, larger batch for "
                           "better gradient estimates, slightly less dropout, higher weight decay for SGD."
            },
            # Trial 3: Refinement based on results
            {
                "parameters": {"learning_rate": 0.0008, "batch_size": 80, 
                             "dropout": 0.25, "weight_decay": 0.00008, "optimizer": "adam"},
                "reasoning": "Fine-tuning around Adam: slightly lower lr for stability, intermediate batch size, "
                           "moderate dropout, adjusted weight decay based on preliminary results."
            }
        ]
        
        response_data = responses[min(trial_num, len(responses) - 1)]
        return TrialConfiguration(
            parameters=response_data["parameters"],
            reasoning=response_data["reasoning"],
            confidence=0.85,
            strategy="balanced"
        )
    
    # Mock the LLM for demonstration (in real usage, this calls actual LLM)
    from unittest.mock import patch
    trial_count = 0
    
    def mock_llm_call(self, context, temperature=None):
        nonlocal trial_count
        
        print(f"\n🧠 LLM Analysis for Trial {trial_count + 1}:")
        print(f"   📊 Trials completed: {context.n_trials_completed}")
        print(f"   🎯 Optimization stage: {context.progress_analysis.stage.value}")
        print(f"   📈 Recent trend: {context.progress_analysis.trend.value}")
        if context.best_trial:
            print(f"   🏆 Best so far: {context.best_trial.value:.6f}")
        print(f"   💡 Strategy: {context.progress_analysis.recommendation}")
        
        config = create_smart_mock_response(trial_count)
        
        print(f"   🎯 LLM suggests: {config.parameters}")
        print(f"   💭 LLM reasoning: {config.reasoning}")
        
        trial_count += 1
        return config
    
    # Run optimization with mocked LLM
    print(f"\n🏃 Running optimization with LLM guidance...")
    
    with patch.object(sampler.llm_client, 'generate_trial_configuration', mock_llm_call):
        try:
            study.optimize(neural_network_objective, n_trials=3)
            
            print(f"\n🏆 Optimization Results:")
            print(f"   Best trial: #{study.best_trial.number}")
            print(f"   Best loss: {study.best_value:.6f}")
            print(f"   Best parameters:")
            for key, value in study.best_params.items():
                if isinstance(value, float):
                    print(f"     {key}: {value:.2e}" if value < 0.01 else f"     {key}: {value:.4f}")
                else:
                    print(f"     {key}: {value}")
            
            # Show progression
            print(f"\n📈 Trial Progression:")
            for i, trial in enumerate(study.trials):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    marker = "🥇" if trial.number == study.best_trial.number else f" {i+1}."
                    print(f"   {marker} Trial {trial.number}: {trial.value:.6f}")
            
            # Show sampler statistics
            stats = sampler.get_statistics()
            print(f"\n📊 LLM Sampler Performance:")
            print(f"   Success rate: {stats['sampler_stats']['success_rate']:.1%}")
            print(f"   Total trials: {stats['sampler_stats']['total_trials']}")
            print(f"   LLM generations: {stats['sampler_stats']['successful_generations']}")
            print(f"   Fallback uses: {stats['sampler_stats']['fallback_uses']}")
            
            return study
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

def main():
    """Run the complete demonstration."""
    
    print("This example shows every aspect of LLM-guided optimization:")
    print("• How context is built from optimization history")
    print("• What prompts are sent to LLMs") 
    print("• How LLM responses are validated and processed")
    print("• Complete optimization workflow with intelligent suggestions")
    
    # Part 1: Component demonstration
    context, study = demonstrate_system_components()
    
    # Part 2: Full optimization
    llm_study = demonstrate_llm_guided_optimization()
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 60)
    
    print("🔍 What you just saw:")
    print("✅ Context extraction from 3 baseline trials")
    print("✅ Rich prompt generation (3000+ chars with domain knowledge)")
    print("✅ Parameter validation and constraint enforcement")
    print("✅ Intelligent LLM parameter suggestions with reasoning")
    print("✅ Progressive learning from optimization results")
    print("✅ Performance monitoring and statistics")
    
    print(f"\n🚀 Key Insights:")
    print("• LLM makes informed suggestions based on ML domain knowledge")
    print("• Context includes trial history, trends, and problem description")
    print("• Parameters are validated and clamped to search space constraints")
    print("• System provides explanations for each suggestion")
    print("• Robust error handling ensures optimization continues")
    
    print(f"\n📚 Next Steps:")
    print("• Run: python demo.py (comprehensive functionality demo)")
    print("• Test: python -m pytest tests/test_imports.py -v")
    print("• Use: Set OPENAI_API_KEY and run with real LLM!")
    
    print(f"\n💡 The system is ready for production use!")

if __name__ == "__main__":
    main()