#!/usr/bin/env python3
"""
Real LLM Demo - Use this with an actual LLM API key!

Set your API key with: export OPENAI_API_KEY="your-key-here"
Or modify the code to use your preferred LLM provider.
"""

import sys
import os
sys.path.insert(0, 'src')

import optuna
from sampler import LLMGuidedSampler
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

print("🤖 Real LLM-Guided Optuna Demo")
print("=" * 50)

def neural_network_objective(trial):
    """
    Simulated neural network training objective.
    This represents training a CNN on CIFAR-10.
    """
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    
    print(f"\n🔬 Training with:")
    print(f"   Learning Rate: {learning_rate:.2e}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Dropout: {dropout:.3f}")
    print(f"   Weight Decay: {weight_decay:.2e}")
    print(f"   Optimizer: {optimizer}")
    
    # Simulate model training (replace with actual training)
    import random
    import math
    import time
    
    # Simulate training time
    time.sleep(1)  # Shorter for demo
    
    # Mock validation loss based on hyperparameters
    # This simulates realistic training dynamics
    base_loss = 0.4
    
    # Learning rate effect (optimal around 1e-3 for Adam)
    lr_penalty = abs(math.log10(learning_rate) + 3) * 0.08
    
    # Batch size effect (optimal around 64-96)
    batch_penalty = abs(batch_size - 80) / 80 * 0.04
    
    # Dropout effect (optimal around 0.2-0.3)
    dropout_penalty = abs(dropout - 0.25) * 0.15
    
    # Weight decay effect (optimal around 1e-4)
    wd_penalty = abs(math.log10(weight_decay) + 4) * 0.03
    
    # Optimizer effect
    optimizer_bonus = {"adam": 0.0, "sgd": 0.05, "rmsprop": 0.02}[optimizer]
    
    # Add realistic noise
    noise = random.gauss(0, 0.015)
    
    validation_loss = (
        base_loss + lr_penalty + batch_penalty + 
        dropout_penalty + wd_penalty + optimizer_bonus + noise
    )
    
    # Ensure positive loss
    validation_loss = max(0.1, validation_loss)
    
    print(f"   📊 Validation Loss: {validation_loss:.6f}")
    return validation_loss

def main():
    """Run the LLM-guided optimization."""
    
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("\nOr modify this script to use a different LLM provider.")
        print("The LLMGuidedSampler supports any LiteLLM-compatible model!")
        print("\nAlternatively, run the mock demo to see the functionality:")
        print("   python demo.py")
        return
    
    print("✅ API key found! Setting up LLM-guided optimization...")
    
    # Create LLM-guided sampler
    sampler = LLMGuidedSampler(
        model="gpt-4o-2024-08-06",  # Use GPT-4o for best results
        temperature=0.3,            # Balance creativity and consistency
        timeout=30,                 # 30 second timeout
        max_retries=3,              # Retry failed requests
        max_context_trials=8        # Include last 8 trials in context
    )
    
    # Create study with rich context for the LLM
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="llm_guided_cnn_optimization"
    )
    
    # Add problem context - this helps the LLM make better decisions!
    study.set_user_attr("problem_type", "neural_network_training")
    study.set_user_attr(
        "problem_description", 
        "Optimizing CNN hyperparameters for CIFAR-10 image classification. "
        "Model: ResNet-18 architecture. Goal: minimize validation loss."
    )
    study.set_user_attr("constraints", [
        "training_time < 2 hours per trial",
        "memory_usage < 8GB",
        "target_accuracy > 90%"
    ])
    study.set_user_attr("domain_knowledge", {
        "dataset": "CIFAR-10 (32x32 RGB images, 10 classes)",
        "model_architecture": "ResNet-18 with batch normalization",
        "previous_best": "~92% accuracy with lr=0.001, bs=64",
        "known_issues": "High learning rates cause instability, very small batch sizes slow convergence"
    })
    
    print(f"🚀 Starting LLM-guided optimization with {sampler}")
    print(f"📚 Context: {study.user_attrs['problem_description']}")
    
    try:
        # Run optimization - each trial will query the LLM!
        print(f"\n🎯 Running 10 trials (this will take a few minutes)...")
        study.optimize(neural_network_objective, n_trials=10)
        
        print(f"\n🏆 Optimization Complete!")
        print(f"Best trial: #{study.best_trial.number}")
        print(f"Best validation loss: {study.best_value:.6f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2e}" if value < 0.01 else f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Show trial progression
        print(f"\n📈 Optimization Progress:")
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        for i, trial in enumerate(completed_trials):
            marker = "🥇" if trial.number == study.best_trial.number else f"{i+1:2d}."
            print(f"{marker} Trial {trial.number}: {trial.value:.6f}")
        
        # Show LLM sampler statistics
        stats = sampler.get_statistics()
        print(f"\n📊 LLM Sampler Performance:")
        print(f"Success rate: {stats['sampler_stats']['success_rate']:.1%}")
        print(f"Average LLM time: {stats['sampler_stats']['average_llm_time']:.1f}s")
        print(f"Total trials: {stats['sampler_stats']['total_trials']}")
        print(f"LLM generations: {stats['sampler_stats']['successful_generations']}")
        print(f"Fallbacks used: {stats['sampler_stats']['fallback_uses']}")
        print(f"Parameters fixed: {stats['validation_stats']['fixed_parameters']}")
        print(f"Parameters clamped: {stats['validation_stats']['clamped_parameters']}")
        
        # Compare with regular Optuna
        print(f"\n⚖️  Comparison with Regular TPE:")
        regular_study = optuna.create_study(direction="minimize")
        print("Running 10 trials with regular TPE sampler...")
        regular_study.optimize(neural_network_objective, n_trials=10, show_progress_bar=False)
        
        print(f"LLM-Guided best: {study.best_value:.6f}")
        print(f"Regular TPE best: {regular_study.best_value:.6f}")
        
        improvement = (regular_study.best_value - study.best_value) / regular_study.best_value * 100
        if improvement > 0:
            print(f"🎉 LLM-Guided improved by {improvement:.1f}%!")
        else:
            print(f"📊 Regular TPE was better by {abs(improvement):.1f}%")
        
        print(f"\n✨ Note: LLM-guided optimization shines with:")
        print("• Domain knowledge incorporation")
        print("• Intelligent exploration strategies")
        print("• Adaptive parameter relationships")
        print("• Human-interpretable reasoning")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Optimization interrupted by user")
        if study.trials:
            print(f"Best result so far: {study.best_value:.6f}")
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        print("This might be due to API limits, network issues, or invalid API key")

if __name__ == "__main__":
    main()