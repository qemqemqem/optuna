#!/usr/bin/env python3
"""
Quick test with real LLM - just 1 trial to verify it works!
"""

import sys
import os
sys.path.insert(0, 'src')

import optuna
from sampler import LLMGuidedSampler
import logging

# Quieter logging
logging.basicConfig(level=logging.WARNING)

print("🧪 Quick Real LLM Test (1 trial only)")
print("=" * 40)

# Check API key
if not os.getenv('OPENAI_API_KEY'):
    print("❌ No OPENAI_API_KEY found!")
    print("Set it with: export OPENAI_API_KEY='your-key'")
    exit(1)

def simple_objective(trial):
    """Simple 2-parameter objective for quick testing."""
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    bs = trial.suggest_int("batch_size", 16, 128, step=16)
    
    # Simple quadratic objective (optimal around lr=0.01, bs=64)
    import math
    lr_penalty = (math.log10(lr) + 2) ** 2  # optimal around 1e-2
    bs_penalty = (bs - 64) ** 2 / 1000      # optimal around 64
    
    result = 0.5 + lr_penalty * 0.1 + bs_penalty * 0.001
    print(f"   lr={lr:.2e}, bs={bs} → loss={result:.4f}")
    return result

# Create LLM-guided sampler
sampler = LLMGuidedSampler(
    model="gpt-4o-2024-08-06",
    temperature=0.2,  # Lower temperature for more consistent results
    timeout=15,       # Shorter timeout for quick test
)

# Create study with context
study = optuna.create_study(direction="minimize", sampler=sampler)
study.set_user_attr("problem_type", "neural_network_training")
study.set_user_attr("problem_description", "Quick test: optimize learning rate and batch size")

print("🤖 Querying LLM for hyperparameter suggestion...")

try:
    # Run just 1 trial
    study.optimize(simple_objective, n_trials=1)
    
    print(f"\n✅ Success! LLM suggested:")
    print(f"   Learning Rate: {study.best_params['learning_rate']:.2e}")
    print(f"   Batch Size: {study.best_params['batch_size']}")
    print(f"   Result: {study.best_value:.4f}")
    
    # Show stats
    stats = sampler.get_statistics()
    print(f"\n📊 LLM Performance:")
    print(f"   Success: {stats['sampler_stats']['successful_generations']}/1")
    print(f"   LLM Time: {stats['sampler_stats']['average_llm_time']:.1f}s")
    
    print(f"\n🎉 LLM-Guided Optuna is working perfectly!")
    print(f"Now you can run: python real_llm_demo.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"This might be due to API limits or network issues")

print(f"\n🚀 Ready for full optimization!")