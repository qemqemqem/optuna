#!/usr/bin/env python3
"""
Simple test script to verify llm_guided functionality works.
"""

import os
import sys
sys.path.insert(0, 'src')

import optuna
from sampler import LLMGuidedSampler
from context_builder import ContextBuilder

print("Testing LLM-Guided Optuna functionality...")

# Test 1: Import test
print("✓ All modules imported successfully")

# Test 2: Basic instantiation
try:
    sampler = LLMGuidedSampler(model="gpt-4o-2024-08-06")
    print("✓ LLMGuidedSampler instantiated successfully")
except Exception as e:
    print(f"✗ LLMGuidedSampler instantiation failed: {e}")

# Test 3: Context builder with empty study
try:
    builder = ContextBuilder()
    study = optuna.create_study()
    context = builder.build_context(study)
    print("✓ Context builder works with empty study")
except Exception as e:
    print(f"✗ Context builder failed with empty study: {e}")

# Test 4: Context builder with trials
try:
    study = optuna.create_study()
    
    def test_objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        bs = trial.suggest_int("batch_size", 16, 128)
        return lr * bs / 1000
    
    study.optimize(test_objective, n_trials=3)
    
    builder = ContextBuilder()
    context = builder.build_context(study)
    print("✓ Context builder works with completed trials")
    print(f"  - Found {len(context.search_space)} search space parameters")
    print(f"  - Found {len(context.recent_trials)} recent trials")
except Exception as e:
    print(f"✗ Context builder failed with trials: {e}")

print("Testing completed!")