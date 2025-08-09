#!/usr/bin/env python3
"""
Basic usage example of LLM-Guided Optuna.

This example demonstrates how to use the LLMGuidedSampler to optimize
hyperparameters for a simple neural network training scenario.
"""

import logging
import os
import sys
import time

import numpy as np


# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sampler import LLMGuidedSampler

import optuna


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def objective(trial):
    """
    Example objective function for neural network hyperparameter optimization.

    This simulates training a neural network with different hyperparameters
    and returns a validation loss (to be minimized).
    """
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 256, step=16)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    # Simulate model training (replace with actual training code)
    logger.info(
        f"Training with lr={learning_rate:.2e}, batch_size={batch_size}, "
        f"dropout={dropout:.3f}, weight_decay={weight_decay:.2e}, optimizer={optimizer}"
    )

    # Simulate training time
    training_time = np.random.uniform(1, 3)
    time.sleep(training_time)

    # Simulate validation loss based on hyperparameters
    # This is a mock function - replace with actual model evaluation
    base_loss = 0.5

    # Learning rate effect (optimal around 1e-3)
    lr_penalty = abs(np.log10(learning_rate) + 3) * 0.1

    # Batch size effect (optimal around 64-128)
    batch_penalty = abs(batch_size - 96) / 96 * 0.05

    # Dropout effect (optimal around 0.2-0.3)
    dropout_penalty = abs(dropout - 0.25) * 0.2

    # Weight decay effect (optimal around 1e-4)
    wd_penalty = abs(np.log10(weight_decay) + 4) * 0.05

    # Optimizer effect
    optimizer_bonus = {"adam": 0.0, "sgd": 0.02, "rmsprop": 0.01}[optimizer]

    # Add some noise
    noise = np.random.normal(0, 0.02)

    validation_loss = (
        base_loss
        + lr_penalty
        + batch_penalty
        + dropout_penalty
        + wd_penalty
        + optimizer_bonus
        + noise
    )

    logger.info(f"Validation loss: {validation_loss:.6f}")
    return validation_loss


def main():
    """Main execution function."""

    # Create LLM-guided sampler
    sampler = LLMGuidedSampler(
        model="gpt-4o-2024-08-06",  # Use your preferred model
        temperature=0.3,  # Balance between creativity and consistency
        timeout=30,  # 30 second timeout per request
        max_retries=3,  # Retry failed requests
    )

    # Create Optuna study with LLM sampler
    study = optuna.create_study(
        direction="minimize", sampler=sampler, study_name="neural_network_optimization"
    )

    # Add problem context for better LLM guidance
    study.set_user_attr("problem_type", "neural_network_training")
    study.set_user_attr(
        "problem_description", "Optimizing CNN hyperparameters for image classification task"
    )
    study.set_user_attr("constraints", ["training_time < 2 hours", "memory_usage < 8GB"])
    study.set_user_attr(
        "domain_knowledge",
        {"dataset": "CIFAR-10", "model_architecture": "ResNet-18", "expected_accuracy": "85-92%"},
    )

    logger.info("Starting LLM-guided optimization...")
    logger.info(f"Using sampler: {sampler}")

    try:
        # Run optimization
        study.optimize(objective, n_trials=20)

        # Print results
        logger.info("Optimization completed!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.6f}")
        logger.info(f"Best parameters: {study.best_params}")

        # Print sampler statistics
        stats = sampler.get_statistics()
        logger.info("Sampler Statistics:")
        logger.info(f"  Success rate: {stats['sampler_stats']['success_rate']:.2%}")
        logger.info(f"  Average LLM time: {stats['sampler_stats']['average_llm_time']:.2f}s")
        logger.info(f"  Fallback uses: {stats['sampler_stats']['fallback_uses']}")
        logger.info(f"  Validation fixes: {stats['validation_stats']['fixed_parameters']}")

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
