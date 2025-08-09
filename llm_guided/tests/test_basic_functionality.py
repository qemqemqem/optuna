#!/usr/bin/env python3
"""
Basic functionality tests for LLM-Guided Optuna components.

This test module verifies that the core components work correctly with
mock LLM responses to avoid requiring actual API calls during testing.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch


# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from context_builder import ContextBuilder
from llm_client import LLMClient
from models import OptimizationContext
from models import SearchSpaceParameter
from models import TrialConfiguration
from parameter_validator import ParameterValidator

import optuna


class TestModels(unittest.TestCase):
    """Test Pydantic model validation and creation."""

    def test_trial_configuration_validation(self):
        """Test TrialConfiguration model validation."""

        # Valid configuration
        config = TrialConfiguration(
            parameters={"learning_rate": 0.001, "batch_size": 32},
            reasoning="Test configuration with moderate learning rate and batch size",
        )

        self.assertEqual(config.parameters["learning_rate"], 0.001)
        self.assertEqual(config.parameters["batch_size"], 32)
        self.assertIsInstance(config.reasoning, str)

        # Test empty parameters validation
        with self.assertRaises(ValueError):
            TrialConfiguration(parameters={}, reasoning="This should fail due to empty parameters")

        # Test boilerplate reasoning detection
        with self.assertRaises(ValueError):
            TrialConfiguration(
                parameters={"lr": 0.01},
                reasoning="These are good parameters that should work well",
            )

    def test_search_space_parameter_creation(self):
        """Test SearchSpaceParameter model creation."""

        # Float parameter
        float_param = SearchSpaceParameter(
            name="learning_rate", type="float", low=1e-5, high=1e-1, log_scale=True
        )

        self.assertEqual(float_param.name, "learning_rate")
        self.assertEqual(float_param.type, "float")
        self.assertTrue(float_param.log_scale)

        # Categorical parameter
        cat_param = SearchSpaceParameter(
            name="optimizer", type="categorical", choices=["adam", "sgd", "rmsprop"]
        )

        self.assertEqual(cat_param.choices, ["adam", "sgd", "rmsprop"])

        # Test validation: categorical without choices should fail
        with self.assertRaises(ValueError):
            SearchSpaceParameter(
                name="bad_categorical",
                type="categorical",
                # Missing choices
            )


class TestContextBuilder(unittest.TestCase):
    """Test context building from Optuna studies."""

    def setUp(self):
        """Set up test fixtures."""
        self.context_builder = ContextBuilder()

    def test_empty_study_context(self):
        """Test context building for empty study."""

        study = optuna.create_study(direction="minimize")
        study.set_user_attr("problem_type", "test_problem")

        context = self.context_builder.build_context(study)

        self.assertEqual(context.n_trials_completed, 0)
        self.assertEqual(context.objective_direction, "MINIMIZE")
        self.assertEqual(context.problem_type, "test_problem")
        self.assertEqual(len(context.recent_trials), 0)
        self.assertIsNone(context.best_trial)

    def test_study_with_trials_context(self):
        """Test context building with completed trials."""

        def mock_objective(trial):
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_int("batch_size", 16, 128)
            return lr * batch_size / 1000  # Mock objective

        study = optuna.create_study(direction="minimize")
        study.set_user_attr("problem_type", "neural_network_training")

        # Run a few trials
        study.optimize(mock_objective, n_trials=3)

        context = self.context_builder.build_context(study)

        self.assertEqual(context.n_trials_completed, 3)
        self.assertEqual(len(context.recent_trials), 3)
        self.assertIsNotNone(context.best_trial)
        self.assertEqual(len(context.search_space), 2)  # learning_rate and batch_size


class TestParameterValidator(unittest.TestCase):
    """Test parameter validation and constraint enforcement."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = ParameterValidator()

        # Create mock search space
        self.search_space = {
            "learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-1, log=True),
            "batch_size": optuna.distributions.IntDistribution(16, 256, step=16),
            "optimizer": optuna.distributions.CategoricalDistribution(["adam", "sgd", "rmsprop"]),
        }

    def test_valid_configuration_validation(self):
        """Test validation of valid configuration."""

        config = TrialConfiguration(
            parameters={"learning_rate": 0.001, "batch_size": 64, "optimizer": "adam"},
            reasoning="Valid configuration for testing",
        )

        validated = self.validator.validate_and_clamp_configuration(config, self.search_space)

        self.assertEqual(validated["learning_rate"], 0.001)
        self.assertEqual(validated["batch_size"], 64)
        self.assertEqual(validated["optimizer"], "adam")

    def test_parameter_clamping(self):
        """Test parameter value clamping to bounds."""

        config = TrialConfiguration(
            parameters={
                "learning_rate": 10.0,  # Too high, should be clamped
                "batch_size": 8,  # Too low, should be clamped
                "optimizer": "adam",
            },
            reasoning="Configuration with out-of-bounds values",
        )

        validated = self.validator.validate_and_clamp_configuration(config, self.search_space)

        self.assertEqual(validated["learning_rate"], 1e-1)  # Clamped to max
        self.assertEqual(validated["batch_size"], 16)  # Clamped to min
        self.assertEqual(validated["optimizer"], "adam")

    def test_missing_parameter_handling(self):
        """Test handling of missing parameters."""

        config = TrialConfiguration(
            parameters={
                "learning_rate": 0.001,
                # Missing batch_size and optimizer
            },
            reasoning="Configuration with missing parameters",
        )

        validated = self.validator.validate_and_clamp_configuration(config, self.search_space)

        self.assertEqual(validated["learning_rate"], 0.001)
        self.assertIn("batch_size", validated)  # Should be filled with default
        self.assertIn("optimizer", validated)  # Should be filled with default

    def test_categorical_parameter_fuzzy_matching(self):
        """Test fuzzy matching for categorical parameters."""

        config = TrialConfiguration(
            parameters={
                "learning_rate": 0.001,
                "batch_size": 64,
                "optimizer": "ADAM",  # Wrong case, should match "adam"
            },
            reasoning="Configuration with case mismatch in categorical parameter",
        )

        validated = self.validator.validate_and_clamp_configuration(config, self.search_space)

        self.assertEqual(validated["optimizer"], "adam")  # Should be normalized


class TestLLMClientMocking(unittest.TestCase):
    """Test LLM client with mocked responses."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm_client = LLMClient(model="gpt-4o-test", temperature=0.3)

    @patch("litellm.completion")
    def test_successful_generation(self, mock_completion):
        """Test successful configuration generation."""

        # Mock successful LiteLLM response
        mock_config = TrialConfiguration(
            parameters={"learning_rate": 0.001, "batch_size": 64, "dropout": 0.2},
            reasoning="Generated configuration for test case with moderate parameters",
            confidence=0.8,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = mock_config
        mock_completion.return_value = mock_response

        # Create mock context
        context = Mock()
        context.objective_direction = "MINIMIZE"
        context.problem_type = "test"
        context.search_space = []
        context.recent_trials = []
        context.progress_analysis = Mock()
        context.progress_analysis.stage = Mock()
        context.progress_analysis.stage.value = "early_exploration"

        result = self.llm_client.generate_trial_configuration(context)

        self.assertIsInstance(result, TrialConfiguration)
        self.assertEqual(result.parameters["learning_rate"], 0.001)
        self.assertIn("moderate parameters", result.reasoning)

    @patch("litellm.completion")
    def test_timeout_handling(self, mock_completion):
        """Test timeout error handling."""

        # Mock timeout error
        import litellm

        mock_completion.side_effect = litellm.Timeout("Request timed out")

        context = Mock()
        context.objective_direction = "MINIMIZE"

        from llm_client import LLMTimeoutError

        with self.assertRaises(LLMTimeoutError):
            self.llm_client.generate_trial_configuration(context)


class TestIntegrationBasics(unittest.TestCase):
    """Basic integration tests for the complete pipeline."""

    @patch("litellm.completion")
    def test_end_to_end_mock_pipeline(self, mock_completion):
        """Test end-to-end pipeline with mocked LLM response."""

        # Mock LLM response
        mock_config = TrialConfiguration(
            parameters={"learning_rate": 0.001, "batch_size": 64},
            reasoning="Mock configuration for integration test",
            confidence=0.7,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = mock_config
        mock_completion.return_value = mock_response

        # Create study and sampler
        from sampler import LLMGuidedSampler

        sampler = LLMGuidedSampler(model="gpt-4o-test")
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # Mock objective function
        def mock_objective(trial):
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_int("batch_size", 16, 128)
            return lr * batch_size  # Simple mock objective

        # Run one trial
        study.optimize(mock_objective, n_trials=1)

        # Verify trial completed
        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.trials[0].state, optuna.trial.TrialState.COMPLETE)
        self.assertIn("learning_rate", study.trials[0].params)
        self.assertIn("batch_size", study.trials[0].params)


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
