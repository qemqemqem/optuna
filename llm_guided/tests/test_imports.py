#!/usr/bin/env python3
"""
Test that all core components can be imported successfully.
"""

import os
import sys
import unittest


# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestImports(unittest.TestCase):
    """Test that all modules can be imported without errors."""

    def test_models_import(self):
        """Test importing models module."""
        try:
            from models import OptimizationContext
            from models import SearchSpaceParameter
            from models import TrialConfiguration

            self.assertTrue(True, "Models imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import models: {e}")

    def test_context_builder_import(self):
        """Test importing context builder module."""
        try:
            from context_builder import ContextBuilder

            self.assertTrue(True, "Context builder imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import context builder: {e}")

    def test_parameter_validator_import(self):
        """Test importing parameter validator module."""
        try:
            from parameter_validator import ErrorRecoveryHandler
            from parameter_validator import ParameterValidator

            self.assertTrue(True, "Parameter validator imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import parameter validator: {e}")

    def test_llm_client_import(self):
        """Test importing LLM client module."""
        try:
            from llm_client import LLMClient
            from llm_client import LLMError

            self.assertTrue(True, "LLM client imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import LLM client: {e}")

    def test_sampler_import(self):
        """Test importing main sampler module."""
        try:
            from sampler import LLMGuidedSampler

            self.assertTrue(True, "LLM sampler imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import sampler: {e}")

    def test_basic_instantiation(self):
        """Test basic instantiation of key classes."""
        try:
            from context_builder import ContextBuilder
            from llm_client import LLMClient
            from models import TrialConfiguration
            from parameter_validator import ParameterValidator
            from sampler import LLMGuidedSampler

            # Test basic instantiation
            config = TrialConfiguration(
                parameters={"test_param": 0.5},
                reasoning="Test configuration for import verification",
            )

            context_builder = ContextBuilder()
            validator = ParameterValidator()
            llm_client = LLMClient(model="gpt-4o-test")
            sampler = LLMGuidedSampler(model="gpt-4o-test")

            self.assertIsNotNone(config)
            self.assertIsNotNone(context_builder)
            self.assertIsNotNone(validator)
            self.assertIsNotNone(llm_client)
            self.assertIsNotNone(sampler)

            print("✓ All basic instantiations successful")

        except Exception as e:
            self.fail(f"Failed basic instantiation test: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
