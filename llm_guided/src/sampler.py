"""
LLM-Guided Sampler: Main integration with Optuna's sampler interface.

This module implements the core LLMGuidedSampler that integrates with Optuna's
optimization framework to provide LLM-guided hyperparameter suggestions.
"""

import logging
import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

from context_builder import ContextBuilder
from llm_client import LLMClient
from llm_client import LLMError
from models import TrialConfiguration
from parameter_validator import ErrorRecoveryHandler
from parameter_validator import ParameterValidator
from parameter_validator import ValidationError

import optuna
from optuna.samplers import BaseSampler
from optuna.trial import TrialState


logger = logging.getLogger(__name__)


class LLMSamplingError(Exception):
    """Exception raised when LLM sampling fails completely."""

    pass


class LLMGuidedSampler(BaseSampler):
    """
    Optuna sampler that uses Large Language Models to suggest hyperparameter configurations.

    This sampler integrates with Optuna's optimization framework to provide LLM-guided
    suggestions for hyperparameter configurations. It calls the LLM for every trial
    to generate complete configurations based on the current optimization context.

    Key Features:
    - Complete trial-level configuration generation
    - Fresh context building from study state
    - Robust error handling and fallbacks
    - Parameter validation and constraint enforcement
    - Performance monitoring and statistics
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.3,
        timeout: int = 30,
        max_retries: int = 3,
        max_context_trials: int = 10,
    ):
        """
        Initialize LLM-guided sampler.

        Args:
            model: LLM model identifier for LiteLLM
            temperature: Sampling temperature for LLM (0.0-1.0)
            timeout: API timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            max_context_trials: Maximum trials to include in LLM context
        """
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_context_trials = max_context_trials

        # Initialize components
        self.context_builder = ContextBuilder(max_context_trials=max_context_trials)
        self.llm_client = LLMClient(model, temperature, timeout, max_retries)
        self.validator = ParameterValidator()

        # Performance tracking
        self.stats = {
            "total_trials": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "validation_fixes": 0,
            "fallback_uses": 0,
            "total_llm_time": 0.0,
            "average_llm_time": 0.0,
        }

        logger.info(f"Initialized LLMGuidedSampler with model: {model}")

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        """
        Sample a parameter value for the given trial.

        This method is called by Optuna for each parameter in the search space.
        We generate the complete configuration on the first parameter request
        and cache it for subsequent parameter requests in the same trial.

        Args:
            study: Optuna study
            trial: Current trial
            param_name: Name of parameter to sample
            param_distribution: Distribution for the parameter

        Returns:
            Sampled parameter value

        Raises:
            LLMSamplingError: If configuration generation fails
        """
        # Generate complete configuration on first parameter request
        if not hasattr(trial, "_llm_config_cache"):
            start_time = time.time()

            try:
                # Generate complete trial configuration
                config = self._generate_complete_trial_configuration(study, trial)
                trial._llm_config_cache = config

                # Update timing stats
                generation_time = time.time() - start_time
                self._update_timing_stats(generation_time)

                self.stats["successful_generations"] += 1
                logger.debug(
                    f"Generated configuration for trial {trial.number} in {generation_time:.2f}s"
                )

            except Exception as e:
                self.stats["failed_generations"] += 1
                logger.error(f"Failed to generate configuration for trial {trial.number}: {e}")
                raise LLMSamplingError(f"LLM configuration generation failed: {e}")

        # Return cached parameter value
        config = trial._llm_config_cache
        if param_name not in config:
            raise LLMSamplingError(
                f"Parameter '{param_name}' not found in generated configuration"
            )

        return config[param_name]

    def _generate_complete_trial_configuration(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> Dict[str, Any]:
        """Generate complete trial configuration using LLM."""

        self.stats["total_trials"] += 1

        try:
            # Build optimization context
            context = self.context_builder.build_context(study)

            # Initialize error recovery handler
            recovery_handler = ErrorRecoveryHandler(study.search_space)

            try:
                # Query LLM for configuration
                llm_response = self.llm_client.generate_trial_configuration(context)

                # Validate and clamp parameters
                validated_config = self.validator.validate_and_clamp_configuration(
                    llm_response, study.search_space
                )

                logger.info(f"Generated configuration: {llm_response.reasoning}")
                return validated_config

            except ValidationError as e:
                # Try error recovery
                logger.warning(f"Validation failed, attempting recovery: {e}")
                try:
                    recovered_config = recovery_handler.handle_validation_error(
                        str(llm_response), e
                    )
                    validated_config = self.validator.validate_and_clamp_configuration(
                        recovered_config, study.search_space
                    )
                    self.stats["validation_fixes"] += 1
                    return validated_config
                except Exception as recovery_error:
                    logger.error(f"Recovery also failed: {recovery_error}")
                    raise

            except LLMError as e:
                # LLM generation failed - use fallback
                logger.error(f"LLM generation failed: {e}")
                fallback_config = self._generate_fallback_configuration(study)
                self.stats["fallback_uses"] += 1
                return fallback_config

        except Exception as e:
            logger.error(f"Complete configuration generation failed: {e}")
            # Last resort: generate fallback
            fallback_config = self._generate_fallback_configuration(study)
            self.stats["fallback_uses"] += 1
            return fallback_config

    def _generate_fallback_configuration(self, study: optuna.Study) -> Dict[str, Any]:
        """Generate fallback configuration when LLM fails."""

        logger.warning("Generating fallback configuration due to LLM failure")

        config = {}
        # Get search space from completed trials
        trials = study.get_trials()
        if trials:
            search_space = trials[-1].distributions
            for param_name, distribution in search_space.items():
                config[param_name] = self.validator._get_default_value(distribution)

        return config

    def _update_timing_stats(self, generation_time: float) -> None:
        """Update timing statistics."""

        self.stats["total_llm_time"] += generation_time

        if self.stats["successful_generations"] > 0:
            self.stats["average_llm_time"] = (
                self.stats["total_llm_time"] / self.stats["successful_generations"]
            )

    def infer_relative_search_space(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> Dict[str, optuna.distributions.BaseDistribution]:
        """
        Infer relative search space for the trial.

        This method is called by Optuna to determine the search space for a trial.
        We return the complete study search space since we generate full configurations.
        """
        # Get search space from completed trials' distributions
        trials = study.get_trials()
        if trials:
            return trials[-1].distributions
        else:
            # No trials yet, return empty search space
            return {}

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: Dict[str, optuna.distributions.BaseDistribution],
    ) -> Dict[str, Any]:
        """
        Sample relative configuration.

        This method can be used as an alternative to sample_independent for
        sampling complete configurations at once.
        """
        # Generate complete configuration if not already cached
        if not hasattr(trial, "_llm_config_cache"):
            config = self._generate_complete_trial_configuration(study, trial)
            trial._llm_config_cache = config

        return trial._llm_config_cache

    def after_trial(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        """
        Called after trial completion.

        This method can be used to perform any cleanup or learning
        from the completed trial.
        """
        # Log trial completion
        if state == TrialState.COMPLETE and values is not None:
            logger.debug(f"Trial {trial.number} completed with value: {values[0]:.6f}")
        elif state == TrialState.FAIL:
            logger.warning(f"Trial {trial.number} failed")
        elif state == TrialState.PRUNED:
            logger.debug(f"Trial {trial.number} was pruned")

        # Clean up cached configuration
        if hasattr(trial, "_llm_config_cache"):
            delattr(trial, "_llm_config_cache")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about sampler performance."""

        total_attempts = self.stats["successful_generations"] + self.stats["failed_generations"]
        success_rate = (
            self.stats["successful_generations"] / total_attempts if total_attempts > 0 else 0.0
        )

        validator_stats = self.validator.get_validation_stats()

        return {
            "sampler_stats": {**self.stats, "success_rate": success_rate},
            "validation_stats": validator_stats,
            "configuration": {
                "model": self.model,
                "temperature": self.temperature,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "max_context_trials": self.max_context_trials,
            },
        }

    def reset_statistics(self) -> None:
        """Reset all performance statistics."""

        self.stats = {
            "total_trials": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "validation_fixes": 0,
            "fallback_uses": 0,
            "total_llm_time": 0.0,
            "average_llm_time": 0.0,
        }

        self.validator.reset_stats()
        logger.info("Reset sampler statistics")

    def __repr__(self) -> str:
        """String representation of the sampler."""
        return (
            f"LLMGuidedSampler("
            f"model={self.model}, "
            f"temperature={self.temperature}, "
            f"timeout={self.timeout}s"
            f")"
        )
