"""
LLM Client: Handles communication with Large Language Models for hyperparameter suggestions.

This module provides a robust interface for querying LLMs with structured output
requirements, error handling, and retry logic.
"""

import json
import logging
import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import TypeVar

import litellm
from models import OptimizationContext
from models import TrialConfiguration
from pydantic import BaseModel


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""

    pass


class LLMRequestError(LLMError):
    """Invalid request to LLM API."""

    pass


class LLMParsingError(LLMError):
    """Failed to parse LLM response."""

    pass


class LLMClient:
    """
    Client for structured LLM interactions using LiteLLM.

    Handles LLM communication with robust error handling, retry logic,
    and structured output parsing using Pydantic models.
    """

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.3,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize LLM client.

        Args:
            model: LLM model identifier for LiteLLM
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure LiteLLM
        litellm.enable_json_schema_validation = True
        litellm.set_verbose = False

        logger.info(f"Initialized LLM client with model: {model}, temperature: {temperature}")

    def generate_trial_configuration(
        self, context: OptimizationContext, temperature: Optional[float] = None
    ) -> TrialConfiguration:
        """
        Generate a trial configuration using the LLM.

        Args:
            context: Complete optimization context
            temperature: Override default temperature

        Returns:
            Validated trial configuration

        Raises:
            LLMError: If generation fails after all retries
        """
        prompt = self._build_configuration_prompt(context)

        try:
            response = self._generate_structured_response(
                prompt=prompt, response_model=TrialConfiguration, temperature=temperature
            )

            logger.debug(f"Generated configuration with {len(response.parameters)} parameters")
            return response

        except Exception as e:
            logger.error(f"Failed to generate trial configuration: {e}")
            raise LLMError(f"Configuration generation failed: {e}")

    def _generate_structured_response(
        self, prompt: str, response_model: Type[T], temperature: Optional[float] = None
    ) -> T:
        """Generate structured response using specified Pydantic model."""

        effective_temperature = temperature if temperature is not None else self.temperature

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"LLM request attempt {attempt + 1}/{self.max_retries}")

                response = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=response_model,
                    temperature=effective_temperature,
                    timeout=self.timeout,
                )

                # Extract structured response
                structured_response = response.choices[0].message.content

                # Additional validation
                self._validate_response(structured_response, response_model)

                logger.debug(f"Successfully generated {response_model.__name__}")
                return structured_response

            except litellm.Timeout as e:
                last_exception = LLMTimeoutError(f"Request timed out after {self.timeout}s")
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Timeout on attempt {attempt + 1}, retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

            except litellm.BadRequestError as e:
                # Don't retry bad requests
                raise LLMRequestError(f"Invalid request: {e}")

            except litellm.AuthenticationError as e:
                # Don't retry auth errors
                raise LLMRequestError(f"Authentication failed: {e}")

            except Exception as e:
                last_exception = LLMError(f"Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Error on attempt {attempt + 1}: {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

        # All retries failed
        raise last_exception

    def _validate_response(self, response: BaseModel, model_class: Type[BaseModel]) -> None:
        """Additional validation for structured responses."""

        if isinstance(response, TrialConfiguration):
            self._validate_trial_configuration(response)

    def _validate_trial_configuration(self, config: TrialConfiguration) -> None:
        """Additional validation for trial configurations."""

        # Check for suspicious parameter values
        for param_name, value in config.parameters.items():
            if isinstance(value, (int, float)):
                if not (-1e6 <= value <= 1e6):  # Reasonable bounds
                    logger.warning(f"Suspicious parameter value: {param_name} = {value}")

                if isinstance(value, float) and (abs(value) < 1e-10 or abs(value) > 1e10):
                    logger.warning(f"Extreme parameter value: {param_name} = {value}")

        # Check reasoning quality
        reasoning_words = len(config.reasoning.split())
        if reasoning_words < 3:
            logger.warning("Reasoning appears too brief")
        elif reasoning_words > 100:
            logger.warning("Reasoning appears too verbose")

    def _build_configuration_prompt(self, context: OptimizationContext) -> str:
        """Build comprehensive prompt for trial configuration generation."""

        sections = []

        # Header
        sections.append("You are an expert hyperparameter optimization assistant.")
        sections.append("")

        # Problem context
        sections.extend(
            [
                "OPTIMIZATION CONTEXT:",
                f"- Objective: {context.objective_direction} '{context.objective_name}'",
                f"- Problem Type: {context.problem_type}",
                f"- Description: {context.problem_description}",
                f"- Trials completed: {context.n_trials_completed}",
                f"- Optimization stage: {context.progress_analysis.stage.value}",
                f"- Recent trend: {context.progress_analysis.trend.value} (strength: {context.progress_analysis.trend_strength:.2f})",
                f"- Trials since improvement: {context.progress_analysis.trials_since_improvement}",
                "",
            ]
        )

        # Search space
        sections.append("SEARCH SPACE:")
        for param in context.search_space:
            param_line = f"- {param.name} ({param.type})"

            if param.type in ["float", "int"]:
                param_line += f": {param.low} to {param.high}"
                if param.log_scale:
                    param_line += " (log scale)"
                if param.step:
                    param_line += f" (step: {param.step})"
            elif param.type == "categorical":
                param_line += f": {param.choices}"

            if param.description:
                param_line += f" - {param.description}"

            sections.append(param_line)
        sections.append("")

        # Best result
        if context.best_trial:
            sections.extend(
                [
                    "BEST RESULT SO FAR:",
                    f"- Value: {context.best_trial.value:.6f}",
                    f"- Parameters: {json.dumps(context.best_trial.parameters, indent=2)}",
                    "",
                ]
            )

        # Recent trials (last 5)
        if context.recent_trials:
            sections.append("RECENT TRIAL RESULTS:")
            recent_subset = context.recent_trials[-5:]  # Last 5 only
            for trial in recent_subset:
                sections.append(
                    f"- Trial {trial.trial_number}: {trial.value:.6f} with {trial.parameters}"
                )
            sections.append("")

        # Progress analysis
        sections.extend(
            [
                "OPTIMIZATION ANALYSIS:",
                f"- Current stage: {context.progress_analysis.stage.value}",
                f"- Trend: {context.progress_analysis.trend.value}",
                f"- Recommendation: {context.progress_analysis.recommendation}",
                "",
            ]
        )

        # Domain knowledge
        if context.domain_knowledge:
            sections.append("DOMAIN KNOWLEDGE:")

            if "best_practices" in context.domain_knowledge:
                sections.append("Best Practices:")
                for practice in context.domain_knowledge["best_practices"][:3]:  # Top 3
                    sections.append(f"- {practice}")

            if "parameter_relationships" in context.domain_knowledge:
                sections.append("Parameter Relationships:")
                for rel, desc in list(context.domain_knowledge["parameter_relationships"].items())[
                    :3
                ]:
                    sections.append(f"- {rel}: {desc}")

            sections.append("")

        # Constraints
        if context.constraints:
            sections.append("CONSTRAINTS:")
            for constraint in context.constraints:
                sections.append(f"- {constraint}")
            sections.append("")

        # Task instructions
        sections.extend(
            [
                "TASK:",
                "Suggest the next hyperparameter configuration to test. Consider:",
                "1. Parameter relationships and known interactions",
                "2. Successful patterns from recent trials",
                "3. Current optimization stage and trends",
                "4. Domain knowledge and best practices",
                "5. Exploration vs exploitation balance",
                "",
                "Provide a complete configuration with specific reasoning for your choices.",
                "Be concrete about why each parameter value makes sense given the context.",
                "",
                "FORMAT REQUIREMENTS:",
                "Your response must be valid JSON with this structure:",
                "{",
                '  "parameters": {"param1": value1, "param2": value2, ...},',
                '  "reasoning": "Specific explanation of parameter choices",',
                '  "confidence": 0.8,',
                '  "strategy": "balanced",',
                '  "expected_performance": "Brief performance prediction"',
                "}",
                "",
                "IMPORTANT:",
                "- Include ALL required parameters from the search space",
                "- Use exact parameter names as specified above",
                "- Provide specific numerical values within the given ranges",
                "- Keep reasoning concise but informative (1-3 sentences)",
                "- Confidence should reflect your certainty (0.0 to 1.0)",
            ]
        )

        prompt = "\n".join(sections)

        # Log prompt length for monitoring
        logger.debug(f"Built prompt with {len(prompt)} characters, ~{len(prompt)//4} tokens")

        return prompt
