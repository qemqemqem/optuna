"""
Parameter Validator: Validates and clamps LLM-generated parameters to search space constraints.

This module ensures that LLM-generated parameters conform to the defined search space
bounds and constraints, providing graceful fallbacks for invalid values.
"""

import json
import logging
import re
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

try:
    from .models import TrialConfiguration
except ImportError:
    from models import TrialConfiguration

import optuna


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Parameter validation failed."""

    pass


class ParameterValidator:
    """
    Validates and clamps LLM-generated parameters to search space bounds.

    This class ensures that all LLM suggestions conform to the defined search
    space constraints, providing fallbacks for invalid or missing parameters.
    """

    def __init__(self):
        """Initialize parameter validator."""
        self.validation_stats = {
            "total_validations": 0,
            "clamped_parameters": 0,
            "fixed_parameters": 0,
            "failed_validations": 0,
        }

    def validate_and_clamp_configuration(
        self,
        config: TrialConfiguration,
        search_space: Dict[str, optuna.distributions.BaseDistribution],
    ) -> Dict[str, Any]:
        """
        Validate complete configuration against search space.

        Args:
            config: LLM-generated trial configuration
            search_space: Optuna search space definition

        Returns:
            Validated and clamped parameter dictionary

        Raises:
            ValidationError: If validation fails completely
        """
        self.validation_stats["total_validations"] += 1

        try:
            validated_config = {}

            # Check all required parameters are present
            missing_params = set(search_space.keys()) - set(config.parameters.keys())
            if missing_params:
                logger.warning(f"Missing parameters: {missing_params}")
                # Add default values for missing parameters
                for param_name in missing_params:
                    default_value = self._get_default_value(search_space[param_name])
                    config.parameters[param_name] = default_value
                    self.validation_stats["fixed_parameters"] += 1

            # Validate each parameter
            for param_name, distribution in search_space.items():
                if param_name not in config.parameters:
                    raise ValidationError(f"Missing required parameter: {param_name}")

                raw_value = config.parameters[param_name]
                validated_value = self._validate_parameter(raw_value, distribution, param_name)
                validated_config[param_name] = validated_value

                # Track if parameter was clamped
                if raw_value != validated_value:
                    self.validation_stats["clamped_parameters"] += 1
                    logger.debug(f"Clamped {param_name}: {raw_value} -> {validated_value}")

            logger.debug(
                f"Successfully validated configuration with {len(validated_config)} parameters"
            )
            return validated_config

        except Exception as e:
            self.validation_stats["failed_validations"] += 1
            logger.error(f"Configuration validation failed: {e}")
            raise ValidationError(f"Failed to validate configuration: {e}")

    def _validate_parameter(
        self, value: Any, distribution: optuna.distributions.BaseDistribution, param_name: str
    ) -> Any:
        """Validate and clamp single parameter."""

        try:
            if isinstance(distribution, optuna.distributions.FloatDistribution):
                return self._validate_float_parameter(value, distribution, param_name)
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                return self._validate_int_parameter(value, distribution, param_name)
            elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                return self._validate_categorical_parameter(value, distribution, param_name)
            else:
                logger.warning(
                    f"Unsupported distribution type for {param_name}: {type(distribution)}"
                )
                return value

        except Exception as e:
            logger.warning(f"Error validating {param_name}: {e}. Using default value.")
            return self._get_default_value(distribution)

    def _validate_float_parameter(
        self, value: Any, dist: optuna.distributions.FloatDistribution, param_name: str
    ) -> float:
        """Validate and clamp float parameter."""

        # Convert to float
        try:
            if isinstance(value, str):
                # Try to parse scientific notation or special values
                value = value.strip().lower()
                if value in ["inf", "infinity", "+inf"]:
                    float_val = float("inf")
                elif value in ["-inf", "-infinity"]:
                    float_val = float("-inf")
                elif value in ["nan", "none", "null"]:
                    float_val = float("nan")
                else:
                    float_val = float(value)
            else:
                float_val = float(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value for {param_name}: {value}, using midpoint")
            return self._get_float_midpoint(dist)

        # Handle special values
        if not self._is_finite(float_val):
            logger.warning(f"Non-finite value for {param_name}: {float_val}, using midpoint")
            return self._get_float_midpoint(dist)

        # Clamp to bounds
        clamped = max(dist.low, min(dist.high, float_val))

        # Handle step constraints
        if dist.step is not None:
            # Round to nearest step
            steps_from_low = round((clamped - dist.low) / dist.step)
            clamped = dist.low + steps_from_low * dist.step
            # Ensure still within bounds after stepping
            clamped = max(dist.low, min(dist.high, clamped))

        return clamped

    def _validate_int_parameter(
        self, value: Any, dist: optuna.distributions.IntDistribution, param_name: str
    ) -> int:
        """Validate and clamp integer parameter."""

        # Convert to int
        try:
            if isinstance(value, str):
                value = value.strip()
                # Handle common string representations
                if value.lower() in ["none", "null", "nan"]:
                    int_val = (dist.low + dist.high) // 2
                else:
                    # Try to convert, handling floats in strings
                    int_val = int(round(float(value)))
            else:
                int_val = int(round(float(value)))
        except (ValueError, TypeError):
            logger.warning(f"Invalid int value for {param_name}: {value}, using midpoint")
            return (dist.low + dist.high) // 2

        # Clamp to bounds
        clamped = max(dist.low, min(dist.high, int_val))

        # Handle step constraints
        if dist.step is not None and dist.step != 1:
            # Round to nearest step
            steps_from_low = round((clamped - dist.low) / dist.step)
            clamped = dist.low + steps_from_low * dist.step
            # Ensure still within bounds after stepping
            clamped = max(dist.low, min(dist.high, clamped))

        return int(clamped)

    def _validate_categorical_parameter(
        self, value: Any, dist: optuna.distributions.CategoricalDistribution, param_name: str
    ) -> Any:
        """Validate categorical parameter."""

        choices = list(dist.choices)

        # Direct match
        if value in choices:
            return value

        # String matching with normalization
        if isinstance(value, str):
            value_normalized = value.strip().lower()

            # Try exact match after normalization
            for choice in choices:
                if isinstance(choice, str) and choice.strip().lower() == value_normalized:
                    return choice

            # Try partial matching
            for choice in choices:
                if isinstance(choice, str):
                    choice_normalized = choice.strip().lower()
                    if (
                        value_normalized in choice_normalized
                        or choice_normalized in value_normalized
                    ):
                        logger.debug(f"Partial match for {param_name}: {value} -> {choice}")
                        return choice

        # No match found - use first choice as default
        logger.warning(
            f"Invalid categorical value for {param_name}: {value}, using default: {choices[0]}"
        )
        return choices[0]

    def _get_default_value(self, distribution: optuna.distributions.BaseDistribution) -> Any:
        """Get reasonable default value for distribution."""

        if isinstance(distribution, optuna.distributions.FloatDistribution):
            return self._get_float_midpoint(distribution)
        elif isinstance(distribution, optuna.distributions.IntDistribution):
            return (distribution.low + distribution.high) // 2
        elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
            return list(distribution.choices)[0]
        else:
            logger.warning(
                f"No default value strategy for distribution type: {type(distribution)}"
            )
            return None

    def _get_float_midpoint(self, dist: optuna.distributions.FloatDistribution) -> float:
        """Get midpoint value for float distribution."""
        if dist.log:
            # Geometric mean for log scale
            import math

            return math.exp((math.log(dist.low) + math.log(dist.high)) / 2)
        else:
            # Arithmetic mean for linear scale
            return (dist.low + dist.high) / 2.0

    def _is_finite(self, value: float) -> bool:
        """Check if float value is finite."""
        import math

        return math.isfinite(value)

    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return self.validation_stats.copy()

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            "total_validations": 0,
            "clamped_parameters": 0,
            "fixed_parameters": 0,
            "failed_validations": 0,
        }


class ErrorRecoveryHandler:
    """
    Handles validation errors with recovery strategies.

    When LLM responses are malformed or validation fails, this class
    attempts various recovery strategies before falling back to defaults.
    """

    def __init__(self, search_space: Dict[str, optuna.distributions.BaseDistribution]):
        """Initialize error recovery handler."""
        self.search_space = search_space
        self.validator = ParameterValidator()

    def handle_validation_error(self, raw_response: str, error: Exception) -> TrialConfiguration:
        """
        Attempt to recover from validation errors.

        Args:
            raw_response: Raw LLM response string
            error: The validation error that occurred

        Returns:
            Recovered trial configuration

        Raises:
            ValidationError: If all recovery strategies fail
        """
        logger.warning(f"Attempting error recovery for: {error}")

        # Strategy 1: Try to parse partial JSON
        config = self._extract_partial_config(raw_response)
        if config:
            return config

        # Strategy 2: Regex extraction of parameter values
        config = self._regex_extract_parameters(raw_response)
        if config:
            return config

        # Strategy 3: Generate fallback configuration
        logger.warning("All recovery strategies failed, generating fallback configuration")
        return self._generate_fallback_config()

    def _extract_partial_config(self, raw_response: str) -> Optional[TrialConfiguration]:
        """Try to extract configuration from partial/malformed JSON."""

        # Try to find JSON-like structures
        json_patterns = [
            r'\{[^{}]*"parameters"[^{}]*\}',
            r"\{.*?\}",
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, raw_response, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if "parameters" in parsed:
                        return self._build_config_from_dict(parsed)
                except json.JSONDecodeError:
                    continue

        return None

    def _regex_extract_parameters(self, raw_response: str) -> Optional[TrialConfiguration]:
        """Extract parameter values using regex patterns."""

        parameters = {}

        for param_name in self.search_space.keys():
            # Try various patterns to extract parameter values
            patterns = [
                rf'"{param_name}":\s*([^\s,}}]+)',
                rf"{param_name}[:\s=]+([^\s,\n]+)",
                rf"{param_name}.*?([0-9.e\-+]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, raw_response, re.IGNORECASE)
                if match:
                    try:
                        value_str = match.group(1).strip("\"'")
                        # Try to convert to appropriate type
                        value = self._parse_parameter_value(
                            value_str, self.search_space[param_name]
                        )
                        parameters[param_name] = value
                        break
                    except Exception:
                        continue

        if len(parameters) >= len(self.search_space) // 2:  # At least half the parameters
            return self._build_config_from_dict(
                {
                    "parameters": parameters,
                    "reasoning": "Recovered from malformed response using regex extraction",
                }
            )

        return None

    def _parse_parameter_value(self, value_str: str, distribution) -> Any:
        """Parse parameter value string based on distribution type."""

        if isinstance(distribution, optuna.distributions.FloatDistribution):
            return float(value_str)
        elif isinstance(distribution, optuna.distributions.IntDistribution):
            return int(round(float(value_str)))
        elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
            return value_str
        else:
            return value_str

    def _build_config_from_dict(self, data: Dict[str, Any]) -> TrialConfiguration:
        """Build TrialConfiguration from dictionary data."""

        # Fill missing parameters with defaults
        parameters = data.get("parameters", {})
        for param_name, distribution in self.search_space.items():
            if param_name not in parameters:
                parameters[param_name] = self.validator._get_default_value(distribution)

        return TrialConfiguration(
            parameters=parameters,
            reasoning=data.get("reasoning", "Recovered configuration from partial response"),
            confidence=data.get("confidence", 0.3),
            strategy=data.get("strategy", "conservative"),
        )

    def _generate_fallback_config(self) -> TrialConfiguration:
        """Generate reasonable fallback configuration."""

        parameters = {}

        for param_name, distribution in self.search_space.items():
            parameters[param_name] = self.validator._get_default_value(distribution)

        return TrialConfiguration(
            parameters=parameters,
            reasoning="Fallback configuration using parameter midpoints due to LLM response failure",
            confidence=0.2,
            strategy="conservative",
        )
