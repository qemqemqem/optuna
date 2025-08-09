"""
Context Builder: Extracts and formats optimization context for LLM queries.

This module handles the extraction of relevant information from Optuna studies
and formats it into structured context that LLMs can use to make informed
hyperparameter suggestions.
"""

from collections import Counter
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from models import create_search_space_parameter
from models import create_trial_result
from models import OptimizationContext
from models import OptimizationStage
from models import ProgressAnalysis
from models import SearchSpaceParameter
from models import TrendType
from models import TrialResult
import numpy as np

import optuna


logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds comprehensive optimization context from Optuna study state.

    This class extracts relevant information from the study history, analyzes
    optimization progress, and formats everything into a structured context
    that LLMs can use to make informed suggestions.
    """

    def __init__(self, max_recent_trials: int = 15, max_context_trials: int = 10):
        """
        Initialize context builder.

        Args:
            max_recent_trials: Maximum recent trials to analyze for trends
            max_context_trials: Maximum trials to include in LLM context
        """
        self.max_recent_trials = max_recent_trials
        self.max_context_trials = max_context_trials

        # Parameter description mappings
        self.parameter_descriptions = {
            "learning_rate": "Controls the step size in gradient descent optimization",
            "batch_size": "Number of samples processed before model weight update",
            "dropout": "Fraction of neurons randomly set to zero during training",
            "weight_decay": "L2 regularization strength to prevent overfitting",
            "momentum": "Momentum factor for gradient descent optimization",
            "epochs": "Number of complete passes through the training dataset",
            "hidden_size": "Number of neurons in hidden layers",
            "num_layers": "Number of layers in the neural network",
            "lr": "Learning rate - controls optimization step size",
            "beta1": "Adam optimizer first moment decay rate",
            "beta2": "Adam optimizer second moment decay rate",
            "epsilon": "Small constant for numerical stability",
        }

    def build_context(self, study: optuna.Study) -> OptimizationContext:
        """
        Build complete optimization context from study state.

        Args:
            study: Optuna study to extract context from

        Returns:
            Complete optimization context for LLM
        """
        logger.debug(f"Building context for study: {study.study_name}")

        # Extract completed trials
        completed_trials = [
            trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        # Build search space representation
        search_space = self._build_search_space(study)

        # Build trial history
        recent_trials = self._build_trial_history(completed_trials)
        best_trial = self._get_best_trial(study)

        # Analyze optimization progress
        progress = self._analyze_optimization_progress(completed_trials, study.direction)

        # Extract problem context
        problem_context = self._extract_problem_context(study)

        context = OptimizationContext(
            objective_name=study.study_name or "optimization_objective",
            objective_direction=study.direction.name,
            n_trials_completed=len(completed_trials),
            search_space=search_space,
            recent_trials=recent_trials,
            best_trial=best_trial,
            progress_analysis=progress,
            **problem_context,
        )

        logger.debug(
            f"Built context with {len(recent_trials)} recent trials, stage: {progress.stage}"
        )
        return context

    def _build_search_space(self, study: optuna.Study) -> List[SearchSpaceParameter]:
        """Build search space representation from Optuna study."""
        search_space_params = []

        # Get search space from completed trials' distributions
        search_space = {}
        trials = study.get_trials()
        if trials:
            # Use the most recent trial's distributions as the search space
            search_space = trials[-1].distributions

        for param_name, distribution in search_space.items():
            description = self.parameter_descriptions.get(param_name.lower())
            param = create_search_space_parameter(param_name, distribution, description)
            search_space_params.append(param)

        return search_space_params

    def _build_trial_history(self, completed_trials: List[optuna.Trial]) -> List[TrialResult]:
        """Build trial history for LLM context."""
        if not completed_trials:
            return []

        # Get most recent trials
        recent_trials = completed_trials[-self.max_context_trials :]

        trial_results = []
        for trial in recent_trials:
            trial_result = create_trial_result(trial)
            trial_results.append(trial_result)

        return trial_results

    def _get_best_trial(self, study: optuna.Study) -> Optional[TrialResult]:
        """Get best trial result."""
        try:
            if study.best_trial is None:
                return None
            return create_trial_result(study.best_trial)
        except ValueError:
            # No trials completed yet
            return None

    def _analyze_optimization_progress(
        self, completed_trials: List[optuna.Trial], direction: optuna.study.StudyDirection
    ) -> ProgressAnalysis:
        """Analyze optimization progress and trends."""

        if len(completed_trials) < 3:
            return ProgressAnalysis(
                stage=OptimizationStage.EARLY_EXPLORATION,
                trend=TrendType.INSUFFICIENT_DATA,
                trend_strength=0.0,
                trials_since_improvement=0,
                convergence_indicator=0.0,
                recommendation="Continue exploring the parameter space broadly",
            )

        values = [trial.value for trial in completed_trials]
        n_trials = len(completed_trials)

        # Determine optimization stage
        stage = self._determine_optimization_stage(n_trials)

        # Analyze trends
        trend_analysis = self._analyze_trend(values, direction)

        # Calculate trials since improvement
        trials_since_improvement = self._calculate_trials_since_improvement(values, direction)

        # Assess convergence
        convergence = self._assess_convergence(values)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            stage, trend_analysis, trials_since_improvement
        )

        return ProgressAnalysis(
            stage=stage,
            trend=trend_analysis["trend"],
            trend_strength=trend_analysis["strength"],
            trials_since_improvement=trials_since_improvement,
            convergence_indicator=convergence,
            best_value_trend=values[-5:] if len(values) >= 5 else values,
            recommendation=recommendation,
        )

    def _determine_optimization_stage(self, n_trials: int) -> OptimizationStage:
        """Determine current optimization stage based on trial count."""
        if n_trials < 10:
            return OptimizationStage.EARLY_EXPLORATION
        elif n_trials < 30:
            return OptimizationStage.ACTIVE_SEARCH
        elif n_trials < 80:
            return OptimizationStage.FOCUSED_OPTIMIZATION
        else:
            return OptimizationStage.REFINEMENT

    def _analyze_trend(
        self, values: List[float], direction: optuna.study.StudyDirection
    ) -> Dict[str, Any]:
        """Analyze optimization trend from recent values."""

        if len(values) < 5:
            return {"trend": TrendType.INSUFFICIENT_DATA, "strength": 0.0}

        # Use last 10 trials or half of all trials, whichever is smaller
        recent_window = min(10, len(values) // 2)
        recent_values = values[-recent_window:]
        older_values = (
            values[-2 * recent_window : -recent_window]
            if len(values) >= 2 * recent_window
            else values[:-recent_window]
        )

        if not older_values:
            return {"trend": TrendType.INSUFFICIENT_DATA, "strength": 0.0}

        recent_avg = np.mean(recent_values)
        older_avg = np.mean(older_values)

        # Calculate improvement based on direction
        if direction == optuna.study.StudyDirection.MINIMIZE:
            improvement_ratio = (older_avg - recent_avg) / max(abs(older_avg), 1e-8)
        else:  # MAXIMIZE
            improvement_ratio = (recent_avg - older_avg) / max(abs(older_avg), 1e-8)

        # Determine trend
        if improvement_ratio > 0.02:  # 2% improvement
            trend = TrendType.IMPROVING
            strength = min(improvement_ratio * 5, 1.0)  # Scale to [0,1]
        elif improvement_ratio < -0.02:  # 2% degradation
            trend = TrendType.DEGRADING
            strength = min(abs(improvement_ratio) * 5, 1.0)
        else:
            trend = TrendType.PLATEAUING
            strength = 1.0 - min(abs(improvement_ratio) * 25, 1.0)  # Inverse for plateauing

        return {"trend": trend, "strength": strength, "improvement_ratio": improvement_ratio}

    def _calculate_trials_since_improvement(
        self, values: List[float], direction: optuna.study.StudyDirection
    ) -> int:
        """Calculate number of trials since last improvement."""
        if not values:
            return 0

        if direction == optuna.study.StudyDirection.MINIMIZE:
            best_value = min(values)
        else:
            best_value = max(values)

        # Find most recent occurrence of best value
        for i in range(len(values) - 1, -1, -1):
            if values[i] == best_value:
                return len(values) - 1 - i

        return len(values)

    def _assess_convergence(self, values: List[float]) -> float:
        """Assess convergence based on value stability."""
        if len(values) < 10:
            return 0.0

        # Look at variance in recent values
        recent_values = values[-10:]
        variance = np.var(recent_values)
        mean_value = np.mean(recent_values)

        # Coefficient of variation (CV)
        if abs(mean_value) > 1e-8:
            cv = np.sqrt(variance) / abs(mean_value)
        else:
            cv = float("inf")

        # Convert CV to convergence indicator (lower CV = higher convergence)
        if cv < 0.01:  # Very stable
            return 0.9
        elif cv < 0.05:  # Moderately stable
            return 0.7
        elif cv < 0.1:  # Somewhat stable
            return 0.4
        else:  # Unstable
            return 0.1

    def _generate_recommendation(
        self,
        stage: OptimizationStage,
        trend_analysis: Dict[str, Any],
        trials_since_improvement: int,
    ) -> str:
        """Generate strategic recommendation based on progress analysis."""

        trend = trend_analysis["trend"]
        strength = trend_analysis["strength"]

        if stage == OptimizationStage.EARLY_EXPLORATION:
            return "Continue broad exploration of parameter space to understand the landscape"

        elif stage == OptimizationStage.ACTIVE_SEARCH:
            if trend == TrendType.IMPROVING:
                return (
                    "Trend is positive - continue current exploration strategy with slight focus"
                )
            elif trials_since_improvement > 10:
                return "Try more aggressive exploration in unexplored parameter regions"
            else:
                return "Maintain balanced exploration and exploitation approach"

        elif stage == OptimizationStage.FOCUSED_OPTIMIZATION:
            if trend == TrendType.PLATEAUING and trials_since_improvement > 15:
                return "Consider refocusing search around top performing configurations"
            elif trend == TrendType.IMPROVING:
                return "Focus optimization around current promising regions"
            else:
                return "Try more aggressive parameter variations to escape local optima"

        else:  # REFINEMENT
            if trials_since_improvement > 20:
                return "Consider fine-tuning parameters around best known configurations"
            else:
                return "Make small refinements to parameters around best regions"

    def _extract_problem_context(self, study: optuna.Study) -> Dict[str, Any]:
        """Extract problem-specific context from study attributes."""

        problem_type = study.user_attrs.get("problem_type", "unknown")
        problem_description = study.user_attrs.get(
            "problem_description", "No description provided"
        )
        constraints = study.user_attrs.get("constraints", [])
        domain_knowledge = study.user_attrs.get("domain_knowledge", {})

        # Add default domain knowledge based on problem type
        if problem_type == "neural_network_training" and not domain_knowledge:
            domain_knowledge = self._get_neural_network_knowledge()

        return {
            "problem_type": problem_type,
            "problem_description": problem_description,
            "constraints": constraints,
            "domain_knowledge": domain_knowledge,
        }

    def _get_neural_network_knowledge(self) -> Dict[str, Any]:
        """Get default neural network domain knowledge."""
        return {
            "parameter_relationships": {
                "learning_rate_batch_size": "Larger batch sizes often need higher learning rates",
                "dropout_capacity": "Higher dropout rates for more complex models",
                "weight_decay_lr": "Higher learning rates may need stronger regularization",
            },
            "typical_ranges": {
                "learning_rate": {
                    "min": 1e-5,
                    "max": 1e-1,
                    "log_scale": True,
                    "sweet_spot": [1e-4, 1e-2],
                },
                "batch_size": {"min": 8, "max": 512, "powers_of_2": True, "sweet_spot": [16, 128]},
                "dropout": {"min": 0.0, "max": 0.7, "sweet_spot": [0.1, 0.5]},
                "weight_decay": {
                    "min": 1e-6,
                    "max": 1e-2,
                    "log_scale": True,
                    "sweet_spot": [1e-5, 1e-3],
                },
            },
            "best_practices": [
                "Start with learning rates around 1e-3 for Adam optimizer",
                "Use batch sizes that are powers of 2 for computational efficiency",
                "Higher dropout for overfitting prevention on complex datasets",
                "Lower learning rates for fine-tuning pre-trained models",
            ],
        }
