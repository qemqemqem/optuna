# Context Building Strategy

## Overview

This document details how we extract and format information from Optuna studies to create rich, actionable context for LLM queries. The context building strategy is critical for LLM performance - providing too little information results in generic suggestions, while too much information exceeds token limits and confuses the model.

## Context Architecture

### Information Hierarchy

```
Primary Context (Always Included)
├── Problem Definition
│   ├── Objective name and direction
│   ├── Problem type and description  
│   └── Search space specification
├── Current State
│   ├── Trial completion count
│   ├── Best result so far
│   └── Optimization stage
└── Recent History
    ├── Last 5-10 trial results
    ├── Performance trends
    └── Parameter patterns

Secondary Context (Conditional)
├── Extended History (if many trials)
├── Parameter Correlations (if sufficient data)
├── Domain Knowledge (user-provided)
└── Constraints and Requirements
```

## Core Context Components

### 1. Problem Definition Context

```python
class ProblemDefinitionBuilder:
    """Builds problem definition context from study metadata."""
    
    def build_problem_context(self, study: optuna.Study) -> Dict[str, Any]:
        """Extract and format problem definition information."""
        
        return {
            'objective': {
                'name': study.study_name or 'optimization_objective',
                'direction': study.direction.name,  # 'MINIMIZE' or 'MAXIMIZE'
                'description': study.user_attrs.get('problem_description', 'No description provided')
            },
            'problem_type': study.user_attrs.get('problem_type', 'general_optimization'),
            'domain': study.user_attrs.get('domain', 'unknown'),
            'search_space': self._format_search_space(study.search_space)
        }
    
    def _format_search_space(self, search_space: Dict) -> List[Dict]:
        """Convert Optuna search space to LLM-friendly format."""
        
        formatted_params = []
        
        for param_name, distribution in search_space.items():
            param_info = {
                'name': param_name,
                'type': self._get_distribution_type(distribution),
                'description': self._get_parameter_description(param_name, distribution)
            }
            
            if isinstance(distribution, optuna.distributions.FloatDistribution):
                param_info.update({
                    'range': [distribution.low, distribution.high],
                    'log_scale': distribution.log,
                    'step': distribution.step
                })
                
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                param_info.update({
                    'range': [distribution.low, distribution.high], 
                    'step': distribution.step
                })
                
            elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                param_info.update({
                    'choices': list(distribution.choices)
                })
            
            formatted_params.append(param_info)
        
        return formatted_params
    
    def _get_parameter_description(self, param_name: str, distribution) -> str:
        """Generate helpful parameter descriptions."""
        
        # Common parameter descriptions
        descriptions = {
            'learning_rate': 'Controls the step size in gradient descent optimization',
            'batch_size': 'Number of samples processed before model weights update',
            'dropout': 'Fraction of neurons randomly set to zero during training',
            'weight_decay': 'L2 regularization strength to prevent overfitting',
            'momentum': 'Momentum factor for gradient descent optimization',
            'epochs': 'Number of complete passes through the training dataset',
            'hidden_size': 'Number of neurons in hidden layers',
            'num_layers': 'Number of layers in the neural network'
        }
        
        if param_name.lower() in descriptions:
            return descriptions[param_name.lower()]
        
        # Generate generic description based on type
        if isinstance(distribution, optuna.distributions.FloatDistribution):
            if distribution.log:
                return f"Floating point parameter (log scale) ranging from {distribution.low} to {distribution.high}"
            else:
                return f"Floating point parameter ranging from {distribution.low} to {distribution.high}"
        elif isinstance(distribution, optuna.distributions.IntDistribution):
            return f"Integer parameter ranging from {distribution.low} to {distribution.high}"
        elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
            return f"Categorical parameter with choices: {list(distribution.choices)}"
        
        return "Parameter with unspecified constraints"
```

### 2. Historical Context Builder

```python
class HistoricalContextBuilder:
    """Builds context from trial history with intelligent filtering."""
    
    def __init__(self, max_recent_trials: int = 10, max_best_trials: int = 5):
        self.max_recent_trials = max_recent_trials
        self.max_best_trials = max_best_trials
    
    def build_historical_context(self, study: optuna.Study) -> Dict[str, Any]:
        """Build comprehensive historical context."""
        
        completed_trials = [
            trial for trial in study.trials 
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        
        if not completed_trials:
            return {'stage': 'initialization', 'message': 'No completed trials yet'}
        
        context = {
            'trial_summary': self._build_trial_summary(completed_trials),
            'best_results': self._build_best_results_context(study, completed_trials),
            'recent_trials': self._build_recent_trials_context(completed_trials),
            'progress_analysis': self._analyze_optimization_progress(completed_trials),
            'parameter_insights': self._analyze_parameter_patterns(completed_trials)
        }
        
        return context
    
    def _build_trial_summary(self, completed_trials: List[optuna.Trial]) -> Dict[str, Any]:
        """Build high-level trial summary statistics."""
        
        values = [trial.value for trial in completed_trials]
        
        return {
            'total_completed': len(completed_trials),
            'value_statistics': {
                'best': min(values),  # Assuming minimization
                'worst': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values)
            },
            'completion_rate': len(completed_trials) / len(completed_trials)  # Will be enhanced with failed trials
        }
    
    def _build_best_results_context(self, 
                                   study: optuna.Study, 
                                   completed_trials: List[optuna.Trial]) -> Dict[str, Any]:
        """Format best trial results for LLM context."""
        
        if not study.best_trial:
            return {'message': 'No best trial available yet'}
        
        best_trial = study.best_trial
        
        # Get top N trials for pattern analysis
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        top_trials = sorted_trials[:self.max_best_trials]
        
        return {
            'best_trial': {
                'value': best_trial.value,
                'parameters': best_trial.params,
                'trial_number': best_trial.number,
                'duration': best_trial.duration.total_seconds() if best_trial.duration else None
            },
            'top_trials': [
                {
                    'value': trial.value,
                    'parameters': trial.params,
                    'trial_number': trial.number
                }
                for trial in top_trials
            ],
            'best_parameter_ranges': self._analyze_best_parameter_ranges(top_trials)
        }
    
    def _build_recent_trials_context(self, completed_trials: List[optuna.Trial]) -> List[Dict]:
        """Format recent trials for LLM context."""
        
        recent_trials = completed_trials[-self.max_recent_trials:]
        
        formatted_trials = []
        for trial in recent_trials:
            formatted_trials.append({
                'trial_number': trial.number,
                'value': trial.value,
                'parameters': trial.params,
                'duration': trial.duration.total_seconds() if trial.duration else None,
                'relative_performance': self._calculate_relative_performance(trial, completed_trials)
            })
        
        return formatted_trials
    
    def _analyze_optimization_progress(self, completed_trials: List[optuna.Trial]) -> Dict[str, Any]:
        """Analyze optimization progress and trends."""
        
        if len(completed_trials) < 3:
            return {
                'stage': 'early_exploration',
                'trend': 'insufficient_data',
                'recommendation': 'Continue exploring the parameter space'
            }
        
        values = [trial.value for trial in completed_trials]
        
        # Analyze trends
        recent_window = min(10, len(values) // 3)
        recent_values = values[-recent_window:]
        older_values = values[-2*recent_window:-recent_window] if len(values) >= 2*recent_window else values[:-recent_window]
        
        trend_analysis = self._determine_trend(recent_values, older_values)
        stage = self._determine_optimization_stage(completed_trials)
        
        return {
            'stage': stage,
            'trend': trend_analysis['trend'],
            'trend_strength': trend_analysis['strength'],
            'trials_since_improvement': self._trials_since_improvement(completed_trials),
            'convergence_indicator': self._assess_convergence(values),
            'recommendation': self._generate_stage_recommendation(stage, trend_analysis)
        }
    
    def _determine_trend(self, recent_values: List[float], older_values: List[float]) -> Dict[str, Any]:
        """Determine optimization trend from value sequences."""
        
        if not older_values:
            return {'trend': 'insufficient_data', 'strength': 0.0}
        
        recent_avg = np.mean(recent_values)
        older_avg = np.mean(older_values)
        
        improvement_ratio = (older_avg - recent_avg) / abs(older_avg)  # Assuming minimization
        
        if improvement_ratio > 0.05:  # 5% improvement
            trend = 'improving'
            strength = min(improvement_ratio, 1.0)
        elif improvement_ratio < -0.05:  # 5% degradation
            trend = 'degrading'
            strength = min(abs(improvement_ratio), 1.0)
        else:
            trend = 'plateauing'
            strength = 1.0 - abs(improvement_ratio) / 0.05
        
        return {
            'trend': trend,
            'strength': strength,
            'improvement_ratio': improvement_ratio
        }
    
    def _determine_optimization_stage(self, completed_trials: List[optuna.Trial]) -> str:
        """Determine current optimization stage."""
        
        n_trials = len(completed_trials)
        
        if n_trials < 10:
            return 'early_exploration'
        elif n_trials < 50:
            return 'active_search'
        elif n_trials < 100:
            return 'focused_optimization'
        else:
            return 'refinement'
    
    def _analyze_parameter_patterns(self, completed_trials: List[optuna.Trial]) -> Dict[str, Any]:
        """Analyze patterns in successful parameter choices."""
        
        if len(completed_trials) < 5:
            return {'message': 'Insufficient trials for parameter pattern analysis'}
        
        # Get top 25% of trials
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        top_quarter = sorted_trials[:max(1, len(sorted_trials) // 4)]
        
        parameter_analysis = {}
        
        # Analyze each parameter
        all_param_names = set()
        for trial in completed_trials:
            all_param_names.update(trial.params.keys())
        
        for param_name in all_param_names:
            param_analysis = self._analyze_single_parameter(
                param_name, top_quarter, completed_trials
            )
            parameter_analysis[param_name] = param_analysis
        
        return {
            'successful_parameter_ranges': parameter_analysis,
            'parameter_correlations': self._calculate_parameter_correlations(completed_trials),
            'insights': self._generate_parameter_insights(parameter_analysis)
        }
    
    def _analyze_single_parameter(self, 
                                 param_name: str, 
                                 top_trials: List[optuna.Trial],
                                 all_trials: List[optuna.Trial]) -> Dict[str, Any]:
        """Analyze patterns for a single parameter."""
        
        # Extract parameter values from top trials
        top_values = [trial.params.get(param_name) for trial in top_trials if param_name in trial.params]
        all_values = [trial.params.get(param_name) for trial in all_trials if param_name in trial.params]
        
        if not top_values:
            return {'message': f'No data for parameter {param_name}'}
        
        # Determine parameter type
        sample_value = top_values[0]
        
        if isinstance(sample_value, (int, float)):
            return self._analyze_numeric_parameter(param_name, top_values, all_values)
        else:
            return self._analyze_categorical_parameter(param_name, top_values, all_values)
    
    def _analyze_numeric_parameter(self, 
                                  param_name: str, 
                                  top_values: List[float], 
                                  all_values: List[float]) -> Dict[str, Any]:
        """Analyze numeric parameter patterns."""
        
        return {
            'type': 'numeric',
            'successful_range': {
                'min': min(top_values),
                'max': max(top_values),
                'mean': np.mean(top_values),
                'std': np.std(top_values),
                'median': np.median(top_values)
            },
            'overall_range': {
                'min': min(all_values),
                'max': max(all_values),
                'mean': np.mean(all_values),
                'std': np.std(all_values)
            },
            'concentration_factor': np.std(all_values) / max(np.std(top_values), 1e-8)  # How concentrated successful values are
        }
    
    def _analyze_categorical_parameter(self, 
                                     param_name: str, 
                                     top_values: List[str], 
                                     all_values: List[str]) -> Dict[str, Any]:
        """Analyze categorical parameter patterns."""
        
        from collections import Counter
        
        top_counts = Counter(top_values)
        all_counts = Counter(all_values)
        
        # Calculate success rates for each category
        success_rates = {}
        for category in all_counts.keys():
            success_rate = top_counts.get(category, 0) / all_counts[category]
            success_rates[category] = success_rate
        
        return {
            'type': 'categorical',
            'successful_choices': dict(top_counts),
            'success_rates': success_rates,
            'recommended_choice': max(success_rates.keys(), key=success_rates.get)
        }
```

### 3. Domain Knowledge Integration

```python
class DomainKnowledgeBuilder:
    """Integrates domain-specific knowledge and constraints."""
    
    def __init__(self):
        self.domain_templates = {
            'neural_network_training': self._neural_network_knowledge,
            'automl': self._automl_knowledge,
            'hyperparameter_tuning': self._general_ml_knowledge,
            'reinforcement_learning': self._rl_knowledge
        }
    
    def build_domain_context(self, study: optuna.Study) -> Dict[str, Any]:
        """Build domain-specific context."""
        
        problem_type = study.user_attrs.get('problem_type', 'general')
        domain = study.user_attrs.get('domain', 'unknown')
        
        context = {
            'problem_type': problem_type,
            'domain': domain,
            'domain_knowledge': self._get_domain_knowledge(problem_type),
            'constraints': study.user_attrs.get('constraints', []),
            'requirements': study.user_attrs.get('requirements', []),
            'additional_context': study.user_attrs.get('llm_context', {})
        }
        
        return context
    
    def _get_domain_knowledge(self, problem_type: str) -> Dict[str, Any]:
        """Get domain-specific knowledge and best practices."""
        
        if problem_type in self.domain_templates:
            return self.domain_templates[problem_type]()
        else:
            return self._general_optimization_knowledge()
    
    def _neural_network_knowledge(self) -> Dict[str, Any]:
        """Neural network training domain knowledge."""
        
        return {
            'parameter_relationships': {
                'learning_rate_batch_size': 'Larger batch sizes typically require higher learning rates',
                'dropout_model_size': 'Larger models often benefit from higher dropout rates',
                'weight_decay_learning_rate': 'Higher learning rates may need stronger weight decay'
            },
            'typical_ranges': {
                'learning_rate': {'range': [1e-5, 1e-1], 'log_scale': True, 'typical': [1e-4, 1e-2]},
                'batch_size': {'range': [8, 512], 'typical': [16, 128], 'powers_of_2': True},
                'dropout': {'range': [0.0, 0.7], 'typical': [0.1, 0.5]},
                'weight_decay': {'range': [1e-6, 1e-2], 'log_scale': True, 'typical': [1e-5, 1e-3]}
            },
            'best_practices': [
                'Start with learning rates around 1e-3 for Adam optimizer',
                'Use batch sizes that are powers of 2 for computational efficiency',
                'Higher dropout rates (0.3-0.5) for complex datasets',
                'Lower learning rates for fine-tuning pre-trained models'
            ],
            'common_patterns': {
                'exploration_phase': 'Try diverse learning rates and batch sizes first',
                'refinement_phase': 'Fine-tune regularization parameters (dropout, weight_decay)',
                'final_phase': 'Optimize learning rate schedules and momentum'
            }
        }
    
    def _general_optimization_knowledge(self) -> Dict[str, Any]:
        """General optimization domain knowledge."""
        
        return {
            'best_practices': [
                'Start with broad parameter ranges to understand the landscape',
                'Focus on the most impactful parameters first',
                'Use logarithmic scales for parameters spanning multiple orders of magnitude',
                'Consider parameter interactions and dependencies'
            ],
            'exploration_strategies': {
                'early_stage': 'Prioritize exploration over exploitation',
                'middle_stage': 'Balance exploration with local optimization',
                'late_stage': 'Focus on fine-tuning around best regions'
            }
        }
```

### 4. Context Formatting and Prompt Integration

```python
class ContextFormatter:
    """Formats context for LLM consumption with token management."""
    
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
    
    def format_context_for_llm(self, 
                              problem_context: Dict,
                              historical_context: Dict,
                              domain_context: Dict) -> str:
        """Format complete context into LLM prompt sections."""
        
        sections = []
        
        # Problem Definition Section
        sections.append(self._format_problem_section(problem_context))
        
        # Historical Context Section  
        if historical_context.get('trial_summary'):
            sections.append(self._format_historical_section(historical_context))
        
        # Domain Knowledge Section
        if domain_context.get('domain_knowledge'):
            sections.append(self._format_domain_section(domain_context))
        
        # Combine sections and manage token count
        full_context = '\n\n'.join(sections)
        
        # Truncate if necessary (implement intelligent truncation)
        if self._estimate_token_count(full_context) > self.max_tokens:
            full_context = self._truncate_context_intelligently(sections)
        
        return full_context
    
    def _format_problem_section(self, problem_context: Dict) -> str:
        """Format problem definition section."""
        
        lines = [
            "OPTIMIZATION PROBLEM:",
            f"- Objective: {problem_context['objective']['direction']} '{problem_context['objective']['name']}'",
            f"- Problem Type: {problem_context['problem_type']}",
            f"- Description: {problem_context['objective']['description']}",
            "",
            "SEARCH SPACE:"
        ]
        
        for param in problem_context['search_space']:
            param_line = f"- {param['name']} ({param['type']})"
            
            if param['type'] in ['float', 'int']:
                param_line += f": {param['range'][0]} to {param['range'][1]}"
                if param.get('log_scale'):
                    param_line += " (log scale)"
            elif param['type'] == 'categorical':
                param_line += f": {param['choices']}"
            
            if param.get('description'):
                param_line += f" - {param['description']}"
            
            lines.append(param_line)
        
        return '\n'.join(lines)
    
    def _format_historical_section(self, historical_context: Dict) -> str:
        """Format historical context section."""
        
        lines = ["OPTIMIZATION PROGRESS:"]
        
        # Summary stats
        summary = historical_context['trial_summary']
        lines.extend([
            f"- Completed trials: {summary['total_completed']}",
            f"- Best value: {summary['value_statistics']['best']:.6f}",
            f"- Current trend: {historical_context['progress_analysis']['trend']}",
            f"- Optimization stage: {historical_context['progress_analysis']['stage']}"
        ])
        
        # Best results
        if historical_context['best_results']['best_trial']:
            lines.append("\nBEST RESULT:")
            best = historical_context['best_results']['best_trial']
            lines.append(f"- Value: {best['value']:.6f}")
            lines.append(f"- Parameters: {json.dumps(best['parameters'], indent=2)}")
        
        # Recent trials (truncated)
        recent_trials = historical_context['recent_trials'][-5:]  # Last 5 only
        if recent_trials:
            lines.append("\nRECENT TRIALS:")
            for trial in recent_trials:
                lines.append(f"- Trial {trial['trial_number']}: {trial['value']:.6f} with {trial['parameters']}")
        
        return '\n'.join(lines)
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough token count estimation (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def _truncate_context_intelligently(self, sections: List[str]) -> str:
        """Intelligently truncate context to fit token limits."""
        
        # Priority order: Problem > Best Results > Recent Trials > Domain Knowledge
        essential_sections = [sections[0]]  # Always keep problem definition
        
        remaining_tokens = self.max_tokens - self._estimate_token_count(sections[0])
        
        # Add sections in priority order
        for section in sections[1:]:
            section_tokens = self._estimate_token_count(section)
            if remaining_tokens >= section_tokens:
                essential_sections.append(section)
                remaining_tokens -= section_tokens
            else:
                # Try to include a truncated version
                truncated_section = self._truncate_section(section, remaining_tokens)
                if truncated_section:
                    essential_sections.append(truncated_section)
                break
        
        return '\n\n'.join(essential_sections)
```

## Adaptive Context Strategies

### Context Adaptation Based on Optimization Stage

```python
class AdaptiveContextBuilder:
    """Adapts context based on optimization progress and performance."""
    
    def adapt_context_for_stage(self, 
                               base_context: Dict, 
                               optimization_stage: str,
                               performance_metrics: Dict) -> Dict:
        """Adapt context emphasis based on optimization stage."""
        
        if optimization_stage == 'early_exploration':
            return self._emphasize_exploration_context(base_context)
        elif optimization_stage == 'active_search':
            return self._emphasize_search_context(base_context)
        elif optimization_stage == 'focused_optimization':
            return self._emphasize_exploitation_context(base_context)
        elif optimization_stage == 'refinement':
            return self._emphasize_refinement_context(base_context)
        else:
            return base_context
    
    def _emphasize_exploration_context(self, context: Dict) -> Dict:
        """Emphasize exploration-relevant information."""
        
        context['strategy_guidance'] = {
            'primary_goal': 'broad_exploration',
            'focus': 'Explore diverse regions of parameter space',
            'avoid': 'Avoid clustering around any single region',
            'diversity_emphasis': True
        }
        
        return context
    
    def _emphasize_exploitation_context(self, context: Dict) -> Dict:
        """Emphasize exploitation of promising regions."""
        
        context['strategy_guidance'] = {
            'primary_goal': 'local_optimization',
            'focus': 'Refine parameters around best results',
            'pattern_matching': True,
            'best_trial_weight': 'high'
        }
        
        return context
```

This comprehensive context building strategy ensures that LLMs receive the most relevant, well-formatted information to make informed hyperparameter suggestions while respecting token limits and maintaining focus on the most impactful aspects of the optimization problem.