# LLM-Guided Optuna: Project Overview

## Vision

Enhance Optuna's hyperparameter optimization by integrating Large Language Models (LLMs) to suggest promising hyperparameter configurations. The goal is to leverage LLMs' domain knowledge and pattern recognition capabilities to accelerate convergence and discover better solutions.

## Core Concept

Instead of relying purely on mathematical optimization algorithms, we combine:
- **Optuna's systematic approach** with proven convergence properties
- **LLM domain knowledge** from vast ML literature and successful configurations
- **Fresh contextual awareness** of the current optimization state

## Key Innovation

**Trial-Level LLM Guidance**: LLMs suggest complete hyperparameter configurations (not individual parameters) by reasoning about parameter relationships and optimization history.

## Project Scope

### In Scope
- Single-objective optimization
- Complete trial configuration generation
- Fresh context reconstruction per trial
- Structured LLM output parsing with validation
- Drop-in Optuna sampler replacement

### Out of Scope (For MVP)
- Multi-objective optimization
- Fallback to traditional samplers
- Complex parameter dependencies
- Advanced acquisition function modification

## Technical Architecture

```
User Code
    ↓
Optuna Study (with LLMGuidedSampler)
    ↓
LLM Query (LiteLLM + Pydantic)
    ↓
Structured Response Parsing
    ↓
Parameter Validation & Clamping
    ↓
Trial Execution
```

## Success Metrics

1. **Faster Convergence**: Reduce trials needed to reach good performance
2. **Better Solutions**: Find superior hyperparameter configurations
3. **Sample Efficiency**: Outperform random and traditional samplers
4. **Reliability**: Maintain consistent performance across problem types

## Design Documents Structure

- `01_mathematical_foundations.md` - Distribution combination theory
- `02_llm_integration_strategies.md` - LLM interaction patterns and best practices
- `03_implementation_architecture.md` - Detailed Optuna integration design
- `04_structured_output_design.md` - Pydantic models and parsing strategy
- `05_context_building_strategy.md` - How to build LLM context from study state

## Key Decisions Made

1. **Trial-Level Generation**: Generate complete configurations, not individual parameters
2. **Every Trial**: Call LLM on every single trial (no mixing with traditional samplers)
3. **Fresh Context**: Rebuild context from study state each time (no persistent state)
4. **No Fallback**: Pure LLM-guided approach to focus on the core innovation
5. **Single Objective**: Focus on single-objective optimization initially
6. **Structured Output**: Use LiteLLM + Pydantic for reliable parsing

## Implementation Phases

### Phase 1: Core Implementation
- Basic LLMGuidedSampler
- Context building from study history
- LiteLLM integration with Pydantic models
- Parameter validation and clamping

### Phase 2: Robustness & Polish
- Error handling and retry logic
- Performance optimization and caching
- Comprehensive testing across problem types
- Documentation and examples

### Phase 3: Advanced Features
- Adaptive prompting strategies
- Cost optimization techniques
- Integration with popular ML frameworks
- Performance benchmarking suite

## Dependencies

- **Core**: `optuna`, `litellm`, `pydantic`
- **Optional**: `instructor` (for enhanced structured outputs)
- **Development**: `pytest`, `numpy`, `scikit-learn` (for testing)

## Risks & Mitigation

1. **LLM API Reliability**: Robust error handling, retry logic
2. **Cost Management**: Caching, efficient prompting
3. **Quality Consistency**: Structured outputs, validation
4. **Performance**: Benchmarking against baselines

## Next Steps

1. Complete detailed design documents
2. Implement core LLMGuidedSampler
3. Create test suite with synthetic objectives
4. Benchmark against standard Optuna samplers
5. Iterate on prompt engineering and context building