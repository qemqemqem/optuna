# Mathematical Foundations for LLM-Guided Optimization

## Problem Formulation

We seek to combine two probability distributions over the hyperparameter space:
- **p_optuna(x)**: Traditional optimization-derived distribution (TPE, GP, etc.)
- **p_llm(x)**: LLM-derived distribution expressing domain knowledge

**Goal**: Create p_combined(x) that leverages both information sources effectively.

## Distribution Combination Methods

### Method 1: Linear Combination (Mixture Model)
```
p_combined(x) = α * p_llm(x) + (1-α) * p_optuna(x)
```

**Properties**:
- Always valid probability distribution if components are
- α ∈ [0,1] controls mixing weight
- Convex combination preserves convexity
- Simple to implement and understand

**Use Case**: Safe combination with guaranteed valid distribution

### Method 2: Multiplicative Combination (Bayesian Product)
```
p_combined(x) ∝ p_llm(x) * p_optuna(x)
p_combined(x) = p_llm(x) * p_optuna(x) / Z
where Z = ∫ p_llm(x) * p_optuna(x) dx
```

**Interpretation**: Treating distributions as independent information sources
**Properties**:
- Concentrates probability where both distributions agree
- Information-theoretically optimal under independence
- Requires normalization constant computation

**Use Case**: When LLM and Optuna provide independent evidence

### Method 3: Geometric Mean (Recommended)
```
p_combined(x) ∝ p_llm(x)^α * p_optuna(x)^(1-α)
```

**Log-space form** (numerically stable):
```
log p_combined(x) = α * log p_llm(x) + (1-α) * log p_optuna(x) + constant
```

**Properties**:
- Interpolates between pure methods at extremes (α=0 gives pure Optuna, α=1 gives pure LLM)
- Smooth blending in log-space
- Geometrically interpretable as weighted geometric mean
- Numerically stable when computed in log-space

**Advantages**:
- More flexible than pure multiplicative
- Preserves relative preferences from both distributions
- Natural parameter for balancing influence

## Convergence Analysis

### Theorem: Preservation of Convergence
If p_optuna(x) converges to the true optimum distribution as n → ∞, then p_combined(x) converges to the same distribution under these conditions:

**For Linear Combination**: α → 0 as n → ∞
**For Multiplicative/Geometric**: p_llm(x*) > ε > 0 at the optimum x*

### Practical Convergence Strategy
Use **adaptive α** that decreases over time:
```python
α(n) = α_initial * exp(-decay_rate * n / N_total)
```

This ensures:
- Early trials benefit from LLM guidance
- Later trials rely more on accumulated Optuna knowledge
- Convergence guarantees are preserved

## The Zero Probability Problem

### Challenge
LLM-derived distributions often have zero probability in unexplored regions, which breaks multiplicative combination:
```
p_combined(x) ∝ p_llm(x) * p_optuna(x) = 0 * p_optuna(x) = 0
```

### Solution: ε-Smoothing
Always add uniform background distribution:
```python
p_llm_smoothed(x) = (1-ε) * p_llm_raw(x) + ε * uniform(x)
```

**Mathematical Properties**:
- Guarantees p_llm_smoothed(x) ≥ ε / volume(search_space) > 0
- Preserves relative preferences when ε is small
- Maintains convergence properties

**Recommended Values**:
- ε = 1e-5 for most applications
- ε = 1e-6 for high-precision requirements
- ε = 1e-4 for robust exploration

## Distribution Extraction from LLM

### Kernel Density Estimation Approach
Given LLM samples S = {x₁, x₂, ..., xₙ}:

```python
def llm_density_kde(x, samples, bandwidth='scott'):
    kde = GaussianKDE(samples, bandwidth=bandwidth)
    raw_density = kde.evaluate(x)
    
    # Apply ε-smoothing
    uniform_density = 1.0 / search_space_volume
    return (1-ε) * raw_density + ε * uniform_density
```

### Bandwidth Selection
**Scott's Rule** adapted for hyperparameter spaces:
```
h = n^(-1/(d+4)) * σ * volume_scale
```
Where:
- n = number of LLM samples
- d = dimensionality
- σ = characteristic scale of search space
- volume_scale = empirical scaling factor (typically 0.1-0.5)

### Alternative: Parametric Fitting
For robust density estimation with few samples:
```python
def llm_density_gmm(samples, n_components='auto'):
    if n_components == 'auto':
        n_components = min(len(samples)//2, 5)
    
    gmm = GaussianMixture(
        n_components=n_components,
        reg_covar=1e-4  # Prevents singular covariance
    )
    gmm.fit(samples)
    
    # Mix with uniform background
    raw_density = np.exp(gmm.score_samples(x))
    return (1-ε) * raw_density + ε * uniform_density
```

## Practical Implementation

### Working in Log-Space
Always compute in log-space to avoid numerical underflow:
```python
def combined_log_density(x, alpha=0.3, epsilon=1e-5):
    log_p_llm = llm_distribution.log_density(x)
    log_p_optuna = optuna_model.log_density(x)
    
    # Add ε-smoothing in log space
    log_uniform = -np.log(search_space_volume)
    log_p_llm_smooth = np.logaddexp(
        np.log(1-epsilon) + log_p_llm,
        np.log(epsilon) + log_uniform
    )
    
    # Geometric mean combination
    return alpha * log_p_llm_smooth + (1-alpha) * log_p_optuna
```

### Sampling Strategy
For complex distributions, use MCMC or importance sampling:
```python
def sample_combined_distribution(n_samples=1):
    # Option 1: Importance sampling from Optuna distribution
    x_candidates = optuna_model.sample(n_samples * 10)
    weights = np.exp(alpha * llm_log_density(x_candidates))
    indices = np.random.choice(len(x_candidates), n_samples, p=weights/weights.sum())
    return x_candidates[indices]
    
    # Option 2: MCMC sampling (for complex cases)
    return mcmc_sample(combined_log_density, n_samples)
```

## Key Mathematical Insights

1. **Information Fusion**: Geometric mean provides principled way to combine independent information sources
2. **Numerical Stability**: Log-space computation prevents underflow in high-dimensional spaces
3. **Convergence Preservation**: Adaptive α ensures long-term convergence to true optimum
4. **Robustness**: ε-smoothing handles sparse LLM sampling gracefully
5. **Flexibility**: α parameter allows tuning exploration vs exploitation balance

## Validation Strategies

### Consistency Checks
```python
def validate_combined_distribution(p_combined, search_space):
    # Test 1: Probability mass sums to 1
    total_mass = integrate_over_space(p_combined, search_space)
    assert abs(total_mass - 1.0) < 1e-3
    
    # Test 2: No negative probabilities
    test_points = sample_search_space(1000)
    densities = p_combined.density(test_points)
    assert np.all(densities >= 0)
    
    # Test 3: Reasonable entropy
    entropy = compute_entropy(p_combined, search_space)
    assert entropy > 0  # Not degenerate
```

### Performance Metrics
- **KL Divergence**: Measure difference from baseline distributions
- **Sample Efficiency**: Trials needed to reach performance thresholds
- **Convergence Rate**: Speed of optimization improvement
- **Robustness**: Performance across different problem types

This mathematical foundation ensures our LLM-guided optimization is theoretically sound while remaining practically implementable.