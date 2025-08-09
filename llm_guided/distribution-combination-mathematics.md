# Mathematical Approaches to Combining LLM and Optuna Distributions

## The Core Problem

We have two probability distributions over the hyperparameter space:
- **p_optuna(x)**: Optuna's learned distribution (from TPE, GP, etc.)
- **p_llm(x)**: LLM's implied distribution over promising hyperparameters

Goal: Combine them into p_combined(x) that leverages both sources of information.

## Method 1: Linear Combination (Mixture Model)

**Formula:**
```
p_combined(x) = α * p_llm(x) + (1-α) * p_optuna(x)
```

**Properties:**
- Always a valid probability distribution (if both components are)
- α ∈ [0,1] controls the mixing weight
- Convex combination preserves convexity
- Simple to implement and understand

**Pros:**
- Mathematically guaranteed to be a proper distribution
- Easy parameter tuning with α
- Preserves support of both distributions

**Cons:**
- Doesn't capture interactions between distributions
- Can lead to multimodal distributions even if components are unimodal
- May not be the most information-theoretically principled

## Method 2: Multiplicative Combination (Bayesian Product)

**Formula:**
```
p_combined(x) ∝ p_llm(x) * p_optuna(x)
p_combined(x) = p_llm(x) * p_optuna(x) / Z
where Z = ∫ p_llm(x) * p_optuna(x) dx
```

**Interpretation:** Treating p_llm(x) as a prior and p_optuna(x) as likelihood, or vice versa.

**Properties:**
- Concentrates probability where both distributions agree
- Naturally handles disagreement by reducing probability
- Information-theoretically optimal under independence assumption

**Pros:**
- Bayesian interpretation: combines independent sources of information
- Automatically focuses on regions of mutual agreement
- Theoretically principled

**Cons:**
- Requires numerical integration for normalization
- Can be overly concentrated if distributions are narrow
- Zero probability where either distribution is zero

## Method 3: Geometric Mean (Log-Linear Combination)

**Formula:**
```
p_combined(x) ∝ p_llm(x)^α * p_optuna(x)^(1-α)
```

**Equivalent log form:**
```
log p_combined(x) = α * log p_llm(x) + (1-α) * log p_optuna(x) + constant
```

**Properties:**
- Interpolates between multiplicative (α=1) and original Optuna (α=0)
- Smooth blending in log-space
- Geometric interpretation: weighted geometric mean

**Pros:**
- More flexible than pure multiplicative approach
- Reduces to standard methods at extremes
- Often numerically stable in log-space

**Cons:**
- Still requires normalization
- Can be dominated by whichever distribution has larger values

## Method 4: Maximum Entropy Combination

**Problem Setup:**
Find p_combined(x) that maximizes entropy subject to:
- Moment constraints from p_llm(x): E_combined[f_i(x)] = E_llm[f_i(x)]
- Moment constraints from p_optuna(x): E_combined[g_j(x)] = E_optuna[g_j(x)]
- Normalization: ∫ p_combined(x) dx = 1

**Solution Form:**
```
p_combined(x) ∝ exp(∑_i λ_i f_i(x) + ∑_j μ_j g_j(x))
```

**Properties:**
- Information-theoretically optimal: least biased given constraints
- Flexible constraint incorporation
- Principled way to combine different types of information

**Pros:**
- Theoretically elegant and principled
- Can incorporate arbitrary moment constraints
- Reduces bias beyond available information

**Cons:**
- Computationally expensive
- Requires choosing appropriate moment constraints
- Complex implementation

## Method 5: Importance Sampling Approach

**Concept:** Sample from one distribution, weight by the other.

**Algorithm:**
1. Sample x_i ~ p_optuna(x)
2. Weight each sample by w_i = p_llm(x_i) / p_optuna(x_i)
3. Resample according to weights

**Properties:**
- Avoids explicit distribution combination
- Maintains computational tractability
- Natural way to incorporate LLM guidance

**Pros:**
- No normalization constants needed
- Easy to implement
- Can handle arbitrary distributions

**Cons:**
- Variance can be high if distributions are very different
- Requires good overlap between distributions
- Not a true distribution combination

## Recommended Approach: Adaptive Geometric Mean

For practical implementation, I recommend the **geometric mean approach with adaptive weighting**:

```python
class AdaptiveGeometricCombination:
    def __init__(self, initial_alpha=0.3, adaptation_rate=0.1):
        self.alpha = initial_alpha
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
    
    def combined_log_density(self, x, log_p_llm, log_p_optuna):
        return self.alpha * log_p_llm + (1 - self.alpha) * log_p_optuna
    
    def update_alpha(self, llm_suggestion_performance):
        # Adapt α based on how well LLM suggestions perform
        if llm_suggestion_performance > self.baseline_performance:
            self.alpha = min(0.8, self.alpha + self.adaptation_rate)
        else:
            self.alpha = max(0.1, self.alpha - self.adaptation_rate)
```

## Mathematical Analysis: Convergence Properties

### Theorem 1: Preservation of Convergence
If p_optuna(x) converges to the true optimum distribution as n → ∞, then:
- Linear combination preserves convergence if α → 0 as n → ∞
- Multiplicative combination preserves convergence if p_llm has support at the optimum
- Geometric mean preserves convergence under similar conditions

### Theorem 2: Bias-Variance Trade-off
The combined distribution exhibits:
- **Bias**: Introduced by p_llm if it's systematically wrong
- **Variance**: Reduced compared to pure Optuna if p_llm provides useful information
- **Optimal α**: Minimizes MSE = bias² + variance

### Practical Convergence Guarantee
For the geometric mean approach:
```
If p_optuna(x) → δ(x*) and p_llm(x*) > ε > 0, 
then p_combined(x) → δ(x*) as long as α → 0 sufficiently slowly
```

## Implementation Strategy

### Step 1: Extract LLM Distribution
Convert LLM suggestions into a density:
```python
def build_llm_density(llm_suggestions, search_space):
    # Fit kernel density estimate to LLM suggestions
    kde = GaussianKDE(llm_suggestions)
    return lambda x: kde.evaluate(x)
```

### Step 2: Combine with Optuna
```python
def sample_combined_distribution(n_samples):
    # Work in log space for numerical stability
    log_p_llm = self.llm_density.log_density(x)
    log_p_optuna = self.optuna_model.log_density(x)
    log_p_combined = self.alpha * log_p_llm + (1-self.alpha) * log_p_optuna
    
    # Sample using MCMC or rejection sampling
    return self.sample_from_log_density(log_p_combined, n_samples)
```

### Step 3: Adaptive Weighting
Monitor performance and adjust α:
- Increase α when LLM suggestions outperform random
- Decrease α when LLM suggestions underperform
- Track convergence metrics to ensure we don't hurt optimization

## Key Mathematical Insights

1. **Information Fusion**: Multiplicative combination is optimal when sources are independent
2. **Risk Management**: Linear combination provides safeguards against bad LLM suggestions
3. **Adaptivity**: Dynamic α allows learning which source is more reliable
4. **Convergence**: Proper asymptotic behavior requires α → 0 eventually

The geometric mean approach strikes the best balance: principled combination with practical implementation and theoretical guarantees.