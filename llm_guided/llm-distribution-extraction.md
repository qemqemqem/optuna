# Extracting Probability Distributions from LLM Outputs

## The Core Challenges

1. **Distribution Extraction**: How do we get p_llm(x) from text/JSON output?
2. **Zero Probability Problem**: LLM samples don't cover entire space → zero densities everywhere else
3. **Sparse Sampling**: LLM gives us ~10 points, need continuous distribution over high-dimensional space

## Solution 1: Kernel Density Estimation with Regularization

### Basic KDE Approach
```python
from sklearn.neighbors import KernelDensity
import numpy as np

class LLMDistributionEstimator:
    def __init__(self, bandwidth='auto', kernel='gaussian', min_density=1e-6):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.min_density = min_density  # Critical: prevents zeros!
        
    def fit(self, llm_suggestions):
        # Fit KDE to LLM samples
        self.kde = KernelDensity(
            bandwidth=self.bandwidth, 
            kernel=self.kernel
        ).fit(llm_suggestions)
        
        # Add uniform background to prevent zeros
        self.uniform_weight = 0.01  # 1% uniform mixing
        
    def density(self, x):
        kde_density = np.exp(self.kde.score_samples(x))
        uniform_density = 1.0 / self.search_space_volume
        
        # Mix with uniform to ensure no zeros
        return (1 - self.uniform_weight) * kde_density + self.uniform_weight * uniform_density
```

**Key Innovation**: The uniform mixing prevents zero densities while preserving the LLM's preferences.

### Adaptive Bandwidth Selection
```python
def estimate_bandwidth(self, llm_suggestions, search_space):
    """Use cross-validation or rule-of-thumb for bandwidth"""
    from sklearn.model_selection import GridSearchCV
    
    # Rule of thumb: Scott's rule adapted for high dimensions
    n_samples, n_dims = llm_suggestions.shape
    scott_bandwidth = n_samples**(-1/(n_dims + 4))
    
    # Scale by search space characteristic length
    char_length = np.mean([param.high - param.low for param in search_space.values()])
    return scott_bandwidth * char_length * 0.1  # Conservative scaling
```

## Solution 2: Parametric Distribution Fitting

### Mixture of Gaussians Approach
```python
from sklearn.mixture import GaussianMixture

class ParametricLLMDistribution:
    def __init__(self, n_components='auto', regularization=1e-4):
        self.n_components = n_components
        self.regularization = regularization
    
    def fit(self, llm_suggestions):
        # Determine number of components
        if self.n_components == 'auto':
            self.n_components = min(len(llm_suggestions) // 2, 5)
        
        # Fit Gaussian mixture
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            reg_covar=self.regularization  # Prevents singular covariance
        ).fit(llm_suggestions)
        
        # Add uniform background component
        self.uniform_weight = 0.05
    
    def density(self, x):
        gmm_density = np.exp(self.gmm.score_samples(x))
        uniform_density = 1.0 / self.search_space_volume
        return (1 - self.uniform_weight) * gmm_density + self.uniform_weight * uniform_density
```

**Advantages**: 
- Smooth, analytical density function
- Automatic complexity control
- Handles multimodal suggestions naturally

## Solution 3: Neural Density Estimation

For high-dimensional spaces, use normalizing flows or neural density estimators:

```python
import torch
import torch.nn as nn

class NeuralLLMDensity(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive output
        )
        self.uniform_weight = 0.02
    
    def forward(self, x):
        neural_density = self.net(x).squeeze()
        uniform_density = torch.ones_like(neural_density) / self.search_space_volume
        return (1 - self.uniform_weight) * neural_density + self.uniform_weight * uniform_density
    
    def fit(self, llm_suggestions, n_epochs=1000):
        # Train to fit high density around LLM suggestions
        optimizer = torch.optim.Adam(self.parameters())
        
        for epoch in range(n_epochs):
            # Positive examples: LLM suggestions should have high density
            pos_loss = -torch.log(self(llm_suggestions)).mean()
            
            # Negative examples: Random points should have lower density
            random_points = self.sample_search_space(len(llm_suggestions) * 2)
            neg_loss = torch.log(self(random_points) + 1e-8).mean()
            
            loss = pos_loss + 0.1 * neg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Solution 4: Direct LLM Density Queries

### Probability Elicitation from LLM
Instead of sampling, directly ask LLM for probability assessments:

```python
def query_llm_density(self, hyperparameter_configs, context):
    prompt = f"""
    Given the optimization context: {context}
    
    Rate the probability (0-100) that each configuration will perform well:
    {hyperparameter_configs}
    
    Consider:
    - Theoretical soundness of parameter combinations
    - Typical ranges for this problem type
    - Relationships between parameters
    
    Return JSON: {{"config_1": probability, "config_2": probability, ...}}
    """
    
    response = self.llm_client.query(prompt)
    probabilities = parse_json(response)
    return self.normalize_probabilities(probabilities)

class DirectLLMDensity:
    def __init__(self, cache_size=10000):
        self.cache = {}
        self.cache_size = cache_size
        
    def density(self, x):
        # Check cache first
        x_key = tuple(x.round(3))  # Round for cache efficiency
        if x_key in self.cache:
            return self.cache[x_key]
        
        # Query LLM for this specific point
        probability = self.query_llm_density([x], self.context)[0]
        
        # Add uniform background
        density = 0.95 * probability + 0.05 * self.uniform_density
        
        # Cache result
        if len(self.cache) < self.cache_size:
            self.cache[x_key] = density
            
        return density
```

## Handling the Zero Probability Problem

### Mathematical Solution: ε-Smoothing

The fundamental insight is that we never want truly zero probabilities. Here's the mathematical framework:

```python
class SmoothedLLMDensity:
    def __init__(self, epsilon=1e-6, smoothing_method='uniform'):
        self.epsilon = epsilon
        self.smoothing_method = smoothing_method
    
    def smooth_density(self, raw_density, search_space):
        if self.smoothing_method == 'uniform':
            # Add uniform background
            uniform_density = 1.0 / self.search_space_volume
            return (1 - self.epsilon) * raw_density + self.epsilon * uniform_density
            
        elif self.smoothing_method == 'gaussian':
            # Convolve with tiny Gaussian
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(raw_density, sigma=0.01)
            return np.maximum(smoothed, self.epsilon)
            
        elif self.smoothing_method == 'exponential':
            # Exponential decay from LLM suggestions
            return raw_density + self.epsilon * np.exp(-distance_to_nearest_llm_sample)
    
    def log_density(self, x):
        # Work in log space to avoid numerical issues
        density = self.smooth_density(self.raw_density(x))
        return np.log(np.maximum(density, self.epsilon))
```

### Theoretical Justification

The smoothing is mathematically justified:
1. **Regularization**: Prevents overfitting to sparse LLM samples
2. **Robustness**: Small ε doesn't affect optimization significantly
3. **Convergence**: As long as ε → 0 slower than sample density → 0, we maintain convergence

## Recommended Implementation Strategy

### Phase 1: Robust KDE with Smoothing
```python
class ProductionLLMDensity:
    def __init__(self):
        self.method = 'adaptive_kde'  # Fall back to GMM if needed
        self.epsilon = 1e-5
        self.min_samples = 5
    
    def fit(self, llm_suggestions, search_space):
        if len(llm_suggestions) < self.min_samples:
            # Too few samples: use uniform with slight bias toward suggestions
            self.method = 'sparse_uniform'
            self.suggestions = llm_suggestions
        else:
            # Enough samples: use KDE
            self.kde_estimator = LLMDistributionEstimator(
                min_density=self.epsilon
            )
            self.kde_estimator.fit(llm_suggestions)
    
    def density(self, x):
        if self.method == 'sparse_uniform':
            # Uniform with exponential bias toward suggestions
            distances = [np.linalg.norm(x - s) for s in self.suggestions]
            min_distance = min(distances)
            bias_factor = np.exp(-min_distance)  # Closer = higher probability
            return self.epsilon * (1 + 10 * bias_factor)
        else:
            return self.kde_estimator.density(x)
    
    def log_density(self, x):
        return np.log(self.density(x))
```

### Key Insights:

1. **Never allow true zeros**: Always mix with uniform background
2. **Adaptive complexity**: Use simpler methods when data is sparse
3. **Cache expensive operations**: LLM queries and density evaluations
4. **Work in log-space**: Prevents numerical underflow in combination
5. **Validate everything**: Check that densities integrate to reasonable values

The smoothed KDE approach gives you a principled way to extract continuous distributions from discrete LLM samples while mathematically guaranteeing no zero probabilities that would break the geometric mean combination.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Analyze methods for extracting distributions from LLM outputs", "status": "completed", "priority": "high"}, {"id": "2", "content": "Address zero probability problem in distribution combination", "status": "completed", "priority": "high"}, {"id": "3", "content": "Design robust distribution estimation from sparse samples", "status": "completed", "priority": "high"}, {"id": "4", "content": "Consider alternative LLM output formats for distribution extraction", "status": "completed", "priority": "medium"}]