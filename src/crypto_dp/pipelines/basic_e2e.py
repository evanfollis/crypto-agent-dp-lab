"""
Basic end-to-end differentiable pipeline demonstration.

This module implements a simple E2E-DP system that flows gradients through:
1. Feature extraction
2. Prediction
3. Decision making (portfolio optimization)
4. Simulated returns
5. Loss computation

Based on CLAUDE.md architecture principles.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Tuple, Dict, Any, NamedTuple
import equinox as eqx
import optax
from dataclasses import dataclass

from ..monitoring.gradient_health import EnhancedGradientMonitor, apply_global_gradient_clip


class MarketState(NamedTuple):
    """Simple market state representation."""
    prices: jnp.ndarray  # [n_assets]
    volumes: jnp.ndarray  # [n_assets]
    volatilities: jnp.ndarray  # [n_assets]
    time_features: jnp.ndarray  # [n_time_features]


class E2EDPModule(eqx.Module):
    """Base class for differentiable modules in the pipeline."""
    
    def check_gradients(self, grads: Any, monitor: EnhancedGradientMonitor) -> Dict[str, float]:
        """Check gradient health for this module."""
        metrics = monitor.compute_metrics(grads, prefix=self.__class__.__name__)
        return {
            'norm_ratio': metrics.norm_ratio,
            'stv': metrics.signal_to_total_variance,
            'sparsity': metrics.gradient_sparsity
        }


class DifferentiableFeatureExtractor(E2EDPModule):
    """Learnable feature extraction from market data."""
    
    price_encoder: eqx.nn.MLP
    volume_encoder: eqx.nn.MLP
    time_encoder: eqx.nn.MLP
    fusion_layer: eqx.nn.Linear
    
    def __init__(self, n_assets: int, feature_dim: int = 32, key=None):
        if key is None:
            key = jax.random.PRNGKey(42)
        
        keys = jax.random.split(key, 4)
        
        # Separate encoders for different data modalities
        self.price_encoder = eqx.nn.MLP(
            in_size=n_assets,
            out_size=feature_dim,
            width_size=16,
            depth=2,
            key=keys[0]
        )
        
        self.volume_encoder = eqx.nn.MLP(
            in_size=n_assets,
            out_size=feature_dim,
            width_size=16,
            depth=2,
            key=keys[1]
        )
        
        self.time_encoder = eqx.nn.MLP(
            in_size=4,  # hour, day, week, month features
            out_size=feature_dim // 2,
            width_size=8,
            depth=1,
            key=keys[2]
        )
        
        # Fusion layer
        self.fusion_layer = eqx.nn.Linear(
            in_features=feature_dim * 2 + feature_dim // 2,
            out_features=feature_dim * 2,
            key=keys[3]
        )
    
    def __call__(self, market_state: MarketState) -> jnp.ndarray:
        """Extract features from market state."""
        # Encode each modality
        price_features = self.price_encoder(market_state.prices)
        volume_features = self.volume_encoder(jnp.log1p(market_state.volumes))
        time_features = self.time_encoder(market_state.time_features)
        
        # Concatenate and fuse
        combined = jnp.concatenate([price_features, volume_features, time_features])
        return jax.nn.relu(self.fusion_layer(combined))


class DifferentiablePredictor(E2EDPModule):
    """Predict expected returns and risks."""
    
    return_predictor: eqx.nn.MLP
    risk_predictor: eqx.nn.MLP
    
    def __init__(self, feature_dim: int, n_assets: int, key=None):
        if key is None:
            key = jax.random.PRNGKey(43)
        
        keys = jax.random.split(key, 2)
        
        self.return_predictor = eqx.nn.MLP(
            in_size=feature_dim,
            out_size=n_assets,
            width_size=32,
            depth=2,
            key=keys[0]
        )
        
        self.risk_predictor = eqx.nn.MLP(
            in_size=feature_dim,
            out_size=n_assets,
            width_size=32,
            depth=2,
            activation=jax.nn.softplus,  # Ensure positive risks
            key=keys[1]
        )
    
    def __call__(self, features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict returns and risks from features."""
        expected_returns = self.return_predictor(features)
        predicted_risks = self.risk_predictor(features) + 1e-4  # Ensure non-zero
        return expected_returns, predicted_risks


class DifferentiableDecisionMaker(E2EDPModule):
    """Make portfolio decisions based on predictions."""
    
    temperature: float = 1.0
    max_position: float = 0.2
    
    def __call__(
        self,
        expected_returns: jnp.ndarray,
        predicted_risks: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Generate portfolio weights using differentiable optimization.
        
        Simple mean-variance optimization with softmax relaxation.
        """
        # Risk-adjusted scores
        scores = expected_returns / (predicted_risks + 1e-8)
        
        # Temperature-scaled softmax for differentiable selection
        scaled_scores = scores / self.temperature
        weights = jax.nn.softmax(scaled_scores)
        
        # Soft position limits using tanh
        weights = self.max_position * jnp.tanh(weights / self.max_position)
        
        # Renormalize
        weights = weights / (jnp.sum(weights) + 1e-8)
        
        return weights


class DifferentiableSimulator(E2EDPModule):
    """Simulate market response and returns."""
    
    market_impact: float = 0.001
    volatility_scaling: float = 0.1
    
    def __call__(
        self,
        weights: jnp.ndarray,
        market_state: MarketState,
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """
        Simulate returns with market impact and noise.
        
        This is a simplified simulator - real version would use JAX-LOB.
        """
        # Base returns (could be learned or historical)
        base_returns = 0.0001 * jax.random.normal(key, weights.shape)
        
        # Market impact (penalty for large positions)
        impact = -self.market_impact * weights**2
        
        # Volatility adjustment
        vol_adjustment = market_state.volatilities * self.volatility_scaling
        noise = vol_adjustment * jax.random.normal(jax.random.split(key)[0], weights.shape)
        
        # Total returns
        returns = base_returns + impact + noise
        
        # Portfolio return
        portfolio_return = jnp.dot(weights, returns)
        
        return portfolio_return


class EndToEndDPPipeline(E2EDPModule):
    """Complete E2E-DP pipeline from data to returns."""
    
    feature_extractor: DifferentiableFeatureExtractor
    predictor: DifferentiablePredictor
    decision_maker: DifferentiableDecisionMaker
    simulator: DifferentiableSimulator
    
    def __init__(self, n_assets: int = 10, feature_dim: int = 64, key=None):
        if key is None:
            key = jax.random.PRNGKey(100)
        
        keys = jax.random.split(key, 4)
        
        self.feature_extractor = DifferentiableFeatureExtractor(
            n_assets, feature_dim, keys[0]
        )
        self.predictor = DifferentiablePredictor(
            feature_dim * 2, n_assets, keys[1]
        )
        self.decision_maker = DifferentiableDecisionMaker()
        self.simulator = DifferentiableSimulator()
    
    def __call__(
        self,
        market_state: MarketState,
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Forward pass through entire pipeline.
        
        Returns:
            portfolio_return: Scalar return
            intermediates: Dict of intermediate values for analysis
        """
        # Extract features
        features = self.feature_extractor(market_state)
        
        # Predict
        expected_returns, predicted_risks = self.predictor(features)
        
        # Decide
        weights = self.decision_maker(expected_returns, predicted_risks)
        
        # Simulate
        portfolio_return = self.simulator(weights, market_state, key)
        
        # Store intermediates for gradient analysis
        intermediates = {
            'features': features,
            'expected_returns': expected_returns,
            'predicted_risks': predicted_risks,
            'weights': weights,
            'portfolio_return': portfolio_return
        }
        
        return portfolio_return, intermediates


def smooth_sharpe_loss(returns: jnp.ndarray, epsilon: float = 1e-6) -> jnp.ndarray:
    """Differentiable Sharpe ratio loss."""
    mean_return = jnp.mean(returns)
    std_return = jnp.sqrt(jnp.var(returns) + epsilon)
    sharpe = mean_return / std_return
    return -sharpe  # Negative for minimization


@dataclass
class TrainingConfig:
    """Configuration for E2E-DP training."""
    n_steps: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-3
    gradient_clip: float = 10.0
    n_assets: int = 10
    feature_dim: int = 64
    seed: int = 42


def generate_synthetic_market_data(
    n_samples: int,
    n_assets: int,
    key: jax.random.PRNGKey
) -> MarketState:
    """Generate synthetic market data for testing."""
    keys = jax.random.split(key, 4)
    
    # Generate correlated price movements
    price_base = 100 * jnp.ones(n_assets)
    price_noise = jax.random.normal(keys[0], (n_samples, n_assets))
    prices = price_base + jnp.cumsum(price_noise * 0.01, axis=0)
    
    # Volumes (log-normal)
    volumes = jnp.exp(jax.random.normal(keys[1], (n_samples, n_assets)) + 10)
    
    # Volatilities
    volatilities = 0.01 + 0.02 * jax.random.uniform(keys[2], (n_samples, n_assets))
    
    # Time features (sin/cos encoding)
    time_idx = jnp.arange(n_samples)
    time_features = jnp.stack([
        jnp.sin(2 * jnp.pi * time_idx / 24),  # Daily
        jnp.cos(2 * jnp.pi * time_idx / 24),
        jnp.sin(2 * jnp.pi * time_idx / (24 * 7)),  # Weekly
        jnp.cos(2 * jnp.pi * time_idx / (24 * 7))
    ], axis=1)
    
    # Return last sample as current state
    return MarketState(
        prices=prices[-1],
        volumes=volumes[-1],
        volatilities=volatilities[-1],
        time_features=time_features[-1]
    )


def train_e2e_pipeline(config: TrainingConfig) -> Tuple[EndToEndDPPipeline, Dict[str, Any]]:
    """Train the E2E-DP pipeline with gradient monitoring."""
    
    key = jax.random.PRNGKey(config.seed)
    
    # Initialize pipeline
    pipeline = EndToEndDPPipeline(
        n_assets=config.n_assets,
        feature_dim=config.feature_dim,
        key=key
    )
    
    # Optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(pipeline, eqx.is_array))
    
    # Gradient monitor
    monitor = EnhancedGradientMonitor()
    
    # Training metrics
    losses = []
    gradient_healths = []
    
    print("Training E2E-DP pipeline...")
    
    for step in range(config.n_steps):
        key, subkey = jax.random.split(key)
        
        # Generate batch of market states
        batch_returns = []
        
        for _ in range(config.batch_size):
            key, data_key, sim_key = jax.random.split(key, 3)
            
            # Generate market state
            market_state = generate_synthetic_market_data(100, config.n_assets, data_key)
            
            # Forward pass
            portfolio_return, _ = pipeline(market_state, sim_key)
            batch_returns.append(portfolio_return)
        
        # Compute loss over batch
        batch_returns = jnp.array(batch_returns)
        
        def loss_fn(pipe):
            # Re-run forward passes for gradient computation
            returns = []
            for i in range(config.batch_size):
                key_i = jax.random.fold_in(subkey, i)
                market_state = generate_synthetic_market_data(100, config.n_assets, key_i)
                ret, _ = pipe(market_state, key_i)
                returns.append(ret)
            
            returns = jnp.array(returns)
            return smooth_sharpe_loss(returns)
        
        # Compute gradients
        loss, grads = eqx.filter_value_and_grad(loss_fn)(pipeline)
        
        # Clip gradients
        clipped_grads, clipped = apply_global_gradient_clip(grads, config.gradient_clip)
        
        # Update
        updates, opt_state = optimizer.update(clipped_grads, opt_state)
        pipeline = eqx.apply_updates(pipeline, updates)
        
        losses.append(float(loss))
        
        # Monitor gradients
        if step % 10 == 0:
            metrics = monitor.compute_metrics(grads)
            is_healthy, issues = metrics.is_healthy()
            gradient_healths.append(is_healthy)
            
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss:.4f}")
                print(f"  Gradient health: {'✓' if is_healthy else '✗'}")
                if not is_healthy:
                    print(f"  Issues: {', '.join(issues)}")
                print(f"  Norm ratio: {metrics.norm_ratio:.2f}")
                print(f"  Clipped: {'Yes' if clipped else 'No'}")
    
    # Final summary
    health_rate = jnp.mean(jnp.array(gradient_healths)) * 100
    print(f"\nTraining completed!")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Gradient health rate: {health_rate:.1f}%")
    
    results = {
        'losses': jnp.array(losses),
        'gradient_monitor': monitor,
        'health_rate': health_rate
    }
    
    return pipeline, results


if __name__ == "__main__":
    # Run basic E2E-DP pipeline training
    config = TrainingConfig(
        n_steps=500,
        batch_size=16,
        learning_rate=1e-3,
        n_assets=5
    )
    
    pipeline, results = train_e2e_pipeline(config)
    
    # Test the trained pipeline
    print("\nTesting trained pipeline...")
    test_key = jax.random.PRNGKey(999)
    test_market = generate_synthetic_market_data(100, config.n_assets, test_key)
    
    test_return, intermediates = pipeline(test_market, test_key)
    
    print(f"Test portfolio return: {test_return:.4f}")
    print(f"Portfolio weights: {intermediates['weights']}")
    print(f"Expected returns: {intermediates['expected_returns']}")
    print(f"Predicted risks: {intermediates['predicted_risks']}")