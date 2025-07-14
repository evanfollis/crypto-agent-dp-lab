"""
Differentiable portfolio optimization for crypto trading.

This module implements end-to-end differentiable portfolio construction
and optimization using JAX. Supports various risk measures, transaction
costs, and portfolio constraints.
"""

from typing import Tuple, Optional, Callable
import logging

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import jit, grad, vmap


logger = logging.getLogger(__name__)


def softmax_weights(scores: jnp.ndarray, temperature: float = 10.0) -> jnp.ndarray:
    """
    Convert raw scores to portfolio weights using temperature-scaled softmax.
    
    Args:
        scores: Raw asset scores [n_assets]
        temperature: Temperature parameter (higher = more concentrated)
    
    Returns:
        Portfolio weights that sum to 1
    """
    scaled_scores = scores * temperature
    # Numerical stability
    scaled_scores = scaled_scores - jnp.max(scaled_scores)
    weights = jnp.exp(scaled_scores)
    return weights / jnp.sum(weights)


def gumbel_softmax_weights(
    scores: jnp.ndarray,
    temperature: float = 1.0,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Gumbel-softmax for differentiable discrete portfolio selection.
    
    Args:
        scores: Raw asset scores [n_assets]
        temperature: Temperature parameter
        key: Random key for Gumbel noise
    
    Returns:
        Differentiable discrete-like weights
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Sample Gumbel noise
    gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, scores.shape)))
    
    # Gumbel-softmax
    y = scores + gumbel_noise
    return softmax_weights(y, temperature)


def long_only_weights(scores: jnp.ndarray) -> jnp.ndarray:
    """
    Convert scores to long-only portfolio weights.
    
    Args:
        scores: Raw asset scores [n_assets]
    
    Returns:
        Long-only weights (all non-negative, sum to 1)
    """
    positive_scores = jnp.maximum(scores, 0.0)
    total = jnp.sum(positive_scores)
    
    # Handle edge case where all scores are negative
    return jnp.where(
        total > 1e-8,
        positive_scores / total,
        jnp.ones_like(scores) / len(scores)  # Equal weights fallback
    )


def long_short_weights(
    scores: jnp.ndarray,
    long_weight: float = 1.0,
    short_weight: float = 1.0
) -> jnp.ndarray:
    """
    Convert scores to long-short portfolio weights.
    
    Args:
        scores: Raw asset scores [n_assets]
        long_weight: Total weight for long positions
        short_weight: Total weight for short positions
    
    Returns:
        Long-short weights
    """
    long_scores = jnp.maximum(scores, 0.0)
    short_scores = jnp.maximum(-scores, 0.0)
    
    # Normalize separately
    long_sum = jnp.sum(long_scores)
    short_sum = jnp.sum(short_scores)
    
    long_weights = jnp.where(
        long_sum > 1e-8,
        long_scores / long_sum * long_weight,
        0.0
    )
    
    short_weights = jnp.where(
        short_sum > 1e-8,
        -short_scores / short_sum * short_weight,
        0.0
    )
    
    return long_weights + short_weights


def sharpe_ratio(returns: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Compute differentiable Sharpe ratio.
    
    Args:
        returns: Asset returns [n_periods, n_assets]
        weights: Portfolio weights [n_assets]
    
    Returns:
        Negative Sharpe ratio (for minimization)
    """
    portfolio_returns = jnp.dot(returns, weights)
    mean_return = jnp.mean(portfolio_returns)
    vol = jnp.std(portfolio_returns)
    
    # Add small epsilon for numerical stability
    sharpe = mean_return / (vol + 1e-8)
    return -sharpe  # Negative for minimization


def information_ratio(
    returns: jnp.ndarray,
    weights: jnp.ndarray,
    benchmark_returns: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute differentiable information ratio.
    
    Args:
        returns: Asset returns [n_periods, n_assets]
        weights: Portfolio weights [n_assets]
        benchmark_returns: Benchmark returns [n_periods]
    
    Returns:
        Negative information ratio (for minimization)
    """
    portfolio_returns = jnp.dot(returns, weights)
    excess_returns = portfolio_returns - benchmark_returns
    
    mean_excess = jnp.mean(excess_returns)
    tracking_error = jnp.std(excess_returns)
    
    ir = mean_excess / (tracking_error + 1e-8)
    return -ir  # Negative for minimization


def max_drawdown(returns: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Compute maximum drawdown (approximated for differentiability).
    
    Args:
        returns: Asset returns [n_periods, n_assets]
        weights: Portfolio weights [n_assets]
    
    Returns:
        Maximum drawdown
    """
    portfolio_returns = jnp.dot(returns, weights)
    
    # Cumulative returns
    cum_returns = jnp.cumprod(1 + portfolio_returns)
    
    # Running maximum (approximated with soft maximum)
    running_max = jnp.maximum.accumulate(cum_returns)
    
    # Drawdowns
    drawdowns = (cum_returns - running_max) / running_max
    
    return jnp.min(drawdowns)  # Most negative drawdown


def transaction_cost_penalty(
    old_weights: jnp.ndarray,
    new_weights: jnp.ndarray,
    cost_rate: float = 0.001
) -> jnp.ndarray:
    """
    Compute transaction cost penalty.
    
    Args:
        old_weights: Previous portfolio weights [n_assets]
        new_weights: New portfolio weights [n_assets]
        cost_rate: Transaction cost rate (e.g., 0.001 = 0.1%)
    
    Returns:
        Transaction cost penalty
    """
    turnover = jnp.sum(jnp.abs(new_weights - old_weights))
    return cost_rate * turnover


def concentration_penalty(weights: jnp.ndarray, max_weight: float = 0.2) -> jnp.ndarray:
    """
    Penalty for overly concentrated portfolios.
    
    Args:
        weights: Portfolio weights [n_assets]
        max_weight: Maximum allowed weight per asset
    
    Returns:
        Concentration penalty
    """
    excess_weights = jnp.maximum(jnp.abs(weights) - max_weight, 0.0)
    return jnp.sum(excess_weights ** 2)


class DifferentiablePortfolio(eqx.Module):
    """
    End-to-end differentiable portfolio optimization model.
    """
    
    scoring_network: eqx.nn.MLP
    weight_transform: Callable
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32),
        n_assets: int = 10,
        weight_transform: str = "softmax",
        key: Optional[jax.random.PRNGKey] = None
    ):
        """
        Initialize differentiable portfolio model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            n_assets: Number of assets
            weight_transform: Weight transformation method
            key: Random key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Scoring network
        self.scoring_network = eqx.nn.MLP(
            in_size=input_dim,
            out_size=n_assets,
            width_size=hidden_dims[0],
            depth=len(hidden_dims),
            key=key
        )
        
        # Weight transformation function
        if weight_transform == "softmax":
            self.weight_transform = softmax_weights
        elif weight_transform == "long_only":
            self.weight_transform = long_only_weights
        elif weight_transform == "long_short":
            self.weight_transform = long_short_weights
        else:
            raise ValueError(f"Unknown weight transform: {weight_transform}")
    
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Generate portfolio weights from input features.
        
        Args:
            features: Input features [input_dim] or [batch_size, input_dim]
        
        Returns:
            Portfolio weights
        """
        scores = self.scoring_network(features)
        
        if scores.ndim == 1:
            # Single sample
            return self.weight_transform(scores)
        else:
            # Batch processing
            return vmap(self.weight_transform)(scores)


def portfolio_objective(
    model: DifferentiablePortfolio,
    features: jnp.ndarray,
    returns: jnp.ndarray,
    old_weights: Optional[jnp.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 0.01
) -> jnp.ndarray:
    """
    Combined portfolio optimization objective.
    
    Args:
        model: DifferentiablePortfolio model
        features: Input features for current period
        returns: Historical returns [n_periods, n_assets]
        old_weights: Previous portfolio weights (for transaction costs)
        alpha: Weight for return-based objective (Sharpe ratio)
        beta: Weight for transaction cost penalty
        gamma: Weight for concentration penalty
    
    Returns:
        Combined objective (to minimize)
    """
    # Generate new weights
    new_weights = model(features)
    
    # Sharpe ratio (primary objective)
    sharpe_loss = sharpe_ratio(returns, new_weights)
    
    # Transaction cost penalty
    if old_weights is not None:
        tc_penalty = transaction_cost_penalty(old_weights, new_weights)
    else:
        tc_penalty = 0.0
    
    # Concentration penalty
    conc_penalty = concentration_penalty(new_weights)
    
    return alpha * sharpe_loss + beta * tc_penalty + gamma * conc_penalty


@jit
def portfolio_step(
    model: DifferentiablePortfolio,
    features: jnp.ndarray,
    returns: jnp.ndarray,
    old_weights: Optional[jnp.ndarray] = None,
    learning_rate: float = 1e-3,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 0.01
) -> Tuple[DifferentiablePortfolio, jnp.ndarray, jnp.ndarray]:
    """
    Single optimization step for portfolio model.
    
    Returns:
        Updated model, loss, and generated weights
    """
    def objective_fn(model):
        return portfolio_objective(model, features, returns, old_weights, alpha, beta, gamma)
    
    loss, grads = eqx.filter_value_and_grad(objective_fn)(model)
    model = eqx.apply_updates(model, grads, learning_rate)
    
    # Generate weights with updated model
    weights = model(features)
    
    return model, loss, weights


def backtest_portfolio(
    model: DifferentiablePortfolio,
    features_sequence: jnp.ndarray,
    returns_sequence: jnp.ndarray,
    lookback_window: int = 252,
    rebalance_freq: int = 1
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Backtest portfolio strategy.
    
    Args:
        model: Trained DifferentiablePortfolio model
        features_sequence: Feature sequence [n_periods, input_dim]
        returns_sequence: Return sequence [n_periods, n_assets]
        lookback_window: Number of periods for return calculation
        rebalance_freq: Rebalancing frequency (1 = daily)
    
    Returns:
        Portfolio returns, weights over time, transaction costs
    """
    n_periods = features_sequence.shape[0]
    n_assets = returns_sequence.shape[1]
    
    portfolio_returns = []
    weights_history = []
    transaction_costs = []
    
    current_weights = jnp.ones(n_assets) / n_assets  # Start with equal weights
    
    for t in range(lookback_window, n_periods):
        if t % rebalance_freq == 0:
            # Get features for current period
            features = features_sequence[t]
            
            # Historical returns for optimization
            historical_returns = returns_sequence[t-lookback_window:t]
            
            # Generate new weights
            new_weights = model(features)
            
            # Compute transaction costs
            tc = transaction_cost_penalty(current_weights, new_weights)
            transaction_costs.append(float(tc))
            
            current_weights = new_weights
        
        weights_history.append(current_weights)
        
        # Compute portfolio return for this period
        if t < n_periods - 1:  # Avoid index out of bounds
            period_return = jnp.dot(current_weights, returns_sequence[t])
            portfolio_returns.append(float(period_return))
    
    return (
        jnp.array(portfolio_returns),
        jnp.array(weights_history),
        jnp.array(transaction_costs)
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    key = jax.random.PRNGKey(42)
    n_periods, n_assets, input_dim = 1000, 10, 20
    
    # Synthetic features and returns
    features = jax.random.normal(key, (n_periods, input_dim))
    returns = 0.01 * jax.random.normal(jax.random.split(key)[0], (n_periods, n_assets))
    
    # Initialize portfolio model
    model = DifferentiablePortfolio(
        input_dim=input_dim,
        n_assets=n_assets,
        key=key
    )
    
    # Single optimization step
    model, loss, weights = portfolio_step(
        model,
        features[0],
        returns[:252],  # Use first 252 periods as history
        learning_rate=1e-3
    )
    
    logger.info(f"Optimization step completed. Loss: {loss:.6f}")
    logger.info(f"Generated weights: {weights}")