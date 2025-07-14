"""
Tests for differentiable portfolio optimization module.
"""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from crypto_dp.models.portfolio import (
    softmax_weights,
    gumbel_softmax_weights,
    long_only_weights,
    long_short_weights,
    sharpe_ratio,
    information_ratio,
    max_drawdown,
    transaction_cost_penalty,
    concentration_penalty,
    DifferentiablePortfolio,
    portfolio_objective,
    portfolio_step,
    backtest_portfolio
)


class TestWeightTransformations:
    """Test various weight transformation functions."""
    
    def test_softmax_weights(self):
        """Test softmax weight transformation."""
        scores = jnp.array([1.0, 2.0, 3.0])
        weights = softmax_weights(scores, temperature=1.0)
        
        assert weights.shape == scores.shape
        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert jnp.all(weights >= 0.0)
        
        # Higher scores should get higher weights
        assert weights[2] > weights[1] > weights[0]
    
    def test_softmax_weights_temperature(self):
        """Test temperature effect in softmax."""
        scores = jnp.array([1.0, 2.0, 3.0])
        
        # High temperature should make weights more concentrated
        weights_high_temp = softmax_weights(scores, temperature=10.0)
        weights_low_temp = softmax_weights(scores, temperature=0.1)
        
        # High temperature should be more concentrated (higher max weight)
        assert jnp.max(weights_high_temp) > jnp.max(weights_low_temp)
    
    def test_gumbel_softmax_weights(self):
        """Test Gumbel-softmax weight transformation."""
        scores = jnp.array([1.0, 2.0, 3.0])
        key = jax.random.PRNGKey(42)
        
        weights = gumbel_softmax_weights(scores, temperature=1.0, key=key)
        
        assert weights.shape == scores.shape
        assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-6)
        assert jnp.all(weights >= 0.0)
    
    def test_long_only_weights(self):
        """Test long-only weight transformation."""
        scores = jnp.array([-1.0, 2.0, 3.0])
        weights = long_only_weights(scores)
        
        assert weights.shape == scores.shape
        assert jnp.allclose(jnp.sum(weights), 1.0)
        assert jnp.all(weights >= 0.0)
        
        # Negative scores should result in zero weights
        assert weights[0] == 0.0
        assert weights[1] > 0.0
        assert weights[2] > 0.0
    
    def test_long_only_weights_all_negative(self):
        """Test long-only weights with all negative scores."""
        scores = jnp.array([-1.0, -2.0, -3.0])
        weights = long_only_weights(scores)
        
        # Should fall back to equal weights
        expected_weight = 1.0 / len(scores)
        assert jnp.allclose(weights, expected_weight)
    
    def test_long_short_weights(self):
        """Test long-short weight transformation."""
        scores = jnp.array([-2.0, 1.0, 3.0])
        weights = long_short_weights(scores, long_weight=1.0, short_weight=1.0)
        
        assert weights.shape == scores.shape
        
        # Should have both positive and negative weights
        assert jnp.any(weights > 0.0)
        assert jnp.any(weights < 0.0)
        
        # Positive scores should give positive weights
        assert weights[1] > 0.0
        assert weights[2] > 0.0
        
        # Negative scores should give negative weights
        assert weights[0] < 0.0


class TestRiskMetrics:
    """Test risk and performance metrics."""
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Create synthetic returns with positive expected return
        returns = jnp.array([
            [0.01, 0.02, -0.01],
            [0.02, -0.01, 0.01],
            [0.03, 0.01, 0.02],
            [-0.01, 0.03, 0.01]
        ])
        weights = jnp.array([0.4, 0.4, 0.2])
        
        sharpe = sharpe_ratio(returns, weights)
        
        assert jnp.isscalar(sharpe)
        assert jnp.isfinite(sharpe)
        # Should be negative (since we return negative Sharpe for minimization)
        assert sharpe <= 0.0
    
    def test_information_ratio(self):
        """Test information ratio calculation."""
        returns = jnp.array([
            [0.01, 0.02],
            [0.02, -0.01],
            [0.03, 0.01],
            [-0.01, 0.03]
        ])
        weights = jnp.array([0.6, 0.4])
        benchmark_returns = jnp.array([0.015, 0.005, 0.02, 0.01])
        
        ir = information_ratio(returns, weights, benchmark_returns)
        
        assert jnp.isscalar(ir)
        assert jnp.isfinite(ir)
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Create returns that lead to a drawdown
        returns = jnp.array([
            [0.10, 0.05],   # Positive returns
            [0.05, 0.02],   # More positive
            [-0.15, -0.10], # Large negative (drawdown)
            [-0.05, -0.03], # More negative
            [0.08, 0.06]    # Recovery
        ])
        weights = jnp.array([0.6, 0.4])
        
        dd = max_drawdown(returns, weights)
        
        assert jnp.isscalar(dd)
        assert jnp.isfinite(dd)
        assert dd <= 0.0  # Drawdown should be negative
    
    def test_transaction_cost_penalty(self):
        """Test transaction cost calculation."""
        old_weights = jnp.array([0.3, 0.4, 0.3])
        new_weights = jnp.array([0.4, 0.3, 0.3])
        
        cost = transaction_cost_penalty(old_weights, new_weights, cost_rate=0.001)
        
        assert jnp.isscalar(cost)
        assert jnp.isfinite(cost)
        assert cost >= 0.0
        
        # No change should result in zero cost
        zero_cost = transaction_cost_penalty(old_weights, old_weights)
        assert zero_cost == 0.0
    
    def test_concentration_penalty(self):
        """Test concentration penalty calculation."""
        # Highly concentrated weights
        concentrated_weights = jnp.array([0.8, 0.1, 0.1])
        penalty = concentration_penalty(concentrated_weights, max_weight=0.2)
        
        assert jnp.isscalar(penalty)
        assert jnp.isfinite(penalty)
        assert penalty > 0.0  # Should penalize concentration
        
        # Diversified weights
        diversified_weights = jnp.array([0.15, 0.15, 0.15, 0.15, 0.4])
        low_penalty = concentration_penalty(diversified_weights, max_weight=0.5)
        
        assert low_penalty < penalty


class TestDifferentiablePortfolio:
    """Test differentiable portfolio model."""
    
    def test_portfolio_initialization(self):
        """Test portfolio model initialization."""
        input_dim = 20
        n_assets = 5
        key = jax.random.PRNGKey(42)
        
        model = DifferentiablePortfolio(
            input_dim=input_dim,
            n_assets=n_assets,
            key=key
        )
        
        assert isinstance(model.scoring_network, eqx.nn.MLP)
        assert callable(model.weight_transform)
    
    def test_portfolio_forward_single(self):
        """Test portfolio forward pass with single sample."""
        input_dim = 10
        n_assets = 3
        key = jax.random.PRNGKey(42)
        
        model = DifferentiablePortfolio(
            input_dim=input_dim,
            n_assets=n_assets,
            key=key
        )
        
        features = jax.random.normal(key, (input_dim,))
        weights = model(features)
        
        assert weights.shape == (n_assets,)
        assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-6)
        assert jnp.all(weights >= 0.0)  # Default is softmax (long-only)
    
    def test_portfolio_forward_batch(self):
        """Test portfolio forward pass with batch."""
        input_dim = 10
        n_assets = 3
        batch_size = 5
        key = jax.random.PRNGKey(42)
        
        model = DifferentiablePortfolio(
            input_dim=input_dim,
            n_assets=n_assets,
            key=key
        )
        
        features = jax.random.normal(key, (batch_size, input_dim))
        weights = model(features)
        
        assert weights.shape == (batch_size, n_assets)
        assert jnp.allclose(jnp.sum(weights, axis=1), 1.0, atol=1e-6)
    
    def test_portfolio_different_transforms(self):
        """Test different weight transformation methods."""
        input_dim = 5
        n_assets = 3
        key = jax.random.PRNGKey(42)
        features = jax.random.normal(key, (input_dim,))
        
        # Test different weight transformations
        transforms = ["softmax", "long_only", "long_short"]
        
        for transform in transforms:
            model = DifferentiablePortfolio(
                input_dim=input_dim,
                n_assets=n_assets,
                weight_transform=transform,
                key=key
            )
            
            weights = model(features)
            assert weights.shape == (n_assets,)
            assert jnp.isfinite(weights).all()


class TestPortfolioOptimization:
    """Test portfolio optimization functions."""
    
    def test_portfolio_objective(self):
        """Test portfolio objective function."""
        input_dim = 5
        n_assets = 3
        n_periods = 20
        key = jax.random.PRNGKey(42)
        
        model = DifferentiablePortfolio(
            input_dim=input_dim,
            n_assets=n_assets,
            key=key
        )
        
        features = jax.random.normal(key, (input_dim,))
        returns = jax.random.normal(jax.random.split(key)[0], (n_periods, n_assets)) * 0.01
        
        objective = portfolio_objective(model, features, returns)
        
        assert jnp.isscalar(objective)
        assert jnp.isfinite(objective)
    
    def test_portfolio_step(self):
        """Test single portfolio optimization step."""
        input_dim = 5
        n_assets = 3
        n_periods = 20
        key = jax.random.PRNGKey(42)
        
        model = DifferentiablePortfolio(
            input_dim=input_dim,
            n_assets=n_assets,
            key=key
        )
        
        features = jax.random.normal(key, (input_dim,))
        returns = jax.random.normal(jax.random.split(key)[0], (n_periods, n_assets)) * 0.01
        
        updated_model, loss, weights = portfolio_step(
            model, features, returns, learning_rate=1e-3
        )
        
        assert isinstance(updated_model, DifferentiablePortfolio)
        assert jnp.isscalar(loss)
        assert jnp.isfinite(loss)
        assert weights.shape == (n_assets,)
        
        # Model should be updated
        original_weights = model(features)
        assert not jnp.allclose(weights, original_weights, atol=1e-6)
    
    def test_portfolio_step_with_transaction_costs(self):
        """Test optimization step with transaction costs."""
        input_dim = 5
        n_assets = 3
        n_periods = 20
        key = jax.random.PRNGKey(42)
        
        model = DifferentiablePortfolio(
            input_dim=input_dim,
            n_assets=n_assets,
            key=key
        )
        
        features = jax.random.normal(key, (input_dim,))
        returns = jax.random.normal(jax.random.split(key)[0], (n_periods, n_assets)) * 0.01
        old_weights = jnp.array([0.3, 0.4, 0.3])
        
        updated_model, loss, weights = portfolio_step(
            model, features, returns, old_weights=old_weights, beta=0.1
        )
        
        assert isinstance(updated_model, DifferentiablePortfolio)
        assert jnp.isscalar(loss)
        assert jnp.isfinite(loss)


class TestBacktesting:
    """Test portfolio backtesting functionality."""
    
    def test_backtest_portfolio(self):
        """Test portfolio backtesting."""
        input_dim = 5
        n_assets = 3
        n_periods = 300  # Need enough periods for lookback
        key = jax.random.PRNGKey(42)
        
        model = DifferentiablePortfolio(
            input_dim=input_dim,
            n_assets=n_assets,
            key=key
        )
        
        features_sequence = jax.random.normal(key, (n_periods, input_dim))
        returns_sequence = jax.random.normal(
            jax.random.split(key)[0], (n_periods, n_assets)
        ) * 0.01
        
        portfolio_returns, weights_history, transaction_costs = backtest_portfolio(
            model, features_sequence, returns_sequence,
            lookback_window=252, rebalance_freq=5
        )
        
        assert len(portfolio_returns) > 0
        assert len(weights_history) > 0
        assert len(transaction_costs) > 0
        
        # Check shapes
        assert jnp.isfinite(jnp.array(portfolio_returns)).all()
        assert weights_history.shape[1] == n_assets
        assert jnp.isfinite(jnp.array(transaction_costs)).all()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_returns(self):
        """Test handling of zero returns."""
        returns = jnp.zeros((10, 3))
        weights = jnp.array([0.3, 0.4, 0.3])
        
        # Should handle zero returns gracefully
        sharpe = sharpe_ratio(returns, weights)
        assert jnp.isfinite(sharpe)
    
    def test_single_asset(self):
        """Test portfolio with single asset."""
        input_dim = 5
        n_assets = 1
        key = jax.random.PRNGKey(42)
        
        model = DifferentiablePortfolio(
            input_dim=input_dim,
            n_assets=n_assets,
            key=key
        )
        
        features = jax.random.normal(key, (input_dim,))
        weights = model(features)
        
        assert weights.shape == (1,)
        assert jnp.allclose(jnp.sum(weights), 1.0)
    
    def test_invalid_weight_transform(self):
        """Test invalid weight transformation."""
        with pytest.raises(ValueError):
            DifferentiablePortfolio(
                input_dim=5,
                n_assets=3,
                weight_transform="invalid"
            )


if __name__ == "__main__":
    pytest.main([__file__])