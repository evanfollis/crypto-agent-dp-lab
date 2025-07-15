"""
Tests for latent graph scaffold module.
"""

import pytest
import jax
import jax.numpy as jnp
import networkx as nx

from crypto_dp.graph.scaffold import (
    LatentGraph,
    bic_loss,
    spectral_regularization,
    graph_step,
    train_graph,
    create_crypto_factor_graph
)


class TestLatentGraph:
    """Test LatentGraph functionality."""
    
    def test_latent_graph_initialization(self):
        """Test LatentGraph initialization."""
        n_factors = 5
        key = jax.random.PRNGKey(42)
        
        graph = LatentGraph(n_factors, key=key)
        
        assert graph.n_factors == n_factors
        assert graph.W.shape == (n_factors, n_factors)
        assert callable(graph.activation)
    
    def test_latent_graph_forward_single(self):
        """Test forward pass with single sample."""
        n_factors = 3
        key = jax.random.PRNGKey(42)
        
        graph = LatentGraph(n_factors, key=key)
        x = jnp.array([1.0, 2.0, 3.0])
        
        output = graph(x)
        
        assert output.shape == (n_factors,)
        assert jnp.isfinite(output).all()
    
    def test_latent_graph_forward_batch(self):
        """Test forward pass with batch of samples."""
        n_factors = 4
        batch_size = 10
        key = jax.random.PRNGKey(42)
        
        graph = LatentGraph(n_factors, key=key)
        x = jax.random.normal(key, (batch_size, n_factors))
        
        output = graph(x)
        
        assert output.shape == (batch_size, n_factors)
        assert jnp.isfinite(output).all()
    
    def test_multi_step_forward(self):
        """Test multi-step message passing."""
        n_factors = 3
        key = jax.random.PRNGKey(42)
        
        graph = LatentGraph(n_factors, key=key)
        x = jnp.array([1.0, 2.0, 3.0])
        
        output = graph.forward_multi_step(x, n_steps=3)
        
        assert output.shape == (n_factors,)
        assert jnp.isfinite(output).all()
    
    def test_get_adjacency_matrix(self):
        """Test getting adjacency matrix."""
        n_factors = 3
        key = jax.random.PRNGKey(42)
        
        graph = LatentGraph(n_factors, key=key)
        adj_matrix = graph.get_adjacency_matrix()
        
        assert adj_matrix.shape == (n_factors, n_factors)
        assert jnp.array_equal(adj_matrix, graph.W)
    
    def test_get_graph_structure(self):
        """Test converting to NetworkX graph."""
        n_factors = 3
        key = jax.random.PRNGKey(42)
        
        graph = LatentGraph(n_factors, key=key)
        nx_graph = graph.get_graph_structure(threshold=0.01)
        
        assert isinstance(nx_graph, nx.DiGraph)
        assert nx_graph.number_of_nodes() == n_factors


class TestLossFunctions:
    """Test loss functions for graph training."""
    
    def test_bic_loss(self):
        """Test BIC loss computation."""
        n_factors = 3
        n_samples = 50
        key = jax.random.PRNGKey(42)
        
        model = LatentGraph(n_factors, key=key)
        x = jax.random.normal(key, (n_samples, n_factors))
        target = jax.random.normal(jax.random.split(key)[0], (n_samples, n_factors))
        
        loss = bic_loss(model, x, target, lambda_reg=1e-2)
        
        assert jnp.isscalar(loss)
        assert jnp.isfinite(loss)
        assert loss >= 0.0
    
    def test_spectral_regularization(self):
        """Test spectral regularization."""
        n_factors = 3
        key = jax.random.PRNGKey(42)
        
        model = LatentGraph(n_factors, key=key)
        reg = spectral_regularization(model, alpha=1e-3)
        
        assert jnp.isscalar(reg)
        assert jnp.isfinite(reg)
        assert reg >= 0.0
    
    def test_bic_loss_gradient(self):
        """Test that BIC loss is differentiable."""
        n_factors = 3
        n_samples = 20
        key = jax.random.PRNGKey(42)
        
        model = LatentGraph(n_factors, key=key)
        x = jax.random.normal(key, (n_samples, n_factors))
        target = jax.random.normal(jax.random.split(key)[0], (n_samples, n_factors))
        
        # Compute gradient
        loss_fn = lambda m: bic_loss(m, x, target)
        grad_fn = jax.grad(loss_fn)
        
        try:
            grads = grad_fn(model)
            # Should not raise an error
            assert hasattr(grads, 'W')
            assert grads.W.shape == model.W.shape
        except Exception as e:
            pytest.fail(f"Gradient computation failed: {e}")


class TestTraining:
    """Test graph training functionality."""
    
    def test_graph_step(self):
        """Test single training step."""
        n_factors = 3
        n_samples = 20
        key = jax.random.PRNGKey(42)
        
        model = LatentGraph(n_factors, key=key)
        x = jax.random.normal(key, (n_samples, n_factors))
        target = jax.random.normal(jax.random.split(key)[0], (n_samples, n_factors))
        
        updated_model, loss = graph_step(model, x, target, learning_rate=1e-3)
        
        assert isinstance(updated_model, LatentGraph)
        assert jnp.isscalar(loss)
        assert jnp.isfinite(loss)
        
        # Model should be updated (weights changed)
        assert not jnp.allclose(model.W, updated_model.W, atol=1e-6)
    
    def test_train_graph_basic(self):
        """Test basic graph training."""
        n_factors = 3
        n_samples = 50
        key = jax.random.PRNGKey(42)
        
        model = LatentGraph(n_factors, key=key)
        x_train = jax.random.normal(key, (n_samples, n_factors))
        y_train = jax.random.normal(jax.random.split(key)[0], (n_samples, n_factors))
        
        trained_model, history = train_graph(
            model, x_train, y_train,
            n_epochs=10,
            learning_rate=1e-3,
            verbose=False
        )
        
        assert isinstance(trained_model, LatentGraph)
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'spectral_radius' in history
        assert len(history['train_loss']) == 10
    
    def test_train_graph_with_validation(self):
        """Test graph training with validation data."""
        n_factors = 3
        n_samples = 30
        key = jax.random.PRNGKey(42)
        
        model = LatentGraph(n_factors, key=key)
        x_train = jax.random.normal(key, (n_samples, n_factors))
        y_train = jax.random.normal(jax.random.split(key)[0], (n_samples, n_factors))
        x_val = jax.random.normal(jax.random.split(key)[1], (10, n_factors))
        y_val = jax.random.normal(jax.random.split(key)[2], (10, n_factors))
        
        trained_model, history = train_graph(
            model, x_train, y_train, x_val, y_val,
            n_epochs=5,
            learning_rate=1e-3,
            patience=3,
            verbose=False
        )
        
        assert isinstance(trained_model, LatentGraph)
        assert 'val_loss' in history
        assert len(history['val_loss']) <= 5  # May stop early
    
    def test_training_convergence(self):
        """Test that training reduces loss."""
        n_factors = 3
        n_samples = 100
        key = jax.random.PRNGKey(42)
        
        # Create synthetic data with some structure
        true_W = jax.random.normal(key, (n_factors, n_factors)) * 0.1
        x_data = jax.random.normal(jax.random.split(key)[0], (n_samples, n_factors))
        y_data = jnp.tanh(x_data @ true_W)  # Target with structure
        
        model = LatentGraph(n_factors, key=key)
        
        trained_model, history = train_graph(
            model, x_data, y_data,
            n_epochs=50,
            learning_rate=1e-2,
            verbose=False
        )
        
        # Loss should generally decrease
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        assert final_loss < initial_loss


class TestCryptoFactorGraph:
    """Test crypto-specific graph functionality."""
    
    def test_create_crypto_factor_graph(self):
        """Test creating crypto factor graph."""
        n_assets = 10
        n_market_factors = 5
        key = jax.random.PRNGKey(42)
        
        graph = create_crypto_factor_graph(n_assets, n_market_factors, key)
        
        assert isinstance(graph, LatentGraph)
        assert graph.n_factors == n_assets + n_market_factors
        assert graph.W.shape == (n_assets + n_market_factors, n_assets + n_market_factors)
    
    def test_crypto_graph_forward_pass(self):
        """Test forward pass with crypto factor graph."""
        n_assets = 5
        n_market_factors = 3
        key = jax.random.PRNGKey(42)
        
        graph = create_crypto_factor_graph(n_assets, n_market_factors, key)
        
        # Simulate crypto features (asset prices + market factors)
        crypto_features = jax.random.normal(key, (n_assets + n_market_factors,))
        
        output = graph(crypto_features)
        
        assert output.shape == (n_assets + n_market_factors,)
        assert jnp.isfinite(output).all()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_factors(self):
        """Test handling of zero factors."""
        with pytest.raises((ValueError, IndexError)):
            LatentGraph(0)
    
    def test_single_factor(self):
        """Test single factor graph."""
        key = jax.random.PRNGKey(42)
        graph = LatentGraph(1, key=key)
        
        x = jnp.array([1.0])
        output = graph(x)
        
        assert output.shape == (1,)
        assert jnp.isfinite(output).all()
    
    def test_large_graph(self):
        """Test large graph handling."""
        n_factors = 100
        key = jax.random.PRNGKey(42)
        
        graph = LatentGraph(n_factors, key=key)
        x = jax.random.normal(key, (n_factors,))
        
        output = graph(x)
        
        assert output.shape == (n_factors,)
        assert jnp.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__])