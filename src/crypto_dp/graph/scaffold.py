"""
Latent graph scaffold for structured differentiable programming.

This module implements a differentiable graph structure that can learn
latent relationships between cryptocurrency assets and market factors.
Uses JAX for automatic differentiation and efficient computation.
"""

from typing import Tuple, Optional, Callable
import logging

import jax
import jax.numpy as jnp
import networkx as nx
import equinox as eqx
import optax
from jax import grad, jit, vmap


logger = logging.getLogger(__name__)


class LatentGraph(eqx.Module):
    """
    Differentiable latent graph for crypto asset relationships.
    
    This module learns a structured representation of relationships between
    crypto assets using a parameterized adjacency matrix that can be optimized
    via gradient descent.
    """
    
    W: jnp.ndarray  # Adjacency weight matrix
    n_factors: int
    activation: Callable
    
    def __init__(
        self,
        n_factors: int,
        activation: Callable = jax.nn.tanh,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """
        Initialize latent graph.
        
        Args:
            n_factors: Number of latent factors/nodes
            activation: Activation function for message passing
            key: Random key for initialization
        """
        if n_factors < 1:
            raise ValueError("n_factors must be >= 1")
            
        if key is None:
            key = jax.random.PRNGKey(42)
            
        self.n_factors = n_factors
        self.activation = activation
        
        # Initialize adjacency matrix with small random weights
        self.W = jax.random.normal(key, (n_factors, n_factors)) * 0.1
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the latent graph.
        
        Args:
            x: Input features [batch_size, n_factors] or [n_factors]
        
        Returns:
            Graph-transformed features
        """
        # Linear message passing with activation
        if x.ndim == 1:
            # Single sample
            h = x @ self.W
        else:
            # Batch processing
            h = jnp.einsum('bi,ij->bj', x, self.W)
            
        return self.activation(h)
    
    def forward_multi_step(self, x: jnp.ndarray, n_steps: int = 3) -> jnp.ndarray:
        """
        Multi-step message passing through the graph.
        
        Args:
            x: Input features
            n_steps: Number of message passing steps
        
        Returns:
            Features after n_steps of graph propagation
        """
        h = x
        for _ in range(n_steps):
            h = self(h)
        return h
    
    def get_adjacency_matrix(self) -> jnp.ndarray:
        """Get the current adjacency matrix."""
        return self.W
    
    def get_graph_structure(self, threshold: float = 0.1) -> nx.DiGraph:
        """
        Convert learned weights to a NetworkX graph for visualization.
        
        Args:
            threshold: Minimum weight magnitude to include edge
        
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_factors))
        
        weights = self.W
        for i in range(self.n_factors):
            for j in range(self.n_factors):
                if abs(weights[i, j]) > threshold:
                    G.add_edge(i, j, weight=float(weights[i, j]))
        
        return G


def bic_loss(
    model: LatentGraph,
    x: jnp.ndarray,
    target: jnp.ndarray,
    lambda_reg: float = 1e-2
) -> jnp.ndarray:
    """
    BIC-like loss function for graph structure learning.
    
    Combines prediction accuracy with complexity penalty to encourage
    sparse, interpretable graph structures.
    
    Args:
        model: LatentGraph model
        x: Input features
        target: Target values
        lambda_reg: Regularization strength
    
    Returns:
        Scalar loss value
    """
    # Forward pass
    preds = model(x)
    
    # Prediction loss (MSE)
    mse = jnp.mean((preds - target) ** 2)
    
    # Complexity penalty (L1 regularization on adjacency matrix)
    complexity = lambda_reg * jnp.sum(jnp.abs(model.W))
    
    # BIC-like penalty (encourages sparsity)
    n_edges = jnp.sum(jnp.abs(model.W) > 1e-6)
    bic_penalty = 0.5 * n_edges * jnp.log(x.shape[0])  # log(n_samples)
    
    return mse + complexity + lambda_reg * bic_penalty


def spectral_regularization(model: LatentGraph, alpha: float = 1e-3) -> jnp.ndarray:
    """
    Spectral regularization to encourage stable graph dynamics.
    
    Args:
        model: LatentGraph model
        alpha: Regularization strength
    
    Returns:
        Regularization term
    """
    # Compute spectral radius (largest eigenvalue magnitude)
    eigenvals = jnp.linalg.eigvals(model.W)
    spectral_radius = jnp.max(jnp.abs(eigenvals))
    
    # Penalty for spectral radius > 1 (instability)
    return alpha * jnp.maximum(0.0, spectral_radius - 1.0) ** 2


@jit
def graph_step(
    model: LatentGraph,
    x: jnp.ndarray,
    target: jnp.ndarray,
    learning_rate: float = 1e-3,
    lambda_reg: float = 1e-2
) -> Tuple[LatentGraph, jnp.ndarray]:
    """
    Single optimization step for the latent graph.
    
    Args:
        model: Current LatentGraph model
        x: Input features
        target: Target values
        learning_rate: Learning rate for gradient descent
        lambda_reg: Regularization strength
    
    Returns:
        Updated model and loss value
    """
    def loss_fn(model):
        return bic_loss(model, x, target, lambda_reg) + spectral_regularization(model)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, _ = optax.sgd(learning_rate).update(grads, None)
    model = eqx.apply_updates(model, updates)
    
    return model, loss


def train_graph(
    model: LatentGraph,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: Optional[jnp.ndarray] = None,
    y_val: Optional[jnp.ndarray] = None,
    n_epochs: int = 1000,
    learning_rate: float = 1e-3,
    lambda_reg: float = 1e-2,
    patience: int = 50,
    verbose: bool = True
) -> Tuple[LatentGraph, dict]:
    """
    Train the latent graph model.
    
    Args:
        model: Initial LatentGraph model
        x_train: Training features
        y_train: Training targets
        x_val: Validation features (optional)
        y_val: Validation targets (optional)
        n_epochs: Maximum number of training epochs
        learning_rate: Learning rate
        lambda_reg: Regularization strength
        patience: Early stopping patience
        verbose: Whether to print training progress
    
    Returns:
        Trained model and training history
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'spectral_radius': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = model
    
    for epoch in range(n_epochs):
        # Training step
        model, train_loss = graph_step(
            model, x_train, y_train, learning_rate, lambda_reg
        )
        
        history['train_loss'].append(float(train_loss))
        
        # Compute spectral radius for monitoring
        eigenvals = jnp.linalg.eigvals(model.W)
        spectral_radius = float(jnp.max(jnp.abs(eigenvals)))
        history['spectral_radius'].append(spectral_radius)
        
        # Validation
        if x_val is not None and y_val is not None:
            val_loss = bic_loss(model, x_val, y_val, lambda_reg)
            history['val_loss'].append(float(val_loss))
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break
        else:
            best_model = model
        
        # Logging
        if verbose and epoch % 100 == 0:
            if x_val is not None:
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                    f"val_loss={val_loss:.6f}, spectral_radius={spectral_radius:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.6f}, "
                    f"spectral_radius={spectral_radius:.4f}"
                )
    
    return best_model, history


def create_crypto_factor_graph(
    n_assets: int,
    n_market_factors: int = 5,
    key: Optional[jax.random.PRNGKey] = None
) -> LatentGraph:
    """
    Create a latent graph specifically for crypto asset relationships.
    
    Args:
        n_assets: Number of crypto assets
        n_market_factors: Number of market-wide factors (e.g., BTC dominance, DeFi, etc.)
        key: Random key for initialization
    
    Returns:
        Initialized LatentGraph for crypto assets
    """
    total_factors = n_assets + n_market_factors
    
    if key is None:
        key = jax.random.PRNGKey(42)
    
    model = LatentGraph(total_factors, key=key)
    
    logger.info(
        f"Created crypto factor graph with {n_assets} assets "
        f"and {n_market_factors} market factors"
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    key = jax.random.PRNGKey(42)
    n_samples, n_factors = 1000, 10
    
    x_data = jax.random.normal(key, (n_samples, n_factors))
    
    # Create synthetic targets with some structure
    true_W = jax.random.normal(jax.random.split(key)[0], (n_factors, n_factors)) * 0.1
    y_data = x_data @ true_W + 0.1 * jax.random.normal(jax.random.split(key)[1], (n_samples, n_factors))
    
    # Initialize and train model
    model = LatentGraph(n_factors, key=key)
    trained_model, history = train_graph(model, x_data, y_data, n_epochs=500)
    
    logger.info("Training completed successfully")