"""
Differentiable pendulum control micro-benchmark.

This module implements a simple pendulum control task to validate
gradient flow through the E2E-DP pipeline before moving to complex
trading systems.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Tuple, Dict, Any
import equinox as eqx
import optax


class PendulumDynamics(eqx.Module):
    """Differentiable pendulum dynamics with implicit integration."""
    
    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81
    damping: float = 0.1
    dt: float = 0.01
    
    def __call__(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        """
        Forward dynamics of pendulum.
        
        Args:
            state: [theta, theta_dot] - angle and angular velocity
            control: [u] - control torque
        
        Returns:
            next_state: [theta_new, theta_dot_new]
        """
        theta, theta_dot = state[0], state[1]
        u = control[0]
        
        # Pendulum dynamics: theta_ddot = -(g/l)*sin(theta) - damping*theta_dot + u/(m*l^2)
        theta_ddot = (
            -(self.gravity / self.length) * jnp.sin(theta)
            - self.damping * theta_dot
            + u / (self.mass * self.length**2)
        )
        
        # Semi-implicit Euler integration (more stable for oscillatory systems)
        theta_dot_new = theta_dot + self.dt * theta_ddot
        theta_new = theta + self.dt * theta_dot_new
        
        return jnp.array([theta_new, theta_dot_new])


class DifferentiableController(eqx.Module):
    """End-to-end differentiable controller."""
    
    policy_network: eqx.nn.MLP
    dynamics: PendulumDynamics
    
    def __init__(self, hidden_dims: Tuple[int, ...] = (32, 32), key=None):
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Neural network policy
        self.policy_network = eqx.nn.MLP(
            in_size=2,  # theta, theta_dot
            out_size=1,  # control torque
            width_size=hidden_dims[0],
            depth=len(hidden_dims),
            activation=jax.nn.tanh,
            key=key
        )
        
        self.dynamics = PendulumDynamics()
    
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """Generate control from state."""
        return self.policy_network(state)
    
    def rollout(self, initial_state: jnp.ndarray, horizon: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate trajectory with learned controller.
        
        Returns:
            states: [horizon+1, 2] - trajectory of states
            controls: [horizon, 1] - sequence of controls
        """
        def step(carry, _):
            state = carry
            control = self(state)
            next_state = self.dynamics(state, control)
            return next_state, (state, control)
        
        final_state, (states, controls) = jax.lax.scan(
            step, initial_state, jnp.arange(horizon)
        )
        
        # Append final state
        states = jnp.concatenate([states, final_state[None, :]], axis=0)
        
        return states, controls


def smooth_control_cost(controls: jnp.ndarray) -> jnp.ndarray:
    """Penalize control effort and roughness."""
    effort = jnp.mean(controls**2)
    smoothness = jnp.mean((controls[1:] - controls[:-1])**2)
    return effort + 0.1 * smoothness


def stabilization_loss(
    controller: DifferentiableController,
    initial_state: jnp.ndarray,
    target_state: jnp.ndarray = None,
    horizon: int = 100
) -> jnp.ndarray:
    """
    Loss function for pendulum stabilization task.
    
    Args:
        controller: Differentiable controller
        initial_state: Starting state
        target_state: Desired final state (default: upright position)
        horizon: Planning horizon
    
    Returns:
        Scalar loss value
    """
    if target_state is None:
        target_state = jnp.array([0.0, 0.0])  # Upright, stationary
    
    states, controls = controller.rollout(initial_state, horizon)
    
    # State tracking loss (focus on final states)
    state_errors = states - target_state[None, :]
    tracking_loss = jnp.mean(state_errors[-20:]**2)  # Focus on last 20% of trajectory
    
    # Control regularization
    control_loss = smooth_control_cost(controls)
    
    # Stability bonus (small velocities near target)
    stability_loss = jnp.mean(jnp.abs(states[-20:, 1]))  # Angular velocity should be small
    
    return tracking_loss + 0.01 * control_loss + 0.1 * stability_loss


class GradientHealthMonitor:
    """Monitor gradient health metrics during training."""
    
    def __init__(self):
        self.history = {
            'gradient_norms': [],
            'gradient_variance': [],
            'signal_to_noise': [],
            'layer_ratios': []
        }
    
    def compute_metrics(self, grads: Any) -> Dict[str, float]:
        """Compute gradient health metrics."""
        # Flatten all gradients
        flat_grads, _ = jax.tree_util.tree_flatten(grads)
        all_grads = jnp.concatenate([g.flatten() for g in flat_grads])
        
        # Basic statistics
        grad_norm = jnp.linalg.norm(all_grads)
        grad_mean = jnp.mean(all_grads)
        grad_var = jnp.var(all_grads)
        
        # Signal-to-noise ratio
        snr = jnp.abs(grad_mean) / (jnp.sqrt(grad_var) + 1e-8)
        
        # Layer-wise analysis
        layer_norms = [jnp.linalg.norm(g.flatten()) for g in flat_grads]
        if len(layer_norms) > 1:
            # Ratio between first and last layer
            layer_ratio = layer_norms[-1] / (layer_norms[0] + 1e-8)
        else:
            layer_ratio = 1.0
        
        metrics = {
            'gradient_norm': float(grad_norm),
            'gradient_variance': float(grad_var),
            'signal_to_noise': float(snr),
            'layer_ratio': float(layer_ratio)
        }
        
        # Update history
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        return metrics
    
    def check_health(self, grads: Any) -> Tuple[bool, str]:
        """Check if gradients are healthy."""
        metrics = self.compute_metrics(grads)
        
        issues = []
        
        # Check for vanishing gradients
        if metrics['gradient_norm'] < 1e-6:
            issues.append("Vanishing gradients detected")
        
        # Check for exploding gradients
        if metrics['gradient_norm'] > 100:
            issues.append("Exploding gradients detected")
        
        # Check layer ratio
        if metrics['layer_ratio'] < 0.01 or metrics['layer_ratio'] > 100:
            issues.append(f"Poor gradient flow between layers (ratio: {metrics['layer_ratio']:.2f})")
        
        # Check SNR
        if metrics['signal_to_noise'] < 0.1:
            issues.append("Low gradient signal-to-noise ratio")
        
        is_healthy = len(issues) == 0
        message = "Gradients healthy" if is_healthy else "; ".join(issues)
        
        return is_healthy, message


def train_step(
    controller: DifferentiableController,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    initial_state: jnp.ndarray,
    target_state: jnp.ndarray,
    horizon: int
) -> Tuple[DifferentiableController, Any, float, Any]:
    """Single training step."""
    
    # Compute loss and gradients
    loss_fn = lambda ctrl: stabilization_loss(ctrl, initial_state, target_state, horizon)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(controller)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    controller = eqx.apply_updates(controller, updates)
    
    return controller, opt_state, loss, grads


def train_pendulum_controller(
    n_steps: int = 1000,
    learning_rate: float = 1e-3,
    horizon: int = 100,
    seed: int = 42
) -> Tuple[DifferentiableController, Dict[str, Any]]:
    """
    Train pendulum controller and monitor gradient health.
    
    Returns:
        Trained controller and training metrics
    """
    key = jax.random.PRNGKey(seed)
    
    # Initialize controller
    controller = DifferentiableController(hidden_dims=(32, 32), key=key)
    
    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(controller, eqx.is_array))
    
    # Gradient health monitor
    monitor = GradientHealthMonitor()
    
    # Training metrics
    losses = []
    gradient_healths = []
    
    # Training loop
    for step in range(n_steps):
        # Random initial state (pendulum hanging down with small perturbation)
        key, subkey = jax.random.split(key)
        initial_state = jnp.array([
            jnp.pi + 0.1 * jax.random.normal(subkey),  # Near bottom
            0.1 * jax.random.normal(jax.random.split(subkey)[0])  # Small velocity
        ])
        
        # Target: upright position
        target_state = jnp.array([0.0, 0.0])
        
        # Training step
        controller, opt_state, loss, grads = train_step(
            controller, opt_state, optimizer, initial_state, target_state, horizon
        )
        
        losses.append(float(loss))
        
        # Monitor gradient health
        if step % 10 == 0:
            is_healthy, message = monitor.check_health(grads)
            gradient_healths.append(is_healthy)
            
            if step % 100 == 0:
                metrics = monitor.compute_metrics(grads)
                print(f"Step {step}: Loss = {loss:.4f}, {message}")
                print(f"  Gradient norm: {metrics['gradient_norm']:.2e}")
                print(f"  Layer ratio: {metrics['layer_ratio']:.2f}")
                print(f"  SNR: {metrics['signal_to_noise']:.2f}")
    
    # Compile results
    results = {
        'losses': jnp.array(losses),
        'gradient_history': monitor.history,
        'health_percentage': jnp.mean(jnp.array(gradient_healths)) * 100
    }
    
    return controller, results


if __name__ == "__main__":
    print("Running differentiable pendulum control benchmark...")
    
    # Train controller
    controller, results = train_pendulum_controller(n_steps=500)
    
    print(f"\nTraining completed!")
    print(f"Final loss: {results['losses'][-1]:.4f}")
    print(f"Gradient health: {results['health_percentage']:.1f}% of checks passed")
    
    # Test learned controller
    test_state = jnp.array([jnp.pi, 0.0])  # Start from bottom
    states, controls = controller.rollout(test_state, horizon=200)
    
    print(f"\nTest rollout from bottom position:")
    print(f"Initial state: θ={test_state[0]:.2f}, θ_dot={test_state[1]:.2f}")
    print(f"Final state: θ={states[-1, 0]:.2f}, θ_dot={states[-1, 1]:.2f}")
    print(f"Average control magnitude: {jnp.mean(jnp.abs(controls)):.2f}")