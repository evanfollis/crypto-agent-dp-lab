"""
Φ-layer integration: Bridging symbolic and neural components.

This module provides the integration point between the Φ-layer and
the E2E-DP system, implementing the hybrid neuro-symbolic loss function.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass

from .layer import PhiLayer


@dataclass
class PhiIntegrationConfig:
    """Configuration for Φ-layer integration."""
    phi_weight: float = 1.0                    # Weight of Φ penalties in total loss
    orthogonality_penalty: float = 0.01        # Penalty for DP-Φ interference
    curriculum_schedule: str = "linear"        # How to schedule Φ weight over time
    min_phi_weight: float = 0.1               # Minimum Φ weight during curriculum
    max_phi_weight: float = 2.0               # Maximum Φ weight during curriculum
    decay_schedule: bool = True               # Whether to decay rule weights
    gradient_monitoring: bool = True          # Whether to monitor gradient health


class PhiGuidedLoss(eqx.Module):
    """
    Φ-guided loss function that combines E2E-DP with symbolic knowledge.
    
    Implements the hybrid loss from CLAUDE.md Section 7.2:
    L_total = L_dp + Σᵢ wᵢ · soft_penalty_i(θ)
    """
    
    phi_layer: PhiLayer
    config: PhiIntegrationConfig
    base_loss_fn: Callable
    step_count: int
    
    def __init__(
        self, 
        phi_layer: PhiLayer, 
        base_loss_fn: Callable,
        config: Optional[PhiIntegrationConfig] = None
    ):
        """
        Initialize Φ-guided loss.
        
        Args:
            phi_layer: The Φ-layer with symbolic rules
            base_loss_fn: Base E2E-DP loss function (e.g., negative Sharpe)
            config: Configuration for integration
        """
        self.phi_layer = phi_layer
        self.base_loss_fn = base_loss_fn
        self.config = config or PhiIntegrationConfig()
        self.step_count = 0
    
    def __call__(
        self, 
        positions: jnp.ndarray, 
        state: Dict[str, jnp.ndarray],
        returns: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute Φ-guided loss.
        
        Args:
            positions: Portfolio positions
            state: Market state dictionary
            returns: Portfolio returns for base loss
            
        Returns:
            Total loss, diagnostic information
        """
        # Compute base E2E-DP loss
        base_loss = self.base_loss_fn(returns)
        
        # Compute Φ-layer penalties
        phi_penalty, phi_info = self.phi_layer(positions, state)
        
        # Apply curriculum scheduling
        phi_weight = self._get_curriculum_weight()
        
        # Combine losses
        total_loss = base_loss + phi_weight * phi_penalty
        
        # Add orthogonality penalty if enabled
        if self.config.orthogonality_penalty > 0:
            orthogonal_penalty = self._compute_dp_phi_orthogonality(
                positions, state, returns
            )
            total_loss += self.config.orthogonality_penalty * orthogonal_penalty
        else:
            orthogonal_penalty = 0.0
        
        # Prepare diagnostic info
        diagnostics = {
            'base_loss': float(base_loss),
            'phi_penalty': float(phi_penalty),
            'phi_weight': float(phi_weight),
            'total_loss': float(total_loss),
            'orthogonal_penalty': float(orthogonal_penalty),
            'step_count': self.step_count,
            'phi_info': phi_info,
            'loss_breakdown': {
                'base_pct': float(base_loss / total_loss) * 100,
                'phi_pct': float(phi_weight * phi_penalty / total_loss) * 100,
                'ortho_pct': float(self.config.orthogonality_penalty * orthogonal_penalty / total_loss) * 100
            }
        }
        
        return total_loss, diagnostics
    
    def _get_curriculum_weight(self) -> float:
        """
        Get current Φ weight based on curriculum schedule.
        
        Returns:
            Current Φ weight
        """
        if self.config.curriculum_schedule == "constant":
            return self.config.phi_weight
        
        elif self.config.curriculum_schedule == "linear":
            # Linear increase from min to max over first 1000 steps
            progress = min(self.step_count / 1000.0, 1.0)
            return (
                self.config.min_phi_weight + 
                progress * (self.config.max_phi_weight - self.config.min_phi_weight)
            )
        
        elif self.config.curriculum_schedule == "exponential":
            # Exponential increase
            alpha = 0.001
            weight = self.config.min_phi_weight * jnp.exp(alpha * self.step_count)
            return jnp.clip(weight, self.config.min_phi_weight, self.config.max_phi_weight)
        
        else:
            return self.config.phi_weight
    
    def _compute_dp_phi_orthogonality(
        self, 
        positions: jnp.ndarray, 
        state: Dict[str, jnp.ndarray],
        returns: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute orthogonality penalty between DP and Φ gradients.
        
        This prevents double-counting when both systems learn the same effect.
        """
        # Gradient of base loss w.r.t. positions
        grad_base = jax.grad(lambda pos: self.base_loss_fn(returns))(positions)
        
        # Gradient of Φ penalty w.r.t. positions  
        grad_phi = jax.grad(lambda pos: self.phi_layer(pos, state)[0])(positions)
        
        # Dot product measures alignment (positive = same direction)
        dot_product = jnp.dot(grad_base, grad_phi)
        
        # Penalty for excessive alignment
        return dot_product ** 2
    
    def step(self) -> 'PhiGuidedLoss':
        """
        Advance one training step (for curriculum scheduling).
        
        Returns:
            Updated PhiGuidedLoss with incremented step count
        """
        new_step_count = self.step_count + 1
        updated_loss = eqx.tree_at(
            lambda loss: loss.step_count,
            self,
            new_step_count
        )
        
        # Apply rule weight decay if enabled
        if self.config.decay_schedule:
            decayed_phi_layer = self.phi_layer.decay_weights()
            updated_loss = eqx.tree_at(
                lambda loss: loss.phi_layer,
                updated_loss,
                decayed_phi_layer
            )
        
        return updated_loss
    
    def update_phi_weights(
        self, 
        performance_metrics: Dict[str, float],
        learning_rate: float = 0.01
    ) -> 'PhiGuidedLoss':
        """
        Update Φ-layer rule weights based on performance.
        
        Args:
            performance_metrics: Dictionary of rule_name -> performance
            learning_rate: Learning rate for weight updates
            
        Returns:
            Updated PhiGuidedLoss with new rule weights
        """
        updated_phi_layer = self.phi_layer.update_attention(
            performance_metrics, learning_rate
        )
        
        return eqx.tree_at(
            lambda loss: loss.phi_layer,
            self,
            updated_phi_layer
        )
    
    def get_explanation(
        self, 
        positions: jnp.ndarray, 
        state: Dict[str, jnp.ndarray]
    ) -> str:
        """
        Generate explanation of loss components.
        
        Args:
            positions: Current positions
            state: Market state
            
        Returns:
            Human-readable explanation
        """
        phi_explanation = self.phi_layer.explain_decision(positions, state)
        phi_weight = self._get_curriculum_weight()
        
        return f"""
Φ-Guided Loss Analysis (Step {self.step_count}):
- Φ weight: {phi_weight:.3f}
- Curriculum schedule: {self.config.curriculum_schedule}

{phi_explanation}

Integration status:
- Orthogonality penalty: {self.config.orthogonality_penalty:.3f}
- Decay schedule: {self.config.decay_schedule}
"""


# Factory functions for common configurations
def create_minimal_phi_guided_loss(
    base_loss_fn: Callable,
    key: Optional[jax.random.PRNGKey] = None
) -> PhiGuidedLoss:
    """
    Create minimal Φ-guided loss with single volatility rule.
    
    Implements the minimal POC from CLAUDE.md Section 7.5.
    """
    from .rules import VolatilityRule
    from .layer import PhiLayer
    
    # Single volatility rule
    rules = {'volatility': VolatilityRule(vol_threshold=2.0, initial_weight=1.0)}
    phi_layer = PhiLayer(rules, key=key)
    
    # Conservative integration config
    config = PhiIntegrationConfig(
        phi_weight=0.5,
        orthogonality_penalty=0.01,
        curriculum_schedule="linear",
        min_phi_weight=0.1,
        max_phi_weight=1.0
    )
    
    return PhiGuidedLoss(phi_layer, base_loss_fn, config)


def create_full_phi_guided_loss(
    base_loss_fn: Callable,
    key: Optional[jax.random.PRNGKey] = None
) -> PhiGuidedLoss:
    """
    Create full Φ-guided loss with multiple rules.
    
    Implements the complete hybrid system from CLAUDE.md Section 7.6.
    """
    from .layer import create_default_phi_layer
    
    phi_layer = create_default_phi_layer(key)
    
    # Full integration config
    config = PhiIntegrationConfig(
        phi_weight=1.0,
        orthogonality_penalty=0.01,
        curriculum_schedule="exponential",
        min_phi_weight=0.1,
        max_phi_weight=2.0,
        decay_schedule=True,
        gradient_monitoring=True
    )
    
    return PhiGuidedLoss(phi_layer, base_loss_fn, config)


# Utility functions
def phi_sharpe_loss(returns: jnp.ndarray, epsilon: float = 1e-6) -> jnp.ndarray:
    """
    Smooth Sharpe ratio loss compatible with Φ-guided loss.
    
    This is the base loss function from basic_e2e.py, adapted for Φ integration.
    """
    mean_return = jnp.mean(returns)
    std_return = jnp.sqrt(jnp.var(returns) + epsilon)
    sharpe = mean_return / std_return
    return -sharpe  # Negative for minimization