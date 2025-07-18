"""
Φ-layer: Collection of rules with attention-based activation.

The PhiLayer aggregates multiple rules and provides:
1. Attention-weighted rule combination
2. Rule decay and meta-learning
3. Gradient attribution for interpretability
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, List, Tuple, Optional, Any
import time
from dataclasses import dataclass

from .rules import PhiRule, PhiRuleMetrics


@dataclass
class PhiLayerMetrics:
    """Metrics for monitoring the entire Φ-layer."""
    rule_metrics: Dict[str, PhiRuleMetrics]
    attention_weights: Dict[str, float]
    total_penalty: float
    active_rules: List[str]
    gradient_health: Dict[str, float]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'num_active_rules': len(self.active_rules),
            'total_penalty': self.total_penalty,
            'max_attention': max(self.attention_weights.values()) if self.attention_weights else 0.0,
            'healthy_rules': sum(1 for rule_metrics in self.rule_metrics.values() 
                               if rule_metrics.is_healthy()[0])
        }


class PhiLayer(eqx.Module):
    """
    Φ-layer: Neuro-symbolic knowledge integration layer.
    
    Combines multiple symbolic rules with learnable attention weights
    and provides differentiable penalties for loss shaping.
    """
    
    rules: Dict[str, PhiRule]
    attention_weights: jnp.ndarray
    rule_names: List[str]
    decay_rate: float
    orthogonality_penalty: float
    
    def __init__(
        self, 
        rules: Dict[str, PhiRule], 
        decay_rate: float = 0.99,
        orthogonality_penalty: float = 0.01,
        key: Optional[jax.random.PRNGKey] = None
    ):
        """
        Initialize Φ-layer.
        
        Args:
            rules: Dictionary of rule_name -> PhiRule
            decay_rate: Decay rate for rule weights over time
            orthogonality_penalty: Penalty for rule interference
            key: Random key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        
        self.rules = rules
        self.rule_names = list(rules.keys())
        self.decay_rate = decay_rate
        self.orthogonality_penalty = orthogonality_penalty
        
        # Initialize attention weights
        n_rules = len(rules)
        self.attention_weights = jax.nn.softmax(
            jax.random.normal(key, (n_rules,)) * 0.1
        )
    
    def __call__(
        self, 
        positions: jnp.ndarray, 
        state: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Apply Φ-layer to compute penalty and rule activations.
        
        Args:
            positions: Current portfolio positions
            state: Market state dictionary
            
        Returns:
            Total penalty, individual rule penalties
        """
        rule_penalties = {}
        rule_activations = {}
        
        # Compute individual rule penalties
        for i, (rule_name, rule) in enumerate(self.rules.items()):
            penalty = rule.apply(positions, state)
            activation = rule.trigger(state)
            
            rule_penalties[rule_name] = penalty
            rule_activations[rule_name] = activation
        
        # Attention-weighted combination
        total_penalty = 0.0
        for i, rule_name in enumerate(self.rule_names):
            attention_weight = self.attention_weights[i]
            total_penalty += attention_weight * rule_penalties[rule_name]
        
        # Add orthogonality penalty to prevent rule interference
        if self.orthogonality_penalty > 0:
            total_penalty += self._compute_orthogonality_penalty(
                positions, state, rule_penalties
            )
        
        return total_penalty, {
            'penalties': rule_penalties,
            'activations': rule_activations,
            'attention_weights': dict(zip(self.rule_names, self.attention_weights))
        }
    
    def _compute_orthogonality_penalty(
        self, 
        positions: jnp.ndarray, 
        state: Dict[str, jnp.ndarray],
        rule_penalties: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Compute orthogonality penalty to prevent rule interference.
        
        This prevents rules from having the same effect (double-counting).
        """
        # Get gradients of each rule penalty w.r.t. positions
        rule_gradients = {}
        for rule_name, rule in self.rules.items():
            grad_fn = jax.grad(lambda pos: rule.apply(pos, state))
            rule_gradients[rule_name] = grad_fn(positions)
        
        # Compute dot products between gradients
        penalty = 0.0
        rule_names = list(rule_gradients.keys())
        
        for i in range(len(rule_names)):
            for j in range(i + 1, len(rule_names)):
                grad_i = rule_gradients[rule_names[i]]
                grad_j = rule_gradients[rule_names[j]]
                
                # Penalty for parallel gradients (same effect)
                dot_product = jnp.dot(grad_i, grad_j)
                penalty += dot_product ** 2
        
        return self.orthogonality_penalty * penalty
    
    def update_attention(
        self, 
        rule_performance: Dict[str, float],
        learning_rate: float = 0.01
    ) -> 'PhiLayer':
        """
        Update attention weights based on rule performance.
        
        Args:
            rule_performance: Dictionary of rule_name -> performance metric
            learning_rate: Learning rate for attention updates
            
        Returns:
            Updated PhiLayer
        """
        # Convert performance to gradients
        performance_array = jnp.array([
            rule_performance.get(name, 0.0) 
            for name in self.rule_names
        ])
        
        # Softmax gradient update
        attention_logits = jnp.log(self.attention_weights + 1e-8)
        updated_logits = attention_logits + learning_rate * performance_array
        updated_weights = jax.nn.softmax(updated_logits)
        
        # Create new layer with updated weights
        new_layer = eqx.tree_at(
            lambda layer: layer.attention_weights,
            self,
            updated_weights
        )
        
        return new_layer
    
    def decay_weights(self) -> 'PhiLayer':
        """
        Apply decay to rule weights to prevent stale concepts.
        
        Returns:
            PhiLayer with decayed rule weights
        """
        decayed_rules = {}
        
        for rule_name, rule in self.rules.items():
            decayed_weight = rule.weight * self.decay_rate
            decayed_rule = eqx.tree_at(
                lambda r: r.weight,
                rule,
                decayed_weight
            )
            decayed_rules[rule_name] = decayed_rule
        
        return eqx.tree_at(
            lambda layer: layer.rules,
            self,
            decayed_rules
        )
    
    def get_active_rules(
        self, 
        state: Dict[str, jnp.ndarray], 
        threshold: float = 0.1
    ) -> List[str]:
        """
        Get list of currently active rules.
        
        Args:
            state: Current market state
            threshold: Activation threshold
            
        Returns:
            List of active rule names
        """
        active_rules = []
        
        for rule_name, rule in self.rules.items():
            activation = rule.trigger(state)
            if float(activation) > threshold:
                active_rules.append(rule_name)
        
        return active_rules
    
    def explain_decision(
        self, 
        positions: jnp.ndarray, 
        state: Dict[str, jnp.ndarray]
    ) -> str:
        """
        Generate human-readable explanation of Φ-layer decision.
        
        Args:
            positions: Current positions
            state: Market state
            
        Returns:
            Explanation string
        """
        _, rule_info = self(positions, state)
        
        explanations = []
        for i, rule_name in enumerate(self.rule_names):
            rule = self.rules[rule_name]
            attention = self.attention_weights[i]
            activation = rule_info['activations'][rule_name]
            
            if float(activation) > 0.1:  # Only explain active rules
                rule_explanation = rule.get_explanation(state)
                explanations.append(
                    f"  [{attention:.1%} attention] {rule_explanation}"
                )
        
        if explanations:
            return "Φ-layer active rules:\n" + "\n".join(explanations)
        else:
            return "Φ-layer: No rules currently active"
    
    def compute_metrics(
        self, 
        positions: jnp.ndarray, 
        state: Dict[str, jnp.ndarray]
    ) -> PhiLayerMetrics:
        """
        Compute comprehensive metrics for monitoring.
        
        Args:
            positions: Current positions
            state: Market state
            
        Returns:
            PhiLayerMetrics object
        """
        total_penalty, rule_info = self(positions, state)
        
        # Compute rule-level metrics
        rule_metrics = {}
        for rule_name, rule in self.rules.items():
            activation = rule_info['activations'][rule_name]
            penalty = rule_info['penalties'][rule_name]
            
            # Compute gradient magnitude for the rule
            grad_fn = jax.grad(lambda pos: rule.apply(pos, state))
            gradient = grad_fn(positions)
            grad_magnitude = float(jnp.linalg.norm(gradient))
            
            rule_metrics[rule_name] = PhiRuleMetrics(
                activation_frequency=float(activation),
                penalty_magnitude=float(penalty),
                gradient_magnitude=grad_magnitude,
                concept_drift=0.0  # TODO: Track over time
            )
        
        return PhiLayerMetrics(
            rule_metrics=rule_metrics,
            attention_weights=rule_info['attention_weights'],
            total_penalty=float(total_penalty),
            active_rules=self.get_active_rules(state),
            gradient_health={}  # TODO: Add gradient health metrics
        )


# Helper functions for common operations
def create_default_phi_layer(key: Optional[jax.random.PRNGKey] = None) -> PhiLayer:
    """Create a default Φ-layer with basic trading rules."""
    from .rules import create_basic_rule_set
    
    rules = create_basic_rule_set()
    return PhiLayer(rules, key=key)


def create_conservative_phi_layer(key: Optional[jax.random.PRNGKey] = None) -> PhiLayer:
    """Create a conservative Φ-layer with risk-focused rules."""
    from .rules import create_conservative_rule_set
    
    rules = create_conservative_rule_set()
    return PhiLayer(rules, decay_rate=0.95, key=key)  # Faster decay for conservative approach