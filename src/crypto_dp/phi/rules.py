"""
Φ-layer rules: Symbolic knowledge encoded as differentiable penalties.

Each rule represents a piece of domain knowledge that can be:
1. Evaluated as a soft constraint penalty
2. Updated based on gradient feedback
3. Interpreted for explanation generation
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PhiRuleMetrics:
    """Metrics for monitoring rule activation and effectiveness."""
    activation_frequency: float
    penalty_magnitude: float
    gradient_magnitude: float
    concept_drift: float
    
    def is_healthy(self) -> Tuple[bool, list]:
        """Check if rule is functioning properly."""
        issues = []
        
        if self.activation_frequency < 0.01:
            issues.append("Rule rarely activates")
        elif self.activation_frequency > 0.95:
            issues.append("Rule always activates")
            
        if self.penalty_magnitude < 1e-6:
            issues.append("Penalty too weak")
        elif self.penalty_magnitude > 100:
            issues.append("Penalty too strong")
            
        if self.gradient_magnitude < 1e-8:
            issues.append("No learning signal")
            
        return len(issues) == 0, issues


class PhiRule(eqx.Module, ABC):
    """
    Abstract base class for Φ-layer rules.
    
    A rule combines:
    - Symbolic concept (e.g., "high volatility")
    - Differentiable trigger function (smooth threshold)
    - Penalty function (shapes loss landscape)
    - Learnable weight (importance/strength)
    """
    
    weight: jnp.ndarray
    name: str
    
    def __init__(self, name: str, initial_weight: float = 1.0):
        self.name = name
        self.weight = jnp.array(initial_weight)
    
    @abstractmethod
    def trigger(self, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Evaluate rule activation based on market state.
        
        Args:
            state: Market state dictionary
            
        Returns:
            Activation strength [0, 1] (differentiable)
        """
        pass
    
    @abstractmethod
    def penalty(self, positions: jnp.ndarray, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Compute penalty for current positions given state.
        
        Args:
            positions: Current portfolio positions
            state: Market state dictionary
            
        Returns:
            Penalty value (positive = violation)
        """
        pass
    
    def apply(self, positions: jnp.ndarray, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Apply rule to current state and positions.
        
        Returns:
            Weighted penalty contribution to loss
        """
        activation = self.trigger(state)
        base_penalty = self.penalty(positions, state)
        
        # Smooth gating: only apply penalty when rule is triggered
        return self.weight * activation * base_penalty
    
    def get_explanation(self, state: Dict[str, jnp.ndarray]) -> str:
        """Generate human-readable explanation of rule activation."""
        activation = float(self.trigger(state))
        return f"{self.name}: {activation:.1%} activated (weight: {float(self.weight):.3f})"


class VolatilityRule(PhiRule):
    """
    Rule: "Reduce position size in high volatility regimes"
    
    Implements the minimal Φ-concept from CLAUDE.md Section 7.5.
    """
    
    vol_threshold: float
    steepness: float
    
    def __init__(
        self, 
        vol_threshold: float = 2.0, 
        steepness: float = 10.0,
        initial_weight: float = 1.0
    ):
        super().__init__("VolatilityRule", initial_weight)
        self.vol_threshold = vol_threshold
        self.steepness = steepness
    
    def trigger(self, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Smooth trigger based on volatility level.
        
        Args:
            state: Must contain 'volatility' key with market volatility
            
        Returns:
            Sigmoid activation based on volatility threshold
        """
        volatility = state.get('volatility', 0.0)
        # Smooth sigmoid trigger (replaces hard threshold)
        return jax.nn.sigmoid(self.steepness * (volatility - self.vol_threshold))
    
    def penalty(self, positions: jnp.ndarray, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Penalty proportional to position size squared.
        
        Encourages position reduction in high volatility.
        """
        risk_budget = state.get('risk_budget', 1.0)
        
        # L2 penalty on positions, normalized by risk budget
        return jnp.sum(positions ** 2) / risk_budget
    
    def get_explanation(self, state: Dict[str, jnp.ndarray]) -> str:
        """Explain volatility rule activation."""
        volatility = float(state.get('volatility', 0.0))
        activation = float(self.trigger(state))
        
        return (
            f"VolatilityRule: Market vol={volatility:.2f} (threshold={self.vol_threshold:.2f}), "
            f"activation={activation:.1%}, suggests {'reducing' if activation > 0.5 else 'maintaining'} positions"
        )


class RiskBudgetRule(PhiRule):
    """
    Rule: "Respect maximum risk budget allocation"
    
    Implements position limit constraints as soft penalties.
    """
    
    max_position_pct: float
    
    def __init__(self, max_position_pct: float = 0.2, initial_weight: float = 2.0):
        super().__init__("RiskBudgetRule", initial_weight)
        self.max_position_pct = max_position_pct
    
    def trigger(self, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Always active - risk limits always apply."""
        return jnp.array(1.0)
    
    def penalty(self, positions: jnp.ndarray, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Penalty for exceeding position limits.
        
        Uses smooth hinge loss for differentiability.
        """
        # Soft hinge loss for position limits
        excess = jnp.maximum(0, jnp.abs(positions) - self.max_position_pct)
        return jnp.sum(excess ** 2)
    
    def get_explanation(self, state: Dict[str, jnp.ndarray]) -> str:
        """Explain risk budget rule."""
        return (
            f"RiskBudgetRule: Max position {self.max_position_pct:.1%} per asset, "
            f"enforces portfolio concentration limits"
        )


class MomentumRule(PhiRule):
    """
    Rule: "Follow momentum in trending markets"
    
    Example of a more complex rule that considers market regime.
    """
    
    momentum_threshold: float
    trend_strength: float
    
    def __init__(
        self, 
        momentum_threshold: float = 0.05, 
        trend_strength: float = 5.0,
        initial_weight: float = 0.5
    ):
        super().__init__("MomentumRule", initial_weight)
        self.momentum_threshold = momentum_threshold
        self.trend_strength = trend_strength
    
    def trigger(self, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Trigger based on momentum strength.
        """
        momentum = state.get('momentum', 0.0)
        return jax.nn.sigmoid(self.trend_strength * (jnp.abs(momentum) - self.momentum_threshold))
    
    def penalty(self, positions: jnp.ndarray, state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Penalty for going against momentum.
        
        Encourages alignment with trend direction.
        """
        momentum = state.get('momentum', 0.0)
        expected_returns = state.get('expected_returns', jnp.zeros_like(positions))
        
        # Penalty for misalignment between positions and expected returns
        alignment = jnp.sum(positions * expected_returns)
        return jnp.maximum(0, -alignment)  # Penalty when negative alignment
    
    def get_explanation(self, state: Dict[str, jnp.ndarray]) -> str:
        """Explain momentum rule."""
        momentum = float(state.get('momentum', 0.0))
        activation = float(self.trigger(state))
        
        direction = "bullish" if momentum > 0 else "bearish"
        return (
            f"MomentumRule: Market momentum={momentum:.3f} ({direction}), "
            f"activation={activation:.1%}, suggests {'following' if activation > 0.5 else 'ignoring'} trend"
        )


# Factory function for creating common rule sets
def create_basic_rule_set() -> Dict[str, PhiRule]:
    """Create a basic set of trading rules."""
    return {
        'volatility': VolatilityRule(vol_threshold=2.0, initial_weight=1.0),
        'risk_budget': RiskBudgetRule(max_position_pct=0.2, initial_weight=2.0),
        'momentum': MomentumRule(momentum_threshold=0.05, initial_weight=0.5)
    }


def create_conservative_rule_set() -> Dict[str, PhiRule]:
    """Create a conservative trading rule set."""
    return {
        'volatility': VolatilityRule(vol_threshold=1.5, initial_weight=2.0),  # More sensitive
        'risk_budget': RiskBudgetRule(max_position_pct=0.1, initial_weight=3.0),  # Stricter limits
    }


def create_aggressive_rule_set() -> Dict[str, PhiRule]:
    """Create an aggressive trading rule set."""
    return {
        'volatility': VolatilityRule(vol_threshold=3.0, initial_weight=0.5),  # Less sensitive
        'risk_budget': RiskBudgetRule(max_position_pct=0.3, initial_weight=1.0),  # Looser limits
        'momentum': MomentumRule(momentum_threshold=0.02, initial_weight=1.5)  # More momentum focus
    }