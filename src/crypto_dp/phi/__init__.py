"""
Φ-layer (Phi-layer): Neuro-symbolic knowledge integration for E2E-DP.

This module implements the hybrid neuro-symbolic architecture described in CLAUDE.md,
combining end-to-end differentiable programming with explicit symbolic knowledge.

The Φ-layer provides:
1. Symbolic concept representation (rules, constraints, domain knowledge)
2. Differentiable penalty functions that shape the loss landscape
3. Bidirectional updates between symbolic and neural components
4. Interpretable explanations for learned behaviors

Architecture:
- PhiRule: Individual symbolic rules with differentiable penalties
- PhiLayer: Collection of rules with attention-based activation
- PhiGuidedLoss: Integration point with E2E-DP loss functions
"""

from .rules import PhiRule, VolatilityRule, RiskBudgetRule
from .layer import PhiLayer
from .integration import PhiGuidedLoss

__all__ = [
    'PhiRule',
    'VolatilityRule', 
    'RiskBudgetRule',
    'PhiLayer',
    'PhiGuidedLoss'
]