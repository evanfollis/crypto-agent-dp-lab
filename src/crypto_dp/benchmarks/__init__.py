"""Micro-benchmarks for E2E-DP validation."""

from .pendulum import (
    PendulumDynamics,
    DifferentiableController,
    GradientHealthMonitor,
    train_pendulum_controller
)

__all__ = [
    'PendulumDynamics',
    'DifferentiableController',
    'GradientHealthMonitor',
    'train_pendulum_controller'
]