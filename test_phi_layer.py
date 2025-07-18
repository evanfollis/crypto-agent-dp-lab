"""
Test script for the Φ-layer implementation.
"""

import sys
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_phi_rules():
    """Test basic Φ-rule functionality."""
    logger.info("Testing Φ-rules...")
    
    try:
        from src.crypto_dp.phi.rules import VolatilityRule, RiskBudgetRule, create_basic_rule_set
        
        # Test volatility rule
        vol_rule = VolatilityRule(vol_threshold=2.0, initial_weight=1.0)
        
        # Test state
        state = {
            'volatility': 2.5,  # High volatility
            'risk_budget': 1.0
        }
        positions = jnp.array([0.1, -0.05, 0.15])
        
        # Test rule components
        activation = vol_rule.trigger(state)
        penalty = vol_rule.penalty(positions, state)
        total_penalty = vol_rule.apply(positions, state)
        
        logger.info(f"  Volatility rule activation: {activation:.3f}")
        logger.info(f"  Volatility rule penalty: {penalty:.3f}")
        logger.info(f"  Total penalty: {total_penalty:.3f}")
        
        # Test explanation
        explanation = vol_rule.get_explanation(state)
        logger.info(f"  Explanation: {explanation}")
        
        # Test rule set creation
        rule_set = create_basic_rule_set()
        logger.info(f"  Created rule set with {len(rule_set)} rules")
        
        logger.info("✓ Φ-rules test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Φ-rules test failed: {e}")
        return False


def test_phi_layer():
    """Test Φ-layer functionality."""
    logger.info("Testing Φ-layer...")
    
    try:
        from src.crypto_dp.phi.layer import PhiLayer, create_default_phi_layer
        from src.crypto_dp.phi.rules import create_basic_rule_set
        
        # Create Φ-layer
        rules = create_basic_rule_set()
        phi_layer = PhiLayer(rules, key=jax.random.PRNGKey(42))
        
        # Test state
        state = {
            'volatility': 2.5,
            'risk_budget': 1.0,
            'momentum': 0.1,
            'expected_returns': jnp.array([0.02, -0.01, 0.03])
        }
        positions = jnp.array([0.1, -0.05, 0.15])
        
        # Test layer call
        total_penalty, rule_info = phi_layer(positions, state)
        
        logger.info(f"  Total penalty: {total_penalty:.3f}")
        logger.info(f"  Rule penalties: {rule_info['penalties']}")
        logger.info(f"  Rule activations: {rule_info['activations']}")
        
        # Test explanation
        explanation = phi_layer.explain_decision(positions, state)
        logger.info(f"  Explanation:\n{explanation}")
        
        # Test metrics
        metrics = phi_layer.compute_metrics(positions, state)
        logger.info(f"  Metrics summary: {metrics.get_summary()}")
        
        logger.info("✓ Φ-layer test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Φ-layer test failed: {e}")
        return False


def test_phi_integration():
    """Test Φ-layer integration with E2E-DP."""
    logger.info("Testing Φ-integration...")
    
    try:
        from src.crypto_dp.phi.integration import (
            PhiGuidedLoss, 
            create_minimal_phi_guided_loss,
            phi_sharpe_loss
        )
        
        # Create minimal Φ-guided loss
        phi_loss = create_minimal_phi_guided_loss(
            phi_sharpe_loss, 
            key=jax.random.PRNGKey(42)
        )
        
        # Test data
        positions = jnp.array([0.1, -0.05, 0.15])
        state = {
            'volatility': 2.5,
            'risk_budget': 1.0
        }
        returns = jnp.array([0.01, -0.005, 0.02, 0.01, -0.01])
        
        # Test loss computation
        total_loss, diagnostics = phi_loss(positions, state, returns)
        
        logger.info(f"  Total loss: {total_loss:.6f}")
        logger.info(f"  Base loss: {diagnostics['base_loss']:.6f}")
        logger.info(f"  Φ penalty: {diagnostics['phi_penalty']:.6f}")
        logger.info(f"  Φ weight: {diagnostics['phi_weight']:.3f}")
        
        # Test explanation
        explanation = phi_loss.get_explanation(positions, state)
        logger.info(f"  Explanation:\n{explanation}")
        
        # Test step progression
        phi_loss_stepped = phi_loss.step()
        logger.info(f"  Step count after step: {phi_loss_stepped.step_count}")
        
        logger.info("✓ Φ-integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Φ-integration test failed: {e}")
        return False


def test_phi_gradient_flow():
    """Test that gradients flow through Φ-layer properly."""
    logger.info("Testing Φ-gradient flow...")
    
    try:
        from src.crypto_dp.phi.integration import create_minimal_phi_guided_loss, phi_sharpe_loss
        
        # Create loss function
        phi_loss = create_minimal_phi_guided_loss(
            phi_sharpe_loss, 
            key=jax.random.PRNGKey(42)
        )
        
        # Test gradient computation
        def loss_fn(positions):
            state = {
                'volatility': 2.5,
                'risk_budget': 1.0
            }
            returns = jnp.array([0.01, -0.005, 0.02, 0.01, -0.01])
            total_loss, _ = phi_loss(positions, state, returns)
            return total_loss
        
        positions = jnp.array([0.1, -0.05, 0.15])
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(positions)
        
        logger.info(f"  Gradients: {gradients}")
        logger.info(f"  Gradient norm: {jnp.linalg.norm(gradients):.6f}")
        
        # Check that gradients are finite and non-zero
        assert jnp.all(jnp.isfinite(gradients)), "Gradients contain NaN or inf"
        assert jnp.linalg.norm(gradients) > 1e-8, "Gradients are too small"
        
        logger.info("✓ Φ-gradient flow test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Φ-gradient flow test failed: {e}")
        return False


def run_all_tests():
    """Run all Φ-layer tests."""
    logger.info("Running Φ-layer validation tests...")
    
    tests = [
        ("Φ-Rules", test_phi_rules),
        ("Φ-Layer", test_phi_layer),
        ("Φ-Integration", test_phi_integration),
        ("Φ-Gradient Flow", test_phi_gradient_flow)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n=== Testing {test_name} ===")
        try:
            results[test_name] = test_func()
            if results[test_name]:
                passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    logger.info(f"\n=== Test Results Summary ===")
    logger.info(f"Passed: {passed}/{len(tests)}")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} {test_name}")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)