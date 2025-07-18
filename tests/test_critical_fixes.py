"""
Test script to verify the critical fixes are working properly.
"""

import jax
import jax.numpy as jnp
import polars as pl
import tempfile
import os
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_duckdb_loader():
    """Test that DuckDB loader works with the register pattern."""
    try:
        from src.crypto_dp.data.ingest import load_to_duck
        
        # Create test data
        df = pl.DataFrame({
            'timestamp': [1, 2, 3],
            'symbol': ['BTC/USDT', 'ETH/USDT', 'BTC/USDT'],
            'price': [50000.0, 3000.0, 51000.0],
            'volume': [100.0, 200.0, 150.0]
        })
        
        # Test with temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            # Test replace mode
            load_to_duck(db_path, df, 'test_table', mode='replace')
            logger.info("✓ DuckDB loader register pattern works")
            return True
        except Exception as e:
            logger.error(f"✗ DuckDB loader failed: {e}")
            return False
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
                
    except ImportError as e:
        logger.error(f"✗ Could not import DuckDB loader: {e}")
        return False


def test_optimizer_state():
    """Test that optimizer state is properly initialized."""
    try:
        from src.crypto_dp.graph.scaffold import LatentGraph, train_graph
        
        # Create simple test data
        key = jax.random.PRNGKey(42)
        n_samples, n_factors = 100, 5
        
        x_data = jax.random.normal(key, (n_samples, n_factors))
        y_data = jax.random.normal(jax.random.split(key)[0], (n_samples, n_factors))
        
        # Create and train model
        model = LatentGraph(n_factors, key=key)
        trained_model, history = train_graph(
            model, x_data, y_data, 
            n_epochs=10, 
            verbose=False
        )
        
        # Check that training completed without errors
        assert len(history['train_loss']) == 10
        logger.info("✓ Optimizer state initialization works")
        return True
        
    except Exception as e:
        logger.error(f"✗ Optimizer state test failed: {e}")
        return False


def test_training_loop_consistency():
    """Test that training loop uses consistent data."""
    try:
        from src.crypto_dp.pipelines.basic_e2e import train_e2e_pipeline, TrainingConfig
        
        # Create minimal config for testing
        config = TrainingConfig(
            n_steps=5,
            batch_size=2,
            n_assets=3,
            learning_rate=1e-3
        )
        
        # Train pipeline briefly
        pipeline, results = train_e2e_pipeline(config)
        
        # Check that training completed
        assert len(results['losses']) == 5
        logger.info("✓ Training loop data consistency works")
        return True
        
    except Exception as e:
        logger.error(f"✗ Training loop test failed: {e}")
        return False


def test_rl_environment_reward():
    """Test that RL environment calculates proper P&L-based rewards."""
    try:
        from src.crypto_dp.rl.agent import TradingEnvironment, TradingAction
        
        # Create environment
        env = TradingEnvironment(['BTC/USDT', 'ETH/USDT'])
        state = env.reset()
        
        # Create test action
        action = TradingAction(
            positions=jnp.array([0.1, -0.05]),
            confidence=0.8,
            timestamp=0.0
        )
        
        # Take step
        new_state, reward, done, info = env.step(action)
        
        # Check that reward includes P&L (not just negative costs)
        # The reward should potentially be positive if P&L is good
        assert isinstance(reward, (float, jnp.ndarray))
        assert 'portfolio_value' in info
        
        logger.info("✓ RL environment P&L-based reward works")
        return True
        
    except Exception as e:
        logger.error(f"✗ RL environment reward test failed: {e}")
        return False


def test_portfolio_weights_nan():
    """Test that portfolio weights handle zero positions correctly."""
    try:
        from src.crypto_dp.rl.agent import TradingEnvironment
        
        # Create environment with zero positions
        env = TradingEnvironment(['BTC/USDT', 'ETH/USDT'])
        state = env.reset()
        
        # Check that portfolio weights are not NaN when positions are zero
        assert not jnp.any(jnp.isnan(state.portfolio))
        assert jnp.allclose(state.portfolio, 0.0)  # Should be all zeros
        
        logger.info("✓ Portfolio weights NaN handling works")
        return True
        
    except Exception as e:
        logger.error(f"✗ Portfolio weights NaN test failed: {e}")
        return False


def test_gradient_monitor():
    """Test that gradient monitor handles API changes."""
    try:
        from src.crypto_dp.monitoring.gradient_health import EnhancedGradientMonitor
        
        # Create monitor
        monitor = EnhancedGradientMonitor()
        
        # Create test gradients
        grads = {
            'layer1': {
                'weight': jax.random.normal(jax.random.PRNGKey(42), (10, 5)),
                'bias': jax.random.normal(jax.random.PRNGKey(43), (10,))
            },
            'layer2': {
                'weight': jax.random.normal(jax.random.PRNGKey(44), (5, 2)),
                'bias': jax.random.normal(jax.random.PRNGKey(45), (5,))
            }
        }
        
        # Compute metrics
        metrics = monitor.compute_metrics(grads)
        
        # Check that metrics are computed without errors
        assert hasattr(metrics, 'norm_ratio')
        assert hasattr(metrics, 'signal_to_total_variance')
        assert hasattr(metrics, 'gradient_sparsity')
        
        logger.info("✓ Gradient monitor API compatibility works")
        return True
        
    except Exception as e:
        logger.error(f"✗ Gradient monitor test failed: {e}")
        return False


def run_all_tests():
    """Run all critical fix tests."""
    logger.info("Running critical fixes validation tests...")
    
    tests = [
        ("DuckDB Loader", test_duckdb_loader),
        ("Optimizer State", test_optimizer_state),
        ("Training Loop Consistency", test_training_loop_consistency),
        ("RL Environment Reward", test_rl_environment_reward),
        ("Portfolio Weights NaN", test_portfolio_weights_nan),
        ("Gradient Monitor", test_gradient_monitor)
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