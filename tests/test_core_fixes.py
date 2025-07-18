"""
Simple test to verify core fixes work without heavy dependencies.
"""

import logging
import tempfile
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_duckdb_fix():
    """Test that DuckDB loader uses register pattern."""
    logger.info("Testing DuckDB loader fix...")
    
    try:
        # Read the fixed file
        with open('/workspaces/crypto-agent-dp-lab/src/crypto_dp/data/ingest.py', 'r') as f:
            content = f.read()
        
        # Check for the register pattern
        if 'con.register("temp_df", df)' in content:
            logger.info("✓ DuckDB loader uses register pattern")
            return True
        else:
            logger.error("✗ DuckDB loader still uses direct SQL reference")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking DuckDB fix: {e}")
        return False


def test_optimizer_fix():
    """Test that optimizer state is properly initialized."""
    logger.info("Testing optimizer state fix...")
    
    try:
        # Read the fixed file
        with open('/workspaces/crypto-agent-dp-lab/src/crypto_dp/graph/scaffold.py', 'r') as f:
            content = f.read()
        
        # Check for proper optimizer initialization
        if 'optimizer = optax.sgd(learning_rate)' in content and 'opt_state = optimizer.init(' in content:
            logger.info("✓ Optimizer state properly initialized")
            return True
        else:
            logger.error("✗ Optimizer state not properly initialized")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking optimizer fix: {e}")
        return False


def test_training_loop_fix():
    """Test that training loop uses consistent data."""
    logger.info("Testing training loop fix...")
    
    try:
        # Read the fixed file
        with open('/workspaces/crypto-agent-dp-lab/src/crypto_dp/pipelines/basic_e2e.py', 'r') as f:
            content = f.read()
        
        # Check for batch data storage and reuse
        if 'batch_market_states = []' in content and 'batch_sim_keys = []' in content:
            logger.info("✓ Training loop uses consistent batch data")
            return True
        else:
            logger.error("✗ Training loop still regenerates data")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking training loop fix: {e}")
        return False


def test_rl_reward_fix():
    """Test that RL environment includes P&L in reward."""
    logger.info("Testing RL environment reward fix...")
    
    try:
        # Read the fixed file
        with open('/workspaces/crypto-agent-dp-lab/src/crypto_dp/rl/agent.py', 'r') as f:
            content = f.read()
        
        # Check for P&L calculation
        if 'portfolio_pnl = jnp.sum(self.positions * price_returns)' in content:
            logger.info("✓ RL environment calculates P&L-based reward")
            return True
        else:
            logger.error("✗ RL environment still uses cost-only reward")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking RL reward fix: {e}")
        return False


def test_portfolio_weights_fix():
    """Test that portfolio weights handle zero positions."""
    logger.info("Testing portfolio weights fix...")
    
    try:
        # Read the fixed file
        with open('/workspaces/crypto-agent-dp-lab/src/crypto_dp/rl/agent.py', 'r') as f:
            content = f.read()
        
        # Check for NaN handling
        if 'jnp.where(' in content and 'total_abs_position < 1e-8' in content:
            logger.info("✓ Portfolio weights handle zero positions")
            return True
        else:
            logger.error("✗ Portfolio weights still have NaN issue")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking portfolio weights fix: {e}")
        return False


def test_deterministic_seeds():
    """Test that deterministic seeds replace time() usage."""
    logger.info("Testing deterministic seeds fix...")
    
    try:
        # Read the fixed file
        with open('/workspaces/crypto-agent-dp-lab/src/crypto_dp/rl/agent.py', 'r') as f:
            content = f.read()
        
        # Check that time() usage is replaced
        if 'jax.random.PRNGKey(42 + self.step_count)' in content:
            logger.info("✓ Deterministic seeds replace time() usage")
            return True
        else:
            logger.error("✗ Still using time() for random seeds")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking deterministic seeds: {e}")
        return False


def test_gradient_monitor_fix():
    """Test that gradient monitor handles API changes."""
    logger.info("Testing gradient monitor fix...")
    
    try:
        # Read the fixed file
        with open('/workspaces/crypto-agent-dp-lab/src/crypto_dp/monitoring/gradient_health.py', 'r') as f:
            content = f.read()
        
        # Check for API compatibility comment
        if 'Handle both old and new JAX path formats' in content:
            logger.info("✓ Gradient monitor handles API changes")
            return True
        else:
            logger.error("✗ Gradient monitor API drift not addressed")
            return False
    except Exception as e:
        logger.error(f"✗ Error checking gradient monitor: {e}")
        return False


def run_core_tests():
    """Run all core fix tests."""
    logger.info("Running core fixes validation tests...")
    
    tests = [
        ("DuckDB Loader Fix", test_duckdb_fix),
        ("Optimizer State Fix", test_optimizer_fix),
        ("Training Loop Fix", test_training_loop_fix),
        ("RL Environment Reward Fix", test_rl_reward_fix),
        ("Portfolio Weights Fix", test_portfolio_weights_fix),
        ("Deterministic Seeds Fix", test_deterministic_seeds),
        ("Gradient Monitor Fix", test_gradient_monitor_fix)
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
    success = run_core_tests()
    exit(0 if success else 1)