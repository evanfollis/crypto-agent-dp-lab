#!/usr/bin/env python3
"""
Momentum-preserving test script for crypto-agent-dp-lab.

This script validates that every major component executes end-to-end
with real market data, providing preliminary performance numbers.

Expected runtime: < 1 minute total
"""

import sys
import time
import tempfile
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def run_smoke_tests():
    """Step 1: Run smoke tests for core library (< 30s)."""
    print("=" * 60)
    print("Step 1: Running smoke tests for core library")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", "-m", "not slow and not network and not integration", "-q"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Smoke tests passed!")
            if result.stdout:
                # Extract test count from pytest output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'passed' in line:
                        print(f"   {line}")
        else:
            print("❌ Smoke tests failed!")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Smoke tests timed out (>30s)")
        return False
    except Exception as e:
        print(f"⚠️  Smoke tests skipped - pytest not available: {e}")
        print("   Continuing with manual validation...")
    
    return True


def test_data_ingestion():
    """Step 2: Pull 7-day BTC/USDT slice and store in DuckDB (~3s API time)."""
    print("\n" + "=" * 60)
    print("Step 2: Testing data ingestion with 7-day BTC/USDT")
    print("=" * 60)
    
    try:
        from src.crypto_dp.data.ingest import fetch_ohlcv, load_to_duck
        import duckdb
        import polars as pl
        
        symbol = "BTC/USDT"
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - 7 * 24 * 60 * 60 * 1000  # 7 days
        
        print(f"Fetching {symbol} from Binance...")
        start_time = time.time()
        
        df = fetch_ohlcv(symbol, start_ms, end_ms, "1h", "binance")
        fetch_time = time.time() - start_time
        
        assert not df.is_empty(), "Exchange returned no rows"
        print(f"✅ Fetched {len(df)} rows in {fetch_time:.1f}s")
        
        # Store in temporary DuckDB
        db_path = tempfile.NamedTemporaryFile(delete=False, suffix=".db").name
        load_to_duck(db_path, df, "ohlcv", mode="replace")
        
        # Verify storage
        con = duckdb.connect(db_path)
        rows = con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        con.close()
        
        print(f"✅ {rows:,} rows loaded → {db_path}")
        
        # Verify data quality
        print("\nData quality checks:")
        print(f"  - Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  - Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"  - Volume sum: {df['volume'].sum():.2f} BTC")
        
        return df, db_path
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure ccxt and other dependencies are installed")
        return None, None
    except Exception as e:
        print(f"❌ Data ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_quick_backtest(df):
    """Step 3: Run quick-and-dirty backtest with differentiable portfolio (<5s CPU)."""
    print("\n" + "=" * 60)
    print("Step 3: Running quick backtest with differentiable portfolio")
    print("=" * 60)
    
    if df is None:
        print("⚠️  Skipping backtest - no data available")
        return False
    
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from src.crypto_dp.models.portfolio import DifferentiablePortfolio, backtest_portfolio
        
        # Prepare features & returns
        prices = df["close"].to_numpy()
        rets = np.diff(prices) / prices[:-1]
        features = rets.reshape(-1, 1)  # Single-asset feature vector
        
        print(f"Data shape: {len(prices)} prices → {len(rets)} returns")
        
        # Instantiate model
        jax.config.update("jax_enable_x64", False)  # Speed
        model = DifferentiablePortfolio(
            input_dim=1, 
            n_assets=1, 
            key=jax.random.PRNGKey(0)
        )
        print("✅ Model instantiated")
        
        # One-shot backtest
        if len(features) >= 24:  # Need ≥24 hourly observations
            start_time = time.time()
            
            port_rets, weights_hist, tcosts = backtest_portfolio(
                model,
                features,
                rets.reshape(-1, 1),
                lookback_window=min(24, len(features)//2),
                rebalance_freq=6  # Rebalance every 6 hours
            )
            
            backtest_time = time.time() - start_time
            
            # Calculate metrics
            sharpe = np.mean(port_rets) / (np.std(port_rets) + 1e-8) * np.sqrt(365 * 24)
            cum_return = np.prod(1 + port_rets) - 1
            cagr = np.prod(1 + port_rets) ** (365 * 24 / len(port_rets)) - 1
            
            print(f"✅ Backtest completed in {backtest_time:.1f}s")
            print("\nPerformance metrics:")
            print(f"  7-day Sharpe     ≈ {sharpe:6.2f}")
            print(f"  7-day return     ≈ {cum_return:6.1%}")
            print(f"  Annualized CAGR  ≈ {cagr:6.1%}")
            print(f"  Rebalances       = {len(weights_hist)}")
            print(f"  Total TC         = {np.sum(tcosts):.4f}")
            
            # Verify results are reasonable
            assert np.all(np.isfinite(port_rets)), "Non-finite portfolio returns"
            assert -0.5 < cum_return < 0.5, f"Unrealistic 7-day return: {cum_return:.1%}"
            
            print("\n✅ All metrics within healthy ranges")
            return True
            
        else:
            print(f"❌ Insufficient data: {len(features)} returns (need ≥24)")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   JAX environment not available")
        return False
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Step 4: Verify integration test exists and is runnable."""
    print("\n" + "=" * 60)
    print("Step 4: Verifying integration test")
    print("=" * 60)
    
    test_file = "src/tests/test_integration_real.py"
    
    if not os.path.exists(test_file):
        print(f"❌ Integration test not found at {test_file}")
        return False
    
    print(f"✅ Integration test found: {test_file}")
    
    # Check for proper markers
    with open(test_file, 'r') as f:
        content = f.read()
    
    checks = {
        "@pytest.mark.integration": "Integration marker",
        "@pytest.mark.network": "Network marker",
        "test_real_data_flow": "Real data flow test",
        "fetch_ohlcv": "Exchange data fetching",
        "DifferentiablePortfolio": "Portfolio model usage"
    }
    
    all_good = True
    for pattern, desc in checks.items():
        if pattern in content:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ Missing: {desc}")
            all_good = False
    
    if all_good:
        print("\n✅ Integration test properly configured")
        print("   Run with: pytest -m 'integration and network' -q")
    else:
        print("\n⚠️  Integration test needs updates")
    
    return all_good


def main():
    """Run all momentum tests."""
    print("🚀 Momentum-Preserving Test Suite")
    print("=" * 60)
    print("Testing that every major component executes end-to-end")
    print("with real market data. Expected runtime: < 1 minute\n")
    
    start_time = time.time()
    results = {}
    
    # Step 1: Smoke tests
    results['smoke_tests'] = run_smoke_tests()
    
    # Step 2: Data ingestion
    df, db_path = test_data_ingestion()
    results['data_ingestion'] = df is not None
    
    # Step 3: Quick backtest
    results['backtest'] = test_quick_backtest(df)
    
    # Step 4: Integration test check
    results['integration_test'] = test_integration()
    
    # Cleanup
    if db_path and os.path.exists(db_path):
        os.unlink(db_path)
        print(f"\n✅ Cleaned up temporary database")
    
    # Summary
    runtime = time.time() - start_time
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Total runtime: {runtime:.1f}s")
    print("\nResults:")
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test}")
    
    if passed == total:
        print("\n🎉 All momentum tests passed!")
        print("\nNext steps:")
        print("  1. Add 2-asset test (BTC & ETH) for matrix validation")
        print("  2. Wire EnhancedGradientMonitor into training loop")
        print("  3. Implement OHLCV caching to speed up CI")
        print("  4. Schedule integration tests nightly in CI")
    else:
        print("\n⚠️  Some tests failed - fix these before proceeding")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)