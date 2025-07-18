"""
Integration test for real data flow - exercises the entire E2E pipeline.

This test validates that:
1. Real data can be fetched from exchanges
2. Data can be stored and retrieved from DuckDB
3. Differentiable portfolio model can be trained on real data
4. Backtest can be executed end-to-end

Marked with @pytest.mark.integration for CI separation.
"""

import pytest
import tempfile
import os
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
import duckdb
import jax
import jax.numpy as jnp
import numpy as np

from src.crypto_dp.data.ingest import fetch_ohlcv, load_to_duck
from src.crypto_dp.models.portfolio import DifferentiablePortfolio, backtest_portfolio


@pytest.mark.integration
@pytest.mark.network
def test_real_data_flow():
    """Test complete data flow from exchange to trained model."""
    
    # Configuration for lightweight test
    symbol = "BTC/USDT"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 7 * 24 * 60 * 60 * 1000  # 7 days
    
    # Step 1: Fetch real data from exchange
    try:
        df = fetch_ohlcv(symbol, start_ms, end_ms, "1h", "binance")
    except Exception as e:
        pytest.skip(f"Exchange API unavailable: {e}")
    
    assert not df.is_empty(), "No data returned from exchange"
    assert len(df) > 0, "Empty dataframe returned"
    assert "close" in df.columns, "Close price column missing"
    assert "timestamp" in df.columns, "Timestamp column missing"
    
    # Step 2: Store and retrieve from DuckDB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        # Store data
        load_to_duck(db_path, df, "ohlcv", "replace")
        
        # Verify storage
        con = duckdb.connect(db_path)
        rows = con.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        con.close()
        
        assert rows == len(df), f"Row count mismatch: {rows} != {len(df)}"
        
        # Step 3: Process data for model training
        prices = df["close"].to_numpy()
        returns = np.diff(prices) / prices[:-1]
        features = returns.reshape(-1, 1)  # Single asset
        
        assert len(features) > 10, "Insufficient data for training"
        assert not np.any(np.isnan(features)), "NaN values in features"
        
        # Step 4: Create and test differentiable portfolio model
        jax.config.update('jax_enable_x64', False)  # Use 32-bit for speed
        key = jax.random.PRNGKey(42)
        
        model = DifferentiablePortfolio(
            input_dim=1,
            n_assets=1,
            key=key
        )
        
        # Verify model can process features
        test_weights = model.scoring_network(features[0])
        assert test_weights.shape == (1,), f"Wrong weight shape: {test_weights.shape}"
        assert jnp.isfinite(test_weights).all(), "Non-finite weights generated"
        
        # Step 5: Run mini backtest
        if len(features) >= 24:  # Need at least 24 hours of data
            try:
                port_returns, weight_history, transaction_costs = backtest_portfolio(
                    model,
                    features,
                    returns.reshape(-1, 1),
                    lookback_window=min(24, len(features) // 2),
                    rebalance_freq=6
                )
                
                assert len(port_returns) > 0, "No portfolio returns generated"
                assert jnp.isfinite(port_returns).all(), "Non-finite portfolio returns"
                assert len(weight_history) > 0, "No weight history generated"
                
                # Calculate basic performance metrics
                cumulative_return = np.prod(1 + port_returns)
                assert cumulative_return > 0, "Invalid cumulative return"
                
            except Exception as e:
                pytest.fail(f"Backtest failed: {e}")
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.integration
def test_minimal_training_loop():
    """Test that a minimal training loop works with synthetic data."""
    
    # Create synthetic data that mimics real market data
    np.random.seed(42)
    n_timesteps = 100
    n_assets = 2
    
    # Generate correlated price returns
    returns = np.random.multivariate_normal(
        mean=[0.0001, 0.0001],  # Small positive drift
        cov=[[0.0004, 0.0002], [0.0002, 0.0004]],  # Realistic volatility
        size=n_timesteps
    )
    features = returns.copy()
    
    # Initialize model
    jax.config.update('jax_enable_x64', False)
    key = jax.random.PRNGKey(42)
    
    model = DifferentiablePortfolio(
        input_dim=n_assets,
        n_assets=n_assets,
        key=key
    )
    
    # Test that we can compute a portfolio step
    from src.crypto_dp.models.portfolio import portfolio_step
    
    try:
        updated_model, loss, diagnostics = portfolio_step(
            model,
            features[0],
            returns[:10],  # Small lookback
            learning_rate=1e-3
        )
        
        assert jnp.isfinite(loss), f"Non-finite loss: {loss}"
        assert hasattr(updated_model, 'scoring_network'), "Model structure corrupted"
        
        # Test that model parameters actually changed
        original_weights = model.scoring_network.weight
        updated_weights = updated_model.scoring_network.weight
        
        weight_change = jnp.linalg.norm(updated_weights - original_weights)
        assert weight_change > 1e-8, f"Parameters didn't update: {weight_change}"
        
    except Exception as e:
        pytest.fail(f"Portfolio step failed: {e}")


@pytest.mark.integration 
@pytest.mark.slow
def test_full_experiment_pipeline():
    """Test the full experiment pipeline with minimal data."""
    
    # This test simulates the complete experiment with smaller scope
    # to ensure the pipeline works end-to-end
    
    try:
        # Import the experiment module
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments" / "first_e2e"))
        
        # Create minimal synthetic dataset
        n_timesteps = 72  # 3 days of hourly data
        n_assets = 2
        
        # Generate synthetic OHLCV data
        np.random.seed(42)
        timestamps = [int(time.time() * 1000) - i * 3600000 for i in range(n_timesteps)]
        timestamps.reverse()
        
        data = []
        symbols = ['BTC/USDT', 'ETH/USDT']
        
        for i, symbol in enumerate(symbols):
            base_price = 50000 if i == 0 else 3000
            prices = base_price * np.exp(np.cumsum(np.random.normal(0, 0.01, n_timesteps)))
            
            for t, (timestamp, price) in enumerate(zip(timestamps, prices)):
                data.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'timeframe': '1h',
                    'exchange': 'binance',
                    'open': price * 0.999,
                    'high': price * 1.001,
                    'low': price * 0.998,
                    'close': price,
                    'volume': np.random.uniform(100, 1000),
                    'datetime': timestamp // 1000
                })
        
        df = pl.DataFrame(data)
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            # Store data
            load_to_duck(db_path, df, "ohlcv", "replace")
            
            # Process data (minimal feature pipeline)
            con = duckdb.connect(db_path)
            prices_df = con.execute("""
                SELECT datetime, symbol, close
                FROM ohlcv
                WHERE timeframe = '1h'
                ORDER BY datetime
            """).df()
            con.close()
            
            pivot_df = (pl.from_pandas(prices_df)
                       .pivot(index='datetime', columns='symbol', values='close')
                       .sort('datetime'))
            
            price_matrix = pivot_df.select(pl.exclude('datetime')).to_numpy()
            returns = np.diff(price_matrix, axis=0) / price_matrix[:-1]
            features = returns.copy()
            
            # Mini training loop
            jax.config.update('jax_enable_x64', False)
            key = jax.random.PRNGKey(42)
            
            model = DifferentiablePortfolio(
                input_dim=n_assets,
                n_assets=n_assets,
                key=key
            )
            
            # Train for just a few steps
            from src.crypto_dp.models.portfolio import portfolio_step
            
            losses = []
            for epoch in range(10):
                t_idx = epoch % (len(features) - 1)
                lookback_returns = returns[max(0, t_idx-10):t_idx+1]
                
                model, loss, _ = portfolio_step(
                    model,
                    features[t_idx],
                    lookback_returns,
                    learning_rate=1e-3
                )
                losses.append(float(loss))
            
            # Verify training worked
            assert len(losses) == 10, "Training loop incomplete"
            assert all(jnp.isfinite(l) for l in losses), "Non-finite losses during training"
            
            # Mini backtest
            port_returns, _, _ = backtest_portfolio(
                model,
                features,
                returns,
                lookback_window=10,
                rebalance_freq=6
            )
            
            assert len(port_returns) > 0, "Backtest produced no returns"
            assert jnp.isfinite(port_returns).all(), "Non-finite backtest returns"
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
                
    except Exception as e:
        pytest.fail(f"Full pipeline test failed: {e}")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])