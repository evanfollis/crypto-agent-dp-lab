# First End-to-End Experiment

This experiment implements the concrete, narrow-scope first E2E experiment proposed in the technical review. It generates real numbers while keeping risk, runtime, and code changes minimal.

## Overview

The experiment fetches real cryptocurrency data, trains a differentiable portfolio model, and produces a backtest with performance visualization.

**Expected Runtime**: < 5 minutes  
**Expected Outputs**: Database, trained model, performance plot, metrics JSON

## Quick Start

```bash
# 1. Set up environment
source experiments/first_e2e/setup_env.sh

# 2. Run the experiment
python experiments/first_e2e/run_experiment.py

# 3. Check results
ls artifacts/
```

## Expected Artifacts

| File | Description |
|------|-------------|
| `artifacts/sandbox_crypto.db` | DuckDB with genuine OHLCV & market-cap data |
| `artifacts/backtest.png` | Performance visualization with metrics |
| `artifacts/experiment_results.json` | Complete experiment metrics and metadata |
| `experiments/first_e2e/logs/experiment.log` | Detailed execution log |

## Configuration

The experiment is configured for a minimal first run:

- **Assets**: BTC/USDT, ETH/USDT, BNB/USDT
- **Timeframe**: 30 days of hourly data
- **Training**: 300 epochs with Adam optimizer
- **Backtest**: 10.5-day lookback, 6-hour rebalancing

## Expected Results

Based on the scope and synthetic testing:

```
Data ingestion: ~2,160 OHLCV rows (30 days × 3 assets × 24 hours)
Training: Final Sharpe loss ≈ -1.12, Gradient health ≈ 95%
Backtest: CAGR ≈ 20-80% (highly variable due to short timeframe)
```

## Integration Testing

Run the integration tests to verify the pipeline:

```bash
# Test real data flow (requires network)
pytest src/tests/test_integration_real.py::test_real_data_flow -v -s

# Test complete pipeline (offline)
pytest src/tests/test_integration_real.py::test_full_experiment_pipeline -v -s

# Run all integration tests
pytest -m integration -v -s
```

## Data Sources

- **CCXT (Binance)**: Public REST API for OHLCV data
- **CoinGecko**: Public API for market cap and volume data

No API keys required - uses public endpoints with rate limiting.

## Guard Rails

The experiment includes several safety measures:

1. **Timeout Protection**: Each section has reasonable timeouts
2. **Error Handling**: Graceful degradation when APIs are unavailable
3. **Data Validation**: Comprehensive checks for data quality
4. **Gradient Monitoring**: Real-time health checks during training
5. **Resource Limits**: Small dataset and short training for first run

## Troubleshooting

### Common Issues

**ImportError**: Ensure you're running from the project root directory
```bash
cd /path/to/crypto-agent-dp-lab
python experiments/first_e2e/run_experiment.py
```

**Network Issues**: The experiment will skip unavailable data sources and continue
```
WARNING - CoinGecko fetch failed: Rate limit exceeded
INFO - Continuing with OHLCV data only
```

**Insufficient Data**: Minimum 100 timesteps required for training
```
ERROR - Insufficient data points: 45. Need at least 100.
```

### Environment Issues

Check dependencies:
```bash
python -c "import jax, equinox, polars, duckdb, matplotlib; print('✅ All dependencies available')"
```

Check project structure:
```bash
ls src/crypto_dp/  # Should show: data, models, monitoring, phi, etc.
```

## Next Steps

After successful completion:

1. **Week +1**: Add technical indicators (RSI, Bollinger Bands) to features
2. **Week +2**: Integrate RiskManager with position limits and stop-losses  
3. **Week +3**: Switch to RL agent for decision layer
4. **Week +4**: Implement Φ-layer integration for hybrid system

## Performance Baselines

The experiment establishes baseline metrics for comparison:

- **Data Pipeline**: Throughput and error rates
- **Model Training**: Convergence speed and gradient health
- **Backtest Performance**: Risk-adjusted returns vs. buy-and-hold

These baselines enable measurement of improvements from future enhancements.

## Files Structure

```
experiments/first_e2e/
├── README.md              # This file
├── setup_env.sh           # Environment setup script
├── run_experiment.py      # Main experiment script
└── logs/                  # Execution logs
    └── experiment.log

artifacts/                 # Generated outputs
├── sandbox_crypto.db      # Market data storage
├── backtest.png          # Performance visualization  
└── experiment_results.json # Metrics and metadata
```