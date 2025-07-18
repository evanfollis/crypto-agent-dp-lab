#!/bin/bash
# Environment setup for first E2E experiment

set -e

echo "=== Setting up environment for first E2E experiment ==="

# Environment variables for API access (set empty for public endpoints)
export CCXT_BINANCE_API_KEY=""
export CCXT_BINANCE_SECRET=""
export COINGECKO_API_BASE="https://api.coingecko.com/api/v3"

# Experiment configuration
export EXPERIMENT_NAME="first_e2e"
export DB_PATH="artifacts/sandbox_crypto.db"
export PLOT_PATH="artifacts/backtest.png"
export RESULTS_PATH="artifacts/experiment_results.json"

echo "Environment variables set:"
echo "  EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "  DB_PATH: $DB_PATH"
echo "  PLOT_PATH: $PLOT_PATH"
echo "  RESULTS_PATH: $RESULTS_PATH"

# Check dependencies
echo "Checking Python dependencies..."

required_packages=(
    "jax"
    "equinox"
    "optax"
    "polars" 
    "duckdb"
    "matplotlib"
    "tqdm"
    "numpy"
)

missing_packages=()

for package in "${required_packages[@]}"; do
    if ! python -c "import $package" 2>/dev/null; then
        missing_packages+=("$package")
    else
        echo "  ✓ $package"
    fi
done

if [ ${#missing_packages[@]} -gt 0 ]; then
    echo "❌ Missing packages: ${missing_packages[*]}"
    echo "Please install missing packages before running experiment"
    exit 1
else
    echo "✅ All required packages available"
fi

# Create directories
mkdir -p artifacts
mkdir -p experiments/first_e2e/logs

echo "✅ Environment setup complete"
echo ""
echo "To run the experiment:"
echo "  1. source experiments/first_e2e/setup_env.sh"
echo "  2. python experiments/first_e2e/run_experiment.py"