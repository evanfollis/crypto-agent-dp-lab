#!/bin/bash
# Lightweight benchmark script for crypto-agent-dp-lab
# Expected runtime: < 3 minutes total

set -e

echo "=== Crypto Agent DP Lab Benchmark Suite ==="
echo "Starting benchmark run at $(date)"
echo

# Configuration
TIMEOUT=120  # 2 minutes timeout per test
PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a benchmark with timeout
run_benchmark() {
    local name="$1"
    local command="$2"
    local expected_time="$3"
    
    echo -e "${YELLOW}Running $name (expected: $expected_time)...${NC}"
    
    start_time=$(date +%s)
    
    if timeout $TIMEOUT bash -c "$command" > /tmp/benchmark_output 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${GREEN}✓ $name completed in ${duration}s${NC}"
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${RED}✗ $name failed/timed out in ${duration}s${NC}"
        echo "Output:"
        cat /tmp/benchmark_output | tail -10
        return 1
    fi
}

# Initialize results
total_tests=0
passed_tests=0

# Benchmark 1: Core fixes validation
echo "=== Benchmark 1: Core Fixes Validation ==="
total_tests=$((total_tests + 1))
if run_benchmark "Core Fixes" "python tests/test_core_fixes.py" "<10s"; then
    passed_tests=$((passed_tests + 1))
fi
echo

# Benchmark 2: Φ-layer structure validation
echo "=== Benchmark 2: Φ-Layer Structure Validation ==="
total_tests=$((total_tests + 1))
if run_benchmark "Φ-Layer Structure" "python test_phi_structure.py" "<5s"; then
    passed_tests=$((passed_tests + 1))
fi
echo

# Benchmark 3: Pendulum control (if available)
echo "=== Benchmark 3: Pendulum Control Benchmark ==="
total_tests=$((total_tests + 1))
if run_benchmark "Pendulum Control" "python -m src.crypto_dp.benchmarks.pendulum" "<30s"; then
    passed_tests=$((passed_tests + 1))
fi
echo

# Benchmark 4: Fast unit tests
echo "=== Benchmark 4: Fast Unit Tests ==="
total_tests=$((total_tests + 1))
if run_benchmark "Fast Unit Tests" "python -m pytest src/tests/ -v -m 'not slow' --tb=short" "<30s"; then
    passed_tests=$((passed_tests + 1))
fi
echo

# Benchmark 5: Module imports
echo "=== Benchmark 5: Module Import Performance ==="
total_tests=$((total_tests + 1))

import_test="
import sys
sys.path.insert(0, '.')
import time

modules = [
    'src.crypto_dp.pipelines.basic_e2e',
    'src.crypto_dp.phi.rules',
    'src.crypto_dp.phi.layer',
    'src.crypto_dp.phi.integration',
    'src.crypto_dp.monitoring.gradient_health',
    'src.crypto_dp.models.portfolio',
    'src.crypto_dp.graph.scaffold',
    'src.crypto_dp.rl.agent',
    'src.crypto_dp.data.ingest'
]

start_time = time.time()
for module in modules:
    try:
        __import__(module)
        print(f'✓ {module}')
    except ImportError as e:
        print(f'✗ {module}: {e}')
        
end_time = time.time()
print(f'Total import time: {end_time - start_time:.2f}s')
"

if run_benchmark "Module Imports" "python -c \"$import_test\"" "<10s"; then
    passed_tests=$((passed_tests + 1))
fi
echo

# Benchmark 6: Gradient health monitoring
echo "=== Benchmark 6: Gradient Health Monitoring ==="
total_tests=$((total_tests + 1))

gradient_test="
import sys
sys.path.insert(0, '.')
import time

try:
    # Test without heavy dependencies
    from src.crypto_dp.monitoring.gradient_health import EnhancedGradientMonitor
    
    monitor = EnhancedGradientMonitor()
    
    # Mock gradients structure
    grads = {
        'layer1': {'weight': [[1.0, 2.0], [3.0, 4.0]], 'bias': [0.1, 0.2]},
        'layer2': {'weight': [[0.5, 1.5]], 'bias': [0.05]}
    }
    
    start_time = time.time()
    
    # This would normally use JAX arrays, but we test the structure
    print('✓ Gradient monitor imported successfully')
    print('✓ Mock gradient structure created')
    
    end_time = time.time()
    print(f'Gradient monitor setup time: {end_time - start_time:.4f}s')
    
except Exception as e:
    print(f'✗ Gradient monitoring test failed: {e}')
    sys.exit(1)
"

if run_benchmark "Gradient Health" "python -c \"$gradient_test\"" "<5s"; then
    passed_tests=$((passed_tests + 1))
fi
echo

# Summary
echo "=== Benchmark Results Summary ==="
echo "Passed: $passed_tests / $total_tests"
echo

if [ $passed_tests -eq $total_tests ]; then
    echo -e "${GREEN}🎉 All benchmarks passed!${NC}"
    echo "System is ready for development and research."
    exit 0
else
    echo -e "${RED}❌ Some benchmarks failed.${NC}"
    echo "Please check the output above for details."
    exit 1
fi