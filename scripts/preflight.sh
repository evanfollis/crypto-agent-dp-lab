#!/bin/bash
set -euo pipefail

echo "🔍 Pre-flight checks starting..."

# Check Python version compatibility
REQ="3.11"
cur=$(python -c 'import sys;print(".".join(map(str,sys.version_info[:2])))')
if [[ "$cur" == "$REQ" ]]; then
    echo "✅ Python version $cur OK"
else
    echo "⚠️  Python $cur may have compatibility issues (expected $REQ)"
fi

# Warn about RC versions
python -c "import sys; rc='rc' in sys.version; exit(0 if not rc else 1)" || {
    echo "⚠️  Python release candidate detected - some packages may fail to install"
    echo "   See PYTHON_VERSION_FIX.md for instructions"
}

# Check Poetry is available
command -v poetry >/dev/null || { echo "❌ Poetry missing"; exit 127; }
echo "✅ Poetry found: $(poetry --version)"

# Check Python version
if command -v python >/dev/null 2>&1; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo "✅ Python found: $PYTHON_VERSION"
    
    # Ensure Python 3.11
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" || {
        echo "❌ Python 3.11+ required"
        exit 1
    }
else
    echo "❌ Python not found"
    exit 1
fi

# Driver and GPU availability checks
if [ -f /proc/driver/nvidia/version ]; then
    echo "🔍 Checking CUDA driver match..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "✅ NVIDIA driver available"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
        
        # Check torch.cuda availability if torch is installed
        python -c "
import sys
try:
    import torch
    if torch.cuda.is_available():
        print('✅ torch.cuda available')
        print(f'   CUDA devices: {torch.cuda.device_count()}')
    else:
        print('⚠️  torch.cuda not available')
except ImportError:
    print('⚠️  PyTorch not installed yet')
" 2>/dev/null || echo "⚠️  Could not check torch.cuda"
    else
        echo "❌ nvidia-smi not found"
        exit 1
    fi
elif [ "$(uname)" = "Darwin" ]; then
    echo "🔍 Checking Metal/MPS availability..."
    python -c "
import sys
try:
    import torch
    if torch.backends.mps.is_available():
        print('✅ MPS available')
    else:
        print('⚠️  MPS not available')
except ImportError:
    print('⚠️  PyTorch not installed yet')
" 2>/dev/null || echo "⚠️  Could not check MPS"
fi

# Test poetry install (dry run)
if [ -f "pyproject.toml" ]; then
    echo "🔍 Testing poetry install..."
    poetry check || { echo "❌ pyproject.toml invalid"; exit 1; }
    poetry install --dry-run >/dev/null 2>&1 || { echo "❌ Poetry install would fail"; exit 1; }
    echo "✅ Poetry install test passed"
else
    echo "⚠️  pyproject.toml not found, skipping poetry install test"
fi

echo "🔍 Poetry install dry-run"
poetry run python - <<'PY'
import sys, platform
print("Python:", sys.version.split()[0], "Platform:", platform.system())
try:
    import jax, numpy
    print("✅ Core packages available")
except ImportError as e:
    print(f"⚠️  Missing package: {e}")
PY
echo "✅ Pre-flight checks complete"