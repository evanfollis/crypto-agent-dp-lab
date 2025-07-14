#!/bin/bash
set -euo pipefail

echo "🔍 Running code format checks..."

# Check if we're in a poetry environment
if ! command -v poetry >/dev/null 2>&1; then
    echo "❌ Poetry not found"
    exit 1
fi

# Run black format check
echo "🔍 Checking code formatting with black..."
if ! poetry run black --check --diff src/ 2>/dev/null; then
    echo "❌ Black formatting issues found"
    exit 1
fi
echo "✅ Black formatting OK"

# Run isort import order check  
echo "🔍 Checking import order with isort..."
if ! poetry run isort --check-only --diff src/ 2>/dev/null; then
    echo "❌ Import order issues found"
    exit 1
fi
echo "✅ Import order OK"

# Run flake8 linting
echo "🔍 Running flake8 linting..."
if ! poetry run flake8 src/ 2>/dev/null; then
    echo "❌ Flake8 linting issues found"
    exit 1
fi
echo "✅ Flake8 linting OK"

echo "✅ All format checks passed"