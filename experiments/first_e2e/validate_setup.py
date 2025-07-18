#!/usr/bin/env python3
"""
Quick validation script for first E2E experiment setup.

This script checks that all components are properly set up
without requiring heavy dependencies or network access.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core imports (these should work in basic Python environment)
        import json
        import time
        import datetime as dt
        import logging
        print("  ✓ Standard library imports")
        
        # Test project structure
        from src.crypto_dp.data.ingest import fetch_ohlcv, load_to_duck
        print("  ✓ Data ingestion module")
        
        from src.crypto_dp.models.portfolio import DifferentiablePortfolio
        print("  ✓ Portfolio model module")
        
        from src.crypto_dp.monitoring.gradient_health import EnhancedGradientMonitor
        print("  ✓ Gradient monitoring module")
        
        print("✅ All required modules import successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_directory_structure():
    """Test that directory structure is correct."""
    print("Testing directory structure...")
    
    required_dirs = [
        "src/crypto_dp/data",
        "src/crypto_dp/models", 
        "src/crypto_dp/monitoring",
        "src/crypto_dp/phi",
        "experiments/first_e2e",
        "artifacts"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
        else:
            print(f"  ✓ {dir_path}")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories present")
        return True


def test_files_exist():
    """Test that required files exist."""
    print("Testing required files...")
    
    required_files = [
        "experiments/first_e2e/setup_env.sh",
        "experiments/first_e2e/run_experiment.py",
        "src/tests/test_integration_real.py",
        "pyproject.toml",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True


def test_experiment_script():
    """Test that the experiment script is syntactically correct."""
    print("Testing experiment script syntax...")
    
    try:
        script_path = "experiments/first_e2e/run_experiment.py"
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Test that it's valid Python syntax
        compile(script_content, script_path, 'exec')
        print("  ✓ Script syntax is valid")
        
        # Check for key functions/sections
        key_components = [
            "def main():",
            "Data Ingestion",
            "Feature Pipeline", 
            "Train Differentiable Portfolio",
            "Backtest",
            "Visualization"
        ]
        
        for component in key_components:
            if component in script_content:
                print(f"  ✓ {component}")
            else:
                print(f"  ⚠️ Missing: {component}")
        
        print("✅ Experiment script structure looks good")
        return True
        
    except Exception as e:
        print(f"❌ Script validation failed: {e}")
        return False


def test_integration_test():
    """Test that integration test is syntactically correct."""
    print("Testing integration test syntax...")
    
    try:
        test_path = "src/tests/test_integration_real.py"
        with open(test_path, 'r') as f:
            test_content = f.read()
        
        compile(test_content, test_path, 'exec')
        print("  ✓ Integration test syntax is valid")
        
        # Check for pytest markers
        if "@pytest.mark.integration" in test_content:
            print("  ✓ Integration marker present")
        else:
            print("  ⚠️ Missing integration marker")
        
        if "test_real_data_flow" in test_content:
            print("  ✓ Real data flow test present")
        else:
            print("  ⚠️ Missing real data flow test")
        
        print("✅ Integration test structure looks good")
        return True
        
    except Exception as e:
        print(f"❌ Integration test validation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=== First E2E Experiment Setup Validation ===\n")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Required Files", test_files_exist),
        ("Module Imports", test_imports),
        ("Experiment Script", test_experiment_script),
        ("Integration Test", test_integration_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== Validation Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        print("\nYou're ready to run the first E2E experiment:")
        print("  1. source experiments/first_e2e/setup_env.sh")
        print("  2. python experiments/first_e2e/run_experiment.py")
        return True
    else:
        print("❌ Some validation tests failed.")
        print("Please fix the issues above before running the experiment.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)