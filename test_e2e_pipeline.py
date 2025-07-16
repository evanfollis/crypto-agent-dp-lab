#!/usr/bin/env python3
"""Test script for basic E2E-DP pipeline."""

import sys
sys.path.append('src')

from crypto_dp.pipelines.basic_e2e import (
    EndToEndDPPipeline, 
    TrainingConfig, 
    train_e2e_pipeline,
    generate_synthetic_market_data
)
import jax

if __name__ == "__main__":
    print("Testing basic E2E-DP pipeline...")
    
    # Quick test with small configuration
    config = TrainingConfig(
        n_steps=100,
        batch_size=8,
        learning_rate=1e-3,
        n_assets=3,
        feature_dim=16
    )
    
    print("Running training...")
    pipeline, results = train_e2e_pipeline(config)
    
    # Test inference
    print("\nTesting inference...")
    test_key = jax.random.PRNGKey(999)
    test_market = generate_synthetic_market_data(50, config.n_assets, test_key)
    
    test_return, intermediates = pipeline(test_market, test_key)
    
    print(f"✅ E2E-DP pipeline working!")
    print(f"Portfolio return: {test_return:.4f}")
    print(f"Portfolio weights sum: {intermediates['weights'].sum():.4f}")
    print(f"Health rate: {results['health_rate']:.1f}%")