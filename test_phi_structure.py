"""
Test script to validate Φ-layer structure without heavy dependencies.
"""

import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_phi_module_structure():
    """Test that Φ-layer module structure is correct."""
    logger.info("Testing Φ-layer module structure...")
    
    try:
        # Check that all files exist
        phi_dir = '/workspaces/crypto-agent-dp-lab/src/crypto_dp/phi'
        
        required_files = [
            '__init__.py',
            'rules.py',
            'layer.py',
            'integration.py'
        ]
        
        for file in required_files:
            file_path = os.path.join(phi_dir, file)
            if not os.path.exists(file_path):
                logger.error(f"Missing file: {file_path}")
                return False
            logger.info(f"✓ Found {file}")
        
        # Check that __init__.py has correct imports
        with open(os.path.join(phi_dir, '__init__.py'), 'r') as f:
            init_content = f.read()
        
        required_imports = [
            'PhiRule',
            'VolatilityRule',
            'RiskBudgetRule',
            'PhiLayer',
            'PhiGuidedLoss'
        ]
        
        for import_name in required_imports:
            if import_name not in init_content:
                logger.error(f"Missing import: {import_name}")
                return False
            logger.info(f"✓ Found import {import_name}")
        
        logger.info("✓ Φ-layer module structure test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Φ-layer module structure test failed: {e}")
        return False


def test_phi_rules_content():
    """Test that rules.py has the required content."""
    logger.info("Testing Φ-rules content...")
    
    try:
        with open('/workspaces/crypto-agent-dp-lab/src/crypto_dp/phi/rules.py', 'r') as f:
            rules_content = f.read()
        
        # Check for key classes and functions
        required_elements = [
            'class PhiRule',
            'class VolatilityRule',
            'class RiskBudgetRule',
            'def trigger',
            'def penalty',
            'def apply',
            'create_basic_rule_set',
            'vol_threshold',
            'jax.nn.sigmoid'
        ]
        
        for element in required_elements:
            if element not in rules_content:
                logger.error(f"Missing element in rules.py: {element}")
                return False
            logger.info(f"✓ Found {element}")
        
        logger.info("✓ Φ-rules content test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Φ-rules content test failed: {e}")
        return False


def test_phi_layer_content():
    """Test that layer.py has the required content."""
    logger.info("Testing Φ-layer content...")
    
    try:
        with open('/workspaces/crypto-agent-dp-lab/src/crypto_dp/phi/layer.py', 'r') as f:
            layer_content = f.read()
        
        # Check for key classes and functions
        required_elements = [
            'class PhiLayer',
            'attention_weights',
            'def __call__',
            'def update_attention',
            'def decay_weights',
            'def explain_decision',
            'def compute_metrics',
            'orthogonality_penalty',
            'create_default_phi_layer'
        ]
        
        for element in required_elements:
            if element not in layer_content:
                logger.error(f"Missing element in layer.py: {element}")
                return False
            logger.info(f"✓ Found {element}")
        
        logger.info("✓ Φ-layer content test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Φ-layer content test failed: {e}")
        return False


def test_phi_integration_content():
    """Test that integration.py has the required content."""
    logger.info("Testing Φ-integration content...")
    
    try:
        with open('/workspaces/crypto-agent-dp-lab/src/crypto_dp/phi/integration.py', 'r') as f:
            integration_content = f.read()
        
        # Check for key classes and functions
        required_elements = [
            'class PhiGuidedLoss',
            'L_total = L_dp + Σᵢ wᵢ · soft_penalty_i(θ)',
            'def __call__',
            'base_loss_fn',
            'phi_penalty',
            'curriculum_schedule',
            'orthogonality_penalty',
            'create_minimal_phi_guided_loss',
            'phi_sharpe_loss'
        ]
        
        for element in required_elements:
            if element not in integration_content:
                logger.error(f"Missing element in integration.py: {element}")
                return False
            logger.info(f"✓ Found {element}")
        
        logger.info("✓ Φ-integration content test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Φ-integration content test failed: {e}")
        return False


def test_phi_claude_md_alignment():
    """Test that implementation aligns with CLAUDE.md specifications."""
    logger.info("Testing alignment with CLAUDE.md specifications...")
    
    try:
        # Check that key concepts from CLAUDE.md are implemented
        
        # Read all phi files
        phi_files = []
        phi_dir = '/workspaces/crypto-agent-dp-lab/src/crypto_dp/phi'
        
        for file in ['rules.py', 'layer.py', 'integration.py']:
            with open(os.path.join(phi_dir, file), 'r') as f:
                phi_files.append(f.read())
        
        all_content = '\n'.join(phi_files)
        
        # Check for CLAUDE.md concepts
        claude_md_concepts = [
            'volatility regime',
            'soft penalty',
            'differentiable',
            'bidirectional',
            'curriculum',
            'orthogonality',
            'meta-learning',
            'gradient attribution',
            'attention weights',
            'symbolic knowledge',
            'neuro-symbolic'
        ]
        
        found_concepts = []
        for concept in claude_md_concepts:
            if concept.lower() in all_content.lower():
                found_concepts.append(concept)
                logger.info(f"✓ Found concept: {concept}")
        
        # Check coverage
        coverage = len(found_concepts) / len(claude_md_concepts)
        logger.info(f"CLAUDE.md concept coverage: {coverage:.1%}")
        
        if coverage < 0.7:
            logger.error("Insufficient alignment with CLAUDE.md")
            return False
        
        logger.info("✓ CLAUDE.md alignment test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ CLAUDE.md alignment test failed: {e}")
        return False


def run_all_tests():
    """Run all Φ-layer structure tests."""
    logger.info("Running Φ-layer structure validation tests...")
    
    tests = [
        ("Φ-Module Structure", test_phi_module_structure),
        ("Φ-Rules Content", test_phi_rules_content),
        ("Φ-Layer Content", test_phi_layer_content),
        ("Φ-Integration Content", test_phi_integration_content),
        ("CLAUDE.md Alignment", test_phi_claude_md_alignment)
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
    success = run_all_tests()
    exit(0 if success else 1)