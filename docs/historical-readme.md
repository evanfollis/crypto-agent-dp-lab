# Crypto Agent DP Lab

#### Part of my broader AI-native research ecosystem. This repo explores systematic trading using the same design principles as SynaPlex.
A research platform for **End-to-End Differentiable Programming (E2E-DP)** applied to algorithmic trading, with hybrid neuro-symbolic architecture.

## 🎯 Project Vision

This project investigates whether end-to-end differentiable programming, enhanced with symbolic knowledge integration (Φ-layer), can achieve statistically-significant performance improvements on complex sequential decision tasks where modular baselines are known to plateau.

**Key Hypothesis**: Holistic optimization via unified gradient descent will achieve both statistical significance (effect size ≥0.3) and economic significance (≥10 bps IR uplift on $100M notional simulation).

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    E2E-DP Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│  Raw Data → Feature Extractor → Predictor → Decision Maker →   │
│  Simulator → Loss                                               │
│      ↑___________________|_______________|______________↓        │
│                    Gradients flow end-to-end                    │
└─────────────────────────────────────────────────────────────────┘
                                    ↕
┌─────────────────────────────────────────────────────────────────┐
│                    Φ-Layer (Symbolic Knowledge)                 │
├─────────────────────────────────────────────────────────────────┤
│  • Volatility Rules        • Risk Budget Rules                 │
│  • Momentum Rules          • Attention Weights                 │
│  • Differentiable Penalties • Explanations                     │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11.10+
- Poetry for dependency management
- CUDA-capable GPU (optional, for acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/crypto-agent-dp-lab.git
cd crypto-agent-dp-lab

# Install dependencies
poetry install

# Run core fixes validation
python test_core_fixes.py

# Run Φ-layer structure test
python test_phi_structure.py
```

### Docker Development

```bash
# For CUDA development
docker compose up -d

# For CPU-only development
docker compose -f docker-compose.cpu.yml up -d
```

## 📊 Running Experiments

### Phase 1: Proof of Concept

```bash
# Run basic E2E-DP pipeline
python src/crypto_dp/pipelines/basic_e2e.py

# Run with Φ-layer integration
python -c "
from src.crypto_dp.phi.integration import create_minimal_phi_guided_loss, phi_sharpe_loss
from src.crypto_dp.pipelines.basic_e2e import train_e2e_pipeline, TrainingConfig

# Test Φ-guided training
phi_loss = create_minimal_phi_guided_loss(phi_sharpe_loss)
print('Φ-layer integration ready!')
"
```

### Phase 1.5: Φ-Layer Integration

```bash
# Test individual Φ-rules
python -c "
from src.crypto_dp.phi.rules import VolatilityRule, create_basic_rule_set
rule = VolatilityRule(vol_threshold=2.0)
print('Volatility rule created')
"

# Test full Φ-layer
python test_phi_layer.py  # Requires JAX environment
```

## 🧪 Testing

### Fast Tests (< 30 seconds)

```bash
# Run core functionality tests
poetry run pytest src/tests/ -v -m "not slow"

# Run critical fixes validation
python test_core_fixes.py
```

### Heavy Tests (GPU/Network required)

```bash
# Run integration tests
poetry run pytest src/tests/ -v -m "slow"

# Run full E2E pipeline
python test_e2e_pipeline.py
```

### Continuous Integration

The project uses GitHub Actions with three-tier testing:

1. **Lint**: Code quality and style checks
2. **Fast Tests**: Unit tests without heavy dependencies
3. **Heavy Tests**: Integration tests with GPU/network requirements

## 📁 Project Structure

```
crypto-agent-dp-lab/
├── src/crypto_dp/
│   ├── pipelines/          # E2E-DP training pipelines
│   │   └── basic_e2e.py   # Main E2E pipeline implementation
│   ├── phi/               # Φ-layer (neuro-symbolic) components
│   │   ├── rules.py       # Symbolic trading rules
│   │   ├── layer.py       # Attention-based rule aggregation
│   │   └── integration.py # Φ-guided loss functions
│   ├── models/            # Portfolio optimization models
│   ├── rl/               # Reinforcement learning agents
│   ├── graph/            # Latent graph structures
│   ├── data/             # Data ingestion and processing
│   └── monitoring/       # Gradient health diagnostics
├── tests/                # Test files
├── scripts/              # Utility scripts
└── .github/workflows/    # CI/CD configuration
```

## 🔬 Key Research Components

### 1. End-to-End Differentiable Pipeline

- **File**: `src/crypto_dp/pipelines/basic_e2e.py`
- **Purpose**: Main E2E-DP implementation with gradient flow monitoring
- **Features**: Feature extraction, prediction, decision making, simulation

### 2. Φ-Layer (Neuro-Symbolic Integration)

- **Files**: `src/crypto_dp/phi/`
- **Purpose**: Symbolic knowledge integration with differentiable penalties
- **Features**: Volatility rules, attention weights, curriculum learning

### 3. Gradient Health Monitoring

- **File**: `src/crypto_dp/monitoring/gradient_health.py`
- **Purpose**: Real-time gradient diagnostics following CLAUDE.md specs
- **Features**: Norm ratios, sparsity tracking, early warning systems

### 4. Portfolio Optimization

- **File**: `src/crypto_dp/models/portfolio.py`
- **Purpose**: Differentiable portfolio construction with risk management
- **Features**: Sharpe optimization, transaction costs, concentration penalties

## 📈 Performance Metrics

The project tracks several key metrics aligned with research objectives:

### Statistical Significance
- **Target**: Hedges' g effect size ≥0.3 on Sharpe ratio
- **Measurement**: Paired bootstrap with 95% CI

### Economic Significance
- **Target**: ≥10 bps IR uplift on $100M notional
- **Measurement**: Out-of-sample backtesting

### Gradient Health
- **Target**: Norm ratio ∈ [0.1, 10]
- **Measurement**: Real-time monitoring during training

### Interpretability
- **Target**: ≥40% overlap in top-10 Jacobian features
- **Measurement**: Comparison with domain expert rankings

## 🛠️ Development Workflow

### Code Quality

```bash
# Run linting
poetry run black .
poetry run isort .
poetry run ruff check .

# Run type checking
poetry run mypy src/crypto_dp/

# Run pre-commit hooks
poetry run pre-commit run --all-files
```

### Adding New Features

1. **Write tests first** (TDD approach)
2. **Implement feature** with proper type hints
3. **Add Φ-layer integration** if applicable
4. **Update documentation** and examples
5. **Run full test suite** before committing

### Debugging

```bash
# Enable JAX debugging
export JAX_DEBUG_NANS=True
export JAX_DISABLE_JIT=True

# Run with gradient monitoring
export CRYPTO_DP_GRADIENT_MONITORING=True
```

## 🧠 Research Phases

### Phase 1: Proof of Concept ✅
- **Status**: Complete
- **Deliverable**: E2E gradients flow through trading system
- **Key Result**: Gradient health monitoring active

### Phase 1.5: Φ-Layer Integration ✅
- **Status**: Complete
- **Deliverable**: Single volatility rule integrated
- **Key Result**: Differentiable symbolic penalties working

### Phase 2: Comparative Study (Next)
- **Status**: Planned
- **Deliverable**: Rigorous comparison vs baselines
- **Target**: Effect size ≥0.3 with p<0.05

### Phase 3: Scalability Analysis
- **Status**: Planned
- **Deliverable**: High-frequency optimization
- **Target**: <1ms inference, stable training

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/your-feature`
3. **Run tests**: `poetry run pytest`
4. **Commit changes**: `git commit -m "Add your feature"`
5. **Push branch**: `git push origin feature/your-feature`
6. **Open Pull Request**

### Code Style

- Follow PEP 8 (enforced by Black)
- Use type hints for all functions
- Add docstrings for public APIs
- Keep functions focused and small
- Test coverage >90% for new code

## 📚 Documentation

- **CLAUDE.md**: Complete research plan and theoretical foundation
- **IMPLEMENTATION_SUMMARY.md**: Technical implementation details
- **CRITICAL_FIXES_SUMMARY.md**: Recent bug fixes and improvements

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🔗 Related Work

- **"Implicit Differentiation in Machine Learning"** (Blondel et al., 2021)
- **"Nonsmooth Implicit Differentiation"** (Pedregosa et al., NeurIPS 2021)
- **"Differentiable Optimization Layers"** (Amos & Kolter, 2017)
- **"JAX-LOB: GPU-Accelerated Limit Order Book Simulation"** (Frey et al., 2023)

## 🎓 Citation

```bibtex
@software{crypto_agent_dp_lab,
  title={Crypto Agent DP Lab: End-to-End Differentiable Programming for Algorithmic Trading},
  author={Research Team},
  year={2024},
  url={https://github.com/your-org/crypto-agent-dp-lab}
}
```

---

**Built with**: JAX, Equinox, Optax, DuckDB, Polars, and ❤️ for differentiable programming research.
