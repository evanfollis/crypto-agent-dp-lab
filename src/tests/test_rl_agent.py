"""
Tests for RL trading agent module.
"""

import pytest
import jax
import jax.numpy as jnp
import time
from unittest.mock import patch

from crypto_dp.rl.agent import (
    TradingAction,
    MarketState,
    RiskLevel,
    TradingEnvironment,
    CryptoTradingAgent,
    RiskManager,
    train_agent
)


class TestDataStructures:
    """Test data structures and enums."""
    
    def test_trading_action(self):
        """Test TradingAction creation."""
        positions = jnp.array([0.1, -0.2, 0.3])
        action = TradingAction(
            positions=positions,
            confidence=0.8,
            timestamp=time.time()
        )
        
        assert jnp.array_equal(action.positions, positions)
        assert action.confidence == 0.8
        assert isinstance(action.timestamp, float)
    
    def test_market_state(self):
        """Test MarketState creation."""
        state = MarketState(
            prices=jnp.array([100.0, 200.0, 50.0]),
            features=jnp.array([1.0, 2.0, 3.0]),
            portfolio=jnp.array([0.3, 0.4, 0.3]),
            cash=10000.0,
            timestamp=time.time()
        )
        
        assert state.prices.shape == (3,)
        assert state.features.shape == (3,)
        assert state.portfolio.shape == (3,)
        assert state.cash == 10000.0
    
    def test_risk_level(self):
        """Test RiskLevel enum."""
        assert RiskLevel.LOW.value == 0.5
        assert RiskLevel.MEDIUM.value == 1.0
        assert RiskLevel.HIGH.value == 1.5
        assert RiskLevel.CRITICAL.value == 2.0


class TestTradingEnvironment:
    """Test trading environment functionality."""
    
    def test_environment_initialization(self):
        """Test trading environment initialization."""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        env = TradingEnvironment(symbols, initial_cash=50000.0)
        
        assert env.symbols == symbols
        assert env.n_assets == 3
        assert env.initial_cash == 50000.0
        assert env.cash == 50000.0
    
    def test_environment_reset(self):
        """Test environment reset."""
        symbols = ['BTC/USDT', 'ETH/USDT']
        env = TradingEnvironment(symbols)
        
        # Modify environment state
        env.cash = 50000.0
        env.step_count = 10
        
        # Reset
        initial_state = env.reset()
        
        assert env.cash == env.initial_cash
        assert env.step_count == 0
        assert isinstance(initial_state, MarketState)
        assert initial_state.prices.shape == (2,)
    
    def test_environment_step(self):
        """Test environment step function."""
        symbols = ['BTC/USDT', 'ETH/USDT']
        env = TradingEnvironment(symbols)
        env.reset()
        
        action = TradingAction(
            positions=jnp.array([0.1, -0.1]),
            confidence=0.7,
            timestamp=time.time()
        )
        
        new_state, reward, done, info = env.step(action)
        
        assert isinstance(new_state, MarketState)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check info dictionary
        assert 'transaction_cost' in info
        assert 'slippage' in info
        assert 'portfolio_value' in info
        assert 'step_count' in info
    
    def test_environment_termination(self):
        """Test environment termination conditions."""
        symbols = ['BTC/USDT']
        env = TradingEnvironment(symbols, max_position_size=0.1)
        env.reset()
        
        # Create action that violates position limit
        large_action = TradingAction(
            positions=jnp.array([0.5]),  # Exceeds max_position_size
            confidence=0.5,
            timestamp=time.time()
        )
        
        _, _, done, _ = env.step(large_action)
        assert done  # Should terminate due to position limit violation


class TestCryptoTradingAgent:
    """Test crypto trading agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        state_dim = 10
        action_dim = 3
        key = jax.random.PRNGKey(42)
        
        agent = CryptoTradingAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            key=key
        )
        
        assert hasattr(agent, 'policy_network')
        assert hasattr(agent, 'value_network')
        assert hasattr(agent, 'risk_module')
    
    def test_agent_get_action_deterministic(self):
        """Test deterministic action generation."""
        state_dim = 5
        action_dim = 3
        key = jax.random.PRNGKey(42)
        
        agent = CryptoTradingAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            key=key
        )
        
        state = jax.random.normal(key, (state_dim,))
        action = agent.get_action(state, deterministic=True)
        
        assert isinstance(action, TradingAction)
        assert action.positions.shape == (action_dim,)
        assert jnp.isfinite(action.positions).all()
        assert isinstance(action.confidence, float)
        
        # Deterministic actions should be repeatable
        action2 = agent.get_action(state, deterministic=True)
        assert jnp.allclose(action.positions, action2.positions)
    
    def test_agent_get_action_stochastic(self):
        """Test stochastic action generation."""
        state_dim = 5
        action_dim = 3
        key = jax.random.PRNGKey(42)
        
        agent = CryptoTradingAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            key=key
        )
        
        state = jax.random.normal(key, (state_dim,))
        action1 = agent.get_action(state, deterministic=False, key=jax.random.PRNGKey(1))
        action2 = agent.get_action(state, deterministic=False, key=jax.random.PRNGKey(2))
        
        # Stochastic actions should be different
        assert not jnp.allclose(action1.positions, action2.positions, atol=1e-6)
    
    def test_agent_get_value(self):
        """Test state value estimation."""
        state_dim = 5
        action_dim = 3
        key = jax.random.PRNGKey(42)
        
        agent = CryptoTradingAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            key=key
        )
        
        state = jax.random.normal(key, (state_dim,))
        value = agent.get_value(state)
        
        assert isinstance(value, float)
        assert jnp.isfinite(value)
    
    def test_agent_market_neutral_constraint(self):
        """Test that agent generates approximately market-neutral positions."""
        state_dim = 5
        action_dim = 4
        key = jax.random.PRNGKey(42)
        
        agent = CryptoTradingAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            key=key
        )
        
        state = jax.random.normal(key, (state_dim,))
        action = agent.get_action(state, deterministic=True)
        
        # Positions should sum to approximately zero (market neutral)
        assert jnp.abs(jnp.sum(action.positions)) < 0.1


class TestRiskManager:
    """Test risk management functionality."""
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        rm = RiskManager(
            max_drawdown=0.05,
            max_daily_loss=0.02,
            max_position_size=0.15
        )
        
        assert rm.max_drawdown == 0.05
        assert rm.max_daily_loss == 0.02
        assert rm.max_position_size == 0.15
    
    def test_risk_check_safe_action(self):
        """Test risk check with safe action."""
        rm = RiskManager()
        
        action = TradingAction(
            positions=jnp.array([0.1, -0.05, 0.08]),
            confidence=0.8,
            timestamp=time.time()
        )
        
        is_safe, warnings, modified_action = rm.check_risk_limits(
            action, portfolio_value=105000.0, volatility=0.5
        )
        
        assert is_safe
        assert len(warnings) == 0
        assert jnp.allclose(modified_action.positions, action.positions)
    
    def test_risk_check_position_size_violation(self):
        """Test risk check with position size violation."""
        rm = RiskManager(max_position_size=0.1)
        
        action = TradingAction(
            positions=jnp.array([0.3, -0.2, 0.1]),  # First position too large
            confidence=0.8,
            timestamp=time.time()
        )
        
        is_safe, warnings, modified_action = rm.check_risk_limits(
            action, portfolio_value=100000.0, volatility=0.5
        )
        
        assert not is_safe
        assert len(warnings) > 0
        assert jnp.max(jnp.abs(modified_action.positions)) <= rm.max_position_size
    
    def test_risk_check_drawdown_violation(self):
        """Test risk check with drawdown violation."""
        rm = RiskManager(max_drawdown=0.05)
        rm.peak_portfolio_value = 100000.0
        
        action = TradingAction(
            positions=jnp.array([0.1, -0.05, 0.08]),
            confidence=0.8,
            timestamp=time.time()
        )
        
        # Portfolio value below drawdown limit
        is_safe, warnings, modified_action = rm.check_risk_limits(
            action, portfolio_value=90000.0, volatility=0.5  # 10% drawdown
        )
        
        assert not is_safe
        assert any("drawdown" in warning.lower() for warning in warnings)
        # Should force flat positions
        assert jnp.allclose(modified_action.positions, 0.0)
    
    def test_risk_check_leverage_violation(self):
        """Test risk check with leverage violation."""
        rm = RiskManager(max_leverage=1.5)
        
        action = TradingAction(
            positions=jnp.array([0.8, -0.9, 0.6]),  # Total exposure = 2.3x
            confidence=0.8,
            timestamp=time.time()
        )
        
        is_safe, warnings, modified_action = rm.check_risk_limits(
            action, portfolio_value=100000.0, volatility=0.5
        )
        
        assert not is_safe
        assert any("leverage" in warning.lower() for warning in warnings)
        # Total exposure should be scaled down
        total_exposure = jnp.sum(jnp.abs(modified_action.positions))
        assert total_exposure <= rm.max_leverage + 1e-6
    
    def test_risk_check_high_volatility(self):
        """Test risk check with high volatility."""
        rm = RiskManager(volatility_threshold=1.0)
        
        action = TradingAction(
            positions=jnp.array([0.1, -0.05, 0.08]),
            confidence=0.8,
            timestamp=time.time()
        )
        
        is_safe, warnings, modified_action = rm.check_risk_limits(
            action, portfolio_value=100000.0, volatility=2.0  # High volatility
        )
        
        assert not is_safe
        assert any("volatility" in warning.lower() for warning in warnings)
        # Positions should be scaled down
        assert jnp.all(jnp.abs(modified_action.positions) < jnp.abs(action.positions))


class TestTraining:
    """Test agent training functionality."""
    
    @patch('crypto_dp.rl.agent.logger')
    def test_train_agent_basic(self, mock_logger):
        """Test basic agent training."""
        symbols = ['BTC/USDT', 'ETH/USDT']
        env = TradingEnvironment(symbols)
        
        state_dim = len(symbols) + 20 + len(symbols) + 1
        action_dim = len(symbols)
        
        agent = CryptoTradingAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            key=jax.random.PRNGKey(42)
        )
        
        # Short training run for testing
        trained_agent, history = train_agent(
            agent, env,
            n_episodes=5,
            learning_rate=1e-3,
            verbose=False
        )
        
        assert isinstance(trained_agent, CryptoTradingAgent)
        assert isinstance(history, dict)
        assert 'episode_rewards' in history
        assert 'episode_lengths' in history
        assert 'policy_losses' in history
        assert len(history['episode_rewards']) == 5
    
    def test_train_agent_history_structure(self):
        """Test training history structure."""
        symbols = ['BTC/USDT']
        env = TradingEnvironment(symbols)
        
        state_dim = len(symbols) + 20 + len(symbols) + 1
        action_dim = len(symbols)
        
        agent = CryptoTradingAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            key=jax.random.PRNGKey(42)
        )
        
        trained_agent, history = train_agent(
            agent, env,
            n_episodes=3,
            verbose=False
        )
        
        # Check history structure
        required_keys = ['episode_rewards', 'episode_lengths', 'policy_losses']
        for key in required_keys:
            assert key in history
            assert len(history[key]) == 3
            assert all(isinstance(x, (int, float)) for x in history[key])


class TestIntegration:
    """Test integration between components."""
    
    def test_agent_environment_interaction(self):
        """Test agent interacting with environment."""
        symbols = ['BTC/USDT', 'ETH/USDT']
        env = TradingEnvironment(symbols)
        
        state_dim = len(symbols) + 20 + len(symbols) + 1
        action_dim = len(symbols)
        
        agent = CryptoTradingAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            key=jax.random.PRNGKey(42)
        )
        
        # Reset environment and get initial state
        state = env.reset()
        
        # Convert state to array for agent
        state_array = jnp.concatenate([
            state.prices,
            state.features,
            state.portfolio,
            jnp.array([state.cash])
        ])
        
        # Agent generates action
        action = agent.get_action(state_array, deterministic=True)
        
        # Environment processes action
        new_state, reward, done, info = env.step(action)
        
        assert isinstance(new_state, MarketState)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_agent_risk_manager_integration(self):
        """Test agent with risk manager."""
        state_dim = 5
        action_dim = 3
        key = jax.random.PRNGKey(42)
        
        agent = CryptoTradingAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            key=key
        )
        
        risk_manager = RiskManager(max_position_size=0.1)
        
        state = jax.random.normal(key, (state_dim,))
        raw_action = agent.get_action(state, deterministic=True)
        
        # Apply risk management
        is_safe, warnings, safe_action = risk_manager.check_risk_limits(
            raw_action, portfolio_value=100000.0, volatility=0.5
        )
        
        # Risk manager should modify action if needed
        assert isinstance(safe_action, TradingAction)
        assert jnp.max(jnp.abs(safe_action.positions)) <= risk_manager.max_position_size


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_environment(self):
        """Test environment with no symbols."""
        with pytest.raises((ValueError, IndexError)):
            TradingEnvironment([])
    
    def test_agent_zero_dimensions(self):
        """Test agent with zero dimensions."""
        with pytest.raises((ValueError, IndexError)):
            CryptoTradingAgent(state_dim=0, action_dim=0)
    
    def test_extreme_risk_parameters(self):
        """Test risk manager with extreme parameters."""
        # Very restrictive risk manager
        rm = RiskManager(
            max_drawdown=0.001,
            max_position_size=0.001,
            max_leverage=0.1
        )
        
        action = TradingAction(
            positions=jnp.array([0.1, -0.1]),
            confidence=0.5,
            timestamp=time.time()
        )
        
        is_safe, warnings, modified_action = rm.check_risk_limits(
            action, portfolio_value=100000.0, volatility=0.5
        )
        
        # Should heavily restrict positions
        assert not is_safe
        assert len(warnings) > 0
        assert jnp.max(jnp.abs(modified_action.positions)) <= rm.max_position_size


if __name__ == "__main__":
    pytest.main([__file__])