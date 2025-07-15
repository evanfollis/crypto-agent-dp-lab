"""
Reinforcement learning agent for crypto trading.

This module implements the core RL loop for training and deploying
crypto trading agents with built-in risk management and execution
simulation capabilities.
"""

from typing import Dict, Any, Tuple, Optional, List, NamedTuple
import logging
from enum import Enum
from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import jit, vmap


logger = logging.getLogger(__name__)


class TradingAction(NamedTuple):
    """Trading action with position sizes and metadata."""
    positions: jnp.ndarray  # Position sizes for each asset
    confidence: float       # Confidence in the action
    timestamp: float       # Action timestamp


class MarketState(NamedTuple):
    """Market state representation."""
    prices: jnp.ndarray          # Current asset prices
    features: jnp.ndarray        # Market features
    portfolio: jnp.ndarray       # Current portfolio weights
    cash: float                  # Available cash
    timestamp: float             # State timestamp


class RiskLevel(Enum):
    """Risk level indicators for position sizing."""
    LOW = 0.5
    MEDIUM = 1.0
    HIGH = 1.5
    CRITICAL = 2.0


@dataclass
class TradingEnvironment:
    """
    Crypto trading environment for RL training.
    
    Simulates realistic trading conditions including:
    - Transaction costs
    - Slippage
    - Latency
    - Market impact
    """
    
    symbols: List[str]
    initial_cash: float = 100000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005    # 0.05%
    max_position_size: float = 0.2   # 20% max per asset
    lookback_window: int = 100
    
    def __post_init__(self):
        if not self.symbols:
            raise ValueError("At least one symbol required")
        self.n_assets = len(self.symbols)
        self.reset()
    
    def reset(self) -> MarketState:
        """Reset environment to initial state."""
        self.cash = self.initial_cash
        self.positions = jnp.zeros(self.n_assets)
        self.portfolio_value = self.initial_cash
        self.step_count = 0
        
        return self._get_state()
    
    def _get_state(self) -> MarketState:
        """Get current market state."""
        # Placeholder - in practice, would fetch real market data
        prices = jnp.ones(self.n_assets)  # Dummy prices
        features = jnp.zeros(20)          # Dummy features
        portfolio_weights = self.positions / jnp.sum(jnp.abs(self.positions) + 1e-8)
        
        return MarketState(
            prices=prices,
            features=features,
            portfolio=portfolio_weights,
            cash=self.cash,
            timestamp=time.time()
        )
    
    def step(self, action: TradingAction) -> Tuple[MarketState, float, bool, Dict[str, Any]]:
        """
        Execute trading action and return new state.
        
        Args:
            action: Trading action to execute
        
        Returns:
            new_state, reward, done, info
        """
        # Simulate transaction costs and slippage
        new_positions = action.positions
        position_change = jnp.abs(new_positions - self.positions)
        
        # Calculate transaction costs
        transaction_cost = jnp.sum(position_change) * self.transaction_cost
        
        # Calculate slippage (simplified)
        slippage = jnp.sum(position_change) * self.slippage_rate
        
        # Update positions
        self.positions = new_positions
        
        # Calculate reward (placeholder - would use actual P&L)
        # For now, use negative of costs as immediate reward
        reward = -(transaction_cost + slippage)
        
        # Update cash (simplified)
        self.cash -= transaction_cost + slippage
        
        self.step_count += 1
        
        # Environment termination conditions
        done = (
            self.step_count >= 1000 or  # Max steps
            self.cash < 0 or            # Margin call
            jnp.any(jnp.abs(self.positions) > self.max_position_size)  # Position limit
        )
        
        info = {
            'transaction_cost': transaction_cost,
            'slippage': slippage,
            'portfolio_value': self.portfolio_value,
            'step_count': self.step_count
        }
        
        return self._get_state(), float(reward), done, info


class CryptoTradingAgent(eqx.Module):
    """
    Deep RL agent for crypto trading using JAX/Equinox.
    """
    
    policy_network: eqx.nn.MLP
    value_network: eqx.nn.MLP
    risk_module: eqx.nn.MLP
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        key: Optional[jax.random.PRNGKey] = None
    ):
        """
        Initialize crypto trading agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of assets)
            hidden_dims: Hidden layer dimensions
            key: Random key for initialization
        """
        if key is None:
            key = jax.random.PRNGKey(42)
        
        keys = jax.random.split(key, 3)
        
        # Policy network (outputs raw position scores)
        self.policy_network = eqx.nn.MLP(
            in_size=state_dim,
            out_size=action_dim,
            width_size=hidden_dims[0],
            depth=len(hidden_dims),
            key=keys[0]
        )
        
        # Value network (state value estimation)
        self.value_network = eqx.nn.MLP(
            in_size=state_dim,
            out_size=1,
            width_size=hidden_dims[0] // 2,
            depth=len(hidden_dims) - 1,
            key=keys[1]
        )
        
        # Risk assessment module
        self.risk_module = eqx.nn.MLP(
            in_size=state_dim,
            out_size=4,  # Risk scores for different factors
            width_size=hidden_dims[0] // 4,
            depth=2,
            key=keys[2]
        )
    
    def __call__(self, state: jnp.ndarray) -> TradingAction:
        """Generate trading action from current state."""
        return self.get_action(state)
    
    def get_action(
        self,
        state: jnp.ndarray,
        deterministic: bool = False,
        key: Optional[jax.random.PRNGKey] = None
    ) -> TradingAction:
        """
        Generate trading action with risk-adjusted position sizing.
        
        Args:
            state: Current market state
            deterministic: Whether to use deterministic policy
            key: Random key for stochastic actions
        
        Returns:
            Trading action
        """
        # Get raw policy scores
        raw_scores = self.policy_network(state)
        
        # Risk assessment
        risk_scores = self.risk_module(state)
        risk_level = jnp.mean(jnp.abs(risk_scores))
        
        # Risk-adjusted position sizing
        max_position = 0.2 / (1.0 + risk_level)  # Lower max when risk is high
        
        if deterministic:
            # Deterministic action (for evaluation)
            positions = jnp.tanh(raw_scores) * max_position
        else:
            # Stochastic action (for exploration)
            if key is None:
                key = jax.random.PRNGKey(int(time.time()))
            
            noise = jax.random.normal(key, raw_scores.shape) * 0.1
            noisy_scores = raw_scores + noise
            positions = jnp.tanh(noisy_scores) * max_position
        
        # Ensure positions sum to approximately zero (market neutral)
        positions = positions - jnp.mean(positions)
        
        # Calculate confidence based on score magnitude
        confidence = float(jnp.mean(jnp.abs(raw_scores)))
        
        return TradingAction(
            positions=positions,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def get_value(self, state: jnp.ndarray) -> float:
        """Estimate state value."""
        value = self.value_network(state)
        return float(value.squeeze())


class RiskManager:
    """
    Risk management system with configurable limits and kill switches.
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.08,
        max_daily_loss: float = 0.02,
        max_position_size: float = 0.2,
        max_leverage: float = 2.0,
        volatility_threshold: float = 2.0
    ):
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.volatility_threshold = volatility_threshold
        
        # Risk state tracking
        self.peak_portfolio_value = 100000.0
        self.daily_start_value = 100000.0
        self.last_reset_time = time.time()
    
    def check_risk_limits(
        self,
        action: TradingAction,
        portfolio_value: float,
        volatility: float
    ) -> Tuple[bool, List[str], TradingAction]:
        """
        Check if action violates risk limits.
        
        Args:
            action: Proposed trading action
            portfolio_value: Current portfolio value
            volatility: Current portfolio volatility
        
        Returns:
            (is_safe, warnings, modified_action)
        """
        warnings = []
        modified_positions = action.positions.copy()
        
        # Update peak value
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
        
        # Check drawdown
        current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        if current_drawdown > self.max_drawdown:
            warnings.append(f"Max drawdown exceeded: {current_drawdown:.2%}")
            # Force flat positions
            modified_positions = jnp.zeros_like(modified_positions)
        
        # Check daily loss
        daily_loss = (self.daily_start_value - portfolio_value) / self.daily_start_value
        if daily_loss > self.max_daily_loss:
            warnings.append(f"Max daily loss exceeded: {daily_loss:.2%}")
            # Reduce position sizes
            modified_positions *= 0.5
        
        # Check position sizes
        max_pos = jnp.max(jnp.abs(modified_positions))
        if max_pos > self.max_position_size:
            warnings.append(f"Position size too large: {max_pos:.2%}")
            # Scale down positions
            scale_factor = self.max_position_size / max_pos
            modified_positions *= scale_factor
        
        # Check leverage (sum of absolute position weights)
        leverage_ratio = jnp.sum(jnp.abs(modified_positions))
        if leverage_ratio > self.max_leverage:
            warnings.append(f"Leverage too high: {leverage_ratio:.2f}x")
            # Scale down to max leverage
            modified_positions *= self.max_leverage / leverage_ratio
        
        # Check volatility
        if volatility > self.volatility_threshold:
            warnings.append(f"High volatility detected: {volatility:.2f}")
            # Reduce position sizes in high volatility
            modified_positions *= 0.7
        
        # Check for daily reset
        current_time = time.time()
        if current_time - self.last_reset_time > 24 * 3600:  # 24 hours
            self.daily_start_value = portfolio_value
            self.last_reset_time = current_time
        
        is_safe = len(warnings) == 0
        
        modified_action = TradingAction(
            positions=modified_positions,
            confidence=action.confidence * (0.5 if warnings else 1.0),
            timestamp=action.timestamp
        )
        
        return is_safe, warnings, modified_action


def train_agent(
    agent: CryptoTradingAgent,
    env: TradingEnvironment,
    n_episodes: int = 1000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    verbose: bool = True
) -> Tuple[CryptoTradingAgent, Dict[str, List[float]]]:
    """
    Train the crypto trading agent using a simple policy gradient method.
    
    Args:
        agent: CryptoTradingAgent to train
        env: Trading environment
        n_episodes: Number of training episodes
        learning_rate: Learning rate for optimization
        gamma: Discount factor
        verbose: Whether to print training progress
    
    Returns:
        Trained agent and training history
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(agent)
    
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': []
    }
    
    for episode in range(n_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # Collect episode data
        states, actions, rewards = [], [], []
        
        done = False
        while not done:
            # Get current state as array
            state_array = jnp.concatenate([
                state.prices,
                state.features,
                state.portfolio,
                jnp.array([state.cash])
            ])
            
            # Get action from agent
            action = agent.get_action(state_array, deterministic=False)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            states.append(state_array)
            actions.append(action.positions)
            rewards.append(reward)
            
            # Update for next iteration
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Convert to arrays
        states = jnp.stack(states)
        actions = jnp.stack(actions)
        rewards = jnp.array(rewards)
        
        # Compute returns (simplified)
        returns = jnp.zeros_like(rewards)
        discounted_sum = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            discounted_sum = rewards[t] + gamma * discounted_sum
            returns = returns.at[t].set(discounted_sum)
        
        # Policy gradient update (simplified)
        def policy_loss_fn(agent):
            values = vmap(agent.get_value)(states)
            advantages = returns - values
            
            # Policy loss (simplified REINFORCE)
            policy_scores = vmap(agent.policy_network)(states)
            log_probs = -0.5 * jnp.sum((policy_scores - actions) ** 2, axis=1)
            policy_loss = -jnp.mean(log_probs * advantages)
            
            # Value loss
            value_loss = jnp.mean((values - returns) ** 2)
            
            return policy_loss + value_loss
        
        # Compute gradients and update
        loss, grads = eqx.filter_value_and_grad(policy_loss_fn)(agent)
        updates, opt_state = optimizer.update(grads, opt_state)
        agent = eqx.apply_updates(agent, updates)
        
        # Record history
        history['episode_rewards'].append(float(episode_reward))
        history['episode_lengths'].append(episode_length)
        history['policy_losses'].append(float(loss))
        
        if verbose and episode % 100 == 0:
            avg_reward = jnp.mean(jnp.array(history['episode_rewards'][-100:]))
            logger.info(f"Episode {episode}: avg_reward={avg_reward:.4f}, loss={loss:.6f}")
    
    return agent, history


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create environment
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    env = TradingEnvironment(symbols)
    
    # Create agent
    state_dim = len(symbols) + 20 + len(symbols) + 1  # prices + features + portfolio + cash
    action_dim = len(symbols)
    
    agent = CryptoTradingAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        key=jax.random.PRNGKey(42)
    )
    
    # Quick test
    state = env.reset()
    state_array = jnp.concatenate([
        state.prices,
        state.features,
        state.portfolio,
        jnp.array([state.cash])
    ])
    
    action = agent.get_action(state_array)
    logger.info(f"Generated action: positions={action.positions}, confidence={action.confidence}")
    
    # Risk manager test
    risk_manager = RiskManager()
    is_safe, warnings, modified_action = risk_manager.check_risk_limits(
        action, 100000.0, 0.5
    )
    
    logger.info(f"Risk check: safe={is_safe}, warnings={warnings}")
    logger.info("Crypto trading agent initialized successfully")