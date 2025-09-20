import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional


class TradingEnvironment:
    """
    A comprehensive trading environment for reinforcement learning with realistic market conditions.

    This environment simulates cryptocurrency trading where an agent can take
    actions (HOLD, BUY, SELL) and receive rewards based on portfolio performance.
    The environment uses processed feature data, maintains trading state, and
    includes realistic market frictions:
    - Transaction costs through configurable commission rates
    - Volume-based price slippage using a square-root market impact model
    - Market data latency to simulate realistic information delays

    The latency model ensures agents operate on delayed market observations while
    executing trades at current market prices, mimicking real-world trading conditions.
    """

    # Action space constants
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, commission_rate: float = 0.001, slippage_factor: float = 0.0005, latency: int = 2):
        """
        Initialize the trading environment with transaction costs, slippage, and latency.

        Args:
            df: DataFrame containing processed features from FeatureEngine
            initial_balance: Starting cash balance for trading
            commission_rate: Commission rate for trades (default: 0.001 = 0.1%)
            slippage_factor: Factor determining slippage magnitude (default: 0.0005 = 0.05%)
            latency: Observation delay in time steps (default: 2 steps)
        """
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_factor = slippage_factor
        self.latency = latency

        # Validate DataFrame contains required columns
        if 'close' not in self.df.columns:
            raise ValueError("DataFrame must contain 'close' price column")

        # Remove any rows with NaN values that could break the environment
        self.df = self.df.dropna()

        if len(self.df) == 0:
            raise ValueError("DataFrame is empty after removing NaN values")

        # Environment state variables
        self.current_step: int = 0
        self.balance: float = initial_balance
        self.shares_held: float = 0.0
        self.portfolio_value: float = initial_balance
        self.previous_portfolio_value: float = initial_balance

        # Trading history for analysis
        self.history: List[Dict[str, Any]] = []

        # Get feature columns (exclude 'close' as it's used for pricing)
        self.feature_columns = [col for col in self.df.columns if col != 'close']
        self.n_features = len(self.feature_columns)

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation (feature vector respecting latency delay)

        Note:
            Due to latency, the initial observation will be from step 0, representing
            the oldest available data when starting trading at step 0.
        """
        # Reset all state variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance

        # Clear trading history
        self.history = []

        # Return initial observation (will be from step 0 due to latency handling)
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment with latency-delayed observations.

        Args:
            action: Action to take (0=HOLD, 1=BUY, 2=SELL)

        Returns:
            Tuple of (next_observation, reward, done, info)

        Note:
            The agent executes trades based on current market prices at step t,
            but receives observations from step (t+1-latency) as next_observation,
            simulating realistic market data delays.
        """
        # Validate action
        if action not in [self.HOLD, self.BUY, self.SELL]:
            raise ValueError(f"Invalid action: {action}. Must be 0 (HOLD), 1 (BUY), or 2 (SELL)")

        # Store previous portfolio value for reward calculation
        self.previous_portfolio_value = self.portfolio_value

        # Execute the trading action
        self._take_action(action)

        # Calculate reward based on portfolio performance
        reward = self._get_reward()

        # Check if episode is done (reached end of data)
        done = self.current_step >= len(self.df) - 1

        # Record this step in history
        self._record_step(action, reward)

        # Move to next time step
        if not done:
            self.current_step += 1

        # Get next observation
        next_observation = self._get_observation() if not done else np.zeros(self.n_features)

        # Prepare info dictionary
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': self.portfolio_value,
            'current_price': self._get_current_price(),
            'step': self.current_step
        }

        return next_observation, reward, done, info

    def _take_action(self, action: int) -> None:
        """
        Execute a trading action with transaction costs and slippage.

        Args:
            action: Action to take (0=HOLD, 1=BUY, 2=SELL)

        Note:
            All BUY and SELL actions incur commission costs and slippage penalties.
            Slippage is calculated based on trade size relative to recent volume.
        """
        current_price = self._get_current_price()

        if action == self.BUY:
            # Buy as many shares as possible with current balance, accounting for commission and slippage
            if self.balance > 0:
                # First estimate shares we can afford (iterative approach due to slippage dependency)
                # Start with no-slippage estimate
                initial_shares_estimate = self.balance / (current_price * (1 + self.commission_rate))

                # Get execution price with slippage for this estimate
                execution_price = self._get_execution_price(initial_shares_estimate, 'buy')

                # Recalculate actual affordable shares with slippage-adjusted price
                shares_to_buy = self.balance / (execution_price * (1 + self.commission_rate))

                # Get final execution price for actual trade size
                final_execution_price = self._get_execution_price(shares_to_buy, 'buy')

                trade_value = shares_to_buy * final_execution_price
                commission_cost = trade_value * self.commission_rate

                self.shares_held += shares_to_buy
                self.balance -= (trade_value + commission_cost)

        elif action == self.SELL:
            # Sell all held shares, accounting for slippage and deducting commission from proceeds
            if self.shares_held > 0:
                # Get execution price with slippage for the number of shares we're selling
                execution_price = self._get_execution_price(self.shares_held, 'sell')

                sale_value = self.shares_held * execution_price
                commission_cost = sale_value * self.commission_rate

                self.balance += (sale_value - commission_cost)
                self.shares_held = 0.0

        # For HOLD action, no changes to balance or shares

        # Update portfolio value (use current market price, not execution price)
        self.portfolio_value = self.balance + (self.shares_held * current_price)

    def _get_execution_price(self, shares_traded: float, trade_type: str) -> float:
        """
        Calculate the execution price accounting for volume-based slippage.

        Args:
            shares_traded: Number of shares being traded
            trade_type: Either 'buy' or 'sell' to determine penalty direction

        Returns:
            Execution price adjusted for slippage

        Note:
            Uses a square-root slippage model based on trade size relative to daily volume.
            Larger trades relative to volume incur higher slippage penalties.
        """
        current_price = self._get_current_price()

        # Calculate average daily volume (rolling 24-period average)
        # Use a minimum window to ensure we have enough data
        volume_window = min(24, self.current_step + 1)
        if volume_window < 1:
            volume_window = 1

        start_idx = max(0, self.current_step - volume_window + 1)
        end_idx = self.current_step + 1

        daily_volume = self.df['volume'].iloc[start_idx:end_idx].mean()

        # Prevent division by zero
        if daily_volume <= 0:
            daily_volume = self.df['volume'].mean()  # Fallback to overall average
            if daily_volume <= 0:
                daily_volume = 1.0  # Final fallback

        # Calculate slippage penalty using square-root model
        volume_ratio = shares_traded / daily_volume
        penalty = self.slippage_factor * (volume_ratio ** 0.5)

        # Apply penalty based on trade direction
        if trade_type.lower() == 'buy':
            # Buy orders push price up
            execution_price = current_price * (1 + penalty)
        elif trade_type.lower() == 'sell':
            # Sell orders push price down
            execution_price = current_price * (1 - penalty)
        else:
            raise ValueError(f"Invalid trade_type: {trade_type}. Must be 'buy' or 'sell'")

        return execution_price

    def _get_reward(self) -> float:
        """
        Calculate the reward for the current step.

        Returns:
            Reward based on change in portfolio value
        """
        # Simple reward: change in portfolio value
        portfolio_change = self.portfolio_value - self.previous_portfolio_value

        # Normalize by initial balance to make rewards scale-independent
        normalized_reward = portfolio_change / self.initial_balance

        return normalized_reward

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (feature vector) with latency delay.

        Returns:
            Feature vector for current time step minus latency delay

        Note:
            Agent receives observations from (current_step - latency) to simulate
            realistic market data delays. For early steps where this would be negative,
            returns observation from step 0.
        """
        # Calculate the observation step accounting for latency
        observation_step = max(0, self.current_step - self.latency)

        if observation_step >= len(self.df):
            return np.zeros(self.n_features)

        # Get features for delayed time step
        observation = self.df[self.feature_columns].iloc[observation_step].values

        # Handle any remaining NaN values
        observation = np.nan_to_num(np.asarray(observation), nan=0.0)

        return observation.astype(np.float32)

    def _get_current_price(self) -> float:
        """
        Get the current close price.

        Returns:
            Current close price
        """
        if self.current_step >= len(self.df):
            return self.df['close'].iloc[-1]

        return self.df['close'].iloc[self.current_step]

    def _record_step(self, action: int, reward: float) -> None:
        """
        Record the current step in trading history.

        Args:
            action: Action taken
            reward: Reward received
        """
        step_record = {
            'step': self.current_step,
            'action': action,
            'reward': reward,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': self.portfolio_value,
            'price': self._get_current_price(),
            'timestamp': self.df.index[self.current_step] if self.current_step < len(self.df) else None
        }
        self.history.append(step_record)

    def get_action_meanings(self) -> List[str]:
        """
        Get human-readable meanings for actions.

        Returns:
            List of action meanings
        """
        return ['HOLD', 'BUY', 'SELL']

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current portfolio state.

        Returns:
            Dictionary containing portfolio metrics
        """
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        current_price = self._get_current_price()

        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'portfolio_value': self.portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'steps_taken': self.current_step,
            'total_steps': len(self.df)
        }

    def get_trading_history(self) -> pd.DataFrame:
        """
        Get trading history as a DataFrame.

        Returns:
            DataFrame containing complete trading history
        """
        if not self.history:
            return pd.DataFrame()

        return pd.DataFrame(self.history)