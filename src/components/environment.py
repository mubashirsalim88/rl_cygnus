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
        Initialize the trading environment with discrete action space and state-dependent actions.

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

        # Define discrete action space: {0: HOLD, 1: BUY, 2: SELL}
        self.action_space_size = 3

        # Composite reward function weights
        self.reward_weights = {'cycle': 1.0, 'time': 0.001, 'unrealized': 0.01, 'drawdown': 0.1, 'opportunity': 0.1}

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

        # New state-tracking variables for complete trade cycles
        self.position_status = 0  # 0 for flat, 1 for long
        self.entry_price = 0.0
        self.steps_in_trade = 0
        self.high_water_mark = 0.0  # Peak price during the current trade

        # Trading history for analysis
        self.history: List[Dict[str, Any]] = []

        # Get feature columns (exclude 'close' as it's used for pricing)
        self.feature_columns = [col for col in self.df.columns if col != 'close']
        # The number of features is the original count + 1 (for position_status)
        self.n_features = len(self.feature_columns) + 1

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

        # Reset new state-tracking variables for complete trade cycles
        self.position_status = 0  # 0 for flat, 1 for long
        self.entry_price = 0.0
        self.steps_in_trade = 0
        self.high_water_mark = 0.0  # Peak price during the current trade

        # Clear trading history
        self.history = []

        # Return initial observation (will be from step 0 due to latency handling)
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment with discrete actions and latency-delayed observations.

        Args:
            action: Integer action {0: HOLD, 1: BUY, 2: SELL}

        Returns:
            Tuple of (next_observation, reward, done, info)

        Note:
            The agent executes trades based on current market prices at step t,
            but receives observations from step (t+1-latency) as next_observation,
            simulating realistic market data delays.
        """
        # Validate action is integer and in valid range
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer, got {type(action)}")

        # Clamp action to valid discrete range
        action = int(np.clip(action, 0, 2))

        # Store previous portfolio value for reward calculation
        self.previous_portfolio_value = self.portfolio_value

        # Execute the trading action and get execution price and entry price for cycle reward
        execution_price, entry_price_for_cycle = self._take_action(action)

        # Calculate reward based on composite reward function
        reward = self._get_reward(action, execution_price, entry_price_for_cycle)

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
            'step': self.current_step,
            'position_status': self.position_status,
            'entry_price': self.entry_price,
            'steps_in_trade': self.steps_in_trade
        }

        return next_observation, reward, done, info

    def _take_action(self, action: int) -> Tuple[float, float]:
        """
        Execute a state-dependent discrete trading action.

        Args:
            action: Integer action {0: HOLD, 1: BUY, 2: SELL}

        Returns:
            tuple: (execution_price, entry_price_for_cycle_reward)
                execution_price: The price at which the trade was executed (0 for HOLD or invalid actions)
                entry_price_for_cycle_reward: Entry price for completed trade (0 if no trade completed)

        Note:
            Actions are state-dependent:
            - BUY (1): Only valid when flat (position_status == 0), uses 100% of balance
            - SELL (2): Only valid when in position (position_status == 1), sells 100% of shares
            - HOLD (0): Always valid, does nothing
        """
        current_price = self._get_current_price()
        execution_price = 0.0
        entry_price_for_cycle = 0.0

        if action == 1:  # BUY
            # Only execute if agent is currently flat
            if self.position_status == 0 and self.balance > 0:
                # Use 100% of available balance for buying
                available_balance = self.balance

                # First estimate shares we can afford
                initial_shares_estimate = available_balance / (current_price * (1 + self.commission_rate))

                # Get execution price with slippage for this estimate
                execution_price = self._get_execution_price(initial_shares_estimate, 'buy')

                # Recalculate actual affordable shares with slippage-adjusted price
                shares_to_buy = available_balance / (execution_price * (1 + self.commission_rate))

                # Get final execution price for actual trade size
                final_execution_price = self._get_execution_price(shares_to_buy, 'buy')

                trade_value = shares_to_buy * final_execution_price
                commission_cost = trade_value * self.commission_rate

                # Execute the trade
                self.shares_held += shares_to_buy
                self.balance -= (trade_value + commission_cost)

                # Update state tracking for new position
                self.position_status = 1
                self.entry_price = final_execution_price
                self.high_water_mark = final_execution_price
                self.steps_in_trade = 0

                execution_price = final_execution_price

        elif action == 2:  # SELL
            # Only execute if agent is in a position
            if self.position_status == 1 and self.shares_held > 0:
                # Store entry price before resetting for cycle reward calculation
                entry_price_for_cycle = self.entry_price

                # Sell 100% of currently held shares
                shares_to_sell = self.shares_held

                # Get execution price with slippage
                execution_price = self._get_execution_price(shares_to_sell, 'sell')

                sale_value = shares_to_sell * execution_price
                commission_cost = sale_value * self.commission_rate

                # Execute the trade
                self.balance += (sale_value - commission_cost)
                self.shares_held = 0.0

                # Reset all cycle trackers
                self.position_status = 0
                self.entry_price = 0.0
                self.steps_in_trade = 0
                self.high_water_mark = 0.0

        # For HOLD action (0) or invalid actions, do nothing

        # Update portfolio value (use current market price, not execution price)
        self.portfolio_value = self.balance + (self.shares_held * current_price)

        # Update trade state if in position
        if self.position_status == 1:
            self.steps_in_trade += 1
            # Update high water mark during the trade
            if current_price > self.high_water_mark:
                self.high_water_mark = current_price

        return execution_price, entry_price_for_cycle

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

    def _get_reward(self, action: int, execution_price: float, entry_price_for_cycle: float) -> float:
        """
        Calculate composite reward using multi-objective reward function.
        """
        # Initialize all reward components to 0
        r_cycle, r_time, r_unrealized, r_drawdown, r_opportunity, r_entry = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # R(entry) - NEW: Add a small incentive for successfully taking a BUY action
        if action == 1 and self.position_status == 1:  # A successful buy just occurred
            r_entry = 0.01  # Small bonus for entering a trade

        # R(cycle) - Event-Driven, on SELL only
        if action == 2 and execution_price > 0 and entry_price_for_cycle > 0:
            transaction_cost = 2 * np.log(1 - self.commission_rate)
            r_cycle = (np.log(execution_price) - np.log(entry_price_for_cycle)) + transaction_cost

        # Shaping Rewards - Per-Step, only when IN a position
        if self.position_status == 1:
            r_time = -np.log(1 + self.steps_in_trade)
            current_price = self._get_current_price()
            unrealized_return = (current_price - self.entry_price) / self.entry_price
            r_unrealized = unrealized_return
            drawdown = (self.high_water_mark - current_price) / self.high_water_mark
            r_drawdown = -drawdown if drawdown > 0 else 0.0

        # R(opportunity) - Penalty for missing opportunities during clear uptrends
        if self.position_status == 0:
            current_price = self._get_current_price()
            if 'EMA_200' in self.df.columns and self.current_step > 0:
                try:
                    ema_200 = self.df.iloc[self.current_step]['EMA_200']
                    prev_price = self.df.iloc[self.current_step - 1]['close']
                    log_return = np.log(current_price / prev_price)
                    if current_price > ema_200 and log_return > 0:
                        # MODIFIED: The penalty is now directly negative
                        r_opportunity = -log_return
                except (IndexError, KeyError):
                    r_opportunity = 0.0

        # NEW: Adjust reward weights to prioritize entry and opportunity cost
        self.reward_weights = {
            'cycle': 1.0, 'time': 0.001, 'unrealized': 0.01,
            'drawdown': 0.1, 'opportunity': 0.5, 'entry': 1.0
        }

        # MODIFIED: Update the final composite reward calculation
        final_reward = (
            self.reward_weights['cycle'] * r_cycle +
            self.reward_weights['time'] * r_time +
            self.reward_weights['unrealized'] * r_unrealized +
            self.reward_weights['drawdown'] * r_drawdown +
            self.reward_weights['opportunity'] * r_opportunity +
            self.reward_weights['entry'] * r_entry
        )

        return final_reward

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

        # NEW: Append the current position status to the feature vector
        # self.position_status is 0 for flat, 1 for long
        full_observation = np.append(observation, self.position_status)

        return full_observation.astype(np.float32)

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