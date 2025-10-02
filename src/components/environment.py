import numpy as np
import pandas as pd
from typing import Tuple, Dict
import gymnasium as gym

class DifferentialSortinoRatio:
    """Calculates the differential Sortino ratio for a dense, per-step reward."""
    def __init__(self, eta=0.1, target_return=0.0):
        self.eta = eta
        self.target_return = target_return
        self.A = 0.0  # EMA of downside returns
        self.B = 0.0  # EMA of squared downside returns
        self.t = 0

    def step(self, portfolio_return: float) -> float:
        dsr = 0.0
        # Only consider downside returns (returns below target)
        downside_return = min(0, portfolio_return - self.target_return)

        if self.t > 0:
            numerator = (self.B - self.A**2)
            if numerator > 1e-8: # Avoid division by zero
                term1 = self.B * (downside_return - self.A)
                term2 = 0.5 * self.A * (downside_return**2 - self.B)
                # We use the negative because we want to MAXIMIZE the Sortino ratio
                # by MINIMIZING the negative reward.
                dsr = -(term1 - term2) / (numerator**1.5)

        # Update moments using ONLY the downside return
        self.A += self.eta * (downside_return - self.A)
        self.B += self.eta * (downside_return**2 - self.B)
        self.t += 1

        return dsr

class TradingEnvironment:
    """
    A trading environment for a long-only trading agent.

    The environment simulates market interactions including commissions, slippage, and latency.
    """
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, commission_rate: float = 0.001, latency: int = 1, profit_bonus: float = 1.0, drawdown_penalty: float = 0.1, trade_size_percentage: float = 0.1):
        self.df = df
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.latency = latency
        self.profit_bonus = profit_bonus
        self.drawdown_penalty = drawdown_penalty
        self.trade_size_percentage = trade_size_percentage
        self.feature_columns = [col for col in df.columns if col not in ['timestamp', 'close']]
        self.n_features = len(self.feature_columns) + 5  # +1 for position_status, +4 for positional features

        # Define the discrete action space
        self.action_space = gym.spaces.Discrete(3)  # 0: SELL, 1: HOLD, 2: BUY

        self.dsr_calculator = DifferentialSortinoRatio()
        self.reset()

    def reset(self) -> np.ndarray:
        """Resets the environment to its initial state."""
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.portfolio_value = self.initial_balance
        self.current_step = 0

        # Trade state variables
        self.position_status = 0  # 0: flat, 1: long
        self.entry_price = 0.0
        self.steps_in_trade = 0
        self.high_water_mark = 0.0

        self.dsr_calculator = DifferentialSortinoRatio()
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get the observation for the current step, considering latency."""
        observation_step = max(0, self.current_step - self.latency)
        if observation_step >= len(self.df):
            # Return zeros with position_status and positional features appended
            base_observation = np.zeros(len(self.feature_columns))
            full_observation = np.append(base_observation, self.position_status)
            positional_features = np.zeros(4)  # P&L, Duration, Drawdown, Norm. Entry Price
            full_observation = np.append(full_observation, positional_features)
            return full_observation.astype(np.float32)

        observation = self.df[self.feature_columns].iloc[observation_step].values
        observation = np.nan_to_num(np.asarray(observation), nan=0.0)

        # Calculate positional features
        current_price = self._get_current_price()
        positional_features = np.zeros(4)  # P&L, Duration, Drawdown, Norm. Entry Price

        if self.position_status == 1 and self.entry_price > 0:
            # Unrealized P&L: percentage profit or loss
            positional_features[0] = (current_price - self.entry_price) / self.entry_price
            # Position duration: number of steps in trade
            positional_features[1] = self.steps_in_trade
            # Drawdown from high-water mark
            if self.high_water_mark > 0:
                positional_features[2] = (self.high_water_mark - current_price) / self.high_water_mark
            # Normalized entry price: entry price relative to current price
            positional_features[3] = self.entry_price / current_price

        # Append position_status and positional features to the observation
        full_observation = np.append(observation, self.position_status)
        full_observation = np.append(full_observation, positional_features)
        return full_observation.astype(np.float32)

    def _get_current_price(self) -> float:
        """Get the 'close' price for the current step."""
        if self.current_step >= len(self.df):
            return self.df['close'].iloc[-1]
        return self.df['close'].iloc[self.current_step]

    def _take_action(self, action: int) -> Tuple[float, float]:
        """
        Executes a trading action based on a discrete integer.
        Action 0: SELL
        Action 1: HOLD
        Action 2: BUY
        """
        execution_price = 0.0
        entry_price_for_cycle = 0.0
        current_price = self._get_current_price()

        if action == 2 and self.position_status == 0:  # BUY
            if self.balance > 0:
                amount_to_invest = self.balance * self.trade_size_percentage
                self.entry_price = current_price
                shares_to_buy = amount_to_invest / self.entry_price
                cost = shares_to_buy * self.entry_price
                commission = cost * self.commission_rate

                if self.balance >= (cost + commission):
                    self.shares_held = shares_to_buy
                    self.balance -= (cost + commission)
                    self.position_status = 1
                    self.steps_in_trade = 0
                    self.high_water_mark = self.entry_price
                    execution_price = self.entry_price

        elif action == 0 and self.position_status == 1:  # SELL
            if self.shares_held > 0:
                shares_to_sell = self.shares_held * self.trade_size_percentage
                sale_value = shares_to_sell * current_price
                commission = sale_value * self.commission_rate

                self.balance += (sale_value - commission)
                self.shares_held -= shares_to_sell
                execution_price = current_price
                entry_price_for_cycle = self.entry_price

                if self.shares_held < 1e-6:
                    self.shares_held = 0
                    self.position_status = 0
                    self.entry_price = 0.0
                    self.steps_in_trade = 0
                    self.high_water_mark = 0.0

        # If action is 1 (HOLD), do nothing.
        return execution_price, entry_price_for_cycle

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Performs one step in the environment.
        """
        # The action is now an integer, no clipping needed.

        # Store previous portfolio value for reward calculation
        self.previous_portfolio_value = self.portfolio_value

        execution_price, entry_price_for_cycle = self._take_action(action)

        # Update portfolio value and trade-specific states
        self.portfolio_value = self.balance + self.shares_held * self._get_current_price()
        if self.position_status == 1:
            self.steps_in_trade += 1
            self.high_water_mark = max(self.high_water_mark, self._get_current_price())

        # Get reward and check if done
        reward = self._get_reward(action, execution_price, entry_price_for_cycle)
        done = self.current_step >= len(self.df) - 2

        # Prepare comprehensive info dictionary
        market_return = ((self._get_current_price() - self.df['close'].iloc[0]) / self.df['close'].iloc[0]) * 100
        agent_return = ((self.portfolio_value - self.initial_balance) / self.initial_balance) * 100

        info = {
            'step': self.current_step,
            'action': action,
            'reward': reward,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'portfolio_value': self.portfolio_value,
            'price': self._get_current_price(),
            'timestamp': self.df['timestamp'].iloc[self.current_step],
            'market_return': market_return,
            'agent_return': agent_return
        }

        # Move to the next step
        self.current_step += 1
        observation = self._get_observation()

        return observation, reward, done, info

    def _get_reward(self, action: int, execution_price: float, entry_price_for_cycle: float) -> float:
        """
        Calculates the reward using the Differential Sortino Ratio.
        """
        # Calculate portfolio return for the current step
        # Ensure previous_portfolio_value is not zero to avoid division errors
        if self.previous_portfolio_value > 1e-8:
            portfolio_return = (self.portfolio_value / self.previous_portfolio_value) - 1
        else:
            portfolio_return = 0.0

        # Get the dense, risk-adjusted reward from the DSR calculator
        reward = self.dsr_calculator.step(portfolio_return)

        return reward