import numpy as np
import pandas as pd
from typing import Tuple, Dict

class TradingEnvironment:
    """
    A trading environment for a long-only trading agent.

    The environment simulates market interactions including commissions, slippage, and latency.
    """
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, commission_rate: float = 0.001, latency: int = 1, profit_bonus: float = 1.0, drawdown_penalty: float = 0.1):
        self.df = df
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.latency = latency
        self.profit_bonus = profit_bonus
        self.drawdown_penalty = drawdown_penalty
        self.feature_columns = [col for col in df.columns if col not in ['timestamp', 'close']]
        self.n_features = len(self.feature_columns) + 5  # +1 for position_status, +4 for new positional features

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

    def _take_action(self, action: float) -> Tuple[float, float]:
        """
        Executes a trading action based on a continuous value.
        Action > 0: BUY (proportional to value)
        Action < 0: SELL (proportional to value)
        """
        execution_price = 0.0
        entry_price_for_cycle = 0.0
        current_price = self._get_current_price()
        action_magnitude = abs(action)

        if action > 0 and self.position_status == 0:  # BUY
            if self.balance > 0:
                # action_magnitude is the percentage of balance to invest
                amount_to_invest = self.balance * action_magnitude
                self.entry_price = current_price
                shares_to_buy = amount_to_invest / self.entry_price
                cost = shares_to_buy * self.entry_price
                commission = cost * self.commission_rate

                self.shares_held = shares_to_buy
                self.balance -= (cost + commission)
                self.position_status = 1
                self.steps_in_trade = 0
                self.high_water_mark = self.entry_price
                execution_price = self.entry_price

        elif action < 0 and self.position_status == 1:  # SELL
            if self.shares_held > 0:
                # action_magnitude is the percentage of shares to sell
                shares_to_sell = self.shares_held * action_magnitude
                sale_value = shares_to_sell * current_price
                commission = sale_value * self.commission_rate

                self.balance += (sale_value - commission)
                self.shares_held -= shares_to_sell
                execution_price = current_price
                entry_price_for_cycle = self.entry_price

                # If we've sold all or nearly all shares, reset the position
                if self.shares_held < 1e-6: # Use a small threshold for float comparison
                    self.shares_held = 0
                    self.position_status = 0
                    self.entry_price = 0.0
                    self.steps_in_trade = 0
                    self.high_water_mark = 0.0

        return execution_price, entry_price_for_cycle

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Performs one step in the environment.
        """
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

    def _get_reward(self, action: float, execution_price: float, entry_price_for_cycle: float) -> float:
        """
        Calculates the reward using the "Realized P&L + Drawdown Penalty" model.
        """
        reward = 0.0

        # Profit-Taking Incentive: reward realized profits from SELL actions
        if action == 2 and entry_price_for_cycle > 0:
            realized_pnl = (execution_price - entry_price_for_cycle) / entry_price_for_cycle
            if realized_pnl > 0:
                reward += realized_pnl * self.profit_bonus

        # Risk Management Incentive: penalize drawdown from high-water mark
        if self.position_status == 1 and self.high_water_mark > 0:
            current_price = self._get_current_price()
            drawdown = (self.high_water_mark - current_price) / self.high_water_mark
            reward -= drawdown * self.drawdown_penalty

        return reward