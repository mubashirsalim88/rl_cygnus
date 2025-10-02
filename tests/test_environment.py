import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from src.components.environment import TradingEnvironment

@pytest.fixture
def sample_dataframe():
    """Creates a sample DataFrame for testing."""
    data = {
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'close': [100.0, 105.0, 102.0],
        'feature1': [0.5, 0.6, 0.4]
    }
    return pd.DataFrame(data)

@pytest.fixture
def trading_env(sample_dataframe):
    """Creates a TradingEnvironment instance for testing."""
    return TradingEnvironment(df=sample_dataframe, initial_balance=1000.0, trade_size_percentage=0.1)

def test_environment_initialization(trading_env):
    """Test that the environment initializes with correct values."""
    assert trading_env.balance == 1000.0
    assert trading_env.shares_held == 0.0
    assert trading_env.position_status == 0

def test_buy_action(trading_env):
    """Test a single BUY action and its effect on the environment state."""
    initial_balance = trading_env.balance
    price_at_step_0 = 100.0 # The price at which the buy occurs

    # Action 2 is BUY
    action = 2
    observation, reward, done, info = trading_env.step(action)

    # Assert that balance has decreased (due to cost + commission)
    assert trading_env.balance < initial_balance
    # Assert that shares are now held
    assert trading_env.shares_held > 0
    # Assert that position status is now 1 (long)
    assert trading_env.position_status == 1

    # The portfolio value is updated using the price from the CURRENT step (step 0).
    expected_portfolio_value = trading_env.balance + (trading_env.shares_held * price_at_step_0)

    # Assert portfolio value is calculated correctly
    assert np.isclose(trading_env.portfolio_value, expected_portfolio_value)
